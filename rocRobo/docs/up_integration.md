# Upstream Integration — rocRobo as a motion backend for RoboSmith

> 目的：把 rocRobo 作为**独立子模块**引入 RoboSmith（Genesis SDG 项目），为其 pick/place/open/close
> 提供 collision-free IK + 避障轨迹。本文从**消费方（RoboSmith）视角**列出当前 gap 与 rocRobo 侧的
> API 封装建议，作为两边对接锚点。rocRobo 内核（IK→trajopt→retiming→plan_batch，见
> [design.md](../design.md)）已成熟；缺的是"可被外部项目干净消费"的薄封装。
>
> 触发：RoboSmith 侧确认 pick/place 失败主因是 **机械手连杆撞打开的抽屉**——场景级避障，
> 手调位姿改不动，正是 rocRobo 的领域。
>
> **当前能力（2026-06-12 全部就绪并验证）**：
> - `solve_ik(pos, quat, *, seed, world) -> (q, ok, quality)`：collision-aware IK seed，world 按调用注入，多回可行性 `quality`。
> - `plan_motion(q_start, waypoints, world) -> PlanResult` + `plan(start,goal,world)`：world **贯通到 trajopt**（注入障碍后优化器把整条臂绕开、已验证）。
> - 可 `import rocrobo`（`MotionGen / PlanConfig / PlanResult / BatchResult`）。
> - 进程边界：`python -m rocrobo.serve --serve` **常驻 warm 进程**（JIT 编译一次后常热，JSON 行收发），`--once` 作 smoke test。
>
> 唯一未做：**机器人/EE 注入（`RobotSpec`，G2）仍默认 Franka**——见下表与 §3A。RoboSmith 可立即开工接入。

## 1. 消费场景（RoboSmith 怎么用）

RoboSmith 段级闭环 runtime：每个 subtask 读实时场景 + 关节态 → 规划该段。它需要 rocRobo 回答两件事：

- **collision-aware IK seed**：给一个 EE 目标位姿 + 当前场景，返回一个**无碰撞**的关节解（替换它现在的
  Genesis 直线 IK，治"pick 接近时撞抽屉"）。
- **避障轨迹**：给起点关节 + 一串 EE 路点 + 场景，返回避开场景几何的关节轨迹。

关键：**场景每段都变**（抽屉按当前 joint 行程伸出不同长度），所以"世界"必须是**每次调用的入参**。

## 2. 四个 gap（决定封装要补什么）

| # | gap | 影响 | 状态 |
|---|---|---|---|
| G1 | **世界是 module-global，不是入参** | RoboSmith 传不进实时抽屉包络 → 无法消费（**致命**）| ✅ **已闭合**：world 贯通 IK + trajopt + evaluate（`solve_ik`/`plan`/`plan_motion`/`plan_batch` 均收 `world` 入参；注入障碍后整条臂绕开已验证）|
| G2 | **机器人/EE 不可注入** | 需与 RoboSmith 的 Franka 帧/DOF/quat 对齐，否则解错 | 🟡 仍默认 Franka；契约已对齐（wxyz/米/7 臂 DOF），`RobotSpec` 注入未做（RoboSmith 当前 Franka，不阻塞）|
| G3 | **两套栈不共进程** | 一进程同时 import jax-rocm + torch-rocm 不可行 → 必须进程/服务边界 | ✅ **已闭合**：`rocrobo.serve` 常驻 warm 进程（warm 调用较冷启动 ~24× 提速已验证）+ `--once` smoke test |
| G4 | **代码非可 import 包** | 外部项目无法 `import rocrobo` 稳定调用 | ✅ **已闭合**：`import rocrobo` 包就绪（回归无变化）|

## 3. rocRobo API 封装建议（按优先级）

### A. 世界与机器人提为显式入参（最高优先，治 G1/G2）

把 v0 里的全局世界/机器人提到参数：

```python
world = World(primitives=[Cuboid(half_extent, pose), Sphere(r, pose), Capsule(...)])
robot = RobotSpec(urdf_path, ee_link, joint_limits)     # 默认 Franka，可注入（待做）
mg    = MotionGen(robot, defaults=...)                  # 一次构建，复用 JIT 编译
res   = mg.plan(start_pose, goal_pose, world=world)     # ✅ world 贯通 trajopt
res   = mg.plan_motion(q_start, waypoints, world=world) # ✅ 多路点避障
q, ok, quality = mg.solve_ik(pos, quat, seed=q0, world=world)  # ✅ 已就绪
```

`World` 复用 rocRobo 已有的"图元世界 SDF"，只是从全局提为入参。这是消费的**前提**。
**已就绪**：`world` 是所有规划入口的 per-call traced 入参（同结构不同数值不重编译），且**真进 trajopt 代价**——注入挡路障碍后优化器把整条臂绕开（非仅挪 EE）已验证。当前 world 图元支持 box / halfspace（抽屉包络 + 地面够用），robot 仍默认 Franka（G2）。

> **`seed` 语义（对消费方很关键）**：rocRobo IK **默认内部多种子**（readme C2：单 seed≈52%，N=16→100%），
> `seed=q0` 只是**可选 warm-start 提示**，不是唯一种子。含义：消费方**不应**自己手动预摆种子来绕局部极小
> （RoboSmith 现在的 `seed[0]=arctan2(y,x)` base 预瞄就是单种子 IK 的人工补丁），多种子由后端系统化解决。
> 建议把"默认多种子 + seed 仅 warm-start"写进契约，避免消费方误以为是 single-seed 而保留旧补丁。

### B. 两个入口对齐 RoboSmith 的 `MotionPlanner` 协议

- `solve_ik(pos, quat, *, seed, world) -> (q, ok, quality)` —— RoboSmith P1 只要这个。✅ 已就绪。
- `plan_motion(q_start, waypoints, world) -> PlanResult` —— RoboSmith P2 要避障轨迹。✅ 已就绪。
- 现有 `plan/plan_batch` 保留为便捷层；`PlanResult(success, traj, metrics, checks, reason)` 契约很好，**勿改**。

**消费方两点补充需求**（来自 RoboSmith 想用 rocRobo 收敛其手写可行性）：

- ✅ **`solve_ik` 已多回可行性质量**：`-> (q, ok, quality)`，`quality = {limit_margin, world_clear, self_clear, pos_err_mm}`，
  与内部 evaluate **同源**、可比可排序、与 `ok` 解耦（是**可比排序量、非绝对安全门**）。RoboSmith 可直接用它给抓取
  候选排序，不必重抄 FK/限位常量。
- **批量可达性查询（次要、生成侧）**：`reachable(poses[N], world) -> (mask[N], quality[N])`，复用
  `plan_batch`/vmap。动机：RoboSmith 想在**布局采样**阶段就把物体放到"机器人够得到且够得好"的点，
  需要对一批候选位姿一次性判可达，避免 N 次串行 CLI 调用。优先级低于 P1/P2，但接口同源。

> **两条 batch 轴别混（都走 vmap，但语义不同）**：
> (a) **种子批**＝同一目标位姿、N 个初值 → 更鲁棒的单解，**已是 `solve_ik` 默认行为**（上 §A 注）；
> (b) **目标批**＝N 个不同位姿各求解 → 给上面 `reachable(poses[N])` 生成侧用，也给 RoboSmith 一次性
> 评一批抓取候选（grasp reachability 过滤）用。消费方需要的是 (b) 的入口；(a) 已隐含在每次 `solve_ik` 里。
> 注：design.md 记多-**种子**(a) 的 GPU 拐点尚未单独压测（既有批量数据是多-**目标**(b)），(b) 路径数据更实。

### C. 进程边界（治 G3，唯一动机是 jax/torch 不共进程；运行态用常驻 warm 进程）

rocRobo 跑在自己的 jax 容器，收 JSON 请求 → 回 JSON，RoboSmith 侧只发请求、**不 import jax**：

```
请求  {robot, world, start_pose, goals|waypoints, exempt}   →
返回  {success, traj, checks, reason}
```

**默认运行态＝常驻 warm worker**（不是"先 one-shot 再说"）：
- **运行态：常驻 warm 进程**，JIT 编译一次后常热，长连读 JSON 行（stdin/socket）。理由：RoboSmith 是
  **批量数据生成管线**（数千 episode，每 episode 多段 × 每段可能批量评候选），且段级闭环**必须在线**
  （每段依赖上一段实现的抽屉行程，躲不开频繁调）。若每次调用都重起进程，**jax 冷启动 + 首次 JIT（数十秒/次）
  × 调用次数**会让管线不可用——所以 JIT 保温是第一天就要的，不是后期优化。
- **one-shot CLI 仅作首次端到端 smoke test**（`docker exec <rocRobo容器> python -m rocrobo.serve < req.json`
  跑通一次接通契约），**不是运行态**。
- warm 进程要兑现红利的两个前提：(1) **请求批量化**——把一段/一批候选合成**一次批量请求**（即 §3B 的
  "目标批"），一次 JIT 摊到 N 个解；(2) **形状稳定**——world 图元数 / batch N 固定或 padding，否则 shape 一变
  就重编译（呼应 §3A"同结构不同数值不重编译"），warm 也白费。

### D. 打包（治 G4）

`f2_e3_plan`/`f3_*`/`rocrobo.py` 收成可 import 的 `rocrobo/` 包，`__init__` 暴露
`MotionGen / World / RobotSpec / PlanResult`。

### E. 契约要钉死的细节

- 世界系 = RoboSmith metric world frame（米）。
- **quat = wxyz**（pyroki/jaxlie 与 RoboSmith `_TOP_DOWN_QUAT` 都是 wxyz，写进契约防回归）。
- 关节顺序与 Franka DOF 对齐（RoboSmith 用 9 维含夹爪；rocRobo 7 臂 DOF，边界处约定补齐方式）。
- **接触豁免**由调用方（RoboSmith）决定：它把"有意接触"的体（把手/推/放目标）从 `world` 里**剔除**后再传，
  即请求里的 `exempt`。rocRobo 不懂任务语义，只对收到的 world 做无碰规划。

## 4. RoboSmith 侧消费方式（仅作 rocRobo 对接参考，细节以 RoboSmith 自己的文档为准）

- `motion/planner.py` 定义 `MotionPlanner` 协议（`solve_ik`/`plan_motion`），与 Genesis IK 平行。
- `motion/rocrobo_backend.py` 适配器：World builder（从 scene_state 实时位姿 + joint 行程出抽屉包络，
  复用其 `resolve_frame` 同源实时位姿；phase-aware 剔除接触体）→ 经 §3C 进程边界调 rocRobo → 回 joint traj。
- 接入点：先把 `select_best_grasp_plan` 的 IK 换成 `solve_ik(..., world)`（直接治臂撞抽屉，轨迹仍
  直线但端点无碰）；其后把 `drag_handle`/place 路点交 `plan_motion`，`execute_motion_chain` 退化为 fallback。

## 5. 落地顺序（建议）

1. ✅ rocRobo **A（world 入参 + `solve_ik`+quality）** + **D 打包（`import rocrobo`）** 已就绪 → RoboSmith 能起最小调用。
2. ✅ rocRobo **C**：`rocrobo.serve` 常驻 warm 进程（+ `--once` smoke test）就绪 → RoboSmith「无碰 IK seed」可联调；
   warm 红利前提（请求批量化 + 形状稳定）见 §3C。
3. ✅ rocRobo **B `plan_motion` + A 贯通 trajopt** 已就绪并验证（注入障碍整臂绕开）→ RoboSmith「避障轨迹」可接入
   open/place/close 全程避障。
4. ⬜ 仅剩 **G2 `RobotSpec` 注入**（非 Franka 时）与 §3B `reachable` 目标批（生成侧），均非当前阻塞，排 backlog。

> **给 RoboSmith agent recheck 的要点**：(1) ✅ §2E 契约已对齐——RoboSmith `_TOP_DOWN_QUAT=[0,1,0,0]` 确为
> wxyz、world frame 用米、Franka 9 维 DOF（7 臂 + 2 指）；rocRobo 7 臂 DOF，边界补齐两指即可。
> (2) `solve_ik(...,world)` 已可联调——确认 RoboSmith 侧能产出 `World(primitives=...)` 实时抽屉包络并剔除
> 接触体（`exempt`）后传入；(3) 进程边界（§3C）：运行态用**常驻 warm 进程**（one-shot CLI 仅 smoke test）——
> RoboSmith 批量数据生成 + 段级在线，per-call JIT 冷启动会拖垮吞吐；确认能把一段/一批候选合成**一次批量请求**
> 且 world/batch 形状稳定，warm 进程才兑现红利。

> **RoboSmith 侧回执（2026-06-12）**：已 review，契约对齐无异议——(1) ✅ 确认；(2) RoboSmith 能从段级闭环
> 实时位姿产出 `World(primitives=...)` 抽屉包络并剔除接触体；(3) 接受**常驻 warm 进程**为运行态、one-shot CLI
> 仅作首次 smoke。**当前阻塞在 rocRobo 侧**：等 §3D 打包（可 `import rocrobo`）+ §3C warm 进程就绪即开工接入；
> RoboSmith 第一步＝把 `select_best_grasp_plan` 的 IK 换成 `solve_ik(...,world)`（无碰 IK seed）。请就绪后通知。

---

## 🔔 rocRobo → RoboSmith 就绪通知（2026-06-12）

**rocRobo 侧此前的全部阻塞已清，RoboSmith 可以开工接入。** 已就绪并验证：

| 能力 | 接口 | 状态 |
|---|---|---|
| 无碰 IK seed + 可行性质量 | `solve_ik(pos, quat, *, seed, world) -> (q, ok, quality)` | ✅ |
| world 贯通轨迹（注入障碍整臂绕开） | `plan(start,goal,world)` / `plan_batch(...,world)` | ✅ |
| 多路点避障轨迹 | `plan_motion(q_start, waypoints, world) -> PlanResult` | ✅ |
| 可 import 包 | `pip install -e core` 后 `from rocrobo import MotionGen, PlanConfig, PlanResult, BatchResult`（或 `PYTHONPATH=.../core`）| ✅ |
| 进程边界（warm 运行态 + smoke） | `python -m rocrobo.serve --serve` / `--once`（JSON 行；`op: solve_ik | plan_motion`）| ✅ |

**契约（钉死）**：quat=wxyz；world frame = 米；rocRobo 7 臂 DOF（RoboSmith 9 维含两指，边界补齐）；world 图元
支持 box / halfspace（抽屉包络 + 地面够用）；接触豁免由 RoboSmith 在传入前从 `world` 剔除（`exempt`）；
`quality` 是**可比排序量、非绝对安全门**（同源 evaluate）。

**warm 进程红利前提**（兑现吞吐）：请求批量化（一段/一批候选合成一次请求）+ world 图元数/batch N 形状稳定（否则重编译）。

**RoboSmith 第一步**：把 `select_best_grasp_plan` 的 IK 换成经 `serve` 的 `solve_ik(...,world)`（治臂撞抽屉）；
其后把 `drag_handle`/place 路点交 `plan_motion`。

**仅剩非阻塞项**：G2 `RobotSpec` 注入（非 Franka 时）、§3B `reachable` 目标批（生成侧）——需要时再排。
对接中如需新增 world 图元类型（sphere/capsule）或调整 serve 协议，回此文档留言即可。

---

## ✅ RoboSmith → rocRobo 开工回执（2026-06-13）

收到就绪通知，已 review，无异议，**RoboSmith 侧启动接入（第一切片 rocrobo_feature1）**（设计见 RoboSmith `docs/features/rocrobo_integration/rocrobo_feature1_collision_aware_ik.md`，母路线图 `rocrobo_integration_design.md`）。

- 契约确认：quat=wxyz / world frame=米 / rocRobo 7 臂 DOF（RoboSmith 9 维边界补两指）/ world 图元 box+halfspace /
  `exempt` 由 RoboSmith 剔除 / `quality` 当排序量不当安全门——全部接受。
- 接法：torch 进程**不 import jax**，经 `rocrobo.serve --serve` warm 进程收发 JSONL（`op: solve_ik`）。
- 第一步（P1）：`motion/rocrobo_backend.py`（World builder + serve client）+ `MotionPlanner` 协议，把
  `select_best_grasp_plan` 的 IK 换成 `solve_ik(...,world)`；端到端在 wbb3 验证 pick 不再撞抽屉。
- **可能的回头需求**（届时来此留言）：(1) 若 box+halfspace 不足以表达抽屉包络/夹爪，请加 sphere/capsule；
  (2) 若 grasp 候选逐个串行 `solve_ik` throughput 不够，需要 §3B 的**目标批**入口（一次评 N 候选）提前于 backlog。
- 仅 Franka，G2 暂不需要。
