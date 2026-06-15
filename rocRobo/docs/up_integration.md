# Upstream Integration — rocRobo 作为 RoboSmith 的 motion backend（待补能力）

> 本文只列 rocRobo 侧**当前待补**的能力（对接锚点）。已就绪基线无需赘述：`world` 作 per-call
> 入参贯通 `solve_ik`/`plan_motion`/`plan`/`plan_batch` 与 trajopt；常驻 warm `rocrobo.serve`
> （JSONL，`op: solve_ik | plan_motion`）；`import rocrobo` 可用。**契约（钉死）**：quat=wxyz、
> world frame=米、rocRobo 7 臂 DOF（RoboSmith 9 维边界补两指）、world 图元 box+halfspace、
> 接触豁免由调用方传入前从 world 剔除（`exempt`）、`quality` 为可比排序量非安全门。
>
> 消费方进展见 RoboSmith `docs/features/rocrobo_integration/`（feature1 抽屉 collision-free）。

## P1 — 抓取负载的附着碰撞（ACO 附着球）｜阻塞 feature1 收尾

> **Status: ✅ rocRobo 侧已支持并验证（feature9，wbb3 E1–E5 全过）。** `plan_motion`/`solve_ik` 新增可选
> `attach` 字段（协议见下方草案，已落地）；消费方 RoboSmith 只需在 pick 后的请求里带上负载包络
> + `T_ee_obj` + 持物夹爪连杆豁免。设计见 [`features/feature9_attached-collision-object.md`](features/feature9_attached-collision-object.md)。

**现象（feature1 实验定位）**：
- M0（空手 reach 进抽屉）已验证 `plan_motion` 让 **Franka 本体**对环境（柜体/抽屉）无碰、干净。
- M1（pick die → place 进抽屉）里**手里抓着的 die 不在碰撞模型**——`plan_motion` 只把 Franka 连杆
  规划成无碰，负载本身仍怼进托盘口/内壁。即"手臂无碰，负载怼障"。

**请求**：支持把抓起的物体作为**刚性附着体**并入机器人碰撞模型（业界 ACO / attached-object 标准做法），
使 `plan_motion`/`solve_ik` 的 world / self 碰撞代价同时覆盖负载——一旦附着，"负载 vs 环境"与
"本体 vs 环境"复用同一套检测。具体：

- **附着**：请求里携带一个"附着负载"，随 `ee_link`（hand）FK 一起平移，位姿 = `T_ee_obj`
  （物体相对末端帧）。它并入当前的 `RobotCollision.from_sphere_decomposition(panda_spheres)`，
  参与 world + self 碰撞代价。
- **几何近似**：球组 / box / 凸包即可（与本体球同构、检测快）；RoboSmith 负责传负载包络 + `T_ee_obj`。
- **自碰豁免**：附着体 vs **持物夹爪那几颗球**的碰撞需豁免（有意接触），其余连杆保留。
- **生命周期**：抓起时 attach、release 时 detach（由调用方在请求里声明/撤销，rocRobo 不懂任务语义）。

**协议草案**（在现有 JSONL 请求里加可选字段，不改已有契约）：

```json
{
  "op": "plan_motion",
  "q_start": [...], "waypoints": [...], "world": [...],
  "attach": {
    "ee_link": "hand",
    "T_ee_obj": [[...4x4...]],
    "spheres": [{"center": [x,y,z], "radius": r}, ...],
    "exempt_self_links": ["<持物夹爪连杆，按 panda_spheres 命名>"]
  }
}
```

（若用 box 近似，`attach.box: {half_extent, pose}` 亦可；与 §world 图元同构。）

## Backlog（非阻塞，需要时再排）

- **G2 `RobotSpec` 注入**（非 Franka 机器人/EE）：当前仅 Franka，feature1/2 不需要。
- **`reachable(poses[N], world) -> (mask[N], quality[N])` 目标批**：生成侧布局采样阶段一次性判一批候选位姿
  可达，复用 `plan_batch`/vmap，免 N 次串行调用。
- **新增 world / 附着体图元 `sphere` / `capsule`**：若 box+halfspace 不足以表达抽屉/夹爪/负载包络时补。
