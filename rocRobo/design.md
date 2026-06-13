# rocRobo — 设计文档（活文档 / 单一真相来源）

> rocRobo = 在 **AMD GPU（ROCm）** 上、不依赖 NV 生态、重做一个 cuRobo 等价的开源运动生成组件。
> 背景与模块/开源调研见 [`docs/curobo-study.md`](docs/curobo-study.md)（cuRobo 拆解 + 开源替代 + ROCm 落地）；模块定位/API/性能见 [`readme.md`](readme.md)。
> 本文件是项目级活文档：架构 + backlog + 选型决策，随迭代持续更新。

## 1. 项目定位

- 目标：collision-free 的 IK / motion plan，跑在 AMD MI 系列（gfx942/gfx950）上。
- 第一版 = **v0**（curobo-v1 的最小子集）：`FK → 多种子 IK → 直线初值 → 轨迹优化(平滑+碰撞) → 输出轨迹`。
- 不锁厂商：底座优先用跨平台栈（PyTorch-ROCm / JAX-ROCm / Triton），避免 CUDA/Warp。

## 2. 架构（按 cuRobo 5 大 feature）

| feature | v0 选型（主线 = JAX-ROCm + PyRoki，feature1 已定） |
|---|---|
| ① IK | 多种子并行 + LM/LBFGS + 碰撞代价 |
| ② 碰撞 | sphere/capsule 机器人 + 图元世界 SDF + 自碰撞 |
| ③ 几何规划 | v0 跳过（直线插值出初值）；后续 OMPL/VAMP(CPU) |
| ④ 轨迹优化 | waypoint + 平滑/碰撞代价 |
| ⑤ motion gen | 极简编排 ①→④ |
| 底座 | **JAX-ROCm + PyRoki**（feature1 验证，GPU 全绿）|

详见 [`docs/curobo-study.md` §1/§4](docs/curobo-study.md)。

## 3. Backlog（优先级 P0/P1/P2）

| 优先级 | feature | slug | 目的 | 设计 / 实验 | 状态 |
|---|---|---|---|---|---|
| ~~P0~~ | feature1 | pyroki-amd-enable | 验证 PyRoki(JAX) 能否在 MI350/gfx950 跑通，定主线路线 | [feature1](features/feature1_pyroki-amd-enable.md) / [part1-exp](exp/part1-exp.md) | ✅ done（GPU 全绿）|
| ~~P0~~ | feature2 | v0-pipeline-skeleton | v0 主干（多种子IK→直线初值→trajopt）+ GPU 批量压测 + demo 视频 | [feature2](features/feature2_v0-pipeline-skeleton.md) / [part2-exp](exp/part2-exp.md) | ✅ done（E1-E4 全绿，主干端到端）|
| ~~P1~~ | feature3 | robust-motion-gen | **总编排层 ⑤**：成功判定 + 端点收紧 + 失败重试 + plan_batch | [feature3](features/feature3_robust-motion-gen.md) / [part3-exp](exp/part3-exp.md) | ✅ done（端点 7cm→<1mm，重试 42%→75%）|
| ~~P1~~ | feature6 | batched-trajopt-parallel | **吃满 GPU 批量红利**：plan_batch 的 trajopt 用 vmap 批量并行，找 pipeline 级 GPU/CPU 拐点 | [feature6](features/feature6_batched-trajopt-parallel.md) / [part6-exp](exp/part6-exp.md) | ✅ done（GPU ~7×吞吐，拐点≈B4）|
| ~~P1~~ | feature7 | batched-plan-integration | 把 feature6 的批量 trajopt 接进 `plan_batch`；`select_backend` 按主导阶段分设阈值（trajopt≈4 / IK≈100）| [feature7](features/feature7_batched-plan-integration.md) / [part7-exp](exp/part7-exp.md) | ✅ done（trajopt 阶段 3.3×，分阶段路由）|
| ~~P2~~ | feature4 | geometric-planner-cpu | 编排里的"**初值**"那一档 ③：零依赖关节空间 RRT-Connect（复用 PyRoki 碰撞）出无碰初值 | [feature4](features/feature4_geometric-planner-cpu.md) / [part4-exp](exp/part4-exp.md) | ✅ done（价值重定：碰撞不靠初值，RRT init=收敛~2×加速的可选档）|
| ~~P2~~ | feature5 | retiming | 编排里的"**输出**"那一档：极简 TOPP 式限速/限加速 retiming（numpy/scipy 自写）；解 feature3 E2 接缝速度尖峰 | [feature5](features/feature5_retiming.md) / [part5-exp](exp/part5-exp.md) | ✅ done（限速解决 9.5→2.2，accel 降 15×）|
| ~~P3~~ | feature5b | pin-blend | 接缝几何圆角：pin 端点时插一小段斜坡而非硬钉单点，消除尖角残余加速度（feature5 E2 遗留）| [feature5b](features/feature5b_pin-blend.md) / [part5b-exp](exp/part5b-exp.md) | ❌ 假设推翻（accel 真因是参数化方法非端点角，不纳入主干；严格限 accel→TOPP-RA backlog）|
| ~~P3~~ | feature7b | plan-mainline | 把 plan_batch_v2 + 分阶段 select_backend + 可选 RRT init + retiming 固化进 v0 主干（现散在 exp 脚本）| [feature7b](features/feature7b_plan-mainline.md) / [part7b-exp](exp/part7b-exp.md) | ✅ done（`rocrobo.py` MotionGen 串通，回归 R1/R2/R3 全过）|
| **P2** | feature8 | consumable-api | 把 v0 封装成外部可消费 API：world/robot 入参化 + `solve_ik`/`plan_motion` + serve CLI + 打包（上游 RoboSmith 需求 [up_integration](docs/up_integration.md)）| [feature8](features/feature8_consumable-api.md) / [part8-exp](exp/part8-exp.md) | 🚧 进行中（P1 G1 world入参+`solve_ik` 已在 wbb3 验证；余 G4打包/serve/trajopt world）|
| P4 | accel-topp | topp-ra | 严格限加速/限 jerk 的时间参数化（TOPP-RA/Ruckig），替换 v0 的简化 retiming（feature5b 定位的 gap）| 待拆解 | backlog（离线装不上，待网络）|

> 流水线视角：`motion_gen` 是总编排，feature4/5 是它编排的两个工位（初值③、输出 retiming），feature3 是流水线本身的调度与质检，feature6 是把质检过的流水线**并行吞吐拉满**。
> 排序逻辑：feature3 先行（success/重试框架）；feature6 紧接（feature3 E4 留下的"批量 trajopt 并行"才是 GPU 批量红利的真正出口）；feature4/5 升级各工位后用 success 框架量化收益。
> 定位：「在 AMD 上用 JAX-ROCm 把 PyRoki 跑通并扩展」，**不追求逐组件自写**，能用 PyRoki 就用。

## 4. 选型决策记录

| 决策 | 结论 | 依据 | 日期 |
|---|---|---|---|
| 计算底座 | **JAX-ROCm（PyRoki 起步）** | feature1 验证 GPU 全绿（gfx950 跑通 PyRoki trajopt） | 2026-06-11 |
| 复用镜像 | `rocrobo/pyroki:rocm7.2.4-jax0.8.2`（含 PyRoki+jaxls，已 commit）| 避免重装；future `docker run` 直接用 | 2026-06-11 |
| 镜像分布 | 已在 **b05-1** 与 **wbb3** 两节点各 commit 一份（节点本地，非共享）| 节点 home/docker 不跨机共享，换节点需重 pull+装或重 commit | 2026-06-11 |
| 跨代验证 | **gfx942/MI308X 开箱可用**（srdc-20）：直接用通用框架镜像 `te-ci rocm-7.2/jax-0.8.0`，无需自定义镜像，三 sample 全绿（详见 findings C1b）| AMD enablement 不锁 gfx950；新节点用现成 jax-rocm 镜像 `pip install -e pyroki/core` 即可，无需专用镜像 | 2026-06-13 |
| 必备环境 | `AMD_COMGR_NAMESPACE=1` + docker `--device=/dev/kfd --device=/dev/dri --group-add video` | 否则 JAX GPU init 挂起（AMD known issue）| 2026-06-11 |
| 远端节点 | 主：b05-1（21:00 后释放）；备：**wbb3**（`gbt350-odcdh5-wbb3.png-odc.dcgpu`，gfx950 / ROCm 7.2.0，64 卡）| b05-1 被占时切 wbb3 | 2026-06-11 |

## 5. 环境基线（远端节点）

| 项 | b05-1（主） | wbb3（备） |
|---|---|---|
| host | `gbt350-odcdh2-b05-1.png-odc.dcgpu` | `gbt350-odcdh5-wbb3.png-odc.dcgpu` |
| GPU / ROCm | MI350X gfx950 / 7.1.1 | gfx950（64 卡）/ 7.2.0 |
| 工作目录（可写）| `/home/zhengjli/rocrobo`（mount→`/work`）| `/home/zhengjli/rocrobo`（mount→`/work`；`/data` 不可建目录）|
| Docker Root | — | `/data/docker`（`/` 仅 91G，镜像进 /data）|
| pyroki | `/work/pyroki`（`pip install -e`）| 同左（已 clone+装）|

### 下次直接起（复用 commit 镜像，单条命令）
```bash
docker run -d --name rocrobo_dev \
  --device=/dev/kfd --device=/dev/dri --group-add video --ipc=host \
  -e AMD_COMGR_NAMESPACE=1 -e HIP_VISIBLE_DEVICES=0 \
  -v /home/zhengjli/rocrobo:/work \
  rocrobo/pyroki:rocm7.2.4-jax0.8.2 sleep infinity
# 跑（共享节点务必加 XLA_PYTHON_CLIENT_PREALLOCATE=false，否则 JAX 预分配 75% 显存会 OOM）：
# docker exec -e AMD_COMGR_NAMESPACE=1 -e HIP_VISIBLE_DEVICES=0 -e XLA_PYTHON_CLIENT_PREALLOCATE=false \
#   rocrobo_dev bash -lc 'cd /work/pyroki/examples && python -u f2_e3_plan.py /work/f2/out/plan.mp4'
```
> 注：pyroki 为 `-e` 安装、指向 `/work/pyroki`，故起容器**必须带 `-v /home/zhengjli/rocrobo:/work`**；其余依赖（jaxls/matplotlib/imageio/numpy）已 baked 进镜像。
