# rocRobo 实战结论速查（cross-feature findings / 单一真相来源）

> rocRobo **跨 feature 关键实验结论**的汇总落点（详细数据在各 `exp/partN-exp.md`，实验日志为本地非跟踪）。
> 规则：每完成一个 feature，把"被数据证实、会影响后续决策"的结论回填一行；避免结论散落在 partN 里被埋没。
> 立项调研/路线见 [curobo-study.md](curobo-study.md)；架构/backlog 见 [../design.md](../design.md)。

## 已验证结论（按 feature）

| # | 结论（被实验证实） | 关键证据 | 对后续的影响 | 来源 |
|---|---|---|---|---|
| C1 | **JAX-ROCm 在 MI350/gfx950 端到端可用**：PyRoki(含 jaxls 求解器) trajopt 全程 GPU 跑通 | trajopt 输出 `(50,8)` 合法轨迹、无 NaN；`jaxls` 线性代数后端在 ROCm 正常 | rocRobo 主线定为 **JAX 路线（PyRoki 起步）**，不走 PyTorch 重写 | [part1 E2](exp/part1-exp.md) |
| C2 | **多种子 IK 大幅提升鲁棒性**：单随机 seed 仅 ~52% 成功，N=16 即 100% | 50 个可达目标、1cm 判定：N=1/4/16/64 → 52/92/100/100% | v0 IK 默认 **N=16**；多种子是 rocRobo 第一个自有能力（对标 cuRobo） | [part2 E2](exp/part2-exp.md) |
| C3 | **GPU 价值只在批量**：小问题 GPU 慢于 CPU，存在 batch 拐点 | 拐点 **B≈64–128**；B=1 GPU 慢 9×，B=512 GPU 3.9ms vs CPU 14.6ms(~3.7×) | rocRobo 必须走 **batched planning**；单次规划用 CPU 反而快 → `plan()` 应按 batch 大小自动选后端 | [part2 E4](exp/part2-exp.md) |
| C4 | **v0 主干 `plan()` 端到端成立**：多种子IK→直线初值→trajopt→渲染 一次出合法越墙轨迹 | 42s，轨迹合法/限位内；端点 ~6cm（AL 软约束松，待收紧） | v0 motion-gen 骨架可用；端点精度作为已知 next-step | [part2 E3](exp/part2-exp.md) |
| C5 | **端点收紧靠硬钉、不靠加迭代**：trajopt 多跑迭代无效，硬钉端点 cfg 才把误差压到 IK 精度 | 加迭代 300 与 baseline 数值完全相同（AL 卡死）；硬钉后端点 7cm/9°→**0.1mm/0.4°** 且仍避碰 | robust `plan()` 端点收紧用 **pin_endpoints**；接缝速度尖峰(0.19rad)交 retiming | [part3 E2](exp/part3-exp.md) |
| C6 | **失败重试编排显著提成功率**：阶梯种子(4→16→64)+success门控 | M=12：single(N=4) 42% → retry **75%**；剩余为不可行目标 | 编排层(⑤)是 rocRobo 自有价值；`PlanResult` 失败返回 best+reason 不假装成功 | [part3 E3](exp/part3-exp.md) |
| C7 | **批量 trajopt 才是 GPU 批量红利的真正出口**：`vmap` 批量优化多条轨迹，GPU ~7×吞吐，**拐点 B≈4**（远早于 IK 的 ≈100）| per-traj GPU 31.5s→4.5s(B1→64)；CPU 15.4→12.7→19.8(无红利)；批量 vs 串行 <1e-3 | trajopt 计算密集、小 batch 即喂满 GPU → `plan_batch` 后端阈值须按主导阶段分设(trajopt≈4 / IK≈100)，不能共用 | [part6 E2](exp/part6-exp.md) |
| C8 | **批量 trajopt 接进 plan_batch 端到端兑现红利**：trajopt 阶段串行→批量 **3.3×**，成功集合不变；后端按主导阶段分阈值 | trajopt 串行 209.7s→批量 63.0s(B=8)；路由 trajopt 翻@4/IK 翻@100；B=16 无 OOM(6.2s/traj)| `plan_batch_v2`+分阶段 `select_backend` 成 rocRobo 批量规划标准路径（待固化进 v0 主干）| [part7 E1/E2](exp/part7-exp.md) |
| C9 | **碰撞可行性不靠几何初值——trajopt 能从穿墙坏初值恢复**：强 world-coll 代价把 29/50 穿墙的直线初值推出墙 | 加高墙下直线/RRT 初值最终**都无碰**；RRT 无碰初值仅让 trajopt 收敛**~2×快**(21.9→11.8s) | 否决"必须几何规划器才能避碰"；RRT-Connect 定位为**可选加速档**，v0 默认直线初值(KISS)；OMPL 留 backlog | [part4 E2](exp/part4-exp.md) |
| C10 | **retiming 解限速、解不了几何尖角**：限速 retiming 把接缝尖峰压到 ≤vmax，加速度降 15×，但 pin 硬钉的几何尖角残余 accel 削不平 | 接缝速度 9.5→2.2 rad/s(超限段 49→0)；accel 365→24(~15×)；几何不变、时长 +12% | retiming 是纯时间后处理→限速可执行；加速度完全合规须真正的时间参数化器(见 C11) | [part5 E1/E2](exp/part5-exp.md) |
| C11 | **pin-blend 不是 accel 的杠杆（假设证伪）**：端点圆角削不动 retimed 加速度，真因是**时间参数化方法**而非端点几何角 | hard/blend/cubic 的 retimed accel 23.8/22.0/44.9 全 >amax(13)；端点/避碰不变 | pin-blend 不纳入主干；严格限 accel/jerk 须 **TOPP-RA/Ruckig**（离线装不上→backlog）| [part5b](exp/part5b-exp.md) |
| C12 | **v0 主干固化无回归 + refine 批量等价**：`rocrobo.py`(MotionGen) 串通 IK重试/可选RRT初值/trajopt/pin/evaluate/retiming/分阶段backend；vmap 批量与串行 trajopt **逐点不等但都合法** | 回归 R1 端点1.08mm、R2(RRT)0.52mm、R3 成功集一致；批量 vs 串行成功目标轨迹差达 ~0.1rad 且跨run变 | no-regression 判据=**成功集一致**而非逐点相等(feature6 <1e-3 仅其 easy 目标成立)；`rocrobo.py` 为 v0 标准入口 | [part7b](exp/part7b-exp.md) |
| C13 | **可消费 API（feature8）打通且零回归**：world 贯通 IK+trajopt+evaluate、`solve_ik` 回 quality、`plan_motion` 多路点、`rocrobo` 可 import、`serve` warm 进程 | b05-1：回归 PASS；注入挡路箱 blind -0.060→aware +0.049 整臂绕开、evaluate 全绿；warm 冷启动 14.7s→0.6s(~24×) | rocRobo 成 RoboSmith 可消费后端（[up_integration](up_integration.md)）；world 图元 box/halfspace | [part8](exp/part8-exp.md) |

## 复现/踩坑要点（环境级，复用价值高）

| 项 | 结论 | 来源 |
|---|---|---|
| GPU init 挂起 | 必须 `AMD_COMGR_NAMESPACE=1`，否则 JAX GPU 初始化静默挂起 >5min（AMD known issue） | [part1 E1](exp/part1-exp.md) |
| 共享节点 OOM | 8 卡满载时 JAX 默认预分配 75% 显存→OOM；须 `XLA_PYTHON_CLIENT_PREALLOCATE=false` 改按需 | [part2 E3](exp/part2-exp.md) |
| 镜像复用 | `rocrobo/pyroki:rocm7.2.4-jax0.8.2`（已在 b05-1 / wbb3 各 commit），起容器须挂 `-v .../rocrobo:/work`（pyroki 为 `-e` 装） | [../design.md](../design.md) |
| 包导入 import 冲突 | 跑实验须从**不含 flat `rocrobo.py`/`f2_e3_plan.py` 的干净目录**、`PYTHONPATH=/work/core` + `ROCROBO_ASSETS=.../assets`，否则脚本目录的旧 flat 模块会抢 `import rocrobo` | [part8 E3](exp/part8-exp.md) |

> 待验证 / 存疑：multi-**seed**（同一目标 N 初值）的 GPU 拐点尚未单独压测，C3 用的是多-**target** 批量（vmap 同构，可外推但未坐实）。
