# rocRobo

> **AMD GPU（ROCm）上的 collision-aware 运动生成**——碰撞感知 IK + 避障轨迹优化，构建于 JAX + [PyRoki](https://github.com/chungmin99/pyroki)，不依赖 NVIDIA 生态。一个开源、NV-free 的 cuRobo 等价 v0。

机器人单臂（Franka）场景下，给一个末端位姿/路点 + 实时世界几何，返回**无碰撞**的关节解与可执行轨迹。面向被外部项目（如 RoboSmith 数据生成管线）作为运动后端消费。

▶ demo：`plan()` 一次输出的无碰撞越墙轨迹 → [samples/demo_plan.mp4](samples/demo_plan.mp4)；collision-blind vs collision-aware 分屏对比 → [samples/demo_marketing.mp4](samples/demo_marketing.mp4)（可运行示例见 [samples/](samples/)）。

**rocRobo 的自有价值**（PyRoki 之上）：
- **AMD enablement**：JAX-ROCm 上把 PyRoki 可微 FK/IK/碰撞/trajopt 全程 GPU 跑通，**gfx950 与 gfx942/MI308X 均开箱可用**。
- **多种子 IK**：N 个随机种子并行求解取最优（单种子 ~52% → N=16 100% 成功率）。
- **批量规划**：`vmap` 批量 trajopt 吃满 GPU 批量红利，按主导阶段自动选 CPU/GPU 后端。
- **极简编排**：成功判定 + 端点硬钉 + 失败重试 + retiming 的一站式 `plan()`，失败返回 best+reason 不假装成功。
- **可消费边界**：world 按调用注入、`solve_ik` 回质量分、JSON warm 进程隔离 jax/torch 栈。

## 能力 / API

```python
from rocrobo import MotionGen, PlanConfig, PlanResult, BatchResult
mg = MotionGen()                 # 一次构建，复用 JIT 编译
```

| 入口 | 签名 | 返回 |
|---|---|---|
| 碰撞感知 IK seed | `mg.solve_ik(pos, quat, *, seed=None, world=None)` | `(q, ok, quality)` |
| 单目标规划 | `mg.plan(start_pose, goal_pose, world=None)` | `PlanResult` |
| 多路点避障运动 | `mg.plan_motion(q_start, waypoints, world=None)` | `PlanResult` |
| 批量规划 | `mg.plan_batch(goals, *, world=None, route=None)` | `BatchResult` |

- `PlanResult(success, traj, metrics, checks, reason, n_retries)`；`res.timed` = retiming 后可执行轨迹。
- `quality = {limit_margin, world_clear, self_clear, pos_err_mm}`：与内部 evaluate **同源**，是给候选**排序用的可比标量**（非绝对安全门）。
- `world` = pyroki 碰撞图元列表（`Box` / `HalfSpace`），**每次调用可变**，贯通 IK + trajopt + evaluate。

**契约 / 约定（消费方必读）**：
- `quat = wxyz`（pyroki/jaxlie 约定）。
- world frame = 米。
- rocRobo = **7 臂 DOF**（消费方若 9 维含两指，边界补齐）。
- 接触豁免由调用方决定：把"有意接触"的体从 `world` 剔除后再传。

## Quick start

### 1. 起容器（复用镜像，单条命令）

```bash
docker run -d --name rocrobo_dev \
  --device=/dev/kfd --device=/dev/dri --group-add video --ipc=host \
  -e AMD_COMGR_NAMESPACE=1 \
  -v /home/<you>/rocrobo:/work \
  rocrobo/pyroki:rocm7.2.4-jax0.8.2 sleep infinity
```

> `AMD_COMGR_NAMESPACE=1` 必须，否则 JAX GPU init 静默挂起（AMD known issue）；pyroki 为 `-e` 安装指向 `/work/pyroki`，故必须挂 `-v .../rocrobo:/work`。

### 2. import 用法（同进程）

`rocrobo` 包在 `core/` 下（src-layout，包名即 `rocrobo`）。装一次即可 `import rocrobo`：

```bash
pip install -e core                          # 之后处处可 import rocrobo（推荐）
# 或免安装：export PYTHONPATH=/work/core
export ROCROBO_ASSETS=/work/pyroki/examples/assets
export XLA_PYTHON_CLIENT_PREALLOCATE=false   # 共享节点防 OOM
```

> 运行时依赖（jax[rocm] / pyroki / jaxls / numpy / imageio）由镜像提供，`pyproject.toml` 不 pin，避免覆盖环境里的 ROCm jax。

```python
from rocrobo import MotionGen
mg = MotionGen()
q, ok, quality = mg.solve_ik(pos, quat, world=world)     # 无碰 IK seed
res = mg.plan(start_pose, goal_pose, world=world)        # 单目标，res.timed 可执行
res = mg.plan_motion(q_start, [wp1, wp2], world=world)   # 多路点避障
```

### 3. serve —— 跨进程消费（jax/torch 不共进程时）

消费方在 torch 进程里**不能** import jax-rocm；用 sidecar 收发 JSON：

```bash
python -m rocrobo.serve --serve   # 常驻 warm 进程（运行态）：JIT 编译一次后常热，逐行 JSON
python -m rocrobo.serve --once    # 一次性 smoke test
```

请求 / 响应（JSONL，`op: solve_ik | plan_motion`）：

```
{"op":"solve_ik","pos":[x,y,z],"quat":[w,x,y,z],"world":[{"type":"box","center":[..],"extent":[..]},
                                                          {"type":"halfspace","point":[..],"normal":[..]}]}
→ {"ok":true,"q":[...],"quality":{...}}
```

> warm 进程兑现吞吐的前提：请求批量化 + world 图元数 / batch N **形状稳定**（否则触发重编译）。

## 性能数据

CPU = 单次规划路径；GPU = MI350X / gfx950。详见 [docs/findings.md](docs/findings.md)。

| 维度 | 数据 | 来源 |
|---|---|---|
| 多种子 IK 成功率 | 单种子 ~52% → **N=16 100%**（50 目标 / 1cm 判定） | C2 |
| 端点精度 | 硬钉端点后 7cm/9° → **~0.1mm/0.4°**（v0 回归 1.08mm）且仍避碰 | C5 / C12 |
| 失败重试 | 阶梯种子(4→16→64) **42% → 75%** | C6 |
| IK GPU 拐点 | B≈64–128（B=512：GPU 3.9ms vs CPU 14.6ms，~3.7×） | C3 |
| 批量 trajopt | per-traj 31.5s→**4.5s**（B1→64，~7×），拐点 **B≈4** | C7 |
| plan_batch 端到端 | trajopt 阶段串行→批量 **3.3×**（209.7s→63.0s，B=8） | C8 |
| 碰撞恢复 | trajopt 把 29/50 穿墙直线初值推出墙；RRT 无碰初值仅收敛~2×快 | C9 |
| retiming | 接缝速度 9.5→2.2 rad/s，加速度 **~15×↓** | C10 |
| serve warm | 冷启动 14.7s → warm 调用 **0.6s（~24×）** | C13 |

## 项目结构

```
core/rocrobo/        # 可 import 的核心包
  __init__.py        #   暴露 MotionGen / PlanConfig / PlanResult / BatchResult
  pipeline.py        #   robot/scene + 多种子 IK + trajopt + quality + batched trajopt + render
  metrics.py         #   evaluate() 成功判定 + PlanResult
  rrt.py             #   可选 RRT-Connect 初值（碰撞复用 pipeline）
  retiming.py        #   限速 / 限加速 时间参数化
  motiongen.py       #   MotionGen 编排入口（python -m rocrobo.motiongen 跑回归自测）
  serve.py           #   JSON 进程边界（--serve warm / --once smoke）
docs/
  up_integration.md  # 上游 RoboSmith 对接契约 / gap 状态
  findings.md        # 跨 feature 实战结论速查（C1–C13）
  curobo-study.md    # 立项调研：cuRobo 拆解 + 开源替代 + ROCm 路线
design.md            # 项目架构 / backlog / 选型决策（活文档）
```

## 文档

- **对接消费方** → [docs/up_integration.md](docs/up_integration.md)
- **实验结论 / 性能细节** → [docs/findings.md](docs/findings.md)
- **架构 / backlog / 环境基线** → [design.md](design.md)
- **设计背景（cuRobo 拆解与路线）** → [docs/curobo-study.md](docs/curobo-study.md)

## reference

* [cuRobo](https://github.com/NVlabs/curobo) ｜ [doc(v1)](https://curobo.org/) ｜ [PyRoki](https://github.com/chungmin99/pyroki)（JAX，对标 cuRobo，rocRobo 主干起点）
* 底座：[JAX ROCm](https://github.com/ROCm/jax) ｜ [PyTorch ROCm](https://pytorch.org/) ｜ [Triton](https://github.com/triton-lang/triton)
* 对照 / 蓝本：[pyroboplan](https://github.com/sea-bass/pyroboplan) ｜ [mplib](https://github.com/haosulab/MPlib) ｜ [OMPL](https://github.com/ompl/ompl) ｜ [Ruckig](https://github.com/pantor/ruckig) ｜ [TOPP-RA](https://github.com/hungpham2511/toppra)
* 更多开源替代清单见 [docs/curobo-study.md](docs/curobo-study.md)。
