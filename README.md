# rocRobo

*中文 ｜ [English](readme-en.md)*

> **AMD GPU（ROCm）上的 collision-aware 运动生成**——碰撞感知 IK + 避障轨迹优化，构建于 JAX + [PyRoki](https://github.com/chungmin99/pyroki)，NV-free 的开源 cuRobo 等价 v0。

给末端位姿/路点 + 实时世界几何，返回**无碰撞**的关节解与可执行轨迹。面向被外部项目（如 RoboSmith 数据生成管线）作为运动后端消费。单臂（Franka）已验证 gfx950（MI350X）与 gfx942（MI300 系，含 MI308X）。

▶ demo：无碰撞越墙轨迹 → [rocRobo/samples/media/demo_plan.mp4](rocRobo/samples/media/demo_plan.mp4) ｜ collision-blind vs aware 对比 → [rocRobo/samples/media/demo_compare.mp4](rocRobo/samples/media/demo_compare.mp4)

**PyRoki 之上的自有价值**：JAX-ROCm 全程 GPU 落地；多种子 IK（单种子 ~52% → N=16 100%）；`vmap` 批量 trajopt；一站式 `plan()`（成功判定 + 端点硬钉 + 重试 + retiming，失败返回 best+reason）；world 按调用注入、JSON warm 进程隔离 jax/torch。

## Dependencies

| Component | Path |
|:---|:---|
| PyRoki | [`pyroki/`](pyroki/) — `git submodule` ([upstream](https://github.com/chungmin99/pyroki)) |
| rocRobo | [`rocRobo/core/`](rocRobo/core/) — Python 包 `rocrobo`（src-layout） |

```bash
git clone --recurse-submodules <repo-url> && cd rocRobo
git submodule update --init --recursive   # 若已克隆但未带子模块
```

## API

```python
from rocrobo import MotionGen
mg = MotionGen()   # 一次构建，复用 JIT
```

| 入口 | 签名 | 返回 |
|---|---|---|
| 碰撞感知 IK | `mg.solve_ik(pos, quat, *, seed=None, world=None, attach=None)` | `(q, ok, quality)` |
| 单目标规划 | `mg.plan(start_pose, goal_pose, world=None)` | `PlanResult` |
| 多路点避障 | `mg.plan_motion(q_start, waypoints, world=None, attach=None)` | `PlanResult` |
| 批量规划 | `mg.plan_batch(goals, *, world=None, route=None)` | `BatchResult` |

**契约**：`quat = wxyz`；world frame = 米，`world` 为 pyroki 图元（`Box`/`HalfSpace`）列表、每次调用可变；rocRobo = **7 臂 DOF**（含两指由消费方边界补齐）；有意接触的体由调用方先从 `world` 剔除。`PlanResult.timed` = retiming 后可执行轨迹；`quality` 与内部 evaluate 同源，用于候选排序。`attach`（可选）= 抓取负载作刚性附着球（ACO）。

## Quick start

镜像由 repo root 的 **`Dockerfile`** 构建（base `rocm/jax:rocm7.2.4-jax0.8.2-py3.12`，构建时装好 `pyroki/` submodule）：

```bash
docker build -t rocrobo/pyroki:rocm7.2.4-jax0.8.2 .   # 或 docker pull 预构建镜像

# 起常驻容器，仓库挂到 /rocrobsim
docker run -d --name rocrobo_dev \
  --device=/dev/kfd --device=/dev/dri --group-add video --ipc=host \
  -e AMD_COMGR_NAMESPACE=1 -e HIP_VISIBLE_DEVICES=0 \
  -v "$PWD:/rocrobsim" \
  rocrobo/pyroki:rocm7.2.4-jax0.8.2 sleep infinity

# 装核心包（src-layout，包名即 rocrobo）
docker exec rocrobo_dev pip install -e /rocrobsim/rocRobo/core
```

> 运行需 `export ROCROBO_ASSETS=/rocrobsim/pyroki/examples/assets`；共享节点加 `XLA_PYTHON_CLIENT_PREALLOCATE=false` 防 OOM。

**同进程 import**：

```python
from rocrobo import MotionGen
mg = MotionGen()
q, ok, quality = mg.solve_ik(pos, quat, world=world)
res = mg.plan_motion(q_start, [wp1, wp2], world=world)   # res.timed 可执行
```

**跨进程 serve**（torch 进程不能 import jax-rocm，用 sidecar 收发 JSONL）：

```bash
python -m rocrobo.serve --serve   # 常驻 warm 进程（JIT 一次后常热）
python -m rocrobo.serve --once    # 一次性 smoke
```

> **作为 RoboSmith sidecar**：RoboSmith 以 `ROCROBO_BACKEND=1` 经 `docker exec` 连入本容器（默认 `rocrobo_dev`）的 warm serve；编排见 RoboSmith 的 `docs/runtime.md`。

## 性能（MI350X / gfx950 实测；功能在 MI300 系 / gfx942 亦验证）

多种子 IK 单种子 ~52% → **N=16 100%**；端点精度硬钉后 **~0.1mm/0.4°** 且仍避碰；批量 trajopt per-traj **~7×**↓；serve 冷启 14.7s → warm **0.6s（~24×）**；retiming 加速度 **~15×**↓。

## 项目结构

```
Dockerfile             # rocrobo/pyroki 镜像
pyroki/                # PyRoki（git submodule）
rocRobo/
  core/rocrobo/        # 核心包：pipeline / metrics / rrt / retiming / motiongen / serve
  samples/             # 可运行 demo + 视频
```

## reference

[cuRobo](https://github.com/NVlabs/curobo) ｜ [PyRoki](https://github.com/chungmin99/pyroki)（rocRobo 主干起点）｜ 底座 [JAX ROCm](https://github.com/ROCm/jax) / [PyTorch ROCm](https://pytorch.org/) ｜ 蓝本 [pyroboplan](https://github.com/sea-bass/pyroboplan) · [mplib](https://github.com/haosulab/MPlib) · [OMPL](https://github.com/ompl/ompl) · [Ruckig](https://github.com/pantor/ruckig) · [TOPP-RA](https://github.com/hungpham2511/toppra)
