# rocRobo samples

两个代表性可运行示例，演示 `rocrobo` 包的核心能力。需在 rocRobo 的 jax-rocm 环境里跑（见根 [README.md](../../README.md) Quick start 起容器）。

## 环境

```bash
pip install -e ../core                                  # 装包后处处可 import rocrobo（推荐）
export ROCROBO_ASSETS=/rocrobsim/pyroki/examples/assets # panda_spheres.json 所在
export AMD_COMGR_NAMESPACE=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false              # 共享节点防 OOM
```

## 示例

| 文件 | 演示 | 产物 |
|---|---|---|
| `demo_plan.py` | 单目标 `plan()` 端到端：多种子 IK → trajopt → retiming → 渲染越墙轨迹 | [`media/demo_plan.mp4`](media/demo_plan.mp4) |
| `demo_compare.py` | collision-blind vs collision-aware 分屏对比（环绕运镜），最直观的「为什么要 rocRobo」 | [`media/demo_compare.mp4`](media/demo_compare.mp4) |

```bash
python samples/demo_plan.py samples/media/demo_plan.mp4
python samples/demo_compare.py samples/media/demo_compare.mp4
```

> 视频产物统一放 [`media/`](media/)，与脚本分开。`solve_ik` / `plan_motion` / `serve` 的最小用法见根 [README.md](../../README.md#api) 的 API 与 Quick start 章节。
