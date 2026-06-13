# rocRobo samples

最小可运行示例，演示 `rocrobo` 包的核心入口。需在 rocRobo 的 jax-rocm 环境里跑（见 [../readme.md](../readme.md) Quick start 起容器）。

## 环境

```bash
pip install -e core                                   # 装包后处处可 import rocrobo（推荐）
# 或免安装：export PYTHONPATH=/work/core
export ROCROBO_ASSETS=/work/pyroki/examples/assets    # panda_spheres.json 所在
export AMD_COMGR_NAMESPACE=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false            # 共享节点防 OOM
```

> 从 repo 根跑 `python samples/xxx.py` 前，先 `pip install -e core`（或设 `PYTHONPATH=$PWD/core`），否则 `import rocrobo` 找不到包。

## 示例

| 文件 | 演示 |
|---|---|
| `demo_plan.py` | 单目标 `plan()` → 渲染越墙轨迹 mp4 |
| `demo_solve_ik.py` | `solve_ik` 碰撞感知 IK + 注入 world + `quality` |
| `demo_plan_motion.py` | `plan_motion` 多路点避障运动 |
| `demo_marketing.py` | collision-blind vs collision-aware 分屏对比（环绕运镜）→ mp4 + 海报 png |
| `serve_client.py` | 跨进程消费：起 `rocrobo.serve` warm 进程并用 JSON 收发（torch 侧消费方范式）|

```bash
python samples/demo_plan.py demo_plan.mp4
python samples/demo_solve_ik.py
python samples/demo_plan_motion.py
python samples/demo_marketing.py samples/demo_marketing.mp4
python samples/serve_client.py
```

## demo 视频

- `demo_plan.mp4` —— v0 `plan()` 在墙体场景一次输出的无碰撞越墙轨迹（多种子 IK → trajopt → retiming）。
- `demo_marketing.mp4` —— 同起点/终点/墙：左 collision-blind 直穿墙体（红色 COLLISION 标注），右 rocRobo collision-aware 优雅绕过。最直观的"为什么要 rocRobo"对比镜头。
