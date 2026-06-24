# rocRobo

*[中文](README.md) ｜ English*

> **Collision-aware motion generation on AMD GPUs (ROCm)** — collision-aware IK + obstacle-avoiding trajectory optimization, built on JAX + [PyRoki](https://github.com/chungmin99/pyroki). An NV-free, open-source cuRobo-equivalent v0.

Given an EE pose / waypoints + live world geometry, it returns **collision-free** joint solutions and an executable trajectory. Designed to be consumed as a motion backend by external projects (e.g. the RoboSmith data-generation pipeline). Single-arm (Franka) validated on gfx950 (MI350X) and gfx942 (MI300 series, incl. MI308X).

▶ demo: collision-free over-the-wall trajectory → [rocRobo/samples/media/demo_plan.mp4](rocRobo/samples/media/demo_plan.mp4) ｜ collision-blind vs aware → [rocRobo/samples/media/demo_compare.mp4](rocRobo/samples/media/demo_compare.mp4)

**Value on top of PyRoki**: full GPU enablement on JAX-ROCm; multi-seed IK (single seed ~52% → N=16 100%); `vmap` batched trajopt; one-shot `plan()` (success gating + endpoint pinning + retries + retiming, returns best+reason on failure); per-call world injection, JSON warm process isolating the jax/torch stacks.

## Dependencies

| Component | Path |
|:---|:---|
| PyRoki | [`pyroki/`](pyroki/) — `git submodule` ([upstream](https://github.com/chungmin99/pyroki)) |
| rocRobo | [`rocRobo/core/`](rocRobo/core/) — Python package `rocrobo` (src-layout) |

```bash
git clone --recurse-submodules <repo-url> && cd rocRobo
git submodule update --init --recursive   # if cloned without submodules
```

## API

```python
from rocrobo import MotionGen
mg = MotionGen()   # build once, reuse the JIT
```

| Entry | Signature | Returns |
|---|---|---|
| Collision-aware IK | `mg.solve_ik(pos, quat, *, seed=None, world=None, attach=None)` | `(q, ok, quality)` |
| Single-goal plan | `mg.plan(start_pose, goal_pose, world=None)` | `PlanResult` |
| Multi-waypoint plan | `mg.plan_motion(q_start, waypoints, world=None, attach=None)` | `PlanResult` |
| Batched plan | `mg.plan_batch(goals, *, world=None, route=None)` | `BatchResult` |

**Contract**: `quat = wxyz`; world frame in meters, `world` is a list of pyroki primitives (`Box`/`HalfSpace`), mutable per call; rocRobo is **7 arm DOF** (the two fingers are padded by the consumer at the boundary); intentional-contact bodies must be removed from `world` by the caller first. `PlanResult.timed` = executable trajectory after retiming; `quality` shares its source with the internal evaluate and is used to rank candidates. `attach` (optional) = grasped payload as rigid attached spheres (ACO).

## Quick start

The image is built from the repo-root **`Dockerfile`** (base `rocm/jax:rocm7.2.4-jax0.8.2-py3.12`, with the `pyroki/` submodule installed at build time):

```bash
docker build -t rocrobo/pyroki:rocm7.2.4-jax0.8.2 .   # or docker pull the prebuilt image

# start a long-lived container, repo mounted at /rocrobsim
docker run -d --name rocrobo_dev \
  --device=/dev/kfd --device=/dev/dri --group-add video --ipc=host \
  -e AMD_COMGR_NAMESPACE=1 -e HIP_VISIBLE_DEVICES=0 \
  -v "$PWD:/rocrobsim" \
  rocrobo/pyroki:rocm7.2.4-jax0.8.2 sleep infinity

# install the core package (src-layout, package name is rocrobo)
docker exec rocrobo_dev pip install -e /rocrobsim/rocRobo/core
```

> Running needs `export ROCROBO_ASSETS=/rocrobsim/pyroki/examples/assets`; on shared nodes add `XLA_PYTHON_CLIENT_PREALLOCATE=false` to avoid OOM.

**In-process import**:

```python
from rocrobo import MotionGen
mg = MotionGen()
q, ok, quality = mg.solve_ik(pos, quat, world=world)
res = mg.plan_motion(q_start, [wp1, wp2], world=world)   # res.timed is executable
```

**Cross-process serve** (a torch process cannot import jax-rocm; use a sidecar exchanging JSONL):

```bash
python -m rocrobo.serve --serve   # long-lived warm process (hot after one JIT)
python -m rocrobo.serve --once    # one-shot smoke
```

> **As a RoboSmith sidecar**: RoboSmith connects with `ROCROBO_BACKEND=1` via `docker exec` into this container's (default `rocrobo_dev`) warm serve; orchestration is in RoboSmith's `docs/runtime.md`.

## Performance (measured on MI350X / gfx950; also functionally validated on MI300 series / gfx942)

Multi-seed IK single seed ~52% → **N=16 100%**; endpoint accuracy after pinning **~0.1mm/0.4°** while still avoiding collisions; batched trajopt per-traj **~7×** faster; serve cold start 14.7s → warm **0.6s (~24×)**; retiming acceleration **~15×** lower.

## Layout

```
Dockerfile             # rocrobo/pyroki image
pyroki/                # PyRoki (git submodule)
rocRobo/
  core/rocrobo/        # core package: pipeline / metrics / rrt / retiming / motiongen / serve
  samples/             # runnable demos + videos
```

## Reference

[cuRobo](https://github.com/NVlabs/curobo) ｜ [PyRoki](https://github.com/chungmin99/pyroki) (rocRobo's starting point) ｜ base [JAX ROCm](https://github.com/ROCm/jax) / [PyTorch ROCm](https://pytorch.org/) ｜ blueprints [pyroboplan](https://github.com/sea-bass/pyroboplan) · [mplib](https://github.com/haosulab/MPlib) · [OMPL](https://github.com/ompl/ompl) · [Ruckig](https://github.com/pantor/ruckig) · [TOPP-RA](https://github.com/hungpham2511/toppra)
