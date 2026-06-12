"""rocRobo — collision-aware motion generation on JAX/PyRoki (ROCm).

The importable core API, extracted from the validated v0 experiment pipeline so
external projects (e.g. RoboSmith) can consume it as a module rather than reaching
into experiment scripts.

    from rocrobo import MotionGen, PlanConfig, PlanResult, BatchResult

    mg = MotionGen()
    q, ok, quality = mg.solve_ik(pos, quat, world=world)   # collision-aware IK seed
    res = mg.plan(start_pose, goal_pose)                    # PlanResult (res.timed = executable)
    batch = mg.plan_batch([g1, g2, ...])                    # BatchResult

Notes:
  - Importing loads the robot URDF + sphere decomposition; set env ROCROBO_ASSETS
    to the dir containing `panda_spheres.json` (defaults to ./assets).
  - `world` is an injectable per-call jit arg (list of pyroki collision primitives).
  - World/RobotSpec explicit types and `plan_motion` are upcoming (feature8 P3/G2).
"""
from .motiongen import MotionGen, PlanConfig, BatchResult
from .metrics import PlanResult

__all__ = ["MotionGen", "PlanConfig", "BatchResult", "PlanResult"]
