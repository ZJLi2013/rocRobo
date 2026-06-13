"""Sample: single-goal plan + render an mp4.

    python samples/demo_plan.py [out.mp4]

Shows the v0 mainline: multi-seed IK -> trajopt (collision-aware) -> retiming,
then renders the executable trajectory crossing the default wall scene.
See samples/README.md for the env (PYTHONPATH / ROCROBO_ASSETS) it expects.
"""
import sys

from rocrobo import MotionGen
from rocrobo import pipeline as rp


def main(out="demo_plan.mp4"):
    mg = MotionGen()
    res = mg.plan(rp.start_pose, rp.goal_pose)
    print("plan:", res, "end_mm:", res.metrics["end_pos_mm"],
          "| timed:", None if res.timed is None else
          f"{res.timed['total_s']}s vel_ok={res.timed['vel_ok']}")
    mg.render(res.traj, out)
    print("wrote", out)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "demo_plan.mp4")
