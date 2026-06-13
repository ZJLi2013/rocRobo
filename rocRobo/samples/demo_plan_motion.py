"""Sample: multi-waypoint collision-aware motion.

    python samples/demo_plan_motion.py

plan_motion(q_start, waypoints, world): starts from a joint config and passes
through a list of EE poses, collision-aware and retimed. Here the mid waypoint is
the FK of the start->goal joint midpoint (guaranteed reachable) for a clean demo.
"""
import numpy as np
import jax.numpy as jnp

from rocrobo import MotionGen
from rocrobo import pipeline as rp


def main():
    mg = MotionGen()
    cfg_start, _ = rp.multiseed_ik(*rp.start_pose, n=16)
    cfg_goal, _ = rp.multiseed_ik(*rp.goal_pose, n=16, near_cfg=cfg_start)

    cfg_mid = 0.5 * (np.asarray(cfg_start) + np.asarray(cfg_goal))
    fk_mid = np.array(rp.robot.forward_kinematics(jnp.array(cfg_mid)))[rp.link_idx]
    waypoints = [(fk_mid[4:7], fk_mid[:4]), rp.goal_pose]

    res = mg.plan_motion(cfg_start, waypoints)
    print("plan_motion:", res, "steps:", len(res.traj),
          "end_mm:", res.metrics["end_pos_mm"], "checks:", res.checks)


if __name__ == "__main__":
    main()
