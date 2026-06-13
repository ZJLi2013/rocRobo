"""Sample: collision-aware IK seed with an injected world.

    python samples/demo_solve_ik.py

Solves IK for a goal EE pose twice: (1) default scene, (2) with a box dropped on
the goal. The injected world changes the solution / feasibility, and `quality`
gives comparable scalars for ranking candidates.
"""
import numpy as np
import pyroki as pk

from rocrobo import MotionGen
from rocrobo import pipeline as rp


def main():
    mg = MotionGen()
    (gp, gq) = rp.goal_pose

    q, ok, quality = mg.solve_ik(gp, gq)
    print("default :", "ok" if ok else "FAIL",
          "pos_err_mm=%.2f" % quality["pos_err_mm"], "quality=", quality)

    block = pk.collision.Box.from_extent(extent=np.array([0.15, 0.15, 0.15]),
                                         position=np.array(gp))
    world = list(rp.world_coll) + [block]
    q2, ok2, quality2 = mg.solve_ik(gp, gq, world=world)
    print("blocked :", "ok" if ok2 else "FAIL (goal occluded, as expected)",
          "pos_err_mm=%.2f" % quality2["pos_err_mm"], "quality=", quality2)


if __name__ == "__main__":
    main()
