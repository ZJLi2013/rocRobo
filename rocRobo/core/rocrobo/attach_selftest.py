"""feature9 regression self-test — ACO attached collision object.

Validates that a grasped payload (sphere group), attached to the EE link and
threaded through the SAME collision detector as the arm, makes IK / trajopt /
evaluate cover "payload vs world" and "payload vs self" (with holding-gripper
exemption). Deterministic + fast (no random goals); pairs with docs/exp/part9-exp.md.

Run inside the rocRobo jax container:
    python -m rocrobo.attach_selftest
(needs ROCROBO_ASSETS pointing at panda_spheres.json, like rocrobo.motiongen).
"""
import json
from pathlib import Path

import numpy as np
import jax.numpy as jnp
import jaxlie

from . import pipeline as v0
from .motiongen import MotionGen

# Small near-gripper payload (E1/E3/E4): stays inside the open default scene.
SPHERES = [{"center": [0.0, 0.0, 0.05], "radius": 0.03}]
# Probe payload (E2): pushed beyond the fingertips so the BARE arm clears a tiny
# box placed at it while the payload sphere penetrates ("手臂无碰、负载怼障").
PROBE_SPHERES = [{"center": [0.0, 0.0, 0.20], "radius": 0.04}]
T_EE_OBJ = np.eye(4).tolist()
FINGERS = [k for k in v0.panda_spheres if "finger" in k]   # holding-gripper links


def _ngeoms(rcoll):
    return int(len(rcoll._geom_to_link_idx))


def _payload_world_center(q, spheres):
    """World position of the payload sphere center at config q (for probe placement)."""
    fk = np.array(v0.robot.forward_kinematics(jnp.array(q)))[v0.link_idx]   # wxyz_xyz
    R = np.array(jaxlie.SO3(jnp.array(fk[:4])).as_matrix())
    return fk[4:7] + R @ np.array(spheres[0]["center"], float)


def main():
    reg = {}
    base = v0.robot_coll
    att = v0.attached_collision(SPHERES, T_EE_OBJ, exempt_self_links=FINGERS)
    att_no_ex = v0.attached_collision(SPHERES, T_EE_OBJ, exempt_self_links=())
    q0 = np.array(v0.robot.joint_var_cls(0).default_factory())

    # E1: payload spheres merged into the EE link's set.
    reg["E1"] = dict(n_base=_ngeoms(base), n_attach=_ngeoms(att),
                     n_added=_ngeoms(att) - _ngeoms(base),
                     pairs_base=len(base.active_idx_i),
                     pairs_attach=len(att.active_idx_i),
                     fingers=FINGERS,
                     ok=bool(_ngeoms(att) == _ngeoms(base) + len(SPHERES)))
    print("E1", json.dumps(reg["E1"]), flush=True)

    # E2: a tiny probe box at the (beyond-fingertip) payload location -> bare arm
    # clears (>0), but with the payload attached the same detector reports a hit (<0).
    att_probe = v0.attached_collision(PROBE_SPHERES, T_EE_OBJ, exempt_self_links=FINGERS)
    c = _payload_world_center(q0, PROBE_SPHERES)
    probe = [v0.pk.collision.Box.from_extent(extent=np.full(3, 0.02), position=c)]
    body_clear = v0.config_quality(q0, world=probe)["world_clear"]
    pay_clear = v0.config_quality(q0, world=probe, rcoll=att_probe)["world_clear"]
    reg["E2"] = dict(body_world_clear=body_clear, payload_world_clear=pay_clear,
                     ok=bool(body_clear > 0.0 and pay_clear < 0.0))
    print("E2", json.dumps(reg["E2"]), flush=True)

    # E3: self-collision exemption. Two facts:
    #  (a) Franka's fingers are IMMEDIATE adjacents of panda_hand -> PyRoki already
    #      auto-ignores them, so the payload(on hand)-vs-finger pairs are exempt for
    #      free (finger exempt_self_links is a no-op: pair count unchanged).
    #  (b) exempt_self_links still removes pairs for a NON-adjacent holding link
    #      (the general gripper case); proven by dropping a named arm link's pairs.
    names = list(v0.robot.links.names)
    counts = np.bincount(np.array(att_no_ex._geom_to_link_idx), minlength=len(names))
    banned = {v0.target_link, "panda_link7"} | {n for n in names if "finger" in n}
    exempt_link = max((int(counts[i]), names[i]) for i in range(len(names))
                      if names[i] not in banned and counts[i] > 0)[1]
    att_user = v0.attached_collision(SPHERES, T_EE_OBJ, exempt_self_links=[exempt_link])
    reg["E3"] = dict(fingers_auto_exempt=bool(len(att.active_idx_i) == len(att_no_ex.active_idx_i)),
                     pairs_no_exempt=len(att_no_ex.active_idx_i),
                     exempt_link=exempt_link, pairs_user_exempt=len(att_user.active_idx_i),
                     ok=bool(len(att.active_idx_i) == len(att_no_ex.active_idx_i)
                             and len(att_user.active_idx_i) < len(att_no_ex.active_idx_i)))
    print("E3", json.dumps(reg["E3"]), flush=True)

    # E4: plan_motion end-to-end with the payload attached (default scene) succeeds.
    mg = MotionGen()
    qs, _ = v0.multiseed_ik(*v0.start_pose, n=16)
    attach = dict(spheres=SPHERES, T_ee_obj=T_EE_OBJ, exempt_self_links=tuple(FINGERS))
    res = mg.plan_motion(qs, [v0.goal_pose], attach=attach)
    reg["E4"] = dict(success=res.success, reason=res.reason,
                     world_min_m=res.metrics.get("world_min_m"),
                     timed_ok=bool(res.timed is not None),
                     ok=bool(res.success and res.timed is not None))
    print("E4", json.dumps(reg["E4"]), flush=True)

    # E5: attach=None leaves the body path identical (default plan still succeeds).
    res0 = mg.plan(v0.start_pose, v0.goal_pose)
    reg["E5"] = dict(success=res0.success, end_mm=res0.metrics["end_pos_mm"],
                     ok=bool(res0.success))
    print("E5", json.dumps(reg["E5"]), flush=True)

    reg["pass"] = bool(all(reg[k]["ok"] for k in ("E1", "E2", "E3", "E4", "E5")))
    print("REGRESSION_PASS", reg["pass"], flush=True)

    outdir = Path("/work/f9/out")
    outdir.mkdir(parents=True, exist_ok=True)
    with open(outdir / "f9_attach_regression.json", "w") as f:
        json.dump(reg, f, indent=2)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
