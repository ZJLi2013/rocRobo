"""feature3 E1: success metrics + PlanResult on top of v0 plan().

evaluate(traj) -> (success, checks, metrics). success = all checks pass.
Validates: feature2 old solution (endpoint ~6cm) is judged FAIL on endpoint,
while collision / limits pass.

Run inside pyroki/examples/ (reuses f2_e3_plan setup).
"""
import numpy as np
import jax
import jax.numpy as jnp

from . import pipeline as v0

robot, link_idx = v0.robot, v0.link_idx
lo, hi = v0.lo, v0.hi
robot_coll, world_coll = v0.robot_coll, v0.world_coll
GROUND, WALL = world_coll[0], world_coll[1]

# thresholds
POS_TOL_MM = 10.0
ROT_TOL_DEG = 5.0
GROUND_TOL = -0.03   # absorb sphere-decomp base artifact (rest pose already ~-0.02)
SELF_TOL = 0.01      # relative to start-config self distance


def quat_angle_deg(q1, q2):
    q1 = np.asarray(q1) / np.linalg.norm(q1)
    q2 = np.asarray(q2) / np.linalg.norm(q2)
    return float(np.degrees(2 * np.arccos(min(1.0, abs(float(q1 @ q2))))))


def ee_pose(cfg):
    fk = np.array(robot.forward_kinematics(jnp.array(cfg)))[link_idx]  # wxyz_xyz
    return fk[4:7], fk[:4]


def evaluate(traj, start_pose, goal_pose, world=None, rcoll=None):
    """world=None -> default scene (wall/ground checks, unchanged). When an injected
    world is given, gate on the min clearance over ALL its primitives (feature8 P3).
    rcoll=None -> module body model; pass an attached_collision() so the self/world
    clearance gates cover the grasped payload too (feature9)."""
    rcoll = robot_coll if rcoll is None else rcoll
    traj_j = jnp.array(traj)
    (sp, sq), (gp, gq) = start_pose, goal_pose
    p0, q0 = ee_pose(traj[0])
    pT, qT = ee_pose(traj[-1])
    start_pos_mm = float(np.linalg.norm(p0 - sp) * 1000)
    end_pos_mm = float(np.linalg.norm(pT - gp) * 1000)
    start_rot_deg = quat_angle_deg(q0, sq)
    end_rot_deg = quat_angle_deg(qT, gq)

    self_d = np.array(rcoll.compute_self_collision_distance(robot, traj_j))
    self_min = float(self_d.min())
    self_min_start = float(self_d[0].min())

    # compute_world_collision_distance asserts on batched cfg -> vmap per timestep
    def _world(geom):
        d = jax.vmap(lambda c: rcoll.compute_world_collision_distance(robot, c, geom))(traj_j)
        return float(np.array(d).min())

    within = bool((traj >= lo - 1e-3).all() and (traj <= hi + 1e-3).all())
    max_step = float(np.abs(np.diff(traj, axis=0)).max())

    checks = {
        "endpoint": (start_pos_mm < POS_TOL_MM and end_pos_mm < POS_TOL_MM
                     and start_rot_deg < ROT_TOL_DEG and end_rot_deg < ROT_TOL_DEG),
        "self_clear": self_min > self_min_start - SELF_TOL,
        "limits": within,
    }
    metrics = dict(
        start_pos_mm=round(start_pos_mm, 2), end_pos_mm=round(end_pos_mm, 2),
        start_rot_deg=round(start_rot_deg, 2), end_rot_deg=round(end_rot_deg, 2),
        self_min_m=round(self_min, 4), self_min_start_m=round(self_min_start, 4),
        max_joint_step=round(max_step, 4), within_limits=within,
    )
    if world is None:
        wall_min, ground_min = _world(WALL), _world(GROUND)
        checks["wall_clear"] = wall_min > 0.0
        checks["ground_clear"] = ground_min > GROUND_TOL
        metrics["wall_min_m"] = round(wall_min, 4)
        metrics["ground_min_m"] = round(ground_min, 4)
    else:
        world_min = min(_world(wc) for wc in world)
        checks["world_clear"] = world_min > GROUND_TOL
        metrics["world_min_m"] = round(world_min, 4)
    success = bool(all(checks.values()))
    return success, checks, metrics


class PlanResult:
    def __init__(self, success, traj, metrics, checks, reason=None, n_retries=0):
        self.success, self.traj, self.metrics = success, traj, metrics
        self.checks, self.reason, self.n_retries = checks, reason, n_retries

    def __repr__(self):
        return f"PlanResult(success={self.success}, reason={self.reason}, n_retries={self.n_retries})"


