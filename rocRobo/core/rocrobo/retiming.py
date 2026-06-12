"""rocRobo retiming: velocity-limited (exact) + acceleration-limited (approximate)
time parameterization (feature5 E1/E2). Velocity bound is exact; rigorous accel/jerk
needs TOPP-RA/Ruckig (backlog). Pure numpy post-processing.
"""
import numpy as np

from . import pipeline as v0

robot, dof = v0.robot, v0.dof
T, DT = v0.TIMESTEPS, v0.DT


def get_vmax():
    for attr in ("velocity_limits", "velocity_limit", "vel_limits"):
        v = getattr(robot.joints, attr, None)
        if v is not None:
            arr = np.abs(np.array(v)).reshape(-1)
            if arr.shape[0] == dof and np.all(arr > 0):
                return arr
    # Panda spec fallback (7 arm joints); pad to dof
    base = np.array([2.175, 2.175, 2.175, 2.175, 2.61, 2.61, 2.61])
    return np.array([base[i] if i < 7 else 2.0 for i in range(dof)])


def retime_vel(wp, vmax, dt):
    wp = np.asarray(wp, dtype=float)
    dq = np.abs(np.diff(wp, axis=0))                 # [K-1, dof]
    seg_dt = np.maximum((dq / vmax).max(axis=1), 1e-9)
    t_knots = np.concatenate([[0.0], np.cumsum(seg_dt)])
    total = float(t_knots[-1])
    n = int(np.ceil(total / dt)) + 1
    t_new = np.linspace(0.0, total, n)
    q_new = np.stack([np.interp(t_new, t_knots, wp[:, d]) for d in range(dof)], axis=1)
    return q_new, t_new, t_knots, total


def max_joint_vel(q, dt):
    return float(np.abs(np.diff(q, axis=0) / dt).max())


# ---- acceleration-limited retiming (feature5 E2) ----

def seg_durations_vel(wp, vmax):
    dq = np.abs(np.diff(wp, axis=0))
    return np.maximum((dq / vmax).max(axis=1), 1e-9)


def accel_limit(wp, seg_dt, amax, iters=300):
    wp = np.asarray(wp, float)
    seg_dt = seg_dt.copy()
    dq = np.diff(wp, axis=0)  # [K-1, dof]
    for _ in range(iters):
        v = dq / seg_dt[:, None]                       # [K-1, dof] segment velocity
        mid = 0.5 * (seg_dt[1:] + seg_dt[:-1])         # [K-2]
        a = (v[1:] - v[:-1]) / mid[:, None]            # [K-2, dof]
        ratio = (np.abs(a) / amax).max(axis=1)         # [K-2]
        if ratio.max() <= 1.0 + 1e-6:
            break
        factor = np.ones(len(seg_dt))
        for k in range(len(ratio)):
            if ratio[k] > 1.0:
                s = np.sqrt(ratio[k])
                factor[k] = max(factor[k], s)
                factor[k + 1] = max(factor[k + 1], s)
        seg_dt = seg_dt * factor
    return seg_dt


def resample(wp, seg_dt, dt):
    t_knots = np.concatenate([[0.0], np.cumsum(seg_dt)])
    total = float(t_knots[-1])
    t_new = np.linspace(0.0, total, int(np.ceil(total / dt)) + 1)
    q = np.stack([np.interp(t_new, t_knots, wp[:, d]) for d in range(wp.shape[1])], axis=1)
    return q, t_new, total


def vel_acc(q, dt):
    v = np.diff(q, axis=0) / dt
    a = np.diff(v, axis=0) / dt
    return v, a
