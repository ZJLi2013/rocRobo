"""feature4 E1: joint-space RRT-Connect feasibility gate.

collision_free(cfg) reuses PyRoki robot_coll (self/wall/ground, feature3 thresholds).
rrt_connect(q_start, q_goal) -> collision-free waypoints (numpy, CPU).
Gate: RRT finds a collision-free path on the wall scene; straight-line init collides.
"""
import numpy as np
import jax
import jax.numpy as jnp

from . import pipeline as v0

robot, link_idx, dof = v0.robot, v0.link_idx, v0.dof
lo, hi = v0.lo, v0.hi
robot_coll, world_coll = v0.robot_coll, v0.world_coll
GROUND, WALL = world_coll[0], world_coll[1]

GROUND_TOL = -0.03
SELF_TOL = 0.01

_self_one = jax.jit(lambda c: robot_coll.compute_self_collision_distance(robot, c[None]).min())
_world_one = jax.jit(lambda c, g: jax.vmap(
    lambda cc: robot_coll.compute_world_collision_distance(robot, cc, g))(c[None]).min())

# rest-pose self distance baseline (sphere-decomp artifact, same as feature3)
_START_SELF = None


def _set_baseline(cfg):
    global _START_SELF
    _START_SELF = float(_self_one(jnp.asarray(cfg)))


def collision_free(cfg):
    c = jnp.asarray(cfg)
    self_ok = float(_self_one(c)) > _START_SELF - SELF_TOL
    wall_ok = float(_world_one(c, WALL)) > 0.0
    ground_ok = float(_world_one(c, GROUND)) > GROUND_TOL
    return bool(self_ok and wall_ok and ground_ok)


def edge_free(qa, qb, res=0.05):
    n = max(2, int(np.linalg.norm(qb - qa) / res))
    for s in np.linspace(0, 1, n + 1)[1:]:
        if not collision_free(qa + s * (qb - qa)):
            return False
    return True


def rrt_connect(q_start, q_goal, step=0.3, max_iter=2000, seed=0):
    rng = np.random.default_rng(seed)
    Ta, Pa = [np.asarray(q_start)], [-1]
    Tb, Pb = [np.asarray(q_goal)], [-1]

    def nearest(tree, q):
        d = np.linalg.norm(np.asarray(tree) - q, axis=1)
        return int(d.argmin())

    def steer(qf, qt):
        v = qt - qf
        n = np.linalg.norm(v)
        return qt if n <= step else qf + step * v / n

    def extend(tree, par, q):
        i = nearest(tree, q)
        qn = steer(tree[i], q)
        if edge_free(tree[i], qn):
            tree.append(qn); par.append(i)
            return ("reached" if np.allclose(qn, q) else "advanced"), len(tree) - 1
        return "trapped", i

    for it in range(max_iter):
        q_rand = lo + (hi - lo) * rng.uniform(size=dof)
        stat, ia = extend(Ta, Pa, q_rand)
        if stat != "trapped":
            qnew = Ta[ia]
            while True:
                stat2, ib = extend(Tb, Pb, qnew)
                if stat2 != "advanced":
                    break
            if stat2 == "reached" and edge_free(Tb[ib], qnew):
                # connected: backtrack both
                pa = []
                k = ia
                while k != -1:
                    pa.append(Ta[k]); k = Pa[k]
                pa = pa[::-1]
                pb = []
                k = ib
                while k != -1:
                    pb.append(Tb[k]); k = Pb[k]
                path = pa + pb
                return np.array(path), it + 1
        Ta, Pa, Tb, Pb = Tb, Pb, Ta, Pa  # swap trees
    return None, max_iter


def shortcut(path, iters=100, res=0.05, seed=1):
    rng = np.random.default_rng(seed)
    path = list(path)
    for _ in range(iters):
        if len(path) <= 2:
            break
        i = rng.integers(0, len(path) - 1)
        j = rng.integers(i + 1, len(path))
        if j - i > 1 and edge_free(path[i], path[j], res):
            path = path[:i + 1] + path[j:]
    return np.array(path)


