"""rocRobo core pipeline: robot/scene + collision-aware multi-seed IK + trajopt +
config quality + batched trajopt + render (the validated v0 building blocks).

The IK step uses N random seeds in parallel and picks the best (rocRobo's own
orchestration, not a single rest-pose seed). `world` is an injectable jit arg.
Requires a CWD containing `assets/panda_spheres.json` (PyRoki examples assets).
"""
import json
import os
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
import jaxlie
import jaxls
import pyroki as pk
from robot_descriptions.loaders.yourdfpy import load_robot_description

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import imageio.v2 as imageio

urdf = load_robot_description("panda_description")
# Contract (readme / up_integration): rocRobo plans the 7 ARM DOF only; the gripper is
# commanded by the consumer. The stock panda URDF actuates panda_finger_joint1 (+ joint2
# mimic), making num_actuated_joints=8 and breaking the 7-DOF q_start contract (a 7-vs-8
# broadcast in multiseed_ik). Weld the fingers to `fixed` at load so IK/trajopt operate
# on exactly the 7 arm joints end-to-end (dof/lo/hi/seeds/FK all collapse to 7).
for _j in urdf.joint_map.values():
    if "finger" in _j.name:
        _j.type = "fixed"
        _j.mimic = None
robot = pk.Robot.from_urdf(urdf)
target_link = "panda_hand"
link_idx = robot.links.names.index(target_link)
dof = int(robot.joints.num_actuated_joints)
lo = np.array(robot.joints.lower_limits)
hi = np.array(robot.joints.upper_limits)

_ASSETS = Path(os.environ.get("ROCROBO_ASSETS", "assets"))
with open(_ASSETS / "panda_spheres.json") as f:
    panda_spheres = json.load(f)
robot_coll = pk.collision.RobotCollision.from_sphere_decomposition(
    sphere_decomposition=panda_spheres, urdf=urdf)


def attached_collision(spheres, T_ee_obj, exempt_self_links=(), ee_link=target_link):
    """Rigid grasped payload as an attached collision object (ACO) — feature9.

    Merge `spheres` into the EE link's sphere set so they ride the EE FK and join
    BOTH world and self collision via the SAME detector as the arm (no new kernel):
      - spheres: [{"center":[x,y,z],"radius":r}, ...] in the PAYLOAD frame.
      - T_ee_obj: 4x4 payload pose relative to the EE frame (panda_hand).
      - exempt_self_links: holding-gripper link names whose self-collision vs the
        payload is exempted (intended contact; payload vs the EE link itself is
        auto-exempt as same-link geoms are never self-paired).
    Returns a RobotCollision to feed IK / trajopt / evaluate (the `rcoll` arg).
    """
    if ee_link not in robot.links.names:   # protocol uses "hand"; map to URDF link
        ee_link = target_link
    T = np.asarray(T_ee_obj, float).reshape(4, 4)
    centers = np.asarray([s["center"] for s in spheres], float)
    centers_ee = centers @ T[:3, :3].T + T[:3, 3]    # payload frame -> EE link frame
    decomp = {k: {"centers": list(v["centers"]), "radii": list(v["radii"])}
              for k, v in panda_spheres.items()}
    link = decomp.setdefault(ee_link, {"centers": [], "radii": []})
    link["centers"] = list(link["centers"]) + centers_ee.tolist()
    link["radii"] = list(link["radii"]) + [float(s["radius"]) for s in spheres]
    ignore = tuple((ee_link, ex) for ex in exempt_self_links)
    return pk.collision.RobotCollision.from_sphere_decomposition(
        sphere_decomposition=decomp, urdf=urdf, user_ignore_pairs=ignore)

wall_h, wall_w, wall_l = 0.4, 0.1, 0.4
wall_pos = np.array([0.5, 0.0, wall_h / 2])
wall_extent = np.array([wall_l, wall_w, wall_h])
world_coll = [
    pk.collision.HalfSpace.from_point_and_normal(np.zeros(3), np.array([0.0, 0.0, 1.0])),
    pk.collision.Box.from_extent(extent=wall_extent, position=wall_pos),
]

down_wxyz = np.array([0.0, 0.0, 1.0, 0.0])
start_pose = (np.array([0.5, -0.35, 0.2]), down_wxyz)
goal_pose = (np.array([0.5, 0.35, 0.2]), down_wxyz)

N_SEEDS = 16
TIMESTEPS, DT = 50, 0.02
rng = np.random.default_rng(0)


def _ik_problem(target_p, target_q, world, rcoll):
    T = jaxlie.SE3.from_rotation_and_translation(jaxlie.SO3(target_q), target_p)
    jv = robot.joint_var_cls(0)
    costs = [
        pk.costs.pose_cost(robot, jv, target_pose=T,
                           target_link_index=jnp.array(link_idx),
                           pos_weight=5.0, ori_weight=1.0),
        pk.costs.rest_cost(jv, rest_pose=jnp.array(jv.default_factory()), weight=0.01),
        pk.costs.self_collision_cost(robot, robot_coll=rcoll, joint_var=jv,
                                     margin=0.02, weight=5.0),
        pk.costs.limit_constraint(robot, jv),
    ]
    costs += [pk.costs.world_collision_constraint(robot, rcoll, jv, wc, 0.05)
              for wc in world]
    return jv, jaxls.LeastSquaresProblem(costs=costs, variables=[jv]).analyze()


def _multiseed_solve(target_p, target_q, seeds, world, rcoll):
    jv, problem = _ik_problem(target_p, target_q, world, rcoll)

    def one(seed):
        return problem.solve(
            initial_vals=jaxls.VarValues.make((jv.with_value(seed),)),
            verbose=False, linear_solver="dense_cholesky")[jv]

    return jax.vmap(one)(seeds)


# world is a traced jit arg: same primitive structure (e.g. ground+box) reuses the
# compiled kernel across different geometry values (feature8 G1 — injectable world).
_multiseed_jit = jax.jit(_multiseed_solve)


def multiseed_ik(target_p, target_q, n=N_SEEDS, near_cfg=None, pos_tol=0.01, world=None, rcoll=None):
    """Solve IK from n random seeds; pick best.

    world: list of pyroki collision primitives (default = module scene world_coll).
    rcoll: robot collision model (default = module body model); pass an
           attached_collision() to make IK avoid the grasped payload too (feature9).
    If near_cfg is given, among seeds meeting pos_tol pick the one closest in
    joint space to near_cfg (keeps start/goal in the same IK branch). Otherwise
    pick the minimum position-error seed.
    """
    world = world_coll if world is None else world
    rcoll = robot_coll if rcoll is None else rcoll
    seeds = jnp.array(lo + (hi - lo) * rng.uniform(size=(n, dof)))
    sols = np.array(_multiseed_jit(jnp.array(target_p), jnp.array(target_q), seeds, world, rcoll))
    fk = np.array(jax.vmap(robot.forward_kinematics)(jnp.array(sols)))
    err = np.linalg.norm(fk[:, link_idx, 4:7] - target_p, axis=-1)
    if near_cfg is not None:
        ok = np.where(err < pos_tol)[0]
        if len(ok) > 0:
            jdist = np.linalg.norm(sols[ok] - near_cfg, axis=-1)
            best = int(ok[int(jdist.argmin())])
            return sols[best], float(err[best])
    best = int(err.argmin())
    return sols[best], float(err[best])


def config_quality(q, world=None, rcoll=None):
    """Per-config feasibility quality for RANKING IK solutions (feature8 P1a).

    Reuses the SAME collision distances as f3 evaluate() + the joint limits — no
    new FK/limit logic — so consumers (RoboSmith) rank grasp candidates on these
    instead of re-deriving FK/limit constants (single source).

    These are COMPARABLE relative quantities for ranking, NOT absolute safe/unsafe
    gates (that is what solve_ik's `ok` + f3 evaluate's checks are for): the sphere
    decomposition carries a fixed negative offset (self-spheres overlap; the ground
    half-space has a base artifact ~-0.02), so a valid config can show negative
    distances. Higher = better. Named scalars:
      limit_margin : min joint distance to a limit (rad; ~0 = riding a limit)
      world_clear  : min robot-vs-world distance (m; sensitive to obstacle proximity)
      self_clear   : min self-collision distance (m; ~constant offset, info only)
    """
    world = world_coll if world is None else world
    rcoll = robot_coll if rcoll is None else rcoll
    q = np.asarray(q, float)
    qj = jnp.asarray(q)
    limit_margin = float(min((q - lo).min(), (hi - q).min()))
    self_min = float(np.array(rcoll.compute_self_collision_distance(robot, qj[None])).min())
    world_min = min(float(np.array(rcoll.compute_world_collision_distance(robot, qj, wc)).min())
                    for wc in world)
    return dict(limit_margin=round(limit_margin, 4),
                world_clear=round(world_min, 4),
                self_clear=round(self_min, 4))


def _trajopt_solve(start_cfg, end_cfg, timesteps, dt, max_iters=100, init_traj=None, world=None, rcoll=None):
    """Pure trajopt solve: start/end cfg are traced inputs, returns jnp (T,dof).

    init_traj: optional (T,dof) initial guess; defaults to joint-space straight line.
    world: list of pyroki collision primitives the swept trajectory must avoid
           (default = module scene world_coll). Injectable per call (feature8 P3).
    rcoll: robot collision model (default = module body model); pass an
           attached_collision() so the swept world cost covers the payload (feature9).
    No np conversion / no pin -> vmap-able over (start_cfgs, end_cfgs) for batching.
    """
    world = world_coll if world is None else world
    rcoll = robot_coll if rcoll is None else rcoll
    start_cfg = jnp.asarray(start_cfg)
    end_cfg = jnp.asarray(end_cfg)
    init_traj = (jnp.linspace(start_cfg, end_cfg, timesteps)
                 if init_traj is None else jnp.asarray(init_traj))
    traj_vars = robot.joint_var_cls(jnp.arange(timesteps))
    robot_b = jax.tree.map(lambda x: x[None], robot)
    coll_b = jax.tree.map(lambda x: x[None], rcoll)

    costs = [
        pk.costs.rest_cost(traj_vars, traj_vars.default_factory()[None],
                           jnp.array([0.01])[None]),
        pk.costs.smoothness_cost(
            robot.joint_var_cls(jnp.arange(1, timesteps)),
            robot.joint_var_cls(jnp.arange(0, timesteps - 1)), jnp.array([1])[None]),
        pk.costs.five_point_acceleration_cost(
            robot.joint_var_cls(jnp.arange(2, timesteps - 2)),
            robot.joint_var_cls(jnp.arange(4, timesteps)),
            robot.joint_var_cls(jnp.arange(3, timesteps - 1)),
            robot.joint_var_cls(jnp.arange(1, timesteps - 3)),
            robot.joint_var_cls(jnp.arange(0, timesteps - 4)),
            dt, jnp.array([0.1])[None]),
        pk.costs.self_collision_cost(robot_b, coll_b, traj_vars, 0.02, 5.0),
        pk.costs.limit_constraint(robot_b, traj_vars),
    ]

    @jaxls.Cost.factory(kind="constraint_eq_zero", name="start_pose_constraint")
    def start_c(vals, var):
        return (vals[var] - start_cfg).flatten()

    @jaxls.Cost.factory(kind="constraint_eq_zero", name="end_pose_constraint")
    def end_c(vals, var):
        return (vals[var] - end_cfg).flatten()

    costs.append(start_c(robot.joint_var_cls(jnp.arange(0, 2))))
    costs.append(end_c(robot.joint_var_cls(jnp.arange(timesteps - 2, timesteps))))
    costs.append(pk.costs.limit_velocity_constraint(
        robot_b, robot.joint_var_cls(jnp.arange(1, timesteps)),
        robot.joint_var_cls(jnp.arange(0, timesteps - 1)), dt))

    def world_resid(vals, rb, cb, wco, prev, curr):
        coll = cb.get_swept_capsules(rb, vals[prev], vals[curr])
        dist = pk.collision.collide(coll.reshape((-1, 1)), wco.reshape((1, -1)))
        return dist.flatten() - 0.05

    for wc in world:
        costs.append(jaxls.Cost(
            world_resid,
            (robot_b, coll_b, jax.tree.map(lambda x: x[None], wc),
             robot.joint_var_cls(jnp.arange(0, timesteps - 1)),
             robot.joint_var_cls(jnp.arange(1, timesteps))),
            kind="constraint_geq_zero", name="world_coll_sweep"))

    sol = (jaxls.LeastSquaresProblem(costs=costs, variables=[traj_vars])
           .analyze()
           .solve(initial_vals=jaxls.VarValues.make((traj_vars.with_value(init_traj),)),
                  termination=jaxls.TerminationConfig(max_iterations=max_iters)))
    return sol[traj_vars]


def trajopt_from_cfgs(start_cfg, end_cfg, timesteps, dt, max_iters=100, pin_endpoints=False):
    traj = np.array(_trajopt_solve(start_cfg, end_cfg, timesteps, dt, max_iters))
    if pin_endpoints:
        traj[0] = np.asarray(start_cfg)
        traj[-1] = np.asarray(end_cfg)
    return traj


def batched_trajopt(start_cfgs, end_cfgs, timesteps=TIMESTEPS, dt=DT, max_iters=100, world=None):
    """vmap _trajopt_solve over (start_cfgs, end_cfgs) -> batched (B,T,dof) (feature6).

    world (shared across the batch) is closed over, not vmapped (feature8 P3).
    """
    f = lambda s, e: _trajopt_solve(s, e, timesteps, dt, max_iters, world=world)
    return jax.vmap(f)(start_cfgs, end_cfgs)


def plan(start_pose, goal_pose, timesteps=TIMESTEPS, dt=DT):
    (sp, sq), (gp, gq) = start_pose, goal_pose
    cfg_start, err_s = multiseed_ik(sp, sq)
    cfg_goal, err_g = multiseed_ik(gp, gq, near_cfg=cfg_start)
    print(f"IK best pos-err  start={err_s*1000:.2f}mm  goal={err_g*1000:.2f}mm", flush=True)
    traj = trajopt_from_cfgs(cfg_start, cfg_goal, timesteps, dt)
    return traj, cfg_start, cfg_goal


def cuboid_faces(center, extent):
    cx, cy, cz = center
    dx, dy, dz = extent / 2.0
    c = np.array([[cx-dx, cy-dy, cz-dz], [cx+dx, cy-dy, cz-dz], [cx+dx, cy+dy, cz-dz],
                  [cx-dx, cy+dy, cz-dz], [cx-dx, cy-dy, cz+dz], [cx+dx, cy-dy, cz+dz],
                  [cx+dx, cy+dy, cz+dz], [cx-dx, cy+dy, cz+dz]])
    return [[c[0], c[1], c[2], c[3]], [c[4], c[5], c[6], c[7]], [c[0], c[1], c[5], c[4]],
            [c[2], c[3], c[7], c[6]], [c[1], c[2], c[6], c[5]], [c[0], c[3], c[7], c[4]]]


def render(traj, out_path):
    names = list(robot.links.names)
    chain = [i for i, n in enumerate(names) if "finger" not in n]
    fk = np.array(jax.vmap(robot.forward_kinematics)(jnp.array(traj)))[..., 4:7]
    ee = fk[:, link_idx, :]
    frames = []
    for t in range(len(traj)):
        fig = plt.figure(figsize=(6.4, 4.8), dpi=100)
        ax = fig.add_subplot(111, projection="3d")
        gx, gy = np.meshgrid(np.linspace(-0.2, 0.9, 2), np.linspace(-0.6, 0.6, 2))
        ax.plot_surface(gx, gy, np.zeros_like(gx), alpha=0.1, color="gray")
        ax.add_collection3d(Poly3DCollection(cuboid_faces(wall_pos, wall_extent),
                            alpha=0.35, facecolor="firebrick", edgecolor="black"))
        cc = fk[t][chain]
        ax.plot(cc[:, 0], cc[:, 1], cc[:, 2], "-o", color="royalblue", lw=3, ms=4)
        ax.plot(ee[:t+1, 0], ee[:t+1, 1], ee[:t+1, 2], color="orange", lw=1.5)
        ax.scatter(*start_pose[0], color="green", s=40, marker="^")
        ax.scatter(*goal_pose[0], color="red", s=40, marker="*")
        ax.set_xlim(-0.2, 0.9); ax.set_ylim(-0.6, 0.6); ax.set_zlim(0, 0.9)
        ax.view_init(elev=22, azim=-60)
        ax.set_title(f"rocRobo v0 plan()  multiseed-IK -> trajopt  t={t+1}/{len(traj)}")
        fig.canvas.draw()
        frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
        plt.close(fig)
    key = [0, len(traj) // 3, 2 * len(traj) // 3, len(traj) - 1]
    cs = np.concatenate([np.concatenate([frames[key[0]], frames[key[1]]], axis=1),
                         np.concatenate([frames[key[2]], frames[key[3]]], axis=1)], axis=0)
    imageio.imwrite(Path(out_path).parent / "plan_contact.png", cs)
    frames += [frames[-1]] * 8
    imageio.mimsave(out_path, frames, fps=12)


