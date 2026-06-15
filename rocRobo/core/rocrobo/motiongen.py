"""rocRobo v0 — converged motion-generation pipeline (feature7b).

A single `MotionGen` entrypoint sediments the mainline features into one API,
reusing the validated building blocks (no re-implementation):
  - IK / trajopt / scene / quality / render : pipeline   (feature2/8)
  - success metrics + PlanResult            : metrics     (feature3)
  - optional RRT-Connect init               : rrt         (feature4)
  - velocity/accel retiming                  : retiming    (feature5)
  - batched (vmap) trajopt                   : pipeline.batched_trajopt (feature6)
  - per-stage backend routing                : here (feature7 thresholds)

Pipeline:  multiseed-IK (retry ladder) -> [opt RRT init] -> trajopt (pin) ->
           evaluate -> [opt retiming] -> PlanResult(+timed trajectory).

Usage:
    from rocrobo import MotionGen
    mg = MotionGen()                         # defaults = v0 standard pipeline
    res = mg.plan(start_pose, goal_pose)     # -> PlanResult (res.timed = executable)
    batch = mg.plan_batch([g1, g2, ...])     # -> BatchResult

`python -m rocrobo.motiongen` runs the feature7b regression self-test.
"""
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import jax
import jax.numpy as jnp

from . import pipeline as v0
from .metrics import evaluate, ee_pose, PlanResult
from . import rrt
from .retiming import get_vmax, seg_durations_vel, accel_limit, resample, vel_acc
from .pipeline import batched_trajopt


@dataclass
class PlanConfig:
    """v0 pipeline knobs. Defaults reproduce the validated mainline behavior."""
    timesteps: int = v0.TIMESTEPS
    dt: float = v0.DT
    ladder: tuple = (16, 64)            # multiseed-IK retry escalation (feature3)
    max_iters: int = 100               # trajopt LM iterations
    use_rrt: bool = False              # optional collision-free RRT init (feature4)
    retime: bool = True                # velocity/accel retiming (feature5)
    amax_scale: float = 5.0            # amax = amax_scale * vmax (no URDF accel limit)
    ik_gpu_threshold: int = 100        # IK GPU crossover (feature2 E4)
    trajopt_gpu_threshold: int = 4     # trajopt batch crossover (feature6)


@dataclass
class BatchResult:
    backends: dict                     # {"ik_backend":..., "traj_backend":...}
    results: list                      # list[PlanResult]

    @property
    def n_success(self):
        return sum(r.success for r in self.results)

    @property
    def success_mask(self):
        return [r.success for r in self.results]


def _resample_path(path, n):
    """Linear-resample a variable-length joint path to n points by cumulative distance."""
    path = np.asarray(path, float)
    seg = np.linalg.norm(np.diff(path, axis=0), axis=1)
    d = np.concatenate([[0.0], np.cumsum(seg)])
    if d[-1] <= 0:
        return np.repeat(path[:1], n, axis=0)
    tt = np.linspace(0.0, d[-1], n)
    return np.stack([np.interp(tt, d, path[:, j]) for j in range(path.shape[1])], axis=1)


class MotionGen:
    """rocRobo v0 motion generator — one converged pipeline over the fixed Panda+wall scene."""

    def __init__(self, config: Optional[PlanConfig] = None):
        self.cfg = config or PlanConfig()
        self.vmax = get_vmax()

    # ---- backend policy (feature7: route per dominant stage) ----
    def select_backend(self, stage, n):
        thr = self.cfg.ik_gpu_threshold if stage == "ik" else self.cfg.trajopt_gpu_threshold
        return "gpu" if n >= thr else "cpu"

    # ---- stages ----
    def _rrt_init(self, cfg_start, cfg_goal):
        rrt._set_baseline(cfg_start)
        path, _ = rrt.rrt_connect(np.asarray(cfg_start), np.asarray(cfg_goal))
        return None if path is None else _resample_path(rrt.shortcut(path), self.cfg.timesteps)

    def _trajopt_route(self, start_cfgs, goal_cfgs_j, backend, world=None):
        c = self.cfg
        if backend == "gpu":
            return np.array(batched_trajopt(start_cfgs, goal_cfgs_j, c.timesteps, c.dt,
                                            c.max_iters, world=world))
        return np.stack([np.array(v0._trajopt_solve(start_cfgs[i], goal_cfgs_j[i],
                                                     c.timesteps, c.dt, c.max_iters, world=world))
                         for i in range(len(goal_cfgs_j))])

    def retime(self, traj):
        """Velocity+accel-limited time parameterization. velocity bound exact (feature5 E1);
        acceleration approximate (rigorous accel/jerk needs TOPP-RA/Ruckig — feature5b/backlog)."""
        vmax, amax = self.vmax, self.vmax * self.cfg.amax_scale
        sd = accel_limit(traj, seg_durations_vel(traj, vmax), amax)
        q, t, total = resample(traj, sd, self.cfg.dt)
        v, a = vel_acc(q, self.cfg.dt)
        return dict(q=q, t=t, total_s=round(float(total), 3),
                    max_vel=round(float(np.abs(v).max()), 3),
                    max_acc=round(float(np.abs(a).max()), 3),
                    vel_ok=bool((np.abs(v) <= vmax.max() + 1e-6).all()))

    def _finalize(self, traj, cfg_start, cfg_goal, start_pose, goal_pose, n_retries=0, world=None, rcoll=None):
        traj = np.array(traj)
        traj[0] = np.asarray(cfg_start)
        traj[-1] = np.asarray(cfg_goal)           # endpoint pin (feature3)
        success, checks, m = evaluate(traj, start_pose, goal_pose, world=world, rcoll=rcoll)
        reason = None if success else ",".join(k for k, ok in checks.items() if not ok)
        res = PlanResult(success, traj, m, checks, reason, n_retries)
        res.timed = self.retime(traj) if self.cfg.retime else None
        return res

    def ik(self, goals, start_pose, n, world=None):
        """Shared-start multiseed IK for a list of goals -> (cfg_start, [goal_cfg])."""
        (sp, sq) = start_pose
        cfg_start, _ = v0.multiseed_ik(sp, sq, n=n, world=world)
        goal_cfgs = [v0.multiseed_ik(gp, gq, n=n, near_cfg=cfg_start, world=world)[0]
                     for (gp, gq) in goals]
        return cfg_start, goal_cfgs

    def solve_ik(self, pos, quat, seed=None, world=None, pos_tol=0.01, attach=None):
        """Collision-aware IK for a single EE pose (RoboSmith P1 entry; feature8 G1/P1a).

        world : list of pyroki collision primitives (default = module scene), injected per call.
        seed  : OPTIONAL warm-start hint only. IK is multi-seed by default (readme C2:
                single-seed ~52%, N=16 -> 100%), so callers should NOT hand-seed to dodge
                local minima — the multi-seed backend systematically handles that.
        attach: OPTIONAL grasped payload as an attached collision object (feature9):
                {"spheres":[{"center","radius"}], "T_ee_obj":4x4, "exempt_self_links":[...],
                 "ee_link":"hand"}. When set, IK + quality also avoid the payload.
        Returns (q, ok, quality):
          ok      = position error within pos_tol (absolute reach gate);
          quality = named COMPARABLE scalars for ranking candidates (limit_margin /
                    world_clear / self_clear / pos_err_mm), reusing evaluate()'s
                    distances — rank on these instead of re-deriving FK/limit
                    constants. Higher = better; not an absolute safe/unsafe gate.
        """
        rcoll = v0.attached_collision(**attach) if attach else None
        q, err = v0.multiseed_ik(pos, quat, n=self.cfg.ladder[0],
                                 near_cfg=seed, pos_tol=pos_tol, world=world, rcoll=rcoll)
        quality = v0.config_quality(q, world=world, rcoll=rcoll)
        quality["pos_err_mm"] = round(float(err * 1000), 3)
        return q, bool(err < pos_tol), quality

    # ---- public API ----
    def plan(self, start_pose, goal_pose, world=None) -> PlanResult:
        """Single goal: IK retry ladder -> [RRT init] -> trajopt -> evaluate -> [retime].

        world: injectable scene (list of pyroki primitives) avoided by IK + trajopt +
               evaluate; None = module default scene (feature8 P3)."""
        (sp, sq), (gp, gq) = start_pose, goal_pose
        res = None
        for i, N in enumerate(self.cfg.ladder):
            cfg_start, _ = v0.multiseed_ik(sp, sq, n=N, world=world)
            cfg_goal, _ = v0.multiseed_ik(gp, gq, n=N, near_cfg=cfg_start, world=world)
            init = self._rrt_init(cfg_start, cfg_goal) if self.cfg.use_rrt else None
            traj = v0._trajopt_solve(cfg_start, cfg_goal, self.cfg.timesteps, self.cfg.dt,
                                     self.cfg.max_iters, init_traj=init, world=world)
            res = self._finalize(traj, cfg_start, cfg_goal, start_pose, goal_pose, i, world=world)
            if res.success:
                break
        return res

    def plan_motion(self, q_start, waypoints, world=None, attach=None) -> PlanResult:
        """Multi-waypoint motion (RoboSmith P2 entry; feature8 P3).

        q_start   : start joint config (DOF vector).
        waypoints : list of EE poses [(pos, quat=wxyz), ...] to pass through in order.
        attach    : OPTIONAL grasped payload as an attached collision object (feature9;
                    same dict as solve_ik). When set, IK + trajopt + evaluate make the
                    payload avoid the world too ("负载 vs 环境" reuses the body detector).
        Collision-aware IK each waypoint (chained near the previous cfg), piecewise
        trajopt with the injected world, concatenate, evaluate(world) + retime.
        Endpoint check is against FK(q_start) and the final waypoint.
        """
        rcoll = v0.attached_collision(**attach) if attach else None
        q_start = np.asarray(q_start)
        cfgs = [q_start]
        for (p, q) in waypoints:
            c, _ = v0.multiseed_ik(p, q, n=self.cfg.ladder[0], near_cfg=cfgs[-1], world=world, rcoll=rcoll)
            cfgs.append(np.asarray(c))
        segs = []
        for a, b in zip(cfgs[:-1], cfgs[1:]):
            seg = np.array(v0._trajopt_solve(a, b, self.cfg.timesteps, self.cfg.dt,
                                             self.cfg.max_iters, world=world, rcoll=rcoll))
            seg[0], seg[-1] = a, b
            segs.append(seg)
        traj = np.concatenate([segs[0]] + [s[1:] for s in segs[1:]], axis=0)
        start_pose = ee_pose(q_start)
        res = self._finalize(traj, cfgs[0], cfgs[-1], start_pose, waypoints[-1], 0, world=world, rcoll=rcoll)
        return res

    def plan_batch(self, goals, start_pose=None, route=None, cfgs=None, world=None) -> BatchResult:
        """B goals from a shared start; trajopt routed batched(vmap)/sequential by size.

        route: force 'gpu'/'cpu' (else auto by trajopt_gpu_threshold).
        cfgs : reuse a precomputed (cfg_start,[goal_cfg]) IK across routes (regression).
        world: injectable scene shared across the batch (feature8 P3).
        """
        start_pose = v0.start_pose if start_pose is None else start_pose
        B = len(goals)
        ik_backend = self.select_backend("ik", B)
        traj_backend = route or self.select_backend("trajopt", B)

        cfg_start, goal_cfgs = (self.ik(goals, start_pose, n=self.cfg.ladder[0], world=world)
                                if cfgs is None else cfgs)
        start_cfgs = jnp.array(np.tile(cfg_start, (B, 1)))
        goal_cfgs_j = jnp.array(np.array(goal_cfgs))

        trajs = self._trajopt_route(start_cfgs, goal_cfgs_j, traj_backend, world=world)
        results = [self._finalize(trajs[i], cfg_start, goal_cfgs[i], start_pose, goals[i], world=world)
                   for i in range(B)]
        return BatchResult(dict(ik_backend=ik_backend, traj_backend=traj_backend), results)

    def render(self, traj, out_path):
        """Render an mp4 of a planned trajectory (passthrough to feature2 renderer)."""
        return v0.render(np.asarray(traj), out_path)


def _reachable_goals(B, seed=11):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(B):
        cfg = v0.lo + (v0.hi - v0.lo) * rng.uniform(size=v0.dof)
        fk = np.array(v0.robot.forward_kinematics(jnp.array(cfg)))[v0.link_idx]
        out.append((tuple(fk[4:7]), tuple(fk[:4])))
    return out


if __name__ == "__main__":
    print("BACKEND", jax.default_backend(), flush=True)
    reg = {}

    # R1: single plan, straight-line init (matches feature3 E1: success, end<1mm)
    mg = MotionGen()
    t0 = time.time()
    r_line = mg.plan(v0.start_pose, v0.goal_pose)
    reg["single_line"] = dict(success=r_line.success, end_mm=r_line.metrics["end_pos_mm"],
                              wall=r_line.metrics["wall_min_m"], retries=r_line.n_retries,
                              timed_max_vel=r_line.timed["max_vel"],
                              vel_ok=r_line.timed["vel_ok"], sec=round(time.time() - t0, 1))
    print("R1 single_line", json.dumps(reg["single_line"]), flush=True)

    # R2: single plan, optional RRT init (feature4) still succeeds + retimes
    mg_rrt = MotionGen(PlanConfig(use_rrt=True))
    t0 = time.time()
    r_rrt = mg_rrt.plan(v0.start_pose, v0.goal_pose)
    reg["single_rrt"] = dict(success=r_rrt.success, end_mm=r_rrt.metrics["end_pos_mm"],
                             wall=r_rrt.metrics["wall_min_m"],
                             timed_max_vel=r_rrt.timed["max_vel"],
                             vel_ok=r_rrt.timed["vel_ok"], sec=round(time.time() - t0, 1))
    print("R2 single_rrt", json.dumps(reg["single_rrt"]), flush=True)

    # R3: no-regression — batched vs sequential trajopt on the SAME IK (feature7 E1).
    # functional contract = same success DECISIONS. trajopt is non-convex (collision costs);
    # vmap changes float reduction order -> batched/sequential may reach different but
    # equally-valid local minima (pointwise diff up to ~0.1rad), so per-point equality is
    # NOT the contract (refines feature6 E1's <1e-3, which held only for its easy goals).
    mg_b = MotionGen(PlanConfig(retime=False))
    goals = _reachable_goals(6)
    cfgs = mg_b.ik(goals, v0.start_pose, n=16)        # one IK, fed to both routes
    res_seq = mg_b.plan_batch(goals, route="cpu", cfgs=cfgs)
    res_bat = mg_b.plan_batch(goals, route="gpu", cfgs=cfgs)
    diffs = [float(np.abs(res_seq.results[i].traj - res_bat.results[i].traj).max())
             for i in range(len(goals))]
    reg["batch_route"] = dict(B=len(goals), auto_traj_backend=mg_b.select_backend("trajopt", 6),
                              auto_ik_backend=mg_b.select_backend("ik", 6),
                              seq_success=res_seq.success_mask, bat_success=res_bat.success_mask,
                              same_success_set=(res_seq.success_mask == res_bat.success_mask),
                              per_goal_traj_diff=[round(d, 4) for d in diffs],
                              traj_diff_max=round(max(diffs), 4))
    print("R3 batch_route", json.dumps(reg["batch_route"]), flush=True)

    reg["pass"] = bool(reg["single_line"]["success"] and reg["single_line"]["vel_ok"]
                       and reg["single_rrt"]["success"]
                       and reg["batch_route"]["same_success_set"])
    print("REGRESSION_PASS", reg["pass"], flush=True)

    outdir = Path("/work/f7/out")
    outdir.mkdir(parents=True, exist_ok=True)
    with open(outdir / "f7b_regression.json", "w") as f:
        json.dump(reg, f, indent=2)
    print("DONE", flush=True)
