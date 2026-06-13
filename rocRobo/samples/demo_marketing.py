"""Marketing demo: collision-blind vs rocRobo collision-aware, side-by-side.

    python samples/demo_marketing.py [out.mp4]

Same start/goal, same wall. LEFT = trajopt with the wall ignored (collision-blind)
-> the arm plows straight through. RIGHT = rocRobo's collision-aware plan -> the
arm arcs over the wall. An orbiting camera + a red "COLLISION" flash on the blind
side make the contrast obvious. Renders an mp4 + a poster PNG.
"""
import sys
import numpy as np
import jax
import jax.numpy as jnp

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import imageio.v2 as imageio

from rocrobo import pipeline as rp


def _chain_fk(traj):
    names = list(rp.robot.links.names)
    chain = [i for i, n in enumerate(names) if "finger" not in n]
    fk = np.array(jax.vmap(rp.robot.forward_kinematics)(jnp.array(traj)))[..., 4:7]
    return fk[:, chain, :], fk[:, rp.link_idx, :]


def _hits_wall(points):
    lo = rp.wall_pos - rp.wall_extent / 2.0
    hi = rp.wall_pos + rp.wall_extent / 2.0
    inside = np.all((points >= lo) & (points <= hi), axis=-1)
    return bool(np.any(inside))


def _panel(ax, chain_t, ee_trail, azim, title, color, collided):
    gx, gy = np.meshgrid(np.linspace(-0.2, 0.9, 2), np.linspace(-0.6, 0.6, 2))
    ax.plot_surface(gx, gy, np.zeros_like(gx), alpha=0.08, color="gray")
    wall_color = "red" if collided else "firebrick"
    ax.add_collection3d(Poly3DCollection(
        rp.cuboid_faces(rp.wall_pos, rp.wall_extent),
        alpha=0.45 if collided else 0.30, facecolor=wall_color, edgecolor="black"))
    if len(ee_trail) > 1:
        ax.plot(ee_trail[:, 0], ee_trail[:, 1], ee_trail[:, 2],
                color="gold", lw=2.5, alpha=0.9)
    ax.plot(chain_t[:, 0], chain_t[:, 1], chain_t[:, 2], "-o",
            color=color, lw=4, ms=5)
    ax.scatter(*rp.start_pose[0], color="lime", s=55, marker="^")
    ax.scatter(*rp.goal_pose[0], color="red", s=70, marker="*")
    ax.set_xlim(-0.2, 0.9); ax.set_ylim(-0.6, 0.6); ax.set_zlim(0, 0.9)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.view_init(elev=20, azim=azim)
    ttl = title + ("   ** COLLISION **" if collided else "")
    ax.set_title(ttl, color="white", fontsize=12, pad=2)


def main(out="demo_marketing.mp4"):
    (sp, sq), (gp, gq) = rp.start_pose, rp.goal_pose
    cfg_start, _ = rp.multiseed_ik(sp, sq)
    cfg_goal, _ = rp.multiseed_ik(gp, gq, near_cfg=cfg_start)

    print("solving collision-aware (with wall)...", flush=True)
    aware = rp.trajopt_from_cfgs(cfg_start, cfg_goal, rp.TIMESTEPS, rp.DT)
    print("solving collision-blind (wall ignored)...", flush=True)
    blind = np.array(rp._trajopt_solve(cfg_start, cfg_goal, rp.TIMESTEPS, rp.DT,
                                       100, world=[]))

    b_chain, b_ee = _chain_fk(blind)
    a_chain, a_ee = _chain_fk(aware)
    T = len(aware)

    frames = []
    for t in range(T):
        azim = -75 + 60.0 * t / max(1, T - 1)
        fig = plt.figure(figsize=(12.8, 5.0), dpi=110, facecolor="#10131a")
        for j, (chain, ee, title, color) in enumerate([
                (b_chain, b_ee, "collision-blind", "#9aa7b4"),
                (a_chain, a_ee, "rocRobo collision-aware", "#3aa0ff")]):
            ax = fig.add_subplot(1, 2, j + 1, projection="3d")
            ax.set_facecolor("#10131a")
            collided = _hits_wall(chain[t]) if j == 0 else False
            _panel(ax, chain[t], ee[:t + 1], azim, title, color, collided)
        fig.suptitle("rocRobo  |  collision-aware motion on AMD GPU (ROCm)",
                     color="white", fontsize=14, y=0.98)
        fig.canvas.draw()
        frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
        plt.close(fig)

    poster = frames[int(T * 0.6)]
    imageio.imwrite(str(out).replace(".mp4", "_poster.png"), poster)
    frames += [frames[-1]] * 12
    imageio.mimsave(out, frames, fps=12)
    print("wrote", out, "(+ poster png)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "demo_marketing.mp4")
