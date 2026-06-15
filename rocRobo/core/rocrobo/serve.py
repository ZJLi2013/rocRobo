"""rocRobo serve: JSON process boundary (feature8 G3/P2).

RoboSmith runs in a torch-rocm process and CANNOT import jax-rocm in the same
interpreter; this sidecar runs in the rocRobo jax container and answers JSON
requests so the caller never imports jax.

Two modes — the ONLY difference is process lifetime:
  --once : read ONE json request from stdin, write one json response, exit.
           Smoke test / one-shot; pays jax import + first-call JIT each invocation.
  --serve: WARM worker (runtime mode). JIT compiles on the first request, then loops
           reading one JSON request per line (JSONL) and writing one JSON response
           per line. RoboSmith's batch data-gen pipeline calls it many times, so the
           tens-of-seconds cold start is amortized. Keep request shapes stable
           (world primitive count / batch size) to avoid recompilation.

Request:  {"op":"solve_ik","pos":[x,y,z],"quat":[w,x,y,z],"seed":[...]|null,"world":<world>}
World:    list of {"type":"box","center":[cx,cy,cz],"extent":[ex,ey,ez]}
                  {"type":"halfspace","point":[px,py,pz],"normal":[nx,ny,nz]}
          (omitted / null -> module default scene)
Attach:   OPTIONAL grasped payload as an attached collision object (feature9):
          {"ee_link":"hand","T_ee_obj":[[..4x4..]],"spheres":[{"center","radius"}],
           "exempt_self_links":["panda_leftfinger",...]}  (null -> body only)
Response: {"ok":bool,"q":[...],"quality":{...},"sec":float}  or  {"error":"..."}

Contract (see up_integration.md §E): quat=wxyz, world frame = metres, 7 arm DOF.
"""
import sys
import json
import time

import numpy as np
import pyroki as pk

from rocrobo import MotionGen


def build_world(spec):
    """JSON world spec -> list of pyroki collision primitives (None = default scene)."""
    if not spec:
        return None
    prims = []
    for p in spec:
        t = p["type"]
        if t == "box":
            prims.append(pk.collision.Box.from_extent(
                extent=np.array(p["extent"], float), position=np.array(p["center"], float)))
        elif t == "halfspace":
            prims.append(pk.collision.HalfSpace.from_point_and_normal(
                np.array(p["point"], float), np.array(p["normal"], float)))
        else:
            raise ValueError(f"unknown primitive type: {t!r} (supported: box, halfspace)")
    return prims


def build_attach(spec):
    """JSON attach spec -> attach dict for MotionGen (None = no payload; feature9).

    {"ee_link":"hand","T_ee_obj":[[..4x4..]],"spheres":[{"center","radius"}],
     "exempt_self_links":["panda_leftfinger",...]}  (all but spheres optional).
    """
    if not spec:
        return None
    return {"spheres": spec["spheres"],
            "T_ee_obj": spec.get("T_ee_obj", np.eye(4).tolist()),
            "exempt_self_links": tuple(spec.get("exempt_self_links", ())),
            "ee_link": spec.get("ee_link", "hand")}


def handle(mg, req):
    op = req.get("op", "solve_ik")
    t0 = time.time()
    if op == "solve_ik":
        world = build_world(req.get("world"))
        attach = build_attach(req.get("attach"))
        seed = np.array(req["seed"], float) if req.get("seed") is not None else None
        q, ok, quality = mg.solve_ik(np.array(req["pos"], float), np.array(req["quat"], float),
                                     seed=seed, world=world, attach=attach)
        return {"ok": bool(ok), "q": np.asarray(q).tolist(),
                "quality": quality, "sec": round(time.time() - t0, 2)}
    if op == "plan_motion":
        world = build_world(req.get("world"))
        attach = build_attach(req.get("attach"))
        waypoints = [(np.array(w["pos"], float), np.array(w["quat"], float))
                     for w in req["waypoints"]]
        res = mg.plan_motion(np.array(req["q_start"], float), waypoints, world=world, attach=attach)
        return {"ok": bool(res.success), "reason": res.reason, "metrics": res.metrics,
                "traj": np.asarray(res.traj).tolist(),
                "timed": (None if res.timed is None
                          else {"q": np.asarray(res.timed["q"]).tolist(),
                                "total_s": res.timed["total_s"], "vel_ok": res.timed["vel_ok"]}),
                "sec": round(time.time() - t0, 2)}
    return {"error": f"unknown op: {op!r}"}


def _write(resp):
    sys.stdout.write(json.dumps(resp) + "\n")
    sys.stdout.flush()


def main(argv):
    mode = argv[1] if len(argv) > 1 else "--once"
    mg = MotionGen()                       # loads URDF; first solve_ik JITs on demand
    if mode == "--once":
        try:
            resp = handle(mg, json.loads(sys.stdin.read()))
        except Exception as e:
            resp = {"error": repr(e)}
        _write(resp)
    elif mode == "--serve":
        sys.stderr.write("rocrobo.serve ready\n")
        sys.stderr.flush()
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            try:
                resp = handle(mg, json.loads(line))
            except Exception as e:
                resp = {"error": repr(e)}
            _write(resp)
    else:
        sys.stderr.write(f"unknown mode {mode!r} (use --once or --serve)\n")
        sys.exit(2)


if __name__ == "__main__":
    main(sys.argv)
