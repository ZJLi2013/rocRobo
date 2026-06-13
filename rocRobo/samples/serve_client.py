"""Sample: consume rocRobo across a process boundary (for torch-side callers).

    python samples/serve_client.py

Launches `python -m rocrobo.serve --serve` (the warm worker) and talks to it over
stdin/stdout JSON lines — the caller never imports jax. Demonstrates the contract a
torch-rocm consumer (e.g. RoboSmith) uses: send {op, pos, quat, world}, read
{ok, q, quality}. The first call pays JIT compile; subsequent same-shape calls are warm.
"""
import json
import os
import subprocess
import sys
import time

from rocrobo import pipeline as rp


def main():
    (gp, gq) = rp.goal_pose
    req = {"op": "solve_ik", "pos": [float(x) for x in gp], "quat": [float(x) for x in gq]}
    blocked = dict(req, world=[{"type": "halfspace", "point": [0, 0, 0], "normal": [0, 0, 1]},
                               {"type": "box", "center": [float(x) for x in gp],
                                "extent": [0.15, 0.15, 0.15]}])

    proc = subprocess.Popen([sys.executable, "-m", "rocrobo.serve", "--serve"],
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=subprocess.DEVNULL, text=True, env=os.environ)

    def call(r):
        t = time.time()
        proc.stdin.write(json.dumps(r) + "\n")
        proc.stdin.flush()
        return json.loads(proc.stdout.readline()), round(time.time() - t, 2)

    r1, t1 = call(req)
    r2, t2 = call(req)
    r3, _ = call(blocked)
    proc.stdin.close()
    proc.wait(timeout=15)

    print("call#1 (cold JIT): ok=%s  %.2fs" % (r1["ok"], t1))
    print("call#2 (warm)    : ok=%s  %.2fs" % (r2["ok"], t2))
    print("blocked world    : ok=%s (goal occluded, as expected)" % r3["ok"])


if __name__ == "__main__":
    main()
