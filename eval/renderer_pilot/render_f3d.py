"""Arm: f3d AO+AA, 512px, white background, 16 views at the cache's angles,
up axis from the pose cache. Output renders/f3d/<key>_v<i>.png"""
import json, math, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).parent
OUT = HERE / "renders/f3d"; OUT.mkdir(parents=True, exist_ok=True)
S = json.load(open(HERE / "sample.json"))
ONLY = sys.argv[2] if len(sys.argv) > 2 else None
if ONLY:
    S["sample"] = [m for m in S["sample"] if ONLY in m["key"]]
UP = {(0, 0, 1): "+Z", (0, 0, -1): "-Z", (0, 1, 0): "+Y", (0, -1, 0): "-Y", (1, 0, 0): "+X", (-1, 0, 0): "-X"}


def one(m):
    up = UP[tuple(int(round(c)) for c in m["up"])]
    for i, (az, el) in enumerate(S["angles"]):
        out = OUT / f"{m['key']}_v{i}.png"
        if out.exists():
            continue
        cmd = ["f3d", m["path"], "--config=none", f"--up={up}", f"--camera-azimuth-angle={math.degrees(az):.3f}",
               f"--camera-elevation-angle={math.degrees(el):.3f}", "--resolution=512,512",
               f"--output={out}", "--ambient-occlusion", "--anti-aliasing",
               "--background-color=1,1,1", "--grid=false", "--axis=false", "--filename=false"]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if r.returncode != 0 or not out.exists():
            print(f"FAIL {m['key']} v{i}: {r.stderr[-200:]}", flush=True)
    return m["key"]


t0 = time.time()
with ThreadPoolExecutor(max_workers=int(sys.argv[1]) if len(sys.argv) > 1 else 4) as ex:
    for n, k in enumerate(ex.map(one, S["sample"]), 1):
        if n % 20 == 0:
            print(f"[{n}/{len(S['sample'])}] {time.time()-t0:.0f}s", flush=True)
print(f"done {time.time()-t0:.0f}s", flush=True)
