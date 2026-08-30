"""Production Open3D pose tiles presented as six individual images ("solo") —
the cell f3d_arbiter left open. Reuses its ask/present machinery with the
tile directory swapped; same 44 models, same report."""
import argparse, itertools, json, threading, time
from concurrent.futures import ThreadPoolExecutor

from common import AX, OUT, load_baselines, load_labels
import f3d_arbiter as F
import gemini_vlm

F.TILE_DIR = OUT / "o3d-tiles2048"
COLS = ["g35-o3d-solo", f"{F.TAG}-o3d-solo"]
gemini_vlm.PRICES["g35-o3d-solo"] = gemini_vlm.PRICES["gemini-3.5-flash"]
gemini_vlm.PRICES[f"{F.TAG}-o3d-solo"] = F.OR_PRICES.get(F.GLM, {"in": 0, "out": 0})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--thumb", default="256,512,1024")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", default="o3d_solo.json")
    ap.add_argument("--cols", default=",".join(COLS))
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args()
    thumbs = [int(t) for t in args.thumb.split(",")]
    cols = args.cols.split(",")
    if args.report_only:
        saved = json.load(open(OUT / args.out))
        gemini_vlm.report(saved["predictions"], thumbs, cols)
        gemini_vlm.cost_report(saved["usage"])
        return
    labels = F.resolve(load_labels())
    base = load_baselines()
    items = [dict(l, **{"arb": base[l["stem"]]["needs_arbiter"],
                        "geo": base[l["stem"]]["geometry"],
                        "ens": base[l["stem"]]["ensemble_2048"]})
             for l in labels if l["stem"] in base]
    tok = gemini_vlm.token()
    usage = {}
    for t in thumbs:
        for col in cols:
            t0, u, done, lock = time.time(), [], itertools.count(1), threading.Lock()

            def one(it, col=col, t=t):
                if col.startswith("g35"):
                    v = F.ask_gemini(F.gemini_parts(it["stem"], t, True), tok, u)
                else:
                    v = F.ask_glm(F.glm_content(it["stem"], t, True), u)
                with lock:
                    print(f"  [{next(done):>3}/{len(items)}] {col}@{t} {it['stem'][:32]:32} "
                          f"{'--' if v is None else AX[v]:>3}  {time.time()-t0:4.0f}s", flush=True)
                return v

            with ThreadPoolExecutor(max_workers=args.workers) as ex:
                res = list(ex.map(one, items))
            for it, v in zip(items, res):
                it[f"{col}_sheet{t}"] = None if v is None else AX[v]
            usage[f"{col}_sheet{t}"] = u
            print(f"{col:14} @{t:<5} {time.time()-t0:5.0f}s "
                  f"({sum(v is not None for v in res)}/{len(items)})", flush=True)
    json.dump({"predictions": items, "usage": usage}, open(OUT / args.out, "w"),
              indent=1, default=str)
    gemini_vlm.report(items, thumbs, cols)
    gemini_vlm.cost_report(usage)


if __name__ == "__main__":
    main()
