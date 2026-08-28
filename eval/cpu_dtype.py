"""Which dtype should SigLIP load in on a CPU, and does the answer move a result?

model-browser's web demo has no GPU: it serves the query API from a small VM,
so the text tower runs on CPU cores. `load_siglip` asked for fp16 whatever the
device — the 4060's dtype, and the one every cache was embedded under. This
harness loads the same checkpoint four ways on the CPU and, per variant,
records what a query costs and what it returns, through the production path
(`Collection`, `embed_texts`, `query.score`, `query.rank`):

  fp16    what shipped before 2026-08-28: `torch_dtype=torch.float16` on cpu
  fp32    what `load_siglip` now picks for a cpu device
  int8    fp32 with the text tower's Linear layers dynamically quantised —
          the usual "make it faster on CPU" move, kept here as the control
          that shows what a *real* change to the embeddings looks like
  text32  fp32 with only `SiglipTextModel` loaded, no vision tower — the
          usual "make it smaller" move

One variant per process, because peak memory is the number that sizes a host
and four loads in one interpreter would share it:

  cpu_dtype.py --dtype fp16        # writes eval/out/cpu_dtype-fp16.json
  cpu_dtype.py --dtype fp32        # ... and so on for each variant
  cpu_dtype.py --report            # tables from whatever JSONs are present

Agreement is read against fp32 on the ranked top-`--top` per query: top-1,
top-10 overlap, Jaccard over the whole list, the largest score movement on
the shared rows, and `best_z` — the number the weak-match line is drawn on.

Recorded run, 2026-08-28 — embed-cache-test (2165 models, the demo corpus at
~/Documents/tests/test-models/miniatures/deduplicated), Ryzen 9 7940HS, torch
default 8 threads, 16 queries: see LEARNINGS, "fp16 on a CPU". Pin the core
count with `OMP_NUM_THREADS=4 taskset -c 0-3` to approximate a 4-vCPU host.
"""
import argparse
import json
import resource
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import query                                        # noqa: E402
from src.cachedir import add_cache_args, apply_run_params    # noqa: E402
from src.collection import Collection                        # noqa: E402

OUT = Path(__file__).resolve().parent / "out"
BASE = "fp32"
VARIANTS = ("fp16", "fp32", "int8", "text32")
QUERIES = [
    "a vampire", "guy with a crossbow crouching", "dragon", "a knight on horseback",
    "skeleton warrior", "wizard casting a spell", "treasure chest", "goblin archer",
    "a giant spider", "dwarf with a hammer", "stone wall ruins", "a werewolf",
    "an elf ranger", "a large demon", "a small house", "barrel",
]


def rss_mb() -> float:
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024
    return float("nan")


def peak_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def load(model_name: str, dtype: str):
    import torch
    from src.embedder import load_siglip

    if dtype == "text32":
        # The checkpoint's model_type is "siglip", so the text class is
        # SiglipTextModel; the full model's get_text_features is just
        # self.text_model(...), which `as_tensor` unwraps either way.
        from transformers import AutoProcessor, SiglipTextModel
        processor = AutoProcessor.from_pretrained(model_name, local_files_only=True)
        tower = SiglipTextModel.from_pretrained(model_name, dtype=torch.float32,
                                                local_files_only=True).eval()

        class TextOnly:
            def get_text_features(self, **kw):
                return tower(**kw)
        return TextOnly(), processor

    want = {"fp16": torch.float16, "fp32": torch.float32, "int8": torch.float32}[dtype]
    model, processor = load_siglip(model_name, "cpu", torch_dtype=want)
    if dtype == "int8":
        model.text_model = torch.quantization.quantize_dynamic(
            model.text_model, {torch.nn.Linear}, dtype=torch.qint8)
    return model, processor


def measure(args) -> None:
    import torch
    from src.embedder import embed_texts

    c = Collection.load(args)
    print(f"cache {args.cache_dir} — {len(c.files)} models, root {c.root}")
    t0 = time.monotonic()
    model, processor = load(args.model, args.dtype)
    loaded_s = time.monotonic() - t0
    after_load = rss_mb()
    print(f"{args.dtype}: loaded in {loaded_s:.1f}s, {after_load:.0f} MB resident, "
          f"threads={torch.get_num_threads()}")

    embed_texts(model, processor, ["warm up"], "cpu")       # first call pays setup
    latencies, per_query = [], {}
    for q in QUERIES:
        t0 = time.perf_counter()
        text_T = embed_texts(model, processor, [q], "cpu").float().cpu().numpy().T
        sims = query.score(c.matrix, text_T, args.pool).ravel()
        ranked = query.rank(sims, top=args.top)
        latencies.append(time.perf_counter() - t0)
        hits = [c.hit(int(j), float(sims[j]), float(ranked.z[j])) for j in ranked.order]
        per_query[q] = {"order": [h["rel_path"] for h in hits],
                        "score": {h["rel_path"]: h["score"] for h in hits},
                        "best_z": float(ranked.z[ranked.best])}
    result = {
        "dtype": args.dtype, "model": args.model, "cache_dir": str(args.cache_dir),
        "models": len(c.files), "threads": torch.get_num_threads(), "top": args.top,
        "pool": args.pool, "load_s": loaded_s, "rss_after_load_mb": after_load,
        "rss_after_queries_mb": rss_mb(), "peak_rss_mb": peak_mb(),
        "latency_s": latencies, "queries": per_query,
    }
    OUT.mkdir(exist_ok=True)
    path = OUT / f"cpu_dtype-{args.dtype}.json"
    path.write_text(json.dumps(result, indent=1))
    print(f"median {statistics.median(latencies):.3f}s  p90 "
          f"{sorted(latencies)[int(len(latencies) * .9) - 1]:.3f}s  "
          f"peak {result['peak_rss_mb']:.0f} MB  -> {path}")


def report() -> None:
    runs = {}
    for v in VARIANTS:
        p = OUT / f"cpu_dtype-{v}.json"
        if p.exists():
            runs[v] = json.loads(p.read_text())
    if BASE not in runs:
        sys.exit(f"no {BASE} run to compare against — run --dtype {BASE} first")
    base = runs[BASE]
    print(f"{base['cache_dir']}: {base['models']} models, {base['threads']} threads, "
          f"top {base['top']}, pool {base['pool']}, {len(QUERIES)} queries\n")
    print(f"{'variant':8} {'median':>8} {'p90':>8} {'load':>6} {'after load':>11} "
          f"{'after queries':>14} {'peak':>8}")
    for v, r in runs.items():
        lat = r["latency_s"]
        print(f"{v:8} {statistics.median(lat):7.3f}s {sorted(lat)[int(len(lat) * .9) - 1]:7.3f}s "
              f"{r['load_s']:5.1f}s {r['rss_after_load_mb']:8.0f} MB "
              f"{r['rss_after_queries_mb']:11.0f} MB {r['peak_rss_mb']:5.0f} MB")

    print(f"\nagreement with {BASE}, per query (top {base['top']}):")
    for q in QUERIES:
        b = base["queries"][q]
        print(f"- {q!r}  {BASE} best_z={b['best_z']:.2f}")
        for v, r in runs.items():
            if v == BASE:
                continue
            a = r["queries"][q]
            bo, ao = b["order"], a["order"]
            top10 = len(set(bo[:10]) & set(ao[:10]))
            jac = len(set(bo) & set(ao)) / max(1, len(set(bo) | set(ao)))
            shared = set(b["score"]) & set(a["score"])
            dmax = max((abs(b["score"][k] - a["score"][k]) for k in shared), default=0.0)
            print(f"    {v:7} top1={'same' if bo[:1] == ao[:1] else 'DIFF'} "
                  f"top10={top10:2d}/10 jaccard={jac:.2f} max|dScore|={dmax:.4f} "
                  f"best_z={a['best_z']:.2f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    add_cache_args(parser, "STL directory (defaults to the last classify run)")
    parser.add_argument("--dtype", choices=VARIANTS, help="measure one variant")
    parser.add_argument("--report", action="store_true", help="compare the recorded runs")
    parser.add_argument("--top", type=int, default=60)
    parser.add_argument("--pool", choices=["mean", "max", "softmax"], default="softmax")
    args = apply_run_params(parser)
    if args.report:
        report()
    elif args.dtype:
        if not args.input:
            sys.exit("no input given, and none recorded by classify_stls.py")
        measure(args)
    else:
        parser.error("one of --dtype or --report")


main()
