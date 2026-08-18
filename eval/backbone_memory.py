"""What each SigLIP tower actually costs in memory.

  python backbone_memory.py

Weights are the number people quote; activations are the number that decides
whether the tower fits alongside anything else on an 8 GB card. patch16-512
sees 1024 image tokens against patch14-384's ~729, so its weights are
identical and its working set is not.

Run this with the GPU otherwise idle — a contended card reports whatever else
was resident.
"""
import argparse, gc

from common import REPO  # noqa: F401 — puts REPO on sys.path for src/ and rig

BACKBONES = ["google/siglip2-so400m-patch14-384", "google/siglip2-so400m-patch16-512"]
MB = 1024 ** 2


def measure(model_id, batch):
    import torch
    from PIL import Image
    import rig

    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()

    # The Embedder is the production tower load. Its `__init__` also embeds the
    # prompt banks, so `weights` here carries the four bank rows on top of the
    # parameters — kilobytes against gigabytes, and the same overhead in both
    # rows, so the comparison this table exists for is untouched.
    e = rig.embedder(model_id, embed_batch=0)
    weights = torch.cuda.memory_allocated() - base

    px = e.processor.image_processor.size["height"]
    patch = e.model.config.vision_config.patch_size
    tokens = (px // patch) ** 2
    params = sum(p.numel() for p in e.model.parameters())

    imgs = [Image.new("RGB", (2048, 2048), "white") for _ in range(batch)]
    torch.cuda.reset_peak_memory_stats()
    e.embed_images(imgs)
    peak = torch.cuda.max_memory_allocated() - base
    reserved = torch.cuda.max_memory_reserved()

    del e
    gc.collect(); torch.cuda.empty_cache()
    return {"params": params, "px": px, "patch": patch, "tokens": tokens,
            "weights": weights, "peak": peak, "reserved": reserved}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=6, help="6 = one model's up-candidate tiles")
    args = ap.parse_args()

    import torch
    if not torch.cuda.is_available():
        raise SystemExit("no CUDA device — this measures VRAM")
    total = torch.cuda.get_device_properties(0).total_memory
    # fp16 only, because that is the load production does (src/embedder.py) and
    # this now measures that load rather than a re-creation of it. The --dtype
    # sweep went with the copy; an fp32 figure is 2x the weights row and was
    # never a number the pipeline had to fit.
    print(f"{torch.cuda.get_device_name(0)}  {total/MB:.0f} MiB total, "
          f"dtype float16, batch {args.batch}\n")

    rows = {m: measure(m, args.batch) for m in BACKBONES}
    hdr = (f"{'backbone':30} {'params':>10} {'input':>7} {'tokens':>7} "
           f"{'weights':>9} {'peak':>9} {'reserved':>9} {'% of card':>10}")
    print(hdr); print("-" * len(hdr))
    for m, r in rows.items():
        print(f"{m.split('/')[-1]:30} {r['params']/1e6:>9.0f}M "
              f"{str(r['px'])+'px':>7} {r['tokens']:>7} "
              f"{r['weights']/MB:>8.0f}M {r['peak']/MB:>8.0f}M "
              f"{r['reserved']/MB:>8.0f}M {100*r['reserved']/total:>9.0f}%")

    a, b = (rows[m] for m in BACKBONES)
    print(f"\nweights differ by {abs(b['weights']-a['weights'])/MB:.0f} MiB "
          f"({abs(b['params']-a['params'])/1e6:.1f}M params) — same tower, different "
          f"patch and position embeddings.\nActivations are where they diverge: "
          f"{b['tokens']}/{a['tokens']} = {b['tokens']/a['tokens']:.2f}x the image tokens, "
          f"peak {b['peak']/a['peak']:.2f}x.")


if __name__ == "__main__":
    main()
