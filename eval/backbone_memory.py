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

from common import REPO  # noqa: F401 — puts REPO on sys.path for classify_stls

BACKBONES = ["google/siglip2-so400m-patch14-384", "google/siglip2-so400m-patch16-512"]
MB = 1024 ** 2


def measure(model_id, batch, dtype):
    import torch
    from PIL import Image
    from transformers import AutoModel, AutoProcessor
    import classify_stls as C

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()

    model = AutoModel.from_pretrained(model_id, torch_dtype=dtype).to(dev).eval()
    proc = AutoProcessor.from_pretrained(model_id)
    weights = torch.cuda.memory_allocated() - base

    px = proc.image_processor.size["height"]
    patch = model.config.vision_config.patch_size
    tokens = (px // patch) ** 2
    params = sum(p.numel() for p in model.parameters())

    imgs = [Image.new("RGB", (2048, 2048), "white") for _ in range(batch)]
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        C.embed_images(model, proc, imgs, dev)
    peak = torch.cuda.max_memory_allocated() - base
    reserved = torch.cuda.max_memory_reserved()

    del model, proc
    gc.collect(); torch.cuda.empty_cache()
    return {"params": params, "px": px, "patch": patch, "tokens": tokens,
            "weights": weights, "peak": peak, "reserved": reserved}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=6, help="6 = one model's up-candidate tiles")
    ap.add_argument("--dtype", default="float16", choices=["float16", "float32"])
    args = ap.parse_args()

    import torch
    if not torch.cuda.is_available():
        raise SystemExit("no CUDA device — this measures VRAM")
    dtype = getattr(torch, args.dtype)
    total = torch.cuda.get_device_properties(0).total_memory
    print(f"{torch.cuda.get_device_name(0)}  {total/MB:.0f} MiB total, "
          f"dtype {args.dtype}, batch {args.batch}\n")

    rows = {m: measure(m, args.batch, dtype) for m in BACKBONES}
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
