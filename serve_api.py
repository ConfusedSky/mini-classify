"""Run the query API over a cache built by classify_stls.py.

Usage:
  python serve_api.py [/path/to/stls] --cache-dir embed-cache2 [--port 8077]

Defaults for every cache-identity flag come from the last classify run's
run-params.json, exactly as test_categories.py does, so the server reads the
cache the classifier wrote without being told twice.

The port binds **before** SigLIP is resident and `/status` answers throughout,
with `ready: false` and `elapsed` until the model lands; `/query` and
`/similar` answer 503 meanwhile. That is deliberate (docs/api/surface.md
§`GET /status`): loading before binding makes a warming server
indistinguishable from a dead one, and the consumer's semantic-search
affordance would flicker off across every restart.

Nothing imports this file — it is the entry point, like classify_stls.py, and
exports nothing. The app itself is `src/api.py:create_app`, which takes a
`ServerState` and so is testable without a GPU.
"""
import argparse
import sys
import threading

from src.api import ServerState, create_app
from src.cachedir import add_cache_args, apply_run_params
from src.collection import Collection


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    add_cache_args(parser, "STL directory (defaults to the last classify run)")
    parser.add_argument("--pool", choices=["mean", "max", "softmax"],
                        default="softmax",
                        help="default view pooling; every request may override")
    parser.add_argument("--host", default="127.0.0.1",
                        help="loopback by default: the caller is model-browser's "
                             "server, not a browser (surface.md)")
    parser.add_argument("--port", type=int, default=8077)
    args = apply_run_params(parser)
    if not args.input:
        sys.exit("no input given, and no directory recorded by classify_stls.py — "
                 "pass the STL directory explicitly")

    state = ServerState(args, pool=args.pool)

    def load_embed():
        # Deferred: importing torch is the slow half of the warmup and must not
        # happen while the port is being bound.
        import torch
        from transformers import AutoModel, AutoProcessor
        from src.embedder import embed_raw, embed_texts
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"loading {args.model} on {device} ...")
        model = (AutoModel.from_pretrained(args.model, torch_dtype=torch.float16)
                 .to(device).eval())
        processor = AutoProcessor.from_pretrained(args.model)

        def embed(texts, raw=False):
            """(dim, n_texts) of unit rows — the shape `query.score` takes.

            Templated by default and verbatim under `raw`, the same choice the
            REPL's `:raw` toggle makes; both are the same matmul downstream, so
            it stays the caller's."""
            fn = embed_raw if raw else embed_texts
            return fn(model, processor, texts, device).float().cpu().numpy().T

        return embed, args.model, device

    threading.Thread(target=state.warm,
                     args=(lambda: Collection.load(args), load_embed),
                     daemon=True, name="warmup").start()

    import uvicorn
    print(f"serving on http://{args.host}:{args.port} — /status answers now, "
          f"queries once ready")
    # One worker, no reload: the matrix and SigLIP load once (surface.md §Stack).
    uvicorn.run(create_app(state), host=args.host, port=args.port,
                workers=1, log_level="info")


if __name__ == "__main__":
    main()
