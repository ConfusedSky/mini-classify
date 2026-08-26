# Loading SigLIP without the hub — 2026-08-26

`serve_api.py` needed the internet to start, against a model that was already
on disk. So did the REPL and the eval harnesses. Nothing was downloading —
`from_pretrained` on a **repo id** revalidates every file's etag against the
hub before serving it from cache, and that round-trip was the dependency.

All measurements here are against `embed-cache512`'s model,
`google/siglip2-so400m-patch16-512` — a complete 4.3 GB snapshot in
`~/.cache/huggingface/hub`, weights and tokenizer and preprocessor config, on
the system disk. A `pytest tests/ -q` run was live during the server test, so
**no wall-clock figure here is a timing** except where the number is a
timeout that expired.

## The failure is a stall, not an error

Offline was simulated with `HF_ENDPOINT=http://127.0.0.1:1` — a dead port, so
every hub request is refused instantly rather than hanging in DNS. The old
call, a bare `AutoProcessor.from_pretrained(repo_id)`, was still going when a
120 s timeout killed it:

```
'[Errno 111] Connection refused' … HEAD …/resolve/main/audio_tokenizer_config.json
Retrying in 1s [Retry 1/5]. … Retrying in 2s [Retry 2/5]. …
```

Two things worth keeping. The retry ladder is **per file**, and transformers
probes for optional files the repo does not contain — `audio_tokenizer_config
.json` above — so the wait multiplies by files that were never going to
exist. And a refused connection is the *fast* version: real offline adds DNS
resolution on top. Nothing distinguishes this from a wedged server; it never
reaches an error you can catch and report.

## The fix, and where it lives

`src/embedder.py:load_siglip(model_name, device)` — `local_files_only=True`
first, retrying without it on `OSError`, which is the shape transformers
raises for a missing snapshot (`LocalEntryNotFoundError`) and a partial one
alike. The retry is what a bare `local_files_only=True` would cost: a model
you have never pulled still downloads on first use.

It is now the one load path. The five call sites that each ran their own
`AutoModel` + `AutoProcessor` pair — `Embedder.__init__`, `serve_api.py`,
`test_categories.py`, `eval/score_precision.py`, `eval/compile_flips.py` —
all go through it, which is the same consolidation `embed_raw`/`embed_texts`
got in the eval-debt cleanup and for the same reason: the offline-first
lookup is a property of loading SigLIP in this project, not of any one entry
point.

Verified against the real model, dead endpoint, in-process: `SiglipModel` and
`SiglipProcessor` both load. End to end, `serve_api.py --cache-dir
embed-cache512` under the same dead endpoint warms to

```json
{"ready":true,"device":"cuda","n_models":3380,"n_views":16,"dim":1152,
 "missing":0,"failure":null}
```

and `POST /query {"text":"a dragon","top":3}` returns *Orguss the Tall —
Green Dragon* at z = 3.93. That last step is the one that matters: it runs
`embed_texts` through the loaded model and processor, so what is proven
offline is the whole SigLIP round trip, not just the constructor.

## The processor loads first, and that is not cosmetic

Inside `load(local_only)` the processor is built before the model. On a
missing or partial snapshot the first call is the one that raises — so with
the model first, a doomed offline attempt pays a full fp16 load onto the 4060
before failing.

That copy does not go away when the exception does. The traceback holds the
raising frame, and the frame holds its locals, for as long as the `except`
block runs — which is exactly when the retry loads the *second* copy.
Demonstrated with a `__del__` sentinel rather than on the card:

```
inside except: first object freed? False
after except block: True
```

So model-first peaks at two resident copies on a path whose whole purpose is
to recover from a failure. Processor-first is cheap CPU-side state, and the
model is built once.

`tests/test_embedder.py::test_load_siglip_tries_the_local_cache_first` pins
both properties by recording `(half, local_files_only)` per call:
`[("processor", True), ("model", True)]` cached, and
`[("processor", True), ("processor", False), ("model", False)]` uncached —
the second sequence is the load-order guarantee written down, since a
model-first implementation cannot produce it. Falsified three ways: dropping
`local_files_only` from the processor call, swapping the load order, and
loading online-first each fail it.

## What is not established

`HF_ENDPOINT` at a dead port proves no request is **attempted** against the
hub, which is the mechanism that was failing. A genuinely disconnected
machine also exercises DNS, one layer below what this blocks. And the
partial-snapshot case — processor files present, weights absent — takes the
retry with the cheap half loaded twice; that path is reasoned about, not
measured.

`HF_HUB_OFFLINE=1` in the environment still works and still does more (it
covers anything in the process that reaches for the hub, not just this
loader). `tests/test_pose_vlm_prompt.py` sets it, along with
`TRANSFORMERS_OFFLINE=1`, so its bogus-model CLI runs fail fast on both
attempts rather than retrying past the network.
