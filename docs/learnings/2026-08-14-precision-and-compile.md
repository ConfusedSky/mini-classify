## Precision, and what it actually flips (2026-08-14)

The question that started it: torch.compile was disqualified because its
embedding drift (max 9.8e-04) matched the closest observed top-1 margin
(9e-04, 128 renders, 0 flips) — but did anything ever actually flip? Three
harnesses later the answer is measured at every level, and the flag shipped.
Harnesses: `eval/score_precision.py`, `eval/compile_flips.py`, and a
compile-mode sweep added to `eval/siglip_bench.py`. Raw results in
`eval/out/`, gitignored — the figures here are the record.

### The system has three measured perturbation scales

| perturbation | size | where it acts |
|---|---|---|
| fp16 scoring cast | 6.2e-05 max per-view sim | every scored model, every run |
| torch.compile | median 7.3e-04, max 3.1e-03 | embeddings, if compiled |
| JPEG q92 vs in-memory | ~8e-03 mean cosine | saved renders only — off the scoring path by design |

For scale: production top-1 margins are median 0.015, p1 1.75e-04, min
8.6e-07 (softmax pool, 2943 models).

### The fp16 scoring cast: 1 flip in 2,943

Production scores by casting cached fp32 embeddings *down* to fp16 for the
GPU matmul (`classify_stls.py:1078,1104`) while the pose ensemble runs fp32
numpy. Scoring every cached model both ways (plus fp64): **one top-1 flip**,
at a 1.9e-05 margin between 'terrain or scenery piece' and 'building or
ruin'; fp32 agreed with fp64 on every model, so fp32 is converged and fp16 is
the only precision effect in the system. Mean pool: 0 flips; max pool: 3, all
sub-1e-04 near-synonym ties. Not worth "fixing": moving to fp32 would change
one arbitrary answer and perturb every fourth-decimal score in the CSV.

### torch.compile: 1 flip in 341, at a margin 200× below the drift

The direct experiment: the 241 worst-margin models plus 100 controls, each
model's 16 saved views preprocessed once and pushed through eager and
compiled SigLIP as identical tensors. **One flip** —
`Slathaai_Casting_Hand_Animancy_L`, 'demon or monster' → 'beast or animal',
eager margin 4.3e-06. Even among the 44 models whose margins sat *below* the
drift magnitude, only that one flipped: being inside the band means the
perturbation can reach you, but its direction has to line up too.

Two methodology notes that earned their keep:

* **The null-run canary.** The first run reported 0.0 drift after a 1 s
  "compile" — `torch.compile(model)` only intercepts `forward()`, and
  `get_image_features` on the wrapper silently stays eager. Compile the bound
  method (`siglip_bench` had it right all along). A drift of exactly zero
  from a supposedly-compiled tower is the tell.
* **JPEG dilution is real, not theoretical.** At-risk selection used cached
  (lossless-pixel) margins, but the jpg substrate perturbs embeddings ~8e-03
  — the flipped model's cached margin was 1.7e-03 and its jpg margin
  4.3e-06, a 400× move. The comparison itself is immune (identical pixels
  through both towers); the *targeting* has to be read against the margin
  the towers actually contested.

### Compile modes: max-autotune is gated off this card

`COMPILE_MODES=default,max-autotune,reduce-overhead siglip_bench.py compile`,
A/B interleaved per mode (the eager baseline slid 23.3 → 24.5 img/s across
blocks as the card warmed — the ratios are comparable, the raw img/s are not):

| mode | speedup | compile cost | max drift |
|---|---|---|---|
| default | 1.092× | 3.7 s (cache warm) | 9.77e-04 |
| max-autotune | 1.116× | 50.7 s | 7.32e-04 |
| reduce-overhead | 1.104× | 13.0 s | 9.77e-04 (byte-identical to default) |

Inductor gates Triton GEMM autotuning behind `is_big_gpu` (≥68 SMs); the
4060 Laptop's 24 SMs close it, so "max-autotune" here tunes only the
pointwise/reduction kernels around the matmuls — 2 points over default for
14× the compile time. reduce-overhead replays default's kernels through CUDA
graphs, hence identical numerics.

### The decision: `--compile` shipped, as a cache regime

The drift was never the hazard — it flips only coin tosses. The real hazard
was the permanent `.npy` cache mixing two numeric regimes ~7e-04 apart. So
the flag lands with the regime in the cache identity:

* `classify_stls.py --compile` (BooleanOptionalAction) compiles the image
  forward, default mode — the robust choice; the mode sweep's extra 1-2% is
  not worth CUDA-graph fragility across the two production batch shapes.
* `cache_key` appends `|compiled` **only when set**, the same trick `elev`
  uses: every pre-existing eager key stays byte-identical, and the two
  regimes can never share a `.npy`.
* `compile` joins `RUN_PARAMS_KEYS`, so the regime sticks to its cache and
  flows to `test_categories` like any other identity key. Flipping regimes
  on a populated cache is an explicit act (`--no-compile` to leave) and
  costs a re-embed, which is the correct price.
* Text embeddings stay eager — they are recomputed at startup, never cached
  per-file.
