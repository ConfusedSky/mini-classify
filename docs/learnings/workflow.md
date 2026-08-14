## REPL affordances that proved useful

- OSC 8 terminal hyperlinks (`file://` URIs, tty-gated) — linking each result
  to its *render* (what SigLIP actually scored) beats linking the STL for
  judging classifications at a glance.
- Paths shown relative to the collection root (pack context), `:find` for
  absolute paths, `:pool` / `:min` to retune live without restart.

## Repo hygiene

- Git repo in `mini-classify/`; gitignore all derived outputs (embed-cache/,
  *.csv, *renders*/, test meshes). uv venvs self-ignore (`.venv/.gitignore`).
- The embedding cache is "derived" but represents the expensive cold pass
  (~1 h for 1000 models) — worth backing up separately once built.

## Environment / tooling

- **uv replaces conda fine for legacy ML repos** — `uv python install 3.8` +
  faithful pins (torch 2.0.0+cu118) all as prebuilt wheels; system
  CUDA 13.3/GCC 16 never involved. Add `numpy<2` for torch-2.0-era stacks.
- **Read the imports before building from source.** Find3D's README demands
  Pointcept pointops (compile) + FlashAttention ("up to 3 hours") — the
  inference path imports neither necessarily: pointops is never imported, and
  flash-attn ships prebuilt wheels (`cu118torch2.0cxx11abiFALSE-cp38`) on
  GitHub releases. Also: PTv3 has a non-flash fallback (`enable_flash=False`).
- uv + multi-index pinned requirements needs `--index-strategy
  unsafe-best-match` (first-index-wins otherwise).
- transformers 5.x: `get_text_features`/`get_image_features` return a
  `BaseModelOutputWithPooling`, not a tensor — unwrap `.pooler_output`.
- Background-shell output capture proved unreliable for long uv installs in
  this harness; foreground with a generous timeout was dependable.
