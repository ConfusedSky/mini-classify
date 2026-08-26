#!/usr/bin/env bash
#
# Masa's quick manual test run — not the project's main entry point, and not
# the primary cache. `embed-cache4` below is a small scratch cache for trying
# things out; the real collection lives in a cache that is not committed. So
# don't read the flags here as production settings, and don't quote a number
# measured against this cache as a fact about the library — the two differ in
# size and in what they hold (2026-08-19: a pose-distribution figure taken
# from this cache was wrong by 3.7x for exactly that reason).

HF_HUB_OFFLINE=1 .venv/bin/python classify_stls.py /home/masa/Documents/tests/test-models/thingiverse/source/ --out results-test.csv --cache-dir embed-cache-test --elevations '20,-20' --render-size 512 --views 8 --pose-vlm off --save-renders --compile --model "google/siglip2-so400m-patch16-512" "$@"
