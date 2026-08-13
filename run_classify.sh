#!/usr/bin/env bash

HF_HUB_OFFLINE=1 .venv/bin/python classify_stls.py /run/media/masa/STLLibrary/ --out results2.csv --cache-dir embed-cache2 --elevations '20,-20' --render-size 384 --views 8 --pose-vlm off --save-renders "$@"
