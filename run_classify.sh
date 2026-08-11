#!/usr/bin/env bash

HF_HUB_OFFLINE=1 .venv/bin/python classify_stls.py /run/media/masa/Files\ and\ S/STL/Loot\ Studios --out results.csv --save-renders my_renders --pose-vlm claude --render-size 1024 --views 8
