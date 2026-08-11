#!/usr/bin/env bash

HF_HUB_OFFLINE=1 .venv/bin/python classify_stls.py /run/media/masa/Files\ and\ S/STL/Loot\ Studios --out results.csv --save-renders my_renders --elevations '20,89,-89,-20' # --render-size 1024 --views 8
