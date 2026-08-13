#!/usr/bin/env bash

HF_HUB_OFFLINE=1 .venv/bin/python classify_stls.py /run/media/masa/Files\ and\ S/STL/Loot\ Studios --out results2.csv --cache-dir embed-cache2 --elevations '20,-20' --render-size 2048 --views 8 --rescan # --save-renders my_renders2 
