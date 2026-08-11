#!/usr/bin/env bash

HF_HUB_OFFLINE=1 .venv/bin/python classify_stls.py /run/media/masa/Files\ and\ S/STL/ --out results.csv --save-renders my_renders
