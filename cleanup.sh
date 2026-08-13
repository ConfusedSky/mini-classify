#!/usr/bin/env bash

# -r because renders now sit in a per-config subdirectory; plain -f cannot
# remove a directory and -f does not silence that error
rm -rf ./my_renders2/* ./embed-cache2/*
