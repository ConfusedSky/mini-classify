#!/usr/bin/env bash
#
# Remove derived artifacts so the next classify run recomputes them.
#
# Everything a run derives lives under the cache directory now. By default this
# clears only the two cheap-to-rebuild parts of it — <cache>/renders/ and
# <cache>/embeds/ — and leaves the three files beside them alone:
#
#   pose-cache.json   the up-candidate renders + a SigLIP forward per model, and
#                     any "vlm" entry also cost a paid arbiter call
#   walk-*.json       a full rescan of the collection, which lives on removable
#                     media
#   run-params.json   run_categories.sh passes no input path and reads the
#                     collection root back from here
#
# Pass one of the pose options to go further than the default.

set -euo pipefail
cd "$(dirname "$0")"  # the ./ paths below mean this repo, not the caller's cwd

CACHE=./embed-cache2
RENDERS="$CACHE/renders"
EMBEDS="$CACHE/embeds"
POSE_CACHE="$CACHE/pose-cache.json"

usage() {
  cat <<'EOF'
usage: cleanup.sh [--clear-caches | --clear-non-vlm-poses] [-n]

Deletes the saved renders and the embedding .npy files. Keeps pose-cache.json,
walk-*.json and run-params.json unless told otherwise.

  --clear-caches         also delete pose-cache.json and the walk-*.json
                         collection scans, for a fully cold next run
  --clear-non-vlm-poses  keep only the pose entries a VLM arbiter produced
                         (source: "vlm"), drop heuristic and ensemble ones.
                         Leaves the walk scans alone
  -n, --dry-run          report what would be deleted, delete nothing
  -h, --help             this message

The two pose options are mutually exclusive. run-params.json is never deleted.
EOF
}

poses=keep
dry=0
while [ $# -gt 0 ]; do
  case "$1" in
    --clear-caches|--clear-non-vlm-poses)
      if [ "$poses" != keep ]; then
        echo "cleanup.sh: --clear-caches and --clear-non-vlm-poses are mutually exclusive" >&2
        exit 2
      fi
      [ "$1" = --clear-caches ] && poses=all || poses=nonvlm
      ;;
    -n|--dry-run) dry=1 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "cleanup.sh: unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
  shift
done

[ "$dry" = 1 ] && echo "dry run — nothing will be deleted"

# -mindepth 1 rather than a ./dir/* glob: it matches dotfiles too, and it does
# not leave an unmatched literal behind when the directory is already empty.
if [ -d "$RENDERS" ]; then
  echo "renders: $(find "$RENDERS" -mindepth 1 -maxdepth 1 | wc -l) config dirs under $RENDERS"
  [ "$dry" = 0 ] && find "$RENDERS" -mindepth 1 -maxdepth 1 -exec rm -rf {} +
else
  echo "renders: $RENDERS not present"
fi

if [ -d "$EMBEDS" ]; then
  echo "embeddings: $(find "$EMBEDS" -maxdepth 1 -name '*.npy' | wc -l) .npy files under $EMBEDS"
  [ "$dry" = 0 ] && find "$EMBEDS" -mindepth 1 -delete
else
  echo "embeddings: $EMBEDS not present"
fi

case "$poses" in
  keep)
    echo "pose + walk caches: kept (--clear-caches or --clear-non-vlm-poses to clear)"
    ;;
  all)
    echo "pose + walk caches: deleting pose-cache.json and" \
         "$(find "$CACHE" -maxdepth 1 -name 'walk-*.json' | wc -l) walk scan(s)"
    if [ "$dry" = 0 ]; then
      rm -f "$POSE_CACHE"
      find "$CACHE" -maxdepth 1 -name 'walk-*.json' -delete
    fi
    ;;
  nonvlm)
    if [ -f "$POSE_CACHE" ]; then
      # Rewrite through a temp file: a crash mid-write would otherwise leave a
      # truncated cache, which is worse than either keeping or deleting it.
      .venv/bin/python - "$POSE_CACHE" "$dry" <<'PY'
import json, os, sys

path, dry = sys.argv[1], sys.argv[2] == "1"
with open(path) as fh:
    entries = json.load(fh)
kept = {k: v for k, v in entries.items()
        if isinstance(v, dict) and v.get("source") == "vlm"}
print(f"pose cache: keeping {len(kept)} vlm entries, "
      f"dropping {len(entries) - len(kept)} heuristic/ensemble")
if not dry:
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(kept, fh)
    os.replace(tmp, path)
PY
    else
      echo "pose cache: $POSE_CACHE not present"
    fi
    echo "walk caches: kept"
    ;;
esac
