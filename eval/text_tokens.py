"""Characters vs tokens for the query text budget (issue #5).

Why this exists: the figures `docs/api/surface.md`, `src/api.py`'s
`QUERY_BODY_MAX` comment and `_within_budget` cite — "100 CJK characters are N
tokens" — were first relayed by hand, and one of them was wrong in a way that
argued against the fix it was quoted to justify. The relayed claim was "100 CJK
characters are 52 tokens", measured from a *spaced* sample (`"龍 "` ×50: 50
ideographs and 50 spaces). Unspaced, which is the filer's own case (100
ideographs, 300 bytes), the same 100 characters are 101 tokens — so the number
put a query that raises inside a budget it is nearly twice.

So the counts live here, where they are re-run rather than re-typed, and the
budget is derived from the model rather than written down: this stays correct
across a change to `JOIN_RESERVE`, the templates, or the checkpoint.

    .venv/bin/python eval/text_tokens.py [--model <hf id>]

Needs only the processor and config in the local HF cache — no weights, no GPU.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.embedder import DEFAULT_MODEL, count_tokens, query_budget  # noqa: E402

# (label, text). Each is either a figure some prose cites or the sample that
# figure was mistakenly taken from.
CASES = [
    ("CJK ×100, unspaced (the filer's case)", "龍" * 100),
    ('CJK "龍 " ×50, spaced (where 52 came from)', "龍 " * 50),
    ("CJK prose, 100 chars", ("这是一个三维打印的模型文件夹里面有很多小雕像和道具" * 5)[:100]),
    ("emoji ×60", "\U0001f409" * 60),
    ("emoji ×30", "\U0001f409" * 30),
    ("ASCII words, 500 chars", ("orc warrior with axe " * 30)[:500]),
    ("ASCII words, 120 chars", ("orc warrior " * 10)[:120]),
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    args = ap.parse_args()

    import types

    from transformers import AutoConfig, AutoProcessor

    proc = AutoProcessor.from_pretrained(args.model, local_files_only=True)
    cfg = AutoConfig.from_pretrained(args.model, local_files_only=True)
    positions = cfg.text_config.max_position_embeddings
    budget = query_budget(types.SimpleNamespace(config=cfg), proc)

    print(f"model      {args.model}")
    print(f"positions  {positions}   query_budget {budget}\n")
    print(f"{'sample':<44}{'chars':>7}{'tokens':>8}{'tok/char':>10}  verdict")
    for label, text in CASES:
        n = count_tokens(proc, text)
        verdict = "ok" if n <= budget else "REFUSED (422)"
        print(f"{label:<44}{len(text):>7}{n:>8}{n / len(text):>10.2f}  {verdict}")

    print("\nCharacters do not predict tokens, and the ratio is not even stable\n"
          "within one script: the two 100-character CJK rows differ by ~2x,\n"
          "because the spaces are what let the tokenizer merge. Only the\n"
          "tokenizer can answer — which is why /query counts and /status\n"
          "publishes the budget instead of asking callers to guess in bytes.")


if __name__ == "__main__":
    main()
