"""Has gemini's thinking per arbiter call grown, or were the models just harder?

    .venv/bin/python eval/gemini_thinking_drift.py

Reruns production's prompt today on a random draw from each population with a
saved run: the labels (`gemini_206.json`) and the gated sample
(`arbiter_agreement_1000_s0.json`). Saved usage rows name no model, so the
comparison is between populations' distributions, not model by model. Drift
moves both; hardness leaves the labels where they were.
"""
import json, random
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from common import OUT, load_labels

import gemini_vlm
from src import pose

LABELS_N, SAMPLE_N = 30, 20


def stats(u):
    t = sorted(x["thoughts"] for x in u)
    return f"n {len(t):3}  mean {np.mean(t):6.0f}  median {t[len(t) // 2]:6}  p90 {t[int(.9 * len(t))]:6}  max {t[-1]:6}"


def main():
    saved = {"labels": json.loads((OUT / "gemini_206.json").read_text())["usage"]["gemini-3.5-flash_sheet512"],
             "sample": json.loads((OUT / "arbiter_agreement_1000_s0.json").read_text())["usage"]["gemini"]}
    rng = random.Random(0)
    draws = {"labels": rng.sample([l["stem"] for l in load_labels()], LABELS_N),
             "sample": rng.sample([i["stem"] for i in json.loads(
                 (OUT / "arbiter_agreement_1000_s0.json").read_text())["items"] if i["gated"]], SAMPLE_N)}
    tok, today = gemini_vlm.token(), {"labels": [], "sample": []}
    with ThreadPoolExecutor(max_workers=4) as ex:
        list(ex.map(lambda a: gemini_vlm.ask(pose.GEMINI_MODEL, OUT / "sheets512" / f"{a[1]}.png",
                                             tok, usage=today[a[0]]),
                    [(k, s) for k, v in draws.items() for s in v]))
    (OUT / "gemini_thinking_drift.json").write_text(json.dumps({"draws": draws, "usage": today}, indent=1))
    p = gemini_vlm.PRICES[pose.GEMINI_MODEL]
    for k in today:
        print(f"{k:7} saved  {stats(saved[k])}\n{k:7} today  {stats(today[k])}")
    spent = sum(x["prompt"] * p["in"] + (x["thoughts"] + x["output"]) * p["out"]
                for u in today.values() for x in u) / 1e6
    print(f"spent ${spent:.2f}")


if __name__ == "__main__":
    main()
