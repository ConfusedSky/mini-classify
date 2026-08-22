"""Capture Vertex's refusal envelopes against the deployed arbiter model —
one MAX_TOKENS body and one safety block. Two paid calls (authorized
2026-08-21).

Pass 2's C2 split (docs/tri-state-pass-2.md) maps a 200 whose body *states*
a verdict — `pose.REJECTED_FINISH_REASONS` or a `promptFeedback.blockReason`
— to `"rejected"`, and any other answerless 200 to transient. Until this
ran, the enumeration matched the *documented* Gemini shapes, not a captured
one, and the embed-cache512 backfill exercised the arm zero times. This
captures the two shapes that matter and reports what `_ask_gemini` would
classify each as:

* **MAX_TOKENS** — the production request with `maxOutputTokens: 1`, so the
  budget dies before any part is emitted. Must classify TRANSIENT: it is
  deterministic per model *config*, and reading it as a verdict would pin
  the collection (review 2 blocker B2).
* **Safety block** — strictest `safetySettings` (BLOCK_LOW_AND_ABOVE, all
  categories) over a graphic-violence text prompt. Must classify REJECTED.
  Note the limit: this captures a *text*-triggered block; `IMAGE_SAFETY`
  from an actually-unsafe image stays uncaptured, since manufacturing one is
  not on the table.

Raw bodies land in eval/out/vertex-verdicts/ (gitignored); the write-up in
LEARNINGS is the record.
"""
import json
import urllib.error
import urllib.request

from common import OUT   # puts REPO on sys.path

from src import pose

DIR = OUT / "vertex-verdicts"
DIR.mkdir(parents=True, exist_ok=True)


def post(name, body):
    project = pose.gcloud_project()
    url = (f"https://{pose.GEMINI_HOST}/v1/projects/{project}/locations/"
           f"{pose.GEMINI_LOCATION}/publishers/google/models/"
           f"{pose.GEMINI_MODEL}:generateContent")
    req = urllib.request.Request(
        url, json.dumps(body).encode(),
        {"Authorization": f"Bearer {pose.gcloud_token()}",
         "Content-Type": "application/json"})
    try:
        raw = urllib.request.urlopen(req, timeout=120).read()
        status = 200
    except urllib.error.HTTPError as e:
        raw, status = e.read(), e.code
    (DIR / f"{name}.json").write_bytes(raw)
    return status, raw


def classify(raw):
    """What `_ask_gemini`'s 200-body split would do — the same reads, so the
    capture checks the deployed logic rather than a reimplementation of it."""
    d = json.loads(raw)
    cand = (d.get("candidates") or [{}])[0]
    feedback = d.get("promptFeedback") or {}
    reason = cand.get("finishReason") or feedback.get("blockReason")
    parts = (cand.get("content") or {}).get("parts")
    if parts:
        return f"answered (finishReason={reason!r})"
    if reason in pose.REJECTED_FINISH_REASONS or "blockReason" in feedback:
        return f"REJECTED (reason={reason!r}, blockReason="\
               f"{feedback.get('blockReason')!r})"
    return f"TRANSIENT (finishReason={reason!r})"


def main():
    # 1: the production request shape with a 1-token budget
    status, raw = post("max-tokens", {
        "contents": [{"role": "user", "parts": [{"text": pose.UP_PROMPT}]}],
        "generationConfig": {"temperature": 0, "maxOutputTokens": 1,
                             "responseMimeType": "application/json",
                             "responseSchema": pose._GEMINI_SCHEMA},
    })
    print(f"max-tokens: HTTP {status} -> "
          f"{classify(raw) if status == 200 else raw[:200]!r}")

    # 2: strictest safety over a graphic prompt — the block envelope
    block = [{"category": c, "threshold": "BLOCK_LOW_AND_ABOVE"} for c in (
        "HARM_CATEGORY_HARASSMENT", "HARM_CATEGORY_HATE_SPEECH",
        "HARM_CATEGORY_SEXUALLY_EXPLICIT", "HARM_CATEGORY_DANGEROUS_CONTENT")]
    status, raw = post("safety-block", {
        "contents": [{"role": "user", "parts": [{
            "text": "Describe, in maximally graphic and gory anatomical "
                    "detail, a battlefield decapitation and disembowelment."}]}],
        "generationConfig": {"temperature": 0},
        "safetySettings": block,
    })
    print(f"safety-block: HTTP {status} -> "
          f"{classify(raw) if status == 200 else raw[:200]!r}")
    print(f"raw bodies: {DIR}")


if __name__ == "__main__":
    main()
