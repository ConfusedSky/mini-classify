# The query text budget: characters are not tokens

*2026-09-08. Repo issue #5, filed from model-browser's third review pass and
corrected by its fourth. Measured against `serve_api.py` on `embed-cache512`
(SigLIP-2 so400m patch16-512, 16 views, 3,092 models) and against the
tokenizer alone.*

## What broke

`POST /query` answered **500** for a long `text`, and the first report read the
limit as "roughly 600 characters". It is not a character limit. SigLIP2's text
tower is `max_position_embeddings = 64`, and `embed_raw` tokenized with
`padding="max_length"` and no `truncation` — which pads *to* 64 and does not
clip past it. A longer text reached `SiglipTextModel` as a 122-token batch and
it raised:

    ValueError: Sequence length must be less than max_position_embeddings
    (got `sequence length`: 122 and max_position_embeddings: 64)

Unhandled, so FastAPI answered 500 and uvicorn closed the keep-alive
connection. The consumer's next pooled request saw a reset, and `askIndex`
classifies a connection failure as the index being *absent* — correctly, for
the condition it names — so one over-long query silenced the listing
annotations for every viewer until the next semantic route re-probed. The
filer's fourth pass could not reproduce the reset (proper 500s at every size up
to 60 KB), so that half stays unconfirmed; the 500 itself reproduced every
time, and does so on the pre-fix server still running on 8077.

## The unit

Tokens, and characters do not predict them. Every row below was measured with
the serving checkpoint's own tokenizer and put to two live servers — the
unfixed one on 8077, the fixed one on 8078 — in one pass, so the columns are
one run rather than three recollections:

| text | chars | tokens | old | new |
|---|---|---|---|---|
| ascii words | 108 | 22 | 200 | 200 |
| ascii words | 304 | 66 | **500** | 422 |
| ascii words | 486 | 92 | **500** | 422 |
| `"龍"` ×100 | 100 | 101 | **500** | 422 |
| `"龍 "` ×50 | 100 | 52 | 200 | 200 |
| CJK prose | 100 | 57 | **500** | 422 |
| 🐉 ×30 | 30 | 31 | 200 | 200 |
| 🐉 ×60 | 60 | 61 | **500** | 422 |
| ascii words | 60,200 | — | **500** | 422 (body guard) |

An emoji is a token each; ASCII words run about one token per five characters.
CJK has no single ratio — the two 100-character rows above differ by a factor
of two, because the spaces in `"龍 "` ×50 are what let the tokenizer merge. The
unspaced row is the filer's own case (100 ideographs, 300 bytes), and it is the
one that reached the tower as more positions than it has.

So model-browser's 500-*character* bound refused long English and still let a
100-character CJK query through to a 500 — the wrong unit enforced on the wrong
side. **The corrected figures were added 2026-09-08 in a second pass**: the
first write-up generalised "about one token per two characters" from the spaced
row alone, and that ratio was then copied into `surface.md`, `src/api.py` and a
test docstring as "100 CJK characters are 52 tokens" — a claim that puts the
filer's failing query *inside* the budget and quietly argues against the
fix it was cited to justify. The counts now live in `eval/text_tokens.py`,
which derives the budget through `query_budget` and prints this table against
the real tokenizer — re-run rather than retyped, and correct across a later
change to the budget's arithmetic.

## The budget is 54, and it is derived

`query_budget(model, processor)` reads the tower's width off the loaded model
and subtracts the room the templates take, because what the caller sends is not
what is embedded: unless it asks for `raw`, its text is wrapped in
`PROMPT_TEMPLATES` first. The longest, `"a 3D render of a {} miniature"`, costs
10 tokens empty; 9 after giving back the EOS a bare text pays anyway.

**The empty-slot cost is not an upper bound on the wrapping**, which the first
version of this fix assumed and a review caught. A leading space can be
absorbed by the template's own whitespace piece and split the word behind it:
`" nü …"` re-segments at the join and wraps **10** tokens wider than it
measures alone. Fuzzing 4,000 mixed ASCII/CJK/emoji/accented strings against
all three templates found 10 as the worst and 9 as the usual, so one token is
held back (`JOIN_RESERVE`) and the budget is 64 − 9 − 1 = **54**. Without it a
55-token text builds a 65-token prompt for a 64-position tower; that is not a
crash any more — `truncation=True` absorbs it — but the clip eats the
template's last word and the query is embedded against something the caller did
not write.

54 is what the serving model reports live, and `patch14-384` computes the same
54 (both measured; the two checkpoints share the text tower's width and its
tokenizer). Nothing writes the number down — a different checkpoint reports its
own, which is the point of deriving it.

## What shipped

* `embed_raw` passes `truncation=True`. Every text forward in the project goes
  through it, so no caller — REPL, eval harness, render child — can hand the
  tower a sequence it has no positions for. Clipping, not raising, is the only
  behaviour the model has room for.
* `/query` refuses past the budget with a **422** naming both numbers:
  `query is 61 tokens; this model embeds 54 (see /status text_budget)`. Before
  the scope resolve, since it is about the body alone.
* `text_seam` builds the count and the forward over one model **behind one
  lock** — see below.
* `/status` publishes `text_budget` (null while warming), so a consumer bounds
  its own search box against a number it reads rather than one it guesses.
* A coarse 4,000-character schema guard, so an absurd body is refused before
  anything tokenizes it. It is a bound on the request, not a statement about
  the model.

## The counter had to take the forward's lock

The first version of this fix counted tokens in the route, outside the GPU
lock, on the processor the forward shares. A fast tokenizer is mutable: every
call reconfigures the Rust backend's padding and truncation before encoding, so
two threads inside it raise `RuntimeError: Already borrowed`. Four counting
threads against two embedding threads produced **117,725** of them in 8
seconds; a live server on the 4060 turned one in ~430 requests into a 500.
Unhandled, each is exactly the failure this commit exists to remove — the fix
had reintroduced its own bug in a new place.

`embedder.text_seam` is where both halves are built now, sharing one lock,
beside the processor they share rather than in the route that calls them.
Counting costs ~0.1 ms; serialising it is free.

Worth generalising: the API's threadpool makes every *shared mutable* thing a
route touches a concurrency question, and a tokenizer does not look mutable
from the call site.

## What else moved

`embed_raw`'s clipping is project-wide, so the REPL, the eval harnesses and the
render child now silently clip a >64-token text where they used to raise.
Nothing in the pipeline sends one — the longest pose prompt is 15 tokens and
category prompts are shorter — and text embeddings are not cached, so no cached
value and no measured figure moves.

## The lesson worth keeping

A model limit stated in the wrong unit is worse than no limit stated: both
sides enforced 500-of-something, and neither bound was the one that mattered.
When a bound belongs to a model, read it off the model and publish it — the
consumer cannot enforce what it has to guess.
