"""The Embedder — synchronous, and the only module that owns torch models
(docs/actor-refactor/interfaces.md §Embedder, actors_proposal.md §Embedder).

Both public methods block for the forward pass — in v1 that *is* the
pipeline's pacing, and torch releases the GIL so the render child renders
on. The uniform return contract (data_structures.md D5): the Embedder
returns what it computes — a normalised `torch.Tensor` on device — and
conversion is the consumer's business (the Poser does the one
`.float().cpu().numpy()`; Done keeps the tensor for its scoring matmul).

`text_embeds` is read-only after `__init__` (interfaces.md); `up_T`/`down_T`
are handed to the Poser at wiring, `front_T`/`back_T` to Done for
`front_view` resolution — plain numpy.

The two text passes are also exported unbound (`as_tensor`, `embed_raw`,
`embed_texts`), because the REPL and the eval harnesses embed one-off text
against a model they loaded themselves. Those free functions came here from
`classify_stls.py` in the eval-debt cleanup, and the methods now delegate to
them — so the CLI holds no embedding code at all and there is one arrangement
of each forward rather than two pinned equal by a parity suite.

`load_siglip` is exported unbound for the same reason, and is the one load
path: the Embedder builds its model through it, and so do the REPL, the
server and the eval harnesses that used to call `from_pretrained` themselves
— which is what gets them all the offline-first cache lookup.
"""
from __future__ import annotations

import threading
from typing import Sequence

import numpy as np
import torch

from src.instrument import stage
from src import pose
# The one copy, kept in `identity` because it is part of every embedding key
# and `cachedir.add_cache_args` declares `--model` with it — reading it there
# must not cost a torch import (D-R1-1, amended 2026-08-18). Re-exported here
# because this is where a caller looks for the Embedder's default.
from src.identity import DEFAULT_MODEL
from src.messages import Embedded, EmbedTilesRequest, EmbedViews, TileEmbeds

# Category prompt templates. The one copy: every templated query in the
# project goes through `embed_texts` below.
PROMPT_TEMPLATES = [
    "a 3D render of a {} miniature",
    "a photo of a {} figurine",
    "a tabletop miniature of a {}",
]


def load_siglip(model_name: str, device: str, torch_dtype=None):
    """`(model, processor)` for `model_name`, off the local HF cache when it
    holds a complete snapshot, in the device's dtype unless one is given.

    fp16 is the 4060's dtype and the one every cache was embedded under. On a
    CPU it is the wrong one twice over: PyTorch has no fast half-precision
    kernels there, so a so400m text query takes 1.14 s against 0.31 s in
    fp32, and the checkpoint is fp32 on disk, so asking for fp16 converts it
    and peaks at **7.7 GB** resident against fp32's 2.8 — enough to OOM an
    8 GB host before its first query. fp32 returns the same rankings to four
    decimals. Measured on embed-cache-test, `eval/cpu_dtype.py`; LEARNINGS,
    "fp16 on a CPU".

    `from_pretrained` on a repo id revalidates every file's etag against the
    hub, so a fully cached model still needs the network to load — the server,
    the REPL and the eval harnesses all failed offline for a round-trip that
    downloads nothing. `local_files_only=True` skips it. The retry keeps a
    model you have never pulled working, since offline raises rather than
    fetches; `OSError` is the shape transformers raises for both a missing
    snapshot (`LocalEntryNotFoundError`) and a partial one. The retried block
    is the two `from_pretrained` calls and nothing else — `OSError` is a broad
    family, and a device transfer or an unrelated filesystem error caught
    alongside them would silently re-enter the loader online, turning "this
    machine has no network" into a second failure with a different cause.

    The `transformers` import stays deferred, as at every other load site: the
    signature already implies torch (interfaces.md §import rules), but the
    module is imported by tools that never load a model.
    """
    from transformers import AutoModel, AutoProcessor

    if torch_dtype is None:
        torch_dtype = torch.float16 if device.startswith("cuda") else torch.float32

    def load(local_only):
        # Processor first: it is the cheap CPU-side half, and on a partial
        # snapshot it is the half that raises. Loading the model first left a
        # copy of the weights alive for the whole retry — the traceback keeps
        # the failed frame — peaking at two.
        processor = AutoProcessor.from_pretrained(model_name,
                                                  local_files_only=local_only)
        model = AutoModel.from_pretrained(model_name, torch_dtype=torch_dtype,
                                          local_files_only=local_only)
        return model, processor

    try:
        model, processor = load(True)
    except OSError:
        model, processor = load(False)
    return model.to(device).eval(), processor


def as_tensor(feat):
    """Some transformers versions return a pooled-output wrapper."""
    return feat if isinstance(feat, torch.Tensor) else feat.pooler_output


# --- the text passes as free functions ---------------------------------------
#
# The Embedder's methods below *are* these, bound to the model it owns. They
# also exist unbound because the REPL (`test_categories.py`) and four eval
# harnesses hold their own model+processor — loaded for their own reasons, with
# their own device — and want one text embedding, not a pipeline stage. That
# pair used to be a genuine second copy living in `classify_stls.py`, kept in
# step by tests/test_embedder.py's parity suite; now the methods delegate here,
# so there is one arrangement of the forward and the suite has nothing left to
# diverge.

def count_tokens(processor, text: str) -> int:
    """Tokens `embed_raw` would hand the model for `text`, its EOS included."""
    return len(processor.tokenizer(text)["input_ids"])


# One token held back from every budget, for the tokenizer's own seam.
#
# The templates' cost is measured against an *empty* slot, and a real text does
# not always cost what it costs alone: a leading space can be absorbed by the
# template's own whitespace piece and split the word behind it, so `" nü …"`
# wraps 10 tokens wider than it measures rather than 9. Found by fuzzing 4,000
# mixed ASCII/CJK/emoji/accented strings against all three templates, where 10
# was the worst seen and 9 the usual. A reserve rather than a smarter probe
# because the failure is data-dependent and one token is cheap; `embed_raw`'s
# clipping is what catches anything this still misses, which is why an
# unreserved budget was a wrong embedding and never a crash.
JOIN_RESERVE = 1


def query_budget(model, processor) -> int:
    """The longest *user* text this pair will embed, in tokens.

    The tower is `max_position_embeddings` wide — 64 on SigLIP2 — and going
    past it is a raise, not a degradation, which is why `embed_raw` clips and
    why `/query` refuses (issue #5). What a caller sends is not what is
    embedded, though: unless it asks for `raw`, its text is wrapped in
    `PROMPT_TEMPLATES` first, so the budget subtracts the worst template's own
    tokens (9 for "a 3D render of a {} miniature": 10 empty, less the EOS a
    bare text pays anyway) and `JOIN_RESERVE` for what the wrapping can cost
    beyond that.

    One number for both modes, not two: `raw` loses ten tokens of headroom it
    could have had, and in exchange the surface states a single limit a
    consumer can bound its own input against (`/status`'s `text_budget`,
    surface.md).

    Read off the loaded model rather than written down here, so a different
    checkpoint reports its own number and no constant can go stale against it.
    """
    positions = model.config.text_config.max_position_embeddings
    return positions - max(count_tokens(processor, t.format(""))
                           for t in PROMPT_TEMPLATES) + 1 - JOIN_RESERVE


def text_seam(model, processor, device):
    """`(embed, tokens, budget)`: the server's whole text pass over one model.

    Here rather than in `serve_api` because of the lock. Both halves go through
    the same fast tokenizer, and a fast tokenizer is *mutable*: every call
    reconfigures the Rust backend's padding and truncation before encoding, so
    two threads in it at once raise `RuntimeError: Already borrowed`. The API
    runs handlers in a threadpool and counts tokens outside its GPU lock — 4
    counting threads against 2 embedding threads produced 117,725 of those in
    8 seconds (measured, 2026-09-08). That is the same unhandled-500 shape
    issue #5 exists to remove, so the count and the forward share one lock, and
    it lives beside the processor they share rather than in the route that
    happens to call them. Counting costs ~0.1 ms; serialising it is free.

    `embed` is `(texts, raw) -> (dim, n_texts)`, the shape `query.score` takes:
    templated by default and verbatim under `raw`, the same choice the REPL's
    `:raw` toggle makes. Both are the same matmul downstream, so it stays the
    caller's."""
    lock = threading.Lock()

    def embed(texts, raw=False):
        fn = embed_raw if raw else embed_texts
        with lock:
            return fn(model, processor, texts, device).float().cpu().numpy().T

    def tokens(text: str) -> int:
        with lock:
            return count_tokens(processor, text)

    return embed, tokens, query_budget(model, processor)


@torch.no_grad()
def embed_raw(model, processor, texts, device):
    """Embed raw text strings (no category templates), row-normalized.

    `truncation=True` because the text tower is 64 positions wide and the
    processor's `padding="max_length"` pads *to* 64 but does not clip past it:
    a longer text reached `SiglipTextModel` as a 122-token batch and it raised
    `ValueError: Sequence length must be less than max_position_embeddings`.
    Through the API that unhandled raise was a 500 and a closed keep-alive
    connection, which the consumer's pooled next request read as the service
    being absent rather than as one bad query (issue #5). Clipping is the only
    behaviour the model has room for; `/query` refuses past the budget with a
    422 naming both numbers (`src.api._within_budget`) and an absurd body at
    the schema (`src.api.QUERY_BODY_MAX`), so a caller sending prose hears the
    limit instead of getting a silent half-answer."""
    inputs = processor(text=list(texts), padding="max_length", truncation=True,
                       return_tensors="pt").to(device)
    feat = as_tensor(model.get_text_features(**inputs))
    return torch.nn.functional.normalize(feat, dim=-1)  # (n_texts, dim)


@torch.no_grad()
def embed_texts(model, processor, categories, device):
    """Category embeddings: each category through PROMPT_TEMPLATES, averaged."""
    embeds = []
    for cat in categories:
        prompts = [t.format(cat) for t in PROMPT_TEMPLATES]
        feat = embed_raw(model, processor, prompts, device).mean(0)
        embeds.append(torch.nn.functional.normalize(feat, dim=-1))
    return torch.stack(embeds)  # (n_categories, dim)


class Embedder:
    """Owns SigLIP, the category text embeddings, and the prompt banks.

    Attributes (all computed once in __init__, read-only thereafter):
        text_embeds: (n_categories, dim) tensor on device, in the model's
            dtype (fp16 on cuda, fp32 on cpu) — Done's scoring matmul runs
            against it.
        up_T, down_T: numpy float32 banks for the Poser's upright ensemble.
        front_T, back_T: numpy float32 banks for Done's front_view resolution.
    """

    def __init__(self, categories: Sequence[str], model_name: str = DEFAULT_MODEL,
                 device: str | None = None, compile_image_forward: bool = False,
                 embed_batch: int = 0):
        # The 4060 via CUDA; `load_siglip` picks the dtype off this.
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.processor = load_siglip(model_name, self.device)
        if compile_image_forward:
            # compile the bound method — wrapping the model only intercepts
            # forward() and get_image_features silently stays eager. Lazy: the
            # first embed call of each batch shape pays the compile. Text
            # embeddings stay eager; they are not cached per-file.
            self.model.get_image_features = torch.compile(self.model.get_image_features)
        # images per SigLIP call on the view path; 0 = whole list at once
        # (`--embed-batch`, default 0)
        self.embed_batch = embed_batch

        with stage("text-embed"):   # the startup stage, exclusive of the model
            self._text_embeds = self._embed_texts(categories)
            # numpy prompt banks: row-normalised text features pulled off the
            # GPU once, at startup.
            self.up_T = self._embed_raw(pose.UPRIGHT_PROMPTS).float().cpu().numpy()
            self.down_T = self._embed_raw(pose.TOPPLED_PROMPTS).float().cpu().numpy()
            self.front_T = self._embed_raw(pose.FRONT_PROMPTS).float().cpu().numpy()
            self.back_T = self._embed_raw(pose.BACK_PROMPTS).float().cpu().numpy()

    @property
    def text_embeds(self) -> torch.Tensor:
        """(n_categories, dim), on device — read-only after __init__."""
        return self._text_embeds

    # --- the two message-shaped entry points --------------------------------

    def embed_tiles(self, m: EmbedTilesRequest) -> TileEmbeds:
        """Embed the stacked up-candidate tiles, order-preserving.

        Whole stack in one forward — the tile path ignores `--embed-batch`,
        which is the view path's knob. The tensor stays on device; the Poser
        pulls it off the GPU.
        """
        return TileEmbeds(file=m.file, index=m.index,
                          embeds=self.embed_images(list(m.tiles)))

    def embed_views(self, m: EmbedViews) -> Embedded:
        """Embed the classification views; the pose rides through untouched.

        The tensor stays on device for Done's scoring matmul; Done's
        `.float().cpu().numpy()` of it is the `.npy` cache write, which is
        why the cache is fp32 whatever dtype the model ran in.
        """
        return Embedded(file=m.file, index=m.index, pose=m.pose,
                        embeds=self.embed_images(m.views, batch=self.embed_batch))

    # --- the forwards: the text pair bound to this model, and the image pass
    #     the evals call directly ------------------------------------------

    def _embed_raw(self, texts: Sequence[str]) -> torch.Tensor:
        """`embed_raw` bound to this Embedder's model, processor and device."""
        return embed_raw(self.model, self.processor, texts, self.device)

    def _embed_texts(self, categories: Sequence[str]) -> torch.Tensor:
        """`embed_texts` bound to this Embedder's model, processor and device."""
        return embed_texts(self.model, self.processor, categories, self.device)

    @torch.no_grad()
    def embed_images(self, images: Sequence[np.ndarray], batch: int = 0) -> torch.Tensor:
        """Row-normalised embeddings, (n_images, dim).

        batch caps how many images go to the GPU at once; 0 sends the whole
        list, which is the historical behaviour and fine at 16-40 images
        (measured peak 2.5 GB of a 7.8 GB card).

        Public, unlike the text passes: the eval harnesses embed pixels they
        rendered themselves and have no message to wrap them in, so this is
        their entry point (`eval/rig.py`). Keeping it private forced a second
        arrangement of the same forward to live in the CLI for them to call —
        which is the copy tests/test_embedder.py used to pin, and now does not
        have to."""
        batch = batch or len(images)
        out = []
        for i in range(0, len(images), batch):
            inputs = self.processor(images=images[i:i + batch],
                                    return_tensors="pt").to(self.device)
            out.append(as_tensor(self.model.get_image_features(**inputs)))
        feat = out[0] if len(out) == 1 else torch.cat(out)
        return torch.nn.functional.normalize(feat, dim=-1)
