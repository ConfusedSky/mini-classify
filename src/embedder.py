"""The Embedder — synchronous, and the only module that owns torch models
(docs/actor-refactor/interfaces.md §Embedder, actors_proposal.md §Embedder).

Extracted from main:classify_stls.py with behaviour kept identical: the fp16
model load (:963-966), `--compile` wrapping the bound `get_image_features`
and nothing else (:967-974), `embed_raw`/`embed_texts`/`embed_images`
(:515-550), and the numpy prompt banks (:1040-1046). Same device pick, same
row-normalisation, same dtypes — so `.float().cpu().numpy()` of an
`Embedded.embeds` stays byte-compatible with the `.npy` cache main's path
writes (main:classify_stls.py:1190).

Both public methods block for the forward pass — in v1 that *is* the
pipeline's pacing, and torch releases the GIL so the render child renders
on. The uniform return contract (data_structures.md D5): the Embedder
returns what it computes — a normalised `torch.Tensor` on device — and
conversion is the consumer's business (the Poser does the one
`.float().cpu().numpy()`; Done keeps the tensor for its scoring matmul).

`text_embeds` is read-only after `__init__` (interfaces.md); `up_T`/`down_T`
are handed to the Poser at wiring, `front_T`/`back_T` to Done for
`front_view` resolution — plain numpy, exactly as in main.

The two text passes are also exported unbound (`as_tensor`, `embed_raw`,
`embed_texts`), because the REPL and the eval harnesses embed one-off text
against a model they loaded themselves. Those free functions came here from
`classify_stls.py` in the eval-debt cleanup, and the methods now delegate to
them — so the CLI holds no embedding code at all and there is one arrangement
of each forward rather than two pinned equal by a parity suite.
"""
from __future__ import annotations

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

# Category prompt templates (main:classify_stls.py:54-58). The one copy: every
# templated query in the project goes through `embed_texts` below.
PROMPT_TEMPLATES = [
    "a 3D render of a {} miniature",
    "a photo of a {} figurine",
    "a tabletop miniature of a {}",
]


def as_tensor(feat):
    """Some transformers versions return a pooled-output wrapper
    (main:classify_stls.py:50)."""
    return feat if isinstance(feat, torch.Tensor) else feat.pooler_output


# --- the text passes as free functions --------------------------------------
#
# The Embedder's methods below *are* these, bound to the model it owns. They
# also exist unbound because the REPL (`test_categories.py`) and four eval
# harnesses hold their own model+processor — loaded for their own reasons, with
# their own device — and want one text embedding, not a pipeline stage. That
# pair used to be a genuine second copy living in `classify_stls.py`, kept in
# step by tests/test_embedder.py's parity suite; now the methods delegate here,
# so there is one arrangement of the forward and the suite has nothing left to
# diverge.

@torch.no_grad()
def embed_raw(model, processor, texts, device):
    """Embed raw text strings (no category templates), row-normalized."""
    inputs = processor(text=list(texts), padding="max_length",
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
        text_embeds: (n_categories, dim) fp16 tensor on device — Done's
            scoring matmul runs against it.
        up_T, down_T: numpy float32 banks for the Poser's upright ensemble.
        front_T, back_T: numpy float32 banks for Done's front_view resolution.
    """

    def __init__(self, categories: Sequence[str], model_name: str = DEFAULT_MODEL,
                 device: str | None = None, compile_image_forward: bool = False,
                 embed_batch: int = 0):
        # Same device pick as main (main:classify_stls.py:960): the 4060 via CUDA.
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        from transformers import AutoModel, AutoProcessor  # deferred, as in main()
        self.model = (AutoModel.from_pretrained(model_name, torch_dtype=torch.float16)
                      .to(self.device).eval())
        self.processor = AutoProcessor.from_pretrained(model_name)
        if compile_image_forward:
            # compile the bound method — wrapping the model only intercepts
            # forward() and get_image_features silently stays eager. Lazy: the
            # first embed call of each batch shape pays the compile. Text
            # embeddings stay eager; they are not cached per-file.
            self.model.get_image_features = torch.compile(self.model.get_image_features)
        # images per SigLIP call on the view path; 0 = whole list at once
        # (--embed-batch, default 0 — main:classify_stls.py:1185-1186)
        self.embed_batch = embed_batch

        with stage("text-embed"):   # the startup stage, exclusive of the model
            self._text_embeds = self._embed_texts(categories)
            # numpy prompt banks (main:classify_stls.py:1040-1046): row-normalised
            # text features pulled off the GPU once, at startup.
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

        Whole stack in one forward, like main's tile path (the score_upright
        closure at main:classify_stls.py:1049 never passed --embed-batch); the
        tensor stays on device — the Poser pulls it off the GPU.
        """
        return TileEmbeds(file=m.file, index=m.index,
                          embeds=self.embed_images(list(m.tiles)))

    def embed_views(self, m: EmbedViews) -> Embedded:
        """Embed the classification views; the pose rides through untouched.

        The tensor stays on device for Done's scoring matmul; Done's
        `.float().cpu().numpy()` of it is the .npy cache write, byte-compatible
        with main's (main:classify_stls.py:1190).
        """
        return Embedded(file=m.file, index=m.index, pose=m.pose,
                        embeds=self.embed_images(m.views, batch=self.embed_batch))

    # --- the extracted forward passes (main:classify_stls.py:515-550) -------

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
