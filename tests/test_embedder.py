"""Embedder tests, two tiers (docs/actor-refactor/interfaces.md §Embedder).

Tier (a): GPU-free contract tests — the transformers stack is faked through
sys.modules, so __init__ runs for real (text embeds, prompt banks) against a
deterministic toy model. These pin the return shapes/dtypes, the
normalisation, the read-only text_embeds, and the batch behaviour of the two
entry points.

Tier (b): one @pytest.mark.gpu test that loads real SigLIP on the 4060 and
asserts parity with the OLD code path — classify_stls.embed_images /
embed_texts / embed_raw called on the *same* model instance — for the same
inputs: allclose plus dtype and norm equality, and byte-compatibility of the
.npy cache write.
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch

import src.embedder
from src import pose
from src.embedder import DEFAULT_MODEL, PROMPT_TEMPLATES, Embedder
from src.messages import Embedded, EmbedTilesRequest, EmbedViews, TileEmbeds
from src.pose import Pose

DIM = 6  # the fake model's embedding width


# --- tier (a): the faked stack ----------------------------------------------

class FakeBatch(dict):
    """Mimics transformers' BatchFeature: a dict with .to(device)."""
    def to(self, device):
        return self


class FakeProcessor:
    def __init__(self):
        self.image_calls: list[int] = []   # images per call, in call order

    def __call__(self, images=None, text=None, padding=None, return_tensors="pt"):
        if images is not None:
            self.image_calls.append(len(images))
            px = torch.stack([
                torch.as_tensor(np.asarray(im), dtype=torch.float32).mean(dim=(0, 1))
                for im in images])                       # (n, 3) channel means
            return FakeBatch(pixel_values=px)
        seeds = torch.tensor([float(sum(t.encode()) % 997) for t in text])
        return FakeBatch(input_ids=seeds)


class FakeModel:
    """Deterministic per-input features, fp16 like the real fp16 load."""
    def to(self, device):
        return self

    def eval(self):
        return self

    def get_text_features(self, input_ids=None, **kw):
        base = torch.arange(1, DIM + 1, dtype=torch.float32)
        return ((input_ids[:, None] + 1.0) * base).to(torch.float16)

    def get_image_features(self, pixel_values=None, **kw):
        feat = torch.cat([pixel_values + 1.0, (pixel_values + 1.0) * 0.5], dim=1)
        return feat.to(torch.float16)                    # (n, 6)


@pytest.fixture
def fake(monkeypatch):
    """An Embedder built over the fake stack, on cpu."""
    proc = FakeProcessor()
    mod = types.ModuleType("transformers")
    mod.AutoModel = types.SimpleNamespace(
        from_pretrained=lambda name, torch_dtype=None: FakeModel())
    mod.AutoProcessor = types.SimpleNamespace(from_pretrained=lambda name: proc)
    monkeypatch.setitem(sys.modules, "transformers", mod)
    emb = Embedder(["dragon", "terrain"], device="cpu")
    return emb, proc


def _tiles(n, seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(n, 8, 8, 3), dtype=np.uint8)


def _pose():
    return Pose(up=(0.0, 0.0, 1.0), confidence=0.9, source="geometry",
                v=pose.POSE_CACHE_VERSION)


def test_embed_tiles_contract(fake):
    emb, _ = fake
    req = EmbedTilesRequest(file=Path("m.stl"), index=7, tiles=_tiles(4))
    out = emb.embed_tiles(req)
    assert isinstance(out, TileEmbeds)
    assert (out.file, out.index) == (req.file, req.index)
    assert isinstance(out.embeds, torch.Tensor)
    assert out.embeds.shape == (4, DIM)
    assert out.embeds.dtype == torch.float16             # the model's dtype
    norms = out.embeds.float().norm(dim=-1)
    assert torch.allclose(norms, torch.ones(4), atol=5e-3)  # row-normalised


def test_embed_views_contract(fake):
    emb, _ = fake
    p = _pose()
    m = EmbedViews(file=Path("m.stl"), index=3, pose=p, views=list(_tiles(3, seed=1)))
    out = emb.embed_views(m)
    assert isinstance(out, Embedded)
    assert (out.file, out.index) == (m.file, m.index)
    assert out.pose is p                                 # rides through untouched
    assert isinstance(out.embeds, torch.Tensor)
    assert out.embeds.shape == (3, DIM)
    assert out.embeds.dtype == torch.float16
    norms = out.embeds.float().norm(dim=-1)
    assert torch.allclose(norms, torch.ones(3), atol=5e-3)


def test_order_preserved(fake):
    emb, _ = fake
    tiles = _tiles(3, seed=2)
    fwd = emb.embed_tiles(EmbedTilesRequest(Path("a"), 0, tiles)).embeds
    rev = emb.embed_tiles(EmbedTilesRequest(Path("a"), 0, tiles[::-1].copy())).embeds
    assert torch.equal(fwd.flip(0), rev)
    assert not torch.equal(fwd[0], fwd[1])               # distinct inputs, distinct rows


def test_text_embeds_read_only(fake):
    emb, _ = fake
    with pytest.raises(AttributeError):
        emb.text_embeds = torch.zeros(1)


def test_text_embeds_shape_and_norm(fake):
    emb, _ = fake
    assert isinstance(emb.text_embeds, torch.Tensor)
    assert emb.text_embeds.shape == (2, DIM)             # one row per category
    assert emb.text_embeds.dtype == torch.float16
    norms = emb.text_embeds.float().norm(dim=-1)
    assert torch.allclose(norms, torch.ones(2), atol=5e-3)


def test_prompt_banks_numpy(fake):
    emb, _ = fake
    for bank, prompts in [(emb.up_T, pose.UPRIGHT_PROMPTS),
                          (emb.down_T, pose.TOPPLED_PROMPTS),
                          (emb.front_T, pose.FRONT_PROMPTS),
                          (emb.back_T, pose.BACK_PROMPTS)]:
        assert isinstance(bank, np.ndarray)              # plain numpy, off the GPU
        assert bank.dtype == np.float32                  # .float().cpu().numpy()
        assert bank.shape == (len(prompts), DIM)
        assert np.allclose(np.linalg.norm(bank, axis=-1), 1.0, atol=5e-3)


def test_views_batching_matches_whole_and_tiles_ignore_it(monkeypatch):
    proc = FakeProcessor()
    mod = types.ModuleType("transformers")
    mod.AutoModel = types.SimpleNamespace(
        from_pretrained=lambda name, torch_dtype=None: FakeModel())
    mod.AutoProcessor = types.SimpleNamespace(from_pretrained=lambda name: proc)
    monkeypatch.setitem(sys.modules, "transformers", mod)
    emb = Embedder(["dragon"], device="cpu", embed_batch=2)

    views = list(_tiles(5, seed=3))
    m = EmbedViews(Path("m.stl"), 0, _pose(), views)
    proc.image_calls.clear()
    batched = emb.embed_views(m).embeds
    assert proc.image_calls == [2, 2, 1]                 # --embed-batch honoured

    emb.embed_batch = 0
    proc.image_calls.clear()
    whole = emb.embed_views(m).embeds
    assert proc.image_calls == [5]                       # 0 = whole list at once
    assert torch.equal(batched, whole)                   # split changes nothing

    # the tile path never batches, matching today's score_upright closure
    proc.image_calls.clear()
    emb.embed_batch = 2
    emb.embed_tiles(EmbedTilesRequest(Path("m.stl"), 0, _tiles(5, seed=4)))
    assert proc.image_calls == [5]


# --- tier (b): real SigLIP on the 4060 --------------------------------------

@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs the 4060")
def test_gpu_parity_with_old_path():
    """New Embedder vs classify_stls' embed_* on the same model instance."""
    import classify_stls as old
    from PIL import Image

    categories = ["dragon", "terrain"]
    emb = Embedder(categories)                           # real load, cuda
    assert emb.device == "cuda"
    # One copy since the dedup pass: the CLI imports this from src.embedder
    # rather than keeping its own, so the old cross-copy assertion would now
    # ask classify_stls for a name it no longer defines (F-1).
    assert PROMPT_TEMPLATES is src.embedder.PROMPT_TEMPLATES

    # -- text parity: category embeddings and the numpy prompt banks
    old_text = old.embed_texts(emb.model, emb.processor, categories, emb.device)
    assert old_text.dtype == emb.text_embeds.dtype == torch.float16
    text_diff = (emb.text_embeds - old_text).abs().max().item()
    print(f"\ntext_embeds max|diff| = {text_diff:.3e}")
    assert torch.equal(emb.text_embeds, old_text)

    # All four banks, not just up_T (D-R1-3): each is handed to a different
    # consumer — up/down to the Poser's ensemble, front/back to Done's
    # front_view — so a drift in any one of them moves cached results.
    for name, prompts in (("up_T", pose.UPRIGHT_PROMPTS),
                          ("down_T", pose.TOPPLED_PROMPTS),
                          ("front_T", pose.FRONT_PROMPTS),
                          ("back_T", pose.BACK_PROMPTS)):
        old_bank = old.embed_raw(emb.model, emb.processor, prompts,
                                 emb.device).float().cpu().numpy()
        new_bank = getattr(emb, name)
        assert new_bank.dtype == old_bank.dtype == np.float32, name
        assert np.array_equal(new_bank, old_bank), name

    # -- image parity: same arrays through both paths
    rng = np.random.default_rng(42)
    arrays = [rng.integers(0, 256, size=(256, 256, 3), dtype=np.uint8)
              for _ in range(3)]
    old_embeds = old.embed_images(emb.model, emb.processor, arrays, emb.device)

    out = emb.embed_tiles(EmbedTilesRequest(Path("m.stl"), 0, np.stack(arrays)))
    assert out.embeds.dtype == old_embeds.dtype == torch.float16
    assert out.embeds.device.type == "cuda"              # stays on device
    tile_diff = (out.embeds - old_embeds).abs().max().item()
    print(f"embed_tiles vs old embed_images max|diff| = {tile_diff:.3e}")
    assert torch.allclose(out.embeds.float(), old_embeds.float(), rtol=0, atol=1e-3)
    assert torch.equal(out.embeds, old_embeds)           # same code path: bitwise
    new_norms = out.embeds.float().norm(dim=-1)
    old_norms = old_embeds.float().norm(dim=-1)
    assert torch.equal(new_norms, old_norms)             # norm equality
    print(f"row norms: new {new_norms.cpu().numpy()}, old {old_norms.cpu().numpy()}")

    # -- views path + the .npy cache write stays byte-compatible
    viewed = emb.embed_views(EmbedViews(Path("m.stl"), 0, _pose(), arrays))
    assert torch.equal(viewed.embeds, old_embeds)
    assert np.array_equal(viewed.embeds.float().cpu().numpy(),
                          old_embeds.float().cpu().numpy())

    # -- today's path fed PIL Images where the child will send arrays: the
    #    processor converts both to the same pixels, so this pins the crossing
    pils = [Image.fromarray(a) for a in arrays]
    old_pil = old.embed_images(emb.model, emb.processor, pils, emb.device)
    pil_diff = (old_pil - out.embeds).abs().max().item()
    print(f"PIL-input vs array-input max|diff| = {pil_diff:.3e}")
    assert torch.allclose(old_pil.float(), out.embeds.float(), rtol=0, atol=1e-3)
