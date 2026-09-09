"""Embedder tests, two tiers (docs/actor-refactor/interfaces.md §Embedder).

Tier (a): GPU-free contract tests — the transformers stack is faked through
sys.modules, so __init__ runs for real (text embeds, prompt banks) against a
deterministic toy model. These pin the return shapes/dtypes, the
normalisation, the read-only text_embeds, and the batch behaviour of the two
entry points.

Tier (b): one @pytest.mark.gpu test that loads real SigLIP on the 4060 and
pins the **eval rig** against the Embedder — `eval/rig.py`'s `embedder()` and
`embed()`, which is what every harness in `eval/` now calls, against
`embed_tiles`/`embed_views`, which is what the pipeline calls. Same instance,
same inputs: bitwise equality plus dtype and norm equality, and
byte-compatibility of the .npy cache write.

That assertion is the one `eval/README.md` has always claimed and could not
prove. It used to point at `classify_stls.embed_images` instead — a second
arrangement of the same forward, kept in the CLI for the harnesses. The
harnesses stopped needing it (2026-08-18), so the parity that matters is no
longer CLI-vs-Embedder but harness-vs-Embedder.

The text half no longer crosses to the CLI either: `embed_raw`/`embed_texts`
moved into `src/embedder.py` as free functions and the Embedder's methods
delegate to them, so what those assertions now pin is the *binding* — that
`__init__` really ran the shared forward with this instance's model,
processor and device.
"""
from __future__ import annotations

import sys
import threading
import time
import types
from pathlib import Path

import numpy as np
import pytest
import torch

import src.embedder
from src import pose
from src.embedder import (DEFAULT_MODEL, JOIN_RESERVE, PROMPT_TEMPLATES,
                          Embedder, count_tokens, load_siglip,
                          query_budget, text_seam)
from src.messages import Embedded, EmbedTilesRequest, EmbedViews, TileEmbeds
from src.pose import Pose

DIM = 6  # the fake model's embedding width


class _WordTokenizer:
    """A tokenizer's one behaviour `count_tokens` uses: ids for a string, with
    the trailing EOS every real one appends."""
    def __call__(self, text):
        return {"input_ids": text.split() + ["</s>"]}


# --- tier (a): the faked stack ----------------------------------------------

class FakeBatch(dict):
    """Mimics transformers' BatchFeature: a dict with .to(device)."""
    def to(self, device):
        return self


class FakeProcessor:
    def __init__(self):
        self.image_calls: list[int] = []   # images per call, in call order
        self.tokenizer = _WordTokenizer()

    def __call__(self, images=None, text=None, padding=None, truncation=None,
                 return_tensors="pt"):
        self.truncation = truncation
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
    # the width `query_budget` reads, in the place a real config keeps it
    config = types.SimpleNamespace(
        text_config=types.SimpleNamespace(max_position_embeddings=64))

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
        from_pretrained=lambda name, torch_dtype=None, local_files_only=False: FakeModel())
    mod.AutoProcessor = types.SimpleNamespace(
        from_pretrained=lambda name, local_files_only=False: proc)
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


def test_query_budget_subtracts_the_worst_template_from_the_towers_width(fake):
    """The number `/query` refuses against and `/status` publishes.

    It is derived, not written down: the tower's own width minus the room the
    templates take, so a checkpoint with different positions or a fourth
    template reports its own budget instead of drifting from a constant here.
    The `+ 1` is the EOS an empty template counts and a bare text pays anyway —
    without it every query loses a token to arithmetic."""
    emb, proc = fake
    tok = _WordTokenizer()
    proc.tokenizer = tok
    model = types.SimpleNamespace(
        config=types.SimpleNamespace(
            text_config=types.SimpleNamespace(max_position_embeddings=64)))
    worst = max(count_tokens(proc, t.format("")) for t in PROMPT_TEMPLATES)
    assert query_budget(model, proc) == 64 - worst + 1 - JOIN_RESERVE


def test_the_budget_bounds_the_templated_text_it_is_computed_for(fake):
    """What the subtraction is *for*: a text at the budget must still fit after
    the templates wrap it. Checked against every template rather than the one
    the max came from — the point is that no template can overflow the tower
    for a text this module said was small enough."""
    _, proc = fake
    proc.tokenizer = _WordTokenizer()
    model = types.SimpleNamespace(
        config=types.SimpleNamespace(
            text_config=types.SimpleNamespace(max_position_embeddings=64)))
    budget = query_budget(model, proc)
    text = " ".join(["orc"] * (budget - 1))              # budget tokens, +EOS
    assert count_tokens(proc, text) == budget
    for t in PROMPT_TEMPLATES:
        assert count_tokens(proc, t.format(text)) <= 64


def test_the_text_pass_clips_to_the_towers_64_positions(fake):
    """Issue #5. `padding="max_length"` pads *to* the tower's 64 positions but
    does not clip past them, so a longer text arrived as a 122-token batch and
    `SiglipTextModel` raised. Every text forward in the project goes through
    here, so the flag belongs here and not at the one caller that noticed: the
    API's 422 bounds what a *query* may say, while this is what stops the model
    from being handed a sequence it has no positions for."""
    emb, proc = fake
    emb._embed_raw(["a dragon " * 200])
    assert proc.truncation is True


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


@pytest.mark.parametrize("cached, expected", [
    (True, [("processor", True), ("model", True)]),
    # Uncached, the offline attempt dies on the processor — which loads first,
    # so the model is only ever built once, and never on the GPU twice.
    (False, [("processor", True), ("processor", False), ("model", False)]),
])
def test_load_siglip_tries_the_local_cache_first(monkeypatch, cached, expected):
    """Both halves load out of the local cache with no hub round-trip; an
    uncached model still downloads, which is what the retry buys over a bare
    local_files_only. Asserting the processor too: it is half the round-trip,
    and dropping its flag is a silent regression the model call cannot catch."""
    proc = FakeProcessor()
    seen = []

    def offline_raises(what, name, local_files_only):
        seen.append((what, local_files_only))
        if local_files_only and not cached:
            raise OSError(f"{name} is not in the local cache")

    def model_from_pretrained(name, torch_dtype=None, local_files_only=False):
        offline_raises("model", name, local_files_only)
        return FakeModel()

    def proc_from_pretrained(name, local_files_only=False):
        offline_raises("processor", name, local_files_only)
        return proc

    mod = types.ModuleType("transformers")
    mod.AutoModel = types.SimpleNamespace(from_pretrained=model_from_pretrained)
    mod.AutoProcessor = types.SimpleNamespace(from_pretrained=proc_from_pretrained)
    monkeypatch.setitem(sys.modules, "transformers", mod)

    model, processor = load_siglip(DEFAULT_MODEL, "cpu")
    assert seen == expected          # all-True = the network was never touched
    assert isinstance(model, FakeModel) and processor is proc


def test_an_oserror_moving_the_model_is_not_a_cache_miss(monkeypatch):
    """The retry answers "the snapshot is not on disk", and only that. An
    `OSError` out of the device transfer is a different fault entirely, and
    retrying it re-enters the loader online — hiding the real cause behind a
    download, or behind a second failure on a machine with no network."""
    seen = []

    class Immovable(FakeModel):
        def to(self, device):
            raise OSError("device transfer failed")

    def model_from_pretrained(name, torch_dtype=None, local_files_only=False):
        seen.append(local_files_only)
        return Immovable()

    mod = types.ModuleType("transformers")
    mod.AutoModel = types.SimpleNamespace(from_pretrained=model_from_pretrained)
    mod.AutoProcessor = types.SimpleNamespace(
        from_pretrained=lambda name, local_files_only=False: FakeProcessor())
    monkeypatch.setitem(sys.modules, "transformers", mod)

    with pytest.raises(OSError, match="device transfer failed"):
        load_siglip(DEFAULT_MODEL, "cpu")
    assert seen == [True]            # loaded once, offline, and never re-tried


@pytest.mark.parametrize("device, expected", [
    ("cpu", torch.float32), ("cuda", torch.float16), ("cuda:0", torch.float16),
])
def test_load_siglip_picks_the_dtype_by_device(monkeypatch, device, expected):
    """fp16 is the 4060's dtype; on a CPU it runs 3.7x slower and peaks at
    7.7 GB converting the fp32 checkpoint (LEARNINGS, "fp16 on a CPU"). An
    explicit dtype still wins, which is how the harness reproduces both."""
    seen = []

    def model_from_pretrained(name, torch_dtype=None, local_files_only=False):
        seen.append(torch_dtype)
        return FakeModel()

    mod = types.ModuleType("transformers")
    mod.AutoModel = types.SimpleNamespace(from_pretrained=model_from_pretrained)
    mod.AutoProcessor = types.SimpleNamespace(
        from_pretrained=lambda name, local_files_only=False: FakeProcessor())
    monkeypatch.setitem(sys.modules, "transformers", mod)

    load_siglip(DEFAULT_MODEL, device)
    load_siglip(DEFAULT_MODEL, device, torch_dtype=torch.bfloat16)
    assert seen == [expected, torch.bfloat16]


def test_views_batching_matches_whole_and_tiles_ignore_it(monkeypatch):
    proc = FakeProcessor()
    mod = types.ModuleType("transformers")
    mod.AutoModel = types.SimpleNamespace(
        from_pretrained=lambda name, torch_dtype=None, local_files_only=False: FakeModel())
    mod.AutoProcessor = types.SimpleNamespace(
        from_pretrained=lambda name, local_files_only=False: proc)
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

def test_the_real_tokenizers_budget_and_what_it_promises():
    """The number itself, and the promise behind it, for the real tokenizer.

    54: the tower's 64 positions, less the 9 the longest template costs around
    a text, less `JOIN_RESERVE`. `patch14-384` and the `patch16-512` that built
    embed-cache512 agree — they differ in the image tower, not the text one.
    A change here is a change to a published contract (`/status`'s
    `text_budget`, surface.md).

    The promise is the second half: a text *at* the budget still fits the tower
    after every template wraps it, the leading-space case included. That case
    is why the reserve exists — `" nü …"` re-segments at the join and wraps a
    token wider than it measures alone, so an unreserved budget let a 65-token
    prompt into a 64-position tower and the clip silently ate the template's
    last word.

    Needs the model's processor and config in the local HF cache and nothing
    else — no weights, no GPU — so it skips rather than downloads."""
    from transformers import AutoConfig, AutoProcessor
    try:
        proc = AutoProcessor.from_pretrained(DEFAULT_MODEL, local_files_only=True)
        cfg = AutoConfig.from_pretrained(DEFAULT_MODEL, local_files_only=True)
    except OSError:
        pytest.skip("the model is not in the local HF cache")
    positions = cfg.text_config.max_position_embeddings
    assert positions == 64
    assert query_budget(types.SimpleNamespace(config=cfg), proc) == 54


def test_a_text_at_the_budget_still_fits_after_the_templates_wrap_it():
    """The promise the budget makes, against the real tokenizer and the
    derived number rather than the pinned one.

    The leading-space prefixes are the point: `" nü …"` re-segments at the join
    — the template's own whitespace piece absorbs the space and splits the word
    behind it — so it wraps a token wider than it measures alone. Without
    `JOIN_RESERVE` this builds a 65-token prompt for a 64-position tower, and
    the clip that keeps it from raising eats the template's last word instead.
    """
    from transformers import AutoConfig, AutoProcessor
    try:
        proc = AutoProcessor.from_pretrained(DEFAULT_MODEL, local_files_only=True)
        cfg = AutoConfig.from_pretrained(DEFAULT_MODEL, local_files_only=True)
    except OSError:
        pytest.skip("the model is not in the local HF cache")
    positions = cfg.text_config.max_position_embeddings
    budget = query_budget(types.SimpleNamespace(config=cfg), proc)

    def at_budget(prefix):
        """`prefix` padded with words until one more would break the budget."""
        text = prefix
        while count_tokens(proc, text + " orc") <= budget:
            text += " orc"
        return text

    for prefix in ["orc", " nü", " y", "龍", "🐉"]:
        text = at_budget(prefix)
        assert count_tokens(proc, text) <= budget
        for t in PROMPT_TEMPLATES:
            assert count_tokens(proc, t.format(text)) <= positions


def test_the_seam_never_lets_two_threads_into_the_tokenizer(fake, monkeypatch):
    """The lock `text_seam` exists for.

    A fast tokenizer reconfigures its Rust backend on every call, so two
    threads inside it raise `RuntimeError: Already borrowed` — and the API
    calls the counter outside its GPU lock, in Starlette's threadpool, while
    the forward runs. Four counting threads against two embedding threads on
    the real tokenizer produced 117,725 of those in 8 seconds (2026-09-08);
    unhandled, each is the 500 issue #5 exists to remove.

    The fake tokenizer cannot raise that, so what is asserted is the property
    that prevents it: no two threads in the shared processor at once. The sleep
    is what makes an unlocked seam fail this rather than pass it by luck."""
    emb, proc = fake
    seen = {"in": 0, "overlap": 0}

    def watch(fn):
        def wrapped(*a, **kw):
            seen["in"] += 1
            seen["overlap"] += seen["in"] > 1
            time.sleep(0.002)
            try:
                return fn(*a, **kw)
            finally:
                seen["in"] -= 1
        return wrapped

    monkeypatch.setattr(FakeProcessor, "__call__", watch(FakeProcessor.__call__))
    proc.tokenizer = watch(proc.tokenizer)
    embed, tokens, budget = text_seam(emb.model, proc, "cpu")
    assert budget == 64 - max(count_tokens(proc, t.format(""))
                              for t in PROMPT_TEMPLATES) + 1 - JOIN_RESERVE

    stop = time.monotonic() + 0.4

    def hammer(fn):
        while time.monotonic() < stop:
            fn()

    threads = ([threading.Thread(target=hammer, args=(lambda: tokens("a dragon"),))
                for _ in range(4)]
               + [threading.Thread(target=hammer, args=(lambda: embed(["a dragon"], True),))
                  for _ in range(2)])
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert seen["overlap"] == 0


def test_the_clip_is_the_tokenizers_and_not_just_a_flag():
    """`truncation=True` is asserted against the fake above; this is the real
    tokenizer actually clipping, which is the behaviour issue #5 turns on. 402
    tokens without it — and that is what reached the tower and raised."""
    from transformers import AutoProcessor
    try:
        proc = AutoProcessor.from_pretrained(DEFAULT_MODEL, local_files_only=True)
    except OSError:
        pytest.skip("the model's processor is not in the local HF cache")
    long = "dragon knight " * 200
    clipped = proc(text=[long], padding="max_length", truncation=True,
                   return_tensors="pt")["input_ids"]
    loose = proc(text=[long], padding="max_length", return_tensors="pt")["input_ids"]
    assert clipped.shape[1] == 64 < loose.shape[1]


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs the 4060")
def test_gpu_parity_of_the_eval_rig_with_the_embedder():
    """eval/rig.py's embedding path vs the Embedder's message-shaped ones."""
    from PIL import Image
    from src.embedder import embed_raw, embed_texts

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "eval"))
    import rig

    categories = ["dragon", "terrain"]
    emb = rig.embedder(categories=categories)            # real load, cuda
    assert isinstance(emb, Embedder)
    assert emb.device == "cuda"
    # One copy since the dedup pass: every templated query in the project goes
    # through src.embedder, so the old cross-copy assertion would now ask
    # classify_stls for a name it no longer defines (F-1).
    assert PROMPT_TEMPLATES is src.embedder.PROMPT_TEMPLATES

    # -- text binding: category embeddings and the numpy prompt banks came off
    #    the shared free functions, with this instance's model/processor/device
    old_text = embed_texts(emb.model, emb.processor, categories, emb.device)
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
        old_bank = embed_raw(emb.model, emb.processor, prompts,
                             emb.device).float().cpu().numpy()
        new_bank = getattr(emb, name)
        assert new_bank.dtype == old_bank.dtype == np.float32, name
        assert np.array_equal(new_bank, old_bank), name

    # -- image parity: the harnesses' call against the pipeline's two
    rng = np.random.default_rng(42)
    arrays = [rng.integers(0, 256, size=(256, 256, 3), dtype=np.uint8)
              for _ in range(3)]
    # what every harness in eval/ runs: float32 numpy off the same forward
    rig_embeds = rig.embed(emb, arrays)
    assert isinstance(rig_embeds, np.ndarray)
    assert rig_embeds.dtype == np.float32                # .float().cpu().numpy()

    out = emb.embed_tiles(EmbedTilesRequest(Path("m.stl"), 0, np.stack(arrays)))
    assert out.embeds.dtype == torch.float16
    assert out.embeds.device.type == "cuda"              # stays on device
    tile_diff = np.abs(rig_embeds - out.embeds.float().cpu().numpy()).max()
    print(f"rig.embed vs embed_tiles max|diff| = {tile_diff:.3e}")
    assert np.array_equal(rig_embeds, out.embeds.float().cpu().numpy())
    norms = out.embeds.float().norm(dim=-1)
    assert np.allclose(np.linalg.norm(rig_embeds, axis=-1), norms.cpu().numpy())
    print(f"row norms: {norms.cpu().numpy()}")

    # -- views path + the .npy cache write stays byte-compatible with what a
    #    harness reads back, which is what makes a harness number comparable to
    #    a cached one
    viewed = emb.embed_views(EmbedViews(Path("m.stl"), 0, _pose(), arrays))
    assert torch.equal(viewed.embeds, out.embeds)
    assert np.array_equal(viewed.embeds.float().cpu().numpy(), rig_embeds)

    # -- harnesses hand PIL Images (they read cached tiles off disk) where the
    #    child sends arrays: the processor converts both to the same pixels, so
    #    this pins the crossing that every disk-cached tile set depends on
    pils = [Image.fromarray(a) for a in arrays]
    pil_embeds = rig.embed(emb, pils)
    pil_diff = np.abs(pil_embeds - rig_embeds).max()
    print(f"PIL-input vs array-input max|diff| = {pil_diff:.3e}")
    assert np.allclose(pil_embeds, rig_embeds, rtol=0, atol=1e-3)
