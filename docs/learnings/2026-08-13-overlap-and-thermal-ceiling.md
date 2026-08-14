## Overlap and the thermal ceiling (2026-08-13)

Two questions from `actors_proposal.md`, measured: does overlapping the iGPU
renderer with SigLIP saturate the 4060, and does the SigLIP side have headroom
of its own. Harnesses: `eval/overlap_spike.py` (render in a child process,
embed in the parent, bounded queue between) and `eval/siglip_bench.py`. Raw
results land in `eval/out/`, which is gitignored — the figures quoted here are
the record.

### One process boundary takes the 4060 from ~57% busy to ~94%

60 pose-cached models, production config (384 px, 8×2 views + 24 tiles per
model), page cache pre-warmed, identical GPU work in both modes. Spike 4 said
threads cannot do this (`render_to_image` holds the GIL ~85–92%); a child
process can. The parent spent 6–8 s of a ~2-minute run waiting on the queue —
the boundary itself is nearly free. At 384 px a view is ~440 KB, so the
"12.6 MB per view" IPC objection in the proposal was a 2048 px number that
does not apply to the run config.

### Utilization and wall-clock parted ways: ~1.2×, not the Amdahl 1.45×

Run order flips the naive comparison — 1.03× with baseline first, 1.37× with
overlap first — because each mode pre-heats the card for the next. Cool-start
against cool-start: **125.0 s → 106.9 s, a true 1.17–1.21×**. The missing
speedup is thermal: saturated, the same embedding work took 101–113 s against
71 s sequential (1.4–1.6× per image), matching the duty-cycled/soaked ratio
measured independently (27.4 vs 18.4 img/s). At 94% busy the remaining idle is
not scheduling — it is an 80 W laptop card trading clocks for duty cycle.
Raising that ceiling is a cooling/power-limit question, not a code question.

### SigLIP is power-clamped, not under-batched

Throughput is flat at 17–19 img/s from batch 1 to 128 (9.3% spread against
10.8% within-point jitter), peak VRAM 4.9 of 8 GiB — never memory-limited. The
card enters SW thermal slowdown within seconds of sustained load and slides
2250 → ~1400 MHz; effective throughput is ~11.5 TFLOP/s against ~28 of
throttled fp16 peak, with attention already `sdpa`. `--embed-batch` is
therefore not a throughput knob (and the pose-tile call at
`classify_stls.py:888` never took it anyway).

### `torch.compile` is 1.10× and disqualified

A consistent 1.10× on both production shapes after ~55 s of compile — but the
embedding drift (max 7.3–9.8e-04) is the same order as the closest observed
top-1 category margin (9e-04 across 128 real renders scored against the real
categories, 0 argmax flips in that sample), and compiled values would seed the
permanent `.npy` cache with numbers inconsistent with every eager run. Not
worth 10%. Available but not taken for the same family of reasons: threading
the preprocessing of chunk i+1 while the GPU runs chunk i inside
`embed_images` — measured 1.044×, ~146 s per full cold run, perturbing
embeddings by 1.2e-04 because batch composition changes the fp16 reduction
order (the same class of change `--embed-batch` already makes).

### The benchmark trap: an ascending sweep measures the cooling curve

The first batch sweep produced a clean "throughput falls with batch size"
line, 25.7 → 17.7 img/s. It was thermal decay in batch order — SW thermal
slowdown from the warmup on, SM clock sliding 2250 → 1320 MHz in lockstep with
the sweep. Every number above comes from a re-run with a 150 s soak and
shuffled, interleaved rounds. On this machine, a benchmark that runs its
configurations in a fixed order is measuring the fan.

### The ceiling moves when the cooling does (same day)

Re-run with improved cooling, same protocol (60 models, both mode orders,
2-minute cooldown between runs), idle temperature 66–71 °C → 58–60 °C.
Cool-start against cool-start:

| | before | cooled | gain |
|---|---|---|---|
| baseline (sequential) | 125.0 s | 115.2 s | 1.09× |
| overlap | 106.9 s | 93.5 s | 1.14× |
| sequential embed (duty-cycled) | 71.0 s | 61.9–65.7 s | ~1.10× |
| saturated embed (back-to-back) | 100.7–112.9 s | 84.8–90.5 s | ~1.20× |

Cooling paid roughly twice as much in the saturated regime as in the
idle-gapped one — which is the regime the overlap pushes the card into, so
the two changes compound: **1.34× over the original sequential baseline**
(125.0 → 93.5 s). The order-dependence of the naive comparison also shrank
(1.15×/1.27× against 1.03×/1.37× before), i.e. less thermal hysteresis
between runs. Not eliminated: the saturation penalty is still ~1.3–1.45×
(from 1.4–1.6×), so further cooling or a power-limit raise keeps paying
roughly proportionally. Raw numbers:
`eval/out/overlap_spike_cooled_run1.json` beside the standard output file.

