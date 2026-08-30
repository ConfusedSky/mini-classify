# description_panel — VLM descriptions of one model: panels, effort, stability (2026-08-30)

Two harnesses around the Proto-Osteotron (`32mm_JuvenileProtoOsteotron1`), the
model whose concept — a crew member with an alien bursting from the head — a
VLM either sees or does not. Write-up: LEARNINGS, "Arbiter backends, sheet
sizes, presentations and effort" and "What a VLM sees in our renders".

| script | what | env |
|---|---|---|
| `panel_matrix.py` | 4 renderers × (control, temperature panel, viewset panel) × 2 efforts; describers are one OpenRouter model, each panel synthesized by GLM **and** gemini-3.6-flash from the same three texts | `DESCRIBE_MODEL` (default GLM-5.3-Flash), `PANEL_OUT`, `WORKERS`, `GLM_PROVIDER` |
| `repeat_control.py` | the control cell 20× per renderer × effort, then `--report`: unique texts, grade histogram, pairwise similarity | `DESCRIBE_MODEL`, `REPEAT_OUT`, `WORKERS`, `GLM_PROVIDER` |

Inputs are the renderer-pilot's 512 px renders (`eval/renderer_pilot/renders/`,
local only) and f3d 1024 px renders made on demand into `f3d1024-renders/`.
Both scripts log every API call (model, provider, secs, tokens, cost) to
`out/<run>.telemetry.json`. `out/` keeps this session's runs: GLM and
gpt-5.6-luna panel matrices, GLM stability under default routing and pinned to
Z.AI. Grades in the write-up are by hand against the rubric there; the regex
pre-grader in the session notes is a triage tool, not the record.
