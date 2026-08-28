# Archived review notes

Snapshots of reviews whose work has shipped. They are kept for one reason:
**410 finding IDs cited from live code and docs are defined in here and
nowhere else.** A comment reading `(K1: unconditional)` or `E-R1-2/E-R1-3`
means nothing without the file that defines the finding.

They are not maintained, and two conventions follow from that:

- **Line numbers inside these files are as-of-then.** They index the code as
  it stood on the file's date — mostly the pre-refactor `classify_stls.py`,
  which was 1277 lines and is now 425. Do not repair them and do not read
  them as navigation; the surrounding prose names what it is talking about.
  Live docs and code cite by *symbol* instead (CLAUDE.md §Conventions).
- **Nothing here is evidence about today's code.** Where a review's finding
  became a durable fact it has a dated write-up in `docs/learnings/` and an
  index line in `LEARNINGS.md`; quote that. The value left here is the
  argument that produced a decision, not the state it described.

## Which file defines which finding IDs

| IDs | defined in |
|---|---|
| `I*`, `J*`, `K*`, `L*`, `M*`, `N*`, `O*` (53 findings, `### K6. …` headings) | `2026-08-14-interfaces.md` |
| `D*` (20 findings, the message and driver shapes) | `2026-08-14-data-structures.md` |
| `C-R1-*`, `E-R1-*`, `F-*` (the implementation round, bold list items) | `2026-08-17-wave1-implementation.md` |

`2026-08-13.md` defines no IDs but is cited by section — §3.1 is the
four-coexisting-renderers measurement behind the teardown constraint in
CLAUDE.md. `2026-08-20-past-week.md` is a retrospective; `docs/tri-state-pass-2.md`
reads its "Not acted on" list.

The live spec these reviews produced is `docs/actor-refactor/interfaces.md`,
which uses the labels without defining them. Read the spec first; come here
only to resolve an ID.
