# Archived review notes

Snapshots of reviews whose work has shipped, kept for one reason: **410
finding IDs cited from live code and docs are defined in here and nowhere
else.** A comment reading `(K1: unconditional)` or `E-R1-2/E-R1-3` means
nothing without the file that defines the finding.

The archive's two rules apply — line numbers are as-of-then and must not be
repaired, and nothing here is evidence about today's code. See `../README.md`.

## Which file defines which finding IDs

| IDs | defined in |
|---|---|
| `I*`, `J*`, `K*`, `L*`, `M*`, `N*`, `O*` (53 findings, `### K6. …` headings) | `2026-08-14-interfaces.md` |
| `D*` (20 findings, the message and driver shapes) | `2026-08-14-data-structures.md` |
| `C-R1-*`, `E-R1-*`, `F-*` (the implementation round, bold list items) | `2026-08-17-wave1-implementation.md` |

`2026-08-13.md` defines no IDs but is cited by section — §3.1 is the
four-coexisting-renderers measurement behind the teardown constraint in
CLAUDE.md. `2026-08-20-past-week.md` is a retrospective; `docs/archive/tri-state-pass-2.md`
reads its "Not acted on" list.

The live spec these reviews produced is `docs/actor-refactor/interfaces.md`,
which uses the labels without defining them. Read the spec first; come here
only to resolve an ID.
