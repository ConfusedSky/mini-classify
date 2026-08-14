## Hardening the unpacker (2026-08-14)

Seven adversarial review passes over `unpack_models.py` — twenty-four code
findings — plus an eighth pass that audited this document's first draft (wrong
counts, an overstated rigor claim, a false invariant, and one of the
reviewer's own phrasings repeated; all corrected in place). Every substantive
finding carried a reproduction — eleven of the twenty-four; the rest were read
off the code — and each batch of fixes bred the next batch of findings. The
arc ended with the collection fully zip-independent: 453 archives, converged
under both selection modes, ~24 GB of models extracted that were previously
locked in zips or destined to destroy each other.

### What the passes found, mechanism by mechanism

**Windows-authored zips are their own genre.** Explorer writes Deflate64
(method 9), which Python's zipfile lists but cannot decompress — dispatched to
`7z`/`7zz`/`7za` through the same `.partial` staging. The same archives omit
the UTF-8 name flag, so zipfile decodes entry names as cp437 while 7z writes
UTF-8: the staged root can carry a name `destination()` never predicted. The
fix is adoption, not detection — take whatever single root was actually
written. The two conditions co-occur in Explorer-authored archives — but
co-occurrence is not a rule: pure-ASCII names decode identically under cp437
and UTF-8, and nothing stops a Deflate64 writer from setting the flag. Which
is the real argument for adoption over prediction: neither condition can be
detected from the other.

**A destructive swap must have no statement where zero copies exist.** The
repair swap moves the original aside, renames the replacement in, and only
then deletes — a mid-swap exception rolls back, and a *hard kill* leaves the
original recoverable in `<dest>.replaced`. The subtle trap: the aside copy is
byte-identical to the archive, so the CRC-verified redundancy check *confirmed*
the false "elsewhere" for a model that was missing from disk. Every safety
mechanism that walks the tree has to know about every artifact the tool
leaves behind (`.partial`, `.replaced`) — an invariant that has to be
re-checked each time either list grows.

**Loud guards surface old bugs.** The guard that refused unexpected staging
contents (instead of silently nesting the tree a level too deep) immediately
exposed a latent `destination()` bug older than the branch: a flat zip holding
one file at its root was "broken differently in every revision" — planned
extract-forever, then misnested under a directory named after itself, then
refused loudly. Only the loud version was diagnosable in one run. A single
top-level name counts as a root only when every entry sits under a directory
prefix.

**Redundancy detection needs bytes, not names.** "Every entry matches by
basename and size" declared unrelated zips redundant — and a false positive
here means a zip silently never extracts, the exact failure the tool exists to
fix. All entries name+size, plus a spread sample CRC-verified against the
archive's own table — a bounded *extra* read, at most five files, and only for
zips that already matched every entry by name and size. Still a
heuristic: `--ignore-elsewhere` is the explicit way past it, named in the very
advisory that reports the skip.

**Flags widen selection; they never move destinations or disable safety.**
Two findings converged on one principle. `--all` had been exempted from the
redundancy check ("unpacks everything regardless") — re-decided: it widens the
selection past the skip tags, nothing else. And `divert_collisions` filtering
by skip tag meant `--all` changed collision-group membership, so the same zip
extracted to different directories under different flags, both copies then
reporting done. Where a zip extracts is a function of the files on disk,
never of the run's flags; whether it extracts is the flags' whole domain.

**Collisions divert to a fixed point.** Fifteen thingiverse zips carrying
their author's name as the root all claimed `j4roid/`; under one-destination
semantics only the last survives. Each colliding zip diverts to its own
stem-named directory — and because a diversion target can itself collide with
another zip's derived root, diversion iterates until stable (terminating
because a diverted zip sits at its own stem and stems are unique per
directory).

### The review loop itself

The reviewer worked in an artifact updated per pass; every substantive
finding carried a reproduction, and three times the landed fix was stronger
than the suggested one (adopting the written root; CRC over path-tails;
restore-then-stop). The
compounding pattern is the lesson: the swap fix created the hard-kill orphan,
the staging guard exposed the destination bug, the `--all` redefinition
removed the only override the redundancy check had. Fixes to destructive
tools breed follow-on findings in exactly the code they touch, and the pass
after a fix is worth as much as the pass that found the original.

One more advisory principle from the seventh pass (V1): a warning that directs
a human to delete something must be provably about something nobody owns — a
contested destination that is some zip's own stem is that zip's live home.

Commits: `e224267`, `d2e6f0a`, `4feb365`, `512c864`, `6928ab1`, `33649bf`,
`4676b4a`; tests grew from 15 at the review's start (11 at the module's birth)
to 34 for this module alone, each new test naming the review finding it pins.
