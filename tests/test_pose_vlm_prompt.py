"""`resolve_pose_vlm`'s auto-failure path — the OpenRouter fallback, and the
prompt for when there is no key for that either (C6,
docs/archive/tri-state-pass-2.md, 2026-08-21) — driven as a **subprocess**.

Nothing may import `classify_stls.py` — tests included (CLAUDE.md): `spawn`
re-imports it as `__mp_main__` in the render child, so its module scope is the
child's startup cost, and a test that imported it would make that rule
unenforceable. So each case runs the CLI the way a person does, and asserts on
the exit code and the text.

The run is stopped a few lines past the decision rather than allowed to
classify anything: `--model` names a repository that does not exist and the
environment is pinned offline, so `main()` dies in the Embedder. That makes
`loading <model> ...` — printed immediately after `resolve_pose_vlm` returns —
the signal that the arbiter decision let the run continue, and its absence the
signal that the run stopped at the prompt. No GPU, no network, no renders.

The interactive branch is driven through a pty (both stdin and stderr must be
ttys for the prompt to fire), which is the one thing a plain pipe cannot fake.
What is NOT covered here: a backgrounded job whose stdin is a tty still
SIGTTIN-stops at the `input()` — the same hazard `cache_root`'s prompt accepts,
and a manual check.
"""
import os
import pty
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
TIMEOUT = 120


def cli(tmp_path, *extra, model="no-such-org/no-such-model", or_key=None):
    """The command and the environment: one STL, an empty cache, and no
    `gcloud` on PATH — which is how the auto probe is made to fail without
    touching the machine's real ADC state.

    `OPENROUTER_KEY_FILE` is pinned for the same reason `PATH` is. These runs
    use a real HOME, so once `auto` gained its GLM fallback the default key
    path would find the developer's own key and degrade to the arbiter instead
    of prompting — every prompt test below would pass on this machine and fail
    on a machine without one, or the other way round. `or_key=None` points it
    at a path that does not exist, which is the no-fallback world these tests
    were written for; a test that wants the fallback passes a real file."""
    stls = tmp_path / "stls"
    stls.mkdir()
    (stls / "a.stl").write_bytes(b"solid x\nendsolid x\n")
    cats = tmp_path / "categories.txt"
    cats.write_text("a knight\na tree\n")
    empty = tmp_path / "no-tools"
    empty.mkdir()

    env = dict(os.environ, PATH=str(empty), HF_HUB_OFFLINE="1",
               TRANSFORMERS_OFFLINE="1",
               OPENROUTER_KEY_FILE=str(or_key or tmp_path / "no-openrouter-key"))
    for var in ("GOOGLE_CLOUD_PROJECT", "GCLOUD_PROJECT"):
        env.pop(var, None)
    cmd = [sys.executable, "classify_stls.py", str(stls),
           "--cache-dir", str(tmp_path / "cache"), "--categories", str(cats),
           "--model", model, *extra]
    return cmd, env


def run_piped(tmp_path, *extra, or_key=None):
    cmd, env = cli(tmp_path, *extra, or_key=or_key)
    return subprocess.run(cmd, cwd=REPO, env=env, capture_output=True,
                          text=True, stdin=subprocess.DEVNULL, timeout=TIMEOUT)


def run_tty(tmp_path, answer, *extra):
    """The same run with stdin and stderr on a pty, so `isatty()` is true on
    both — the condition the prompt requires."""
    cmd, env = cli(tmp_path, *extra)
    master, slave = pty.openpty()
    p = subprocess.Popen(cmd, cwd=REPO, env=env, stdin=slave, stderr=slave,
                         stdout=subprocess.PIPE, text=True)
    os.close(slave)
    os.write(master, answer)
    try:
        out = p.stdout.read()
        p.wait(timeout=TIMEOUT)
    finally:
        os.close(master)
        if p.poll() is None:
            p.kill()
    return p.returncode, out


def test_a_non_interactive_auto_failure_stops_and_names_both_ways_out(tmp_path):
    """A silent degrade is what made an `auto` run indistinguishable from an
    `off` one: every pose it resolves is marked `arbitrated: false`, deferring
    a whole collection's escalations on a gcloud failure nobody was told
    about. With nobody to ask, the run stops and names the two intents the
    existing options already express — no new flag (decided)."""
    out = run_piped(tmp_path)
    assert out.returncode != 0
    assert "--pose-vlm off" in out.stderr
    assert "gcloud auth application-default login" in out.stderr
    assert "arbitrated: false" in out.stderr        # what continuing would mean
    assert "loading" not in out.stdout              # stopped before SigLIP


def test_the_prompt_defaults_to_no(tmp_path):
    """Bare Enter is a decline: an arbiter-on run is what the marks are for,
    and continuing without one is the choice that has to be made explicitly."""
    rc, stdout = run_tty(tmp_path, b"\n")
    assert rc != 0
    assert "continue without the arbiter? [y/N]" in stdout
    assert "loading" not in stdout


def test_declining_the_prompt_stops_the_run(tmp_path):
    rc, stdout = run_tty(tmp_path, b"n\n")
    assert rc != 0
    assert "loading" not in stdout


def test_accepting_the_prompt_continues_degraded(tmp_path):
    """y carries on with no arbiter — reaching the model load, which is the
    line after the decision. The prompt itself is written by `input()`, which
    puts it on **stdout** whatever the explanation went to; that is why the
    explanation is printed to stderr separately (review 2, N10)."""
    rc, stdout = run_tty(tmp_path, b"y\n")
    assert "continue without the arbiter? [y/N]" in stdout
    assert "loading no-such-org/no-such-model" in stdout


def test_auto_takes_the_openrouter_fallback_instead_of_asking(tmp_path):
    """The prompt is the last resort, not the second one.

    A run that degrades to no arbiter defers every gated pose in it; a run
    that degrades to GLM-5.3-Flash still rescues +3 -> 41/44 against gemini's
    +4 -> 42/44, for $0.06 and no GPU (LEARNINGS, "Arbiter backends, sheet
    sizes, presentations and effort", 2026-08-30). So when there is a key,
    `auto` announces the fallback and carries on — reaching the model load,
    the line after the decision — and never reaches the question."""
    key = tmp_path / "or-key"
    key.write_text("sk-or-not-a-real-key\n")
    out = run_piped(tmp_path, or_key=key)
    assert "gemini unavailable" in out.stdout
    assert "falling back to the OpenRouter arbiter" in out.stdout
    assert "z-ai/glm-5.3-flash on OpenRouter" in out.stdout
    assert "continue without the arbiter" not in out.stdout + out.stderr
    assert "loading no-such-org/no-such-model" in out.stdout


def test_the_degrade_drops_a_gemini_model_pin_and_says_so(tmp_path):
    """The backend degrades; a `--pose-vlm-model` chosen for gemini must not
    survive it. OpenRouter ids are org/model and a gemini id never is, so a
    surviving pin reaches OpenRouter as an HTTP 400 — which `pose._ask_glm`
    has to map to `VLMRejected`, a non-transient 4xx being the API judging the
    request, and `poser.Poser._fold` writes that permanently as
    `arbitrated: "rejected"`. Nothing throttles it either: every rejection
    resets the breaker, so the run pins one model per escalation, forever
    (adversarial review, 2026-08-31)."""
    key = tmp_path / "or-key"
    key.write_text("sk-or-not-a-real-key\n")
    out = run_piped(tmp_path, "--pose-vlm-model", "gemini-3.5-pro", or_key=key)
    assert "ignoring --pose-vlm-model gemini-3.5-pro" in out.stdout
    # the pin is gone, so the backend's own default is what gets announced
    assert "z-ai/glm-5.3-flash on OpenRouter" in out.stdout
    assert "loading no-such-org/no-such-model" in out.stdout


def test_explicit_glm_rejects_a_model_pin_that_is_not_an_openrouter_id(tmp_path):
    """Both halves were typed, so this is an error rather than a silent drop —
    the same fail-fast the gemini arm gets, and for the same reason: the
    alternative is a run that writes a permanent `rejected` per escalation."""
    out = run_piped(tmp_path, "--pose-vlm", "glm",
                    "--pose-vlm-model", "gemini-3.5-pro")
    assert out.returncode != 0
    assert "--pose-vlm glm:" in out.stderr
    assert "gemini-3.5-pro is not an OpenRouter model id" in out.stderr
    assert "loading" not in out.stdout


def test_explicit_gemini_rejects_an_openrouter_shaped_model_pin(tmp_path):
    """The mirror of the glm check, and the same catastrophe on the other side
    of the wire: an org/model id in the Vertex URL names a model the project
    does not serve, the 4xx is non-transient so the retry contract reads it as
    a verdict, and `poser.Poser._fold` writes `arbitrated: "rejected"`
    permanently — one per escalation, unthrottled, because each rejection
    resets the breaker.

    `gcloud` is absent from PATH here, so the assertion that its failure is
    *not* what stopped the run is what pins the check ahead of the
    credentials: without it this run stops at the ADC probe instead, and on a
    machine with working ADC it would not stop at all."""
    out = run_piped(tmp_path, "--pose-vlm", "gemini",
                    "--pose-vlm-model", "z-ai/glm-5.3-flash")
    assert out.returncode != 0
    assert "--pose-vlm gemini:" in out.stderr
    assert "z-ai/glm-5.3-flash is an OpenRouter-shaped org/model id" in out.stderr
    assert "gcloud" not in out.stderr            # the pin, not the credentials
    assert "loading" not in out.stdout


def test_explicit_glm_without_a_key_fails_at_startup(tmp_path):
    """Symmetric with the gemini arm: asking for an arbiter by name and not
    getting one is an error, and the message names the file to fix."""
    out = run_piped(tmp_path, "--pose-vlm", "glm")
    assert out.returncode != 0
    assert "--pose-vlm glm:" in out.stderr
    assert "no-openrouter-key" in out.stderr        # the path it looked at
    assert "continue without the arbiter" not in out.stdout + out.stderr
    assert "loading" not in out.stdout


def test_explicit_off_asks_nothing(tmp_path):
    """`off` is already a decision; only `auto`'s silent degrade needed one."""
    out = run_piped(tmp_path, "--pose-vlm", "off")
    assert "loading no-such-org/no-such-model" in out.stdout
    assert "continue without the arbiter" not in out.stdout + out.stderr
    assert "gemini unavailable" not in out.stderr


def test_explicit_gemini_still_fails_at_startup(tmp_path):
    """Unchanged: asking for the arbiter by name and not getting one is an
    error, not a question — the answer is already on the command line."""
    out = run_piped(tmp_path, "--pose-vlm", "gemini")
    assert out.returncode != 0
    assert "--pose-vlm gemini:" in out.stderr
    assert "continue without the arbiter" not in out.stdout + out.stderr
    assert "loading" not in out.stdout


def test_repose_without_an_arbiter_stops_and_names_the_problem(tmp_path):
    """`--repose` exists to re-judge cached judgments, so a run with nothing
    to judge with is a typo rather than a no-op. Ignoring it silently would
    report success while leaving every entry the wrong arbiter answered
    exactly as it was — which is the whole thing the flag was asked for."""
    out = run_piped(tmp_path, "--pose-vlm", "off", "--repose")
    assert out.returncode != 0
    assert "--repose needs an arbiter" in out.stderr
    assert "--pose-vlm off" in out.stderr        # and which choice caused it
    assert "loading" not in out.stdout           # stopped before SigLIP


def test_repose_refuses_an_auto_that_degraded(tmp_path):
    """`--repose auto` is the one way to re-judge with a backend nobody typed
    (adversarial review finding 3, 2026-08-31). `auto` degrades silently by
    design, so a morning when ADC has expired would have the run walk the
    collection *replacing* gemini's judgments (+4 -> 42/44) with the
    measured-worse OpenRouter arbiter's (+3 -> 41/44) and report it as
    progress — spending exactly the judgments the flag exists to buy.
    Re-judging with the fallback has to be typed."""
    key = tmp_path / "or-key"
    key.write_text("sk-or-not-a-real-key\n")
    out = run_piped(tmp_path, "--repose", or_key=key)       # --pose-vlm auto
    assert out.returncode != 0
    assert "auto degraded to glm" in out.stderr
    assert "--pose-vlm glm" in out.stderr        # the way to mean it
    assert "42/44" in out.stderr and "41/44" in out.stderr
    assert "loading" not in out.stdout           # stopped before SigLIP


def test_repose_with_an_explicit_fallback_is_allowed(tmp_path):
    """The refusal is about the *undeclared* backend, not about glm: typing it
    is the decision, and the run continues."""
    key = tmp_path / "or-key"
    key.write_text("sk-or-not-a-real-key\n")
    out = run_piped(tmp_path, "--pose-vlm", "glm", "--repose", or_key=key)
    assert "auto degraded" not in out.stderr
    assert "loading no-such-org/no-such-model" in out.stdout


def test_repose_refuses_a_forced_up_axis(tmp_path):
    """A forced `--up-axis` builds every pose from the flag — `cache_checker`
    never opens the pose store — so there are no cached judgments for
    `--repose` to re-arbitrate and the flag cannot act. The arbiter is real
    here (explicit glm, a key), so the refusal cannot be the arbiterless one:
    what is refused is the combination. Announcing a re-judgment the run will
    not do is the failure being prevented."""
    key = tmp_path / "or-key"
    key.write_text("sk-or-not-a-real-key\n")
    out = run_piped(tmp_path, "--pose-vlm", "glm", "--repose",
                    "--up-axis", "z", or_key=key)
    assert out.returncode != 0
    assert "--repose does nothing under a forced --up-axis" in out.stderr
    assert "re-arbitrating cached poses" not in out.stdout   # never announced
    assert "loading" not in out.stdout                       # before SigLIP


def test_repose_with_an_arbiter_announces_the_judge_it_will_compare_against(tmp_path):
    """The comparison value is the run's whole `backend/model` string, and a
    reader has to be able to see which one it is: two gemini runs on different
    models re-buy each other's entries, and that is only obvious if the run
    says whose judgments it is keeping."""
    key = tmp_path / "or-key"
    key.write_text("sk-or-not-a-real-key\n")
    out = run_piped(tmp_path, "--pose-vlm", "glm", "--repose", or_key=key)
    assert "--repose: re-arbitrating cached poses not judged by " \
        "glm/z-ai/glm-5.3-flash" in out.stdout
    assert "loading no-such-org/no-such-model" in out.stdout
