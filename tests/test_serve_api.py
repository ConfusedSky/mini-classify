"""serve_api.py: the entry point's wiring.

Nothing imports this file by design — it is a CLI, like `classify_stls.py` —
so the only things worth pinning are the two that a refactor elsewhere can
break silently: its module scope stays torch-free, and its argparse block
still declares what a caller (and the README) says it does.

Both run it as a subprocess, because importing it in-process would prove
nothing about a module whose whole contract is how it behaves when *run*.
"""
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def run(*args, timeout=120):
    return subprocess.run([sys.executable, *args], cwd=REPO, capture_output=True,
                          text=True, timeout=timeout)


def test_importing_the_entry_point_costs_no_torch():
    """The heavy imports are deferred inside `load_embed` so the port binds
    before SigLIP loads (surface.md §`GET /status`). A module-scope torch
    import would move that cost in front of the bind and make a warming
    server indistinguishable from a dead one — the thing bind-before-warm
    exists to prevent."""
    code = ("import sys, serve_api; "
            "bad=[k for k in sys.modules if k in ('torch','open3d') "
            "or k.startswith(('torch.','open3d.'))]; "
            "print(','.join(bad)); sys.exit(1 if bad else 0)")
    r = run("-c", code)
    assert r.returncode == 0, f"forbidden imports: {r.stdout}\n{r.stderr}"


def test_the_help_declares_the_documented_flags():
    """The README and docs/api/surface.md both name these; argparse is where
    they are actually defined, and nothing else would notice them drifting."""
    r = run("serve_api.py", "--help")
    assert r.returncode == 0, r.stderr
    out = r.stdout
    for flag in ("--cache-dir", "--host", "--port", "--pool", "--rescan"):
        assert flag in out, flag
    assert "8077" in out, "the documented default port"
    assert "127.0.0.1" in out, "loopback default"


def test_it_refuses_to_start_with_no_input_and_no_recorded_run():
    """`apply_run_params` defaults the directory from the last classify run;
    with neither, the server has no collection to serve and says so rather
    than starting and answering every query with nothing."""
    r = run("serve_api.py", "--cache-dir", "no-such-cache-dir")
    assert r.returncode != 0
    assert "no input given" in (r.stdout + r.stderr)
