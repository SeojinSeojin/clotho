"""Back up the git history to RESULTS_REPO as a bundle, for sessions where `git push` has no credentials.

  python experiments/sharded/git_backup.py      -> RESULTS_REPO git_bundles/clotho_main.bundle (overwritten)

Restore on a new machine: download the bundle, then `git fetch <bundle> main:bundle_main` in a clone and
fast-forward or merge. The bundle holds every commit reachable from main.
"""
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hub_sync as hs  # noqa: E402

REPO = Path(__file__).resolve().parents[2]

if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as d:
        b = Path(d) / "clotho_main.bundle"
        subprocess.run(["git", "-C", str(REPO), "bundle", "create", str(b), "main"], check=True,
                       capture_output=True)
        head = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"], check=True,
                              capture_output=True, text=True).stdout.strip()
        hs.commit_files(hs.get_results_repo(), {"git_bundles/clotho_main.bundle": b}, f"git bundle main @ {head}")
        print(f"[git_backup] uploaded bundle of main @ {head} ({b.stat().st_size / 1e6:.1f} MB)")
