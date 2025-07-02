#!/usr/bin/env python3
"""
batch_scrape.py – run github_prs.py for a list of popular OSS projects.

Usage
-----
$ python batch_scrape.py          # sequential (safest)
$ python batch_scrape.py --jobs 3 # run up to 3 repos in parallel
"""

from __future__ import annotations
import argparse, subprocess, sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# --------------------------------------------------------------------------- #
# 1.  Edit the targets list to taste
# --------------------------------------------------------------------------- #
TARGETS: list[tuple[str, str, int]] = [
    ("kubernetes",    "kubernetes",     2000),
    ("pytorch",       "pytorch",        2000),
    ("elastic",       "elasticsearch",  2000),
    ("hashicorp",     "terraform",      2000),
    ("facebook",      "react",          2000),
    ("grafana",       "grafana",        2000),
    ("prometheus",    "prometheus",     2000),
    ("angular",       "angular",        2000),
    ("apache",        "spark",          2000),
    ("golang",        "go",             2000),
]

SCRIPT = Path("scrape/github_prs.py")  # relative path to your scraper script

# --------------------------------------------------------------------------- #
# 2.  Helper that runs one scrape command & prints progress
# --------------------------------------------------------------------------- #
def run_single(org: str, repo: str, limit: int) -> None:
    cmd = [sys.executable, SCRIPT, org, repo, "--limit", str(limit)]
    print(f"▶  Scraping {org}/{repo} (limit={limit}) …")
    try:
        subprocess.run(cmd, check=True)
        print(f"✓  Done {org}/{repo}")
    except subprocess.CalledProcessError as exc:
        print(f"✗  {org}/{repo} failed with exit code {exc.returncode}")

# --------------------------------------------------------------------------- #
# 3.  Main entry-point (sequential or parallel)
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(
        description="Batch-run github_prs.py over multiple repos")
    ap.add_argument("--jobs", "-j", type=int, default=1,
                    help="Number of parallel jobs (default 1 = sequential)")
    args = ap.parse_args()

    if args.jobs == 1:
        for org, repo, limit in TARGETS:
            run_single(org, repo, limit)
    else:
        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            futs = {pool.submit(run_single, *t): t for t in TARGETS}
            for fut in as_completed(futs):
                pass  # run_single already prints its own result lines

if __name__ == "__main__":
    main()
