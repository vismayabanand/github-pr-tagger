#!/usr/bin/env python3
"""
scrape/github_prs.py
--------------------
Pull requests → CSV, one row per PR.

Usage
-----
python scrape/github_prs.py <owner> <repo> --limit 1000
# example:
python scrape/github_prs.py microsoft vscode --limit 1500
"""
import os, csv, time, argparse, requests
from typing import List, Dict
from dotenv import load_dotenv

# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------
PER_PAGE = 100                               # GitHub max page size
SLEEP_BETWEEN_PAGES = 0.3                    # be polite to API
CSV_FIELDS = ["id", "title", "body", "labels"]

load_dotenv()                                # read .env in project root
TOKEN = os.getenv("GH_TOKEN")
if not TOKEN:
    raise RuntimeError("GH_TOKEN missing. Add it to your .env file.")

HEADERS = {
    "Authorization": f"Bearer {TOKEN}",
    "Accept": "application/vnd.github+json",
    "X-GitHub-Api-Version": "2022-11-28",
}

# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def fetch_page(owner: str, repo: str, page: int) -> List[Dict]:
    """Return a list of PR objects (possibly empty) for the given page."""
    url = f"https://api.github.com/repos/{owner}/{repo}/pulls"
    params = {"state": "all", "per_page": PER_PAGE, "page": page}
    r = requests.get(url, headers=HEADERS, params=params, timeout=15)
    r.raise_for_status()
    return r.json()

def scrape(owner: str, repo: str, limit: int) -> List[Dict]:
    """Fetch up to *limit* PRs and return simplified dicts."""
    rows = []
    current_page = 1
    while len(rows) < limit:
        page_items = fetch_page(owner, repo, current_page)
        if not page_items:                         # out of pages
            break

        for pr in page_items:
            rows.append(
                {
                    "id": pr["number"],
                    "title": pr["title"] or "",
                    "body": pr.get("body") or "",
                    "labels": ";".join(lbl["name"] for lbl in pr["labels"]),
                }
            )
            if len(rows) >= limit:
                break

        current_page += 1
        time.sleep(SLEEP_BETWEEN_PAGES)

    return rows

def write_csv(rows: List[Dict], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scrape GitHub pull requests to CSV.")
    parser.add_argument("owner", help="GitHub owner/org")
    parser.add_argument("repo", help="Repository name")
    parser.add_argument("--limit", type=int, default=1000, help="Max PRs to fetch")
    args = parser.parse_args()

    print(f"Fetching up to {args.limit} PRs from {args.owner}/{args.repo} …")
    data = scrape(args.owner, args.repo, args.limit)
    out_file = f"data/{args.owner}_{args.repo}_prs.csv"
    write_csv(data, out_file)
    print(f"✅ Saved {len(data):,} rows → {out_file}")
