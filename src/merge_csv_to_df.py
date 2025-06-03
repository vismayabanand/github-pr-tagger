#!/usr/bin/env python3
"""
src/merge.py
-------------
Concatenate every *_prs.csv file under data/ into one
big Arrow/Parquet file for faster downstream work.

Usage:
    python src/merge.py
"""

import glob, os, pandas as pd

RAW_GLOB = "data/*_prs.csv"
OUT_PATH = "data/all_repos_raw.parquet"

paths = glob.glob(RAW_GLOB)
if not paths:
    raise SystemExit("❌  No CSVs found in data/*.csv – run scraper first.")

print(f"Merging {len(paths)} files …")
df = pd.concat((pd.read_csv(p) for p in paths), ignore_index=True)
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
df.to_parquet(OUT_PATH, index=False)
print(f"✅  Saved {len(df):,} rows → {OUT_PATH}")
