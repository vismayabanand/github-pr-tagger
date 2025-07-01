#!/usr/bin/env python3
"""
src/clean_split.py
------------------
Load the raw Parquet produced by merge.py, normalise & filter labels,
then write train/test Parquet files to  data/clean_parqs/.
"""

import os, re, glob
from collections import Counter
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

# ----------------------------------------------------------------------
RAW_PATH  = Path("data/all_repos_raw.parquet")
OUT_DIR   = Path("data/clean_parqs")
MIN_FREQ  = 500          # keep labels that appear ≥ MIN_FREQ times
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------
# hard synonyms (exact matches)
MAP = {
    "bug": "bug", "bugfix": "bug", "kind/bug": "bug",
    "enhancement": "feature", "feat": "feature", "kind/feature": "feature",
    "documentation": "docs", "doc": "docs", "docs": "docs",
    "test": "tests", "tests": "tests",
    "refactor": "refactor", "maintenance": "refactor", "debt": "refactor",
}

# prefix families  →  canonical tag
MAP_PREFIX = {
    "area/":     "area",
    "sig/":      "sig",
    "priority/": "priority",
    "size/":     "size",
}

def map_labels(lbls):
    """
    • collapse common prefixes   (e.g. 'sig/node'   → 'sig')
    • apply exact-match MAP      (e.g. 'bugfix' → 'bug')
    • deduplicate, drop empties
    """
    canon = []
    for l in lbls:
        if not l:
            continue
        # prefix collapse
        l2 = next((v for k, v in MAP_PREFIX.items() if l.startswith(k)), l)
        canon.append(MAP.get(l2, l2))    # exact synonym map
    return sorted(set(canon))
# ------------------------------------------------------------------


if not RAW_PATH.exists():
    raise SystemExit("❌  Raw Parquet not found. Run merge.py first.")

print(f"Loading {RAW_PATH} …")
df = pd.read_parquet(RAW_PATH)

# --- label normalisation ------------------------------------------------
df["labels_list"] = (
    df["labels"].fillna("")
      .str.lower()
      .str.replace(r"\s+", "-", regex=True)
      .str.split(";")
)
df["labels_norm"] = df["labels_list"].apply(map_labels)

# --- frequency filter ---------------------------------------------------
freq = Counter(l for labs in df["labels_norm"] for l in labs)
keep_labels = {l for l, c in freq.items() if c >= MIN_FREQ}

df = df[df["labels_norm"].map(lambda L: any(l in keep_labels for l in L))]
df = df[df["labels_norm"].map(bool)]          # drop rows with 0 kept labels

print(f"After cleaning: {len(df):,} rows, {len(keep_labels)} labels kept.")
for l in sorted(keep_labels):
    print(f"  {l:<10}: {freq[l]}")

# --- train / test split --------------------------------------------------
train, test = train_test_split(
    df[["title", "body", "labels_norm"]],
    test_size=0.2,
    random_state=42,
    shuffle=True,
)

(train.to_parquet(OUT_DIR / "train.parquet", index=False))
(test .to_parquet(OUT_DIR / "test.parquet",  index=False))

print(f"✅  Saved cleaned Parquet files → {OUT_DIR}")
print(f"   train: {len(train):,} rows")
print(f"   test : {len(test):,} rows")
