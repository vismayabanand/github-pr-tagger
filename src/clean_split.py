#!/usr/bin/env python3
"""
Clean and split the merged PR dataset.
------------------------------------------------------------------------
* Normalises labels (exact synonyms + prefix collapse)
* Keeps **TOP_N** most frequent labels to avoid train/test drift
* 80/20 stratified split → Parquet files for downstream training
"""

import os, glob, re
from collections import Counter
import pandas as pd
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------
# Params
RAW_PATH   = "data/all_repos_raw.parquet"
OUT_DIR    = "data/clean_parqs"
TOP_N      = 20          # ← keep N busiest labels
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------
# Synonym + prefix maps  ── tweak here only ─────────────────────────────
MAP = {
    # bugs / features
    "bug": "bug", "type:-bug": "bug", "type:-bug/fix": "bug",
    "bugfix": "bug", "kind/bug": "bug",
    "enhancement": "feature", "01---enhancement": "feature",
    # docs / tests
    "documentation": "docs", "doc": "docs", "docs": "docs",
    "test": "tests", "tests": "tests",
    # misc
    "performance": "perf",
}

MAP_PREFIX = {
    "area/":     "area",
    "a-":        "area",          # a-io, a-ast …
    "sig/":      "sig",
    "module:":   "module",
    "priority/": "priority",
    "size/":     "size",
    "topic:-":   "topic",
    "type:-":    "type",
    "stat:":     "status",
    "risk:-":    "risk",
    "backend/":  "backend",
}

# ---------------------------------------------------------------------
# Helper

def map_labels(lbls: list[str]) -> list[str]:
    """Lower‑case, collapse prefixes, apply exact synonyms, dedup"""
    canon: set[str] = set()
    for l in lbls:
        if not l:
            continue
        l = l.lower().strip()
        # prefix collapse
        l2 = next((v for k, v in MAP_PREFIX.items() if l.startswith(k)), l)
        # exact synonym
        canon.add(MAP.get(l2, l2))
    return sorted(canon)

# ---------------------------------------------------------------------
# Load and clean
print(f"Loading {RAW_PATH} …")
df = pd.read_parquet(RAW_PATH)

# split label strings → list[str]
df["labels_list"] = (
    df.labels.fillna("")
      .str.replace(r"\s+", "-", regex=True)
      .str.split(";|")  # split on semicolon or pipe
)

# map + normalise
df["labels_norm"] = df.labels_list.apply(map_labels)

# --- keep busiest TOP_N labels ---------------------------------------
freq = Counter(l for labs in df.labels_norm for l in labs)
keep = {l for l, _ in freq.most_common(TOP_N)}
df = df[df.labels_norm.map(lambda L: any(l in keep for l in L))]
print(f"After cleaning: {len(df):,} rows, {len(keep)} labels kept.")
for l, c in freq.most_common(TOP_N):
    print(f"  {l:<15s}: {c}")

# --- train / test split ---------------------------------------------
train, test = train_test_split(
    df[["title", "body", "labels_norm"]],
    test_size=0.20,
    random_state=42,
    stratify=df.labels_norm.map(tuple),
)

train.to_parquet(f"{OUT_DIR}/train.parquet", index=False)
test .to_parquet(f"{OUT_DIR}/test.parquet",  index=False)
print("\n✅  Saved cleaned Parquet files →", OUT_DIR)
print(f"   train: {len(train):,} rows\n   test : {len(test):,} rows")