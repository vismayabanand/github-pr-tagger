#!/usr/bin/env python3
"""
Clean and split the merged PR dataset
-------------------------------------
• Normalises labels (synonyms + prefix collapse)
• Filters out 1–2-char garbage labels
• Keeps TOP_N most-frequent labels
• 80/20 stratified split → Parquet
"""

import os, glob
from collections import Counter
import pandas as pd
from sklearn.model_selection import train_test_split

RAW_GLOB = "data/*_prs.csv"
OUT_DIR  = "data/clean_parqs"
TOP_N    = 20                    # keep busiest labels

os.makedirs(OUT_DIR, exist_ok=True)

# ---------- label maps ----------
MAP_PREFIX = {
    "area/": "area",     "a-": "area",
    "sig/":  "sig",
    "priority/": "priority",
    "size/": "size",
    "module:": "module",
    "topic:-": "topic",
    "type:-":  "type",
    "stat:":   "status",
    "risk:-":  "risk",
    "backend/": "backend",
}

MAP = {
    "🤖:docs": "docs",
    "robot:docs": "docs",
    "bug": "bug", "type:-bug": "bug", "type:-bug/fix": "bug",
    "enhancement": "feature", "01---enhancement": "feature",
    "documentation": "docs", "doc": "docs", "docs": "docs",
    "test": "tests", "tests": "tests",
}

# ---------- helpers ----------
def map_labels(lbls):
    """Collapse prefixes, synonyms, drop empties & ≤2-char noise."""
    canon = set()
    for l in lbls:
        if not l:
            continue
        l2 = next((v for k, v in MAP_PREFIX.items() if l.startswith(k)), l)
        l3 = MAP.get(l2, l2)
        if len(l3) >= 3:          # <<<  NEW: ignore 1–2-char tokens
            canon.add(l3)
    return sorted(canon)

# ---------- load & normalise ----------
paths = glob.glob(RAW_GLOB)
if not paths:
    raise SystemExit("❌  No CSVs found – run scraper first.")

print(f"Loading {len(paths)} CSV files …")
df = pd.concat((pd.read_csv(p) for p in paths), ignore_index=True)

df["labels_list"] = (
    df.labels.fillna("")
      .str.lower()
      .str.replace(r"\\s+", "-", regex=True)
      .str.split(r"[;,]", regex=True)            # split on ; or ,
)

df["labels_norm"] = df.labels_list.apply(map_labels)
df = df[df.labels_norm.map(bool)]                # drop rows with 0 labels

# ---------- keep TOP_N busiest labels ----------
freq = Counter(l for labs in df.labels_norm for l in labs)
keep = {l for l, _ in freq.most_common(TOP_N)}
df   = df[df.labels_norm.map(lambda L: any(l in keep for l in L))]

print(f"After cleaning: {len(df):,} rows, {len(keep)} labels kept.")
for l, c in freq.most_common(TOP_N):
    print(f"  {l:<16}: {c}")

# ---------- guarantee ≥2 samples per strata ----------
combo_freq = df.labels_norm.map(tuple).value_counts()
df = df[combo_freq[df.labels_norm.map(tuple)].values >= 2]

# ---------- train / test split ----------
train, test = train_test_split(
    df[["title", "body", "labels_norm"]],
    test_size=0.20,
    random_state=42,
    stratify=df.labels_norm.map(tuple)
)

train.to_parquet(f"{OUT_DIR}/train.parquet", index=False)
test.to_parquet (f"{OUT_DIR}/test.parquet",  index=False)
print("✅  Saved train/test Parquet →", OUT_DIR)
print("Done! You can now run the baseline classifier.")