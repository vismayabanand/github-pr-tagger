#!/usr/bin/env python3
# src/fix_bundle.py

import joblib
import torch
from pathlib import Path

# 1. Load existing bundle (works on your local machine)
bundle_path = Path(__file__).parent / "minilm.joblib"
bundle = joblib.load(bundle_path)

# 2. Move the encoder (SentenceTransformer) to CPU
encoder = bundle["encoder"]
if hasattr(encoder, "to"):
    encoder = encoder.to("cpu")

# 3. Replace in the bundle
bundle["encoder"] = encoder

# 4. Write out a new CPU-only bundle (gzip compression)
out_path = Path(__file__).parent / "minilm_cpu.joblib"
joblib.dump(bundle, out_path, compress=3)
print(f" Saved CPU-only bundle to {out_path}")
