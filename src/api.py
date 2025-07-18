import pathlib
import joblib
from typing import List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="GitHub PR Tagger API")


class PullRequest(BaseModel):
    """Schema for the JSON body expected by `/label`."""

    title: str
    body: str | None = ""


# --------------------
# Model loading helpers
# --------------------
_MODEL = None  # Lazily‑loaded model cache


def get_model():
    """Load the encoder, classifier and MultiLabelBinarizer once and cache them."""
    global _MODEL
    if _MODEL is None:
        path = pathlib.Path(__file__).parent / "minilm_cpu.joblib"
        _MODEL = joblib.load(path)
    return _MODEL


# --------------------
# Inference parameters
# --------------------
THRESHOLD: float = 0.20  # Minimum probability to accept a label
TOP_K: int = 3           # Ensure at least this many labels get returned when all probs < threshold


@app.post("/label", response_model=List[str])
def label(pr: PullRequest):
    """Predict labels for a pull‑request title/body.

    The endpoint will return a JSON list of label strings. When no label clears the
    probability *THRESHOLD*, it falls back to the *TOP_K* most‑likely labels so that
    callers always get a non‑empty list (and never `null`).
    """

    model = get_model()
    enc = model["encoder"]
    clf = model["clf"]
    mlb = model["binarizer"]

    text = f"{pr.title} {pr.body or ''}".strip()
    if not text:
        raise HTTPException(status_code=400, detail="Both title and body are empty.")

    # 1️⃣  Encode the text → 2️⃣ predict probabilities per label
    emb = enc.encode([text])
    probs = clf.predict_proba(emb)[0]  # shape = (n_labels,)

    # 3️⃣  Primary selection: all labels with prob ≥ THRESHOLD
    mask = probs >= THRESHOLD
    labels = mlb.classes_[mask].tolist()

    # 4️⃣  Fallback: return TOP_K highest‑probability labels if list is empty
    if not labels:
        top_idx = probs.argsort()[-TOP_K:][::-1]
        labels = mlb.classes_[top_idx].tolist()

    return labels


@app.get("/health")
def health() -> str:
    """Lightweight health check for containers or K8s probes."""
    return "ok"
