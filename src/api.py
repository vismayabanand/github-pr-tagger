from fastapi import FastAPI
import joblib, numpy as np, os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ---------- Load model bundle ----------
bundle  = joblib.load("src/minilm.joblib")          # or baseline_tfidf.joblib
enc, clf, mlb = bundle["encoder"], bundle["clf"], bundle["binarizer"]

# ---------- Create the web-app first ----------
app = FastAPI()                #  ← this line was missing

# ---------- Config ----------
#THRESH = 0.30          # global min-confidence
#TOP_K  = 2             # return at most 2 labels
THRESH  = 0.30      # return any label ≥30 %
TOP_K   = 5         # but cap list length to 5

# ---------- Routes ----------
@app.get("/")                          # quick health-check
def root():
    return {"status": "ok"}

@app.post("/label")
def label(pr: dict):
    text = (pr.get("title", "") + " " + pr.get("body", "")).strip()
    emb  = enc.encode([text])
    prob = clf.predict_proba(emb)[0]            # 1-D array
    #idx  = [i for i, p in enumerate(prob) if p >= THRESH]
    # keep highest-prob if more than TOP_K
    #idx  = sorted(idx, key=lambda i: prob[i], reverse=True)[:TOP_K]

    top = prob.argmax()
    if prob[top] >= 0.15:
        idx = [top]
    else:
        idx = []
    return {"labels": mlb.classes_[idx].tolist(),
            "scores": prob[idx].round(3).tolist()}
