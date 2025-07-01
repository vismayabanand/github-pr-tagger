from fastapi import FastAPI
import joblib, numpy as np, os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ---------- Load model bundle ----------
bundle  = joblib.load("src/minilm.joblib")          # or baseline_tfidf.joblib
enc, clf, mlb = bundle["encoder"], bundle["clf"], bundle["binarizer"]

# ---------- Create the web-app first ----------
app = FastAPI()                #  ← this line was missing

# ---------- Config ----------
THRESH = 0.30          # global min-confidence
TOP_K  = 2             # return at most 2 labels

# ---------- Routes ----------
@app.get("/")                          # quick health-check
def root():
    return {"status": "ok"}

@app.post("/label")
def label(pr: dict):
    text = (pr.get("title", "") + " " + pr.get("body", "")).strip()
    emb  = enc.encode([text])
    prob = clf.predict_proba(emb)[0]            # 1-D array
    top  = prob.argsort()[-TOP_K:][::-1]        # best TOP_K
    idx  = [i for i in top if prob[i] >= 0.15]  # apply 0.15 cut-off
    return {"labels": mlb.classes_[idx].tolist(),
            "scores": prob[idx].round(3).tolist()}
