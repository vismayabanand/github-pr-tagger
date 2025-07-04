from fastapi import FastAPI
import joblib, pathlib, threading

app = FastAPI()
_model_lock = threading.Lock()
_bundle = None                      # not loaded yet

def get_model():
    global _bundle
    with _model_lock:
        if _bundle is None:         # first request
            _bundle = joblib.load(pathlib.Path(__file__).parent / "minilm.joblib")
    return _bundle

@app.get("/")
def root():
    return {"status": "ok"}

@app.post("/label")
def label(pr: dict):
    bundle = get_model()
    enc, clf, mlb = bundle["encoder"], bundle["clf"], bundle["binarizer"]
    text  = (pr.get("title","") + " " + pr.get("body","")).strip()
    prob  = clf.predict_proba(enc.encode([text]))[0]
    idx   = prob.argsort()[-2:][::-1]
    idx   = [i for i in idx if prob[i] >= 0.25]
    return {"labels": mlb.classes_[idx].tolist(),
            "scores": prob[idx].round(3).tolist()}
