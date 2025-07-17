import pathlib, joblib
from fastapi import FastAPI

app = FastAPI()
_model = None

def get_model():
    global _model
    if _model is None:
        path = pathlib.Path(__file__).parent / "minilm_cpu.joblib"
        _model = joblib.load(path)
    return _model

@app.post("/label")
def label(pr: dict):
    enc, clf, mlb = get_model()["encoder"], get_model()["clf"], get_model()["binarizer"]
    text = (pr.get("title","") + " " + pr.get("body","")).strip()
    emb = enc.encode([text])
    # … rest is unchanged …
