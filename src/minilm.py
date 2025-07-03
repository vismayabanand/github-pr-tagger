#!/usr/bin/env python3
"""
Train MiniLM-based PR-tagger and save to src/minilm.joblib
"""
from pathlib import Path
import pandas as pd, joblib
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score
import os; 
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "clean_parqs"

train = pd.read_parquet(DATA / "train.parquet")
test  = pd.read_parquet(DATA / "test.parquet")

X_train_txt = (train.title + " " + train.body).fillna("").tolist()
X_test_txt  = (test.title  + " " + test.body ).fillna("").tolist()

print("⇢ Encoding PR text with MiniLM …")
enc = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
X_train = enc.encode(X_train_txt, batch_size=64, show_progress_bar=True)
X_test  = enc.encode(X_test_txt,  batch_size=64, show_progress_bar=True)

mlb = MultiLabelBinarizer()
y_train = mlb.fit_transform(train.labels_norm)
y_test  = mlb.transform(test.labels_norm)
clf = OneVsRestClassifier(
         LogisticRegression(C=4.0, max_iter=1000, n_jobs=-1))
clf.fit(X_train, y_train)

pred = clf.predict(X_test)
score = f1_score(y_test, pred, average="micro")
print(f"MiniLM micro-F1: {score:.3f}")

joblib.dump({"encoder": enc, "clf": clf, "binarizer": mlb},
            ROOT / "src" / "minilm.joblib")
print("✅  saved → src/minilm.joblib")