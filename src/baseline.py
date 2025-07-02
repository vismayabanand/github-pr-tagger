#!/usr/bin/env python3
"""
Baseline multi-label classifier:
  • TF-IDF (1–2-grams, 60 k features)
  • One-vs-Rest Logistic Regression
  • Saves model to src/baseline_tfidf.joblib
"""

from pathlib import Path
import pandas as pd, joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score, classification_report
from sklearn.multiclass import OneVsRestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parent.parent   # pr-tagger root
DATA = ROOT / "data" / "clean_parqs"

train = pd.read_parquet(DATA / "train.parquet")
test  = pd.read_parquet(DATA / "test.parquet")

X_train = (train["title"] + " " + train["body"]).fillna("")
X_test  = (test ["title"] + " " + test ["body"]).fillna("")

mlb = MultiLabelBinarizer()
y_train = mlb.fit_transform(train["labels_norm"])
y_test  = mlb.transform(test["labels_norm"])

pipe = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=60_000, ngram_range=(1, 2))),
    ("clf",  OneVsRestClassifier(
                 LogisticRegression(max_iter=400, n_jobs=-1, verbose=0)))
])
print("Training …")
pipe.fit(X_train, y_train)

y_pred = pipe.predict(X_test)
score = f1_score(y_test, y_pred, average="micro")
print(f"\n=== Results on TEST set ===")
print(f"micro-F1: {score:.3f}\n")

print(classification_report(
        y_test, y_pred, target_names=mlb.classes_))


model_path = ROOT / "src" / "baseline_tfidf.joblib"
joblib.dump({"model": pipe, "binarizer": mlb}, model_path)
print("\n  Saved model →", model_path)