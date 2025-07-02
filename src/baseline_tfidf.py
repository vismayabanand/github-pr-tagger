#!/usr/bin/env python3
"""
Baseline TF‑IDF → Logistic Regression multi‑label classifier
----------------------------------------------------------------
* Vectoriser: 1–3‑gram TF‑IDF, 120 k features
* Classifier: One‑vs‑Rest LogisticRegression (C=4, balanced)
* Prints micro‑F1 and saves compressed joblib bundle
"""

from pathlib import Path
import joblib, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score, classification_report

DATA_DIR = Path("data/clean_parqs")
MODEL_PATH = Path("src/baseline_tfidf.joblib")

train = pd.read_parquet(DATA_DIR / "train.parquet")
test  = pd.read_parquet(DATA_DIR / "test.parquet")

X_train = (train.title + " " + train.body).fillna("")
X_test  = (test.title  + " " + test.body ).fillna("")

mlb = MultiLabelBinarizer()
Y_train = mlb.fit_transform(train.labels_norm)
Y_test  = mlb.transform(test.labels_norm)

pipe = Pipeline([
    ("tfidf", TfidfVectorizer(
        max_features=120_000,
        ngram_range=(1, 3),
        stop_words="english")),
    ("clf", OneVsRestClassifier(
        LogisticRegression(
            C=4.0,
            class_weight="balanced",
            max_iter=2000,
            n_jobs=-1)))
])

print("Training …")
pipe.fit(X_train, Y_train)

Y_pred = pipe.predict(X_test)
print("\n=== Results on TEST set ===")
print("micro‑F1:", round(f1_score(Y_test, Y_pred, average="micro"), 3))

# Optional detailed table (comment out if noisy)
# print(classification_report(Y_test, Y_pred, target_names=mlb.classes_))

# save compressed model bundle
joblib.dump({"model": pipe, "binarizer": mlb}, MODEL_PATH, compress=3)
print("\n✅  Saved model →", MODEL_PATH)