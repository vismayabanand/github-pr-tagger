#  Intelligent GitHub PR Auto-Labeler

Automatically predicts and applies labels to new pull requests, cutting manual triage to zero for busy maintainers.

| Stage        | Stack                                                                 |
|--------------|-----------------------------------------------------------------------|
| **Data**     | Python, `pandas`, `pyarrow`, GitHub API                               |
| **Model**    | Sentence-Transformers **MiniLM L6** → One-vs-Rest Logistic Regression |
| **Service**  | `FastAPI` + `uvicorn`, Docker                                         |
| **CI/CD**    | GitHub Actions (scrape ➜ train ➜ build), Railway                     |
| **Integration** | GitHub Action to auto-label PRs                                    |

---

## 1.  Live Demo

```bash
# hit the public endpoint (~100 ms):
curl -X POST https://github-pr-tagger-production.up.railway.app/label \
     -H "Content-Type: application/json" \
     -d '{"title":"Fix crash when saving empty file",
          "body":"Adds regression test and closes #1234."}'

# ➜ {"labels":["bug","tests"],"scores":[0.34,0.29]}
```

---

## 2.  Project Layout

```text
.
├── data/                   # raw & cleaned Parquet files
├── scrape/                 # GitHub API scrapers
├── src/
│   ├── clean_split.py      # label mapping + train/test split
│   ├── baseline_tfidf.py   # classic TF-IDF + LogisticRegression baseline
│   ├── minilm_train.py     # MiniLM training script
│   ├── minilm.joblib       # ~40 MB CPU-only model bundle
│   └── api.py              # FastAPI inference service
└── .github/workflows/
    ├── pipeline.yml        # nightly scrape + retrain
    └── auto_label.yml      # PR-triggered labeling action
```

---

## 3.  Quick Start (Local)

```bash
git clone https://github.com/<your-username>/github-tagger.git
cd github-tagger

# Set up environment
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt   # torch + MiniLM CPU wheels only

# Run API server
uvicorn src.api:app --reload     # launches at http://127.0.0.1:8000
```

Test locally:
```bash
curl -X POST http://127.0.0.1:8000/label \
     -H 'Content-Type: application/json' \
     -d '{"title":"docs: clarify YAML config", "body":""}'
```

---

## 4.  Deploy to Production (Railway)

```bash
railway init    # detects FastAPI → auto-generates Nixpacks
railway run up
```

Then set the tagger URL:
```bash
export TAGGER_URL=https://<project>.up.railway.app/label
```

> **Note**: `requirements.txt` pins `torch==2.3.1+cpu` and points pip to https://download.pytorch.org/whl/cpu — keeping the build under 1 GB.

---

## 5.  Add to Any Repository (GitHub)

1. Navigate to `Settings → Secrets → Actions`
2. Add:
   ```env
   TAGGER_URL = https://<your-service>.up.railway.app/label
   ```
3. Copy `auto_label.yml` into `.github/workflows/`.

 Every new PR will now be labeled in ~10 seconds via the deployed API!

---

## 6.  Retrain on Fresh Data 

```bash
# Scrape more GitHub repos
python scrape/github_prs.py numpy numpy --limit 2000

# Merge → Clean → Split
python src/merge_csv_to_df.py
python src/clean_split.py --min_freq 500

# Retrain MiniLM model
python src/minilm_train.py

# Commit & redeploy
git add src/minilm.joblib
git commit -m "retrain model"
git push   # CI will rebuild & redeploy to Railway
```

---

Enjoy auto-labeling your PRs 
