# Batch Prediction Pipeline

A batch NLP prediction pipeline that reads text documents from a SQLite database, classifies them using a trained TF-IDF + Logistic Regression model, and writes the results back to the database on a recurring schedule.

## Project Structure

```
ml-practice-7/
├── setup_db.py      # Creates the DB and seeds initial documents
├── train_model.py   # Trains the NLP pipeline and saves model.pkl
├── predict.py       # Batch prediction logic (reads unpredicted docs, writes results)
├── scheduler.py     # Runs predict.py every 5 minutes + simulates new incoming data
└── requirements.txt
```

## Dataset

**20 Newsgroups** — a classic NLP benchmark of ~3,400 news articles across 4 categories:

| Label | Topic |
|---|---|
| `comp.graphics` | Computer graphics |
| `rec.sport.baseball` | Baseball |
| `sci.med` | Medicine & science |
| `talk.politics.misc` | Politics |

## Database Schema

**input_data**

| Column | Type | Description |
|---|---|---|
| id | INTEGER | Primary key |
| text | TEXT | Raw news article text |

**predictions**

| Column | Type | Description |
|---|---|---|
| id | INTEGER | Primary key |
| input_id | INTEGER | FK → input_data.id |
| prediction | TEXT | Predicted category name |
| prediction_timestamp | DATETIME | When the prediction was generated |

## Setup & Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the model

```bash
python train_model.py
```

Downloads the 20 Newsgroups dataset, trains a TF-IDF + Logistic Regression pipeline, and saves it as `model.pkl`. Typical test accuracy: **~90 %**.

### 3. Initialize the database

```bash
python setup_db.py
```

Creates `ml_pipeline.db` and seeds the first 50 documents into `input_data`.

### 4. Start the scheduler

```bash
python scheduler.py
```

- Runs an immediate prediction pass on startup.
- Inserts 20 new documents each cycle to simulate a live stream.
- Re-runs automatically every **5 minutes**.

### Run a single prediction manually

```bash
python predict.py
```

## How It Works

1. `setup_db.py` creates the SQLite database and seeds it with initial text documents.
2. `train_model.py` trains a scikit-learn `Pipeline` (TF-IDF vectorizer → Logistic Regression) on the 20 Newsgroups dataset and serialises it with `joblib`.
3. `predict.py` queries all rows in `input_data` with no matching entry in `predictions`, runs them through the model, and inserts the predicted category names with a timestamp.
4. `scheduler.py` uses the `schedule` library to repeat this every 5 minutes and injects new documents between runs to simulate real incoming data.

## Model

- Dataset: [20 Newsgroups](https://scikit-learn.org/stable/datasets/real_world.html#the-20-newsgroups-text-dataset) (4 categories)
- Pipeline: `TfidfVectorizer` (unigrams + bigrams, top 10k features) → `LogisticRegression`
- Typical test accuracy: ~90 %
