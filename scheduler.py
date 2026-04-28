"""
Batch prediction scheduler.

Runs run_batch_prediction() every 5 minutes.
Between runs it inserts a fresh batch of documents into input_data to
simulate an incoming data stream.
"""

import sqlite3
import time

import schedule
from sklearn.datasets import fetch_20newsgroups

from predict import run_batch_prediction

DB_PATH = "ml_pipeline.db"
CATEGORIES = ["rec.sport.baseball", "sci.med", "talk.politics.misc", "comp.graphics"]

_data = fetch_20newsgroups(subset="all", categories=CATEGORIES, remove=("headers", "footers", "quotes"))
_all_docs = _data.data
_next_index = 50  # first 50 already seeded by setup_db.py


def _add_new_docs(batch_size: int = 20) -> None:
    global _next_index
    end = min(_next_index + batch_size, len(_all_docs))
    new_docs = [(doc,) for doc in _all_docs[_next_index:end]]

    if not new_docs:
        print("All documents have been inserted — no new docs added this cycle.")
        return

    conn = sqlite3.connect(DB_PATH)
    conn.executemany("INSERT INTO input_data (text) VALUES (?)", new_docs)
    conn.commit()
    conn.close()
    print(f"Inserted {len(new_docs)} new document(s) (indices {_next_index}–{end - 1}).")
    _next_index = end


def job() -> None:
    print("\n--- Batch prediction job triggered ---")
    _add_new_docs(batch_size=20)
    run_batch_prediction()


if __name__ == "__main__":
    job()

    schedule.every(5).minutes.do(job)
    print("\nScheduler running — next execution in 5 minutes. Press Ctrl+C to stop.")

    while True:
        schedule.run_pending()
        time.sleep(1)
