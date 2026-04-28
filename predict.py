import sqlite3
from datetime import datetime

import joblib

DB_PATH = "ml_pipeline.db"
MODEL_PATH = "model.pkl"


def run_batch_prediction():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute(
        """
        SELECT id, text
        FROM input_data
        WHERE id NOT IN (SELECT input_id FROM predictions)
        """
    )
    rows = cur.fetchall()

    if not rows:
        print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] No new documents to predict.")
        conn.close()
        return

    artifact = joblib.load(MODEL_PATH)
    pipeline = artifact["pipeline"]
    target_names = artifact["target_names"]

    ids = [r[0] for r in rows]
    texts = [r[1] for r in rows]
    preds = pipeline.predict(texts)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    cur.executemany(
        """
        INSERT INTO predictions (input_id, prediction, prediction_timestamp)
        VALUES (?, ?, ?)
        """,
        [(ids[i], target_names[preds[i]], ts) for i in range(len(ids))],
    )

    conn.commit()
    conn.close()
    print(f"[{ts}] Predicted {len(ids)} document(s).")


if __name__ == "__main__":
    run_batch_prediction()
