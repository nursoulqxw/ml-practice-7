import sqlite3

from sklearn.datasets import fetch_20newsgroups

DB_PATH = "ml_pipeline.db"
CATEGORIES = ["rec.sport.baseball", "sci.med", "talk.politics.misc", "comp.graphics"]


def setup_database():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute("DROP TABLE IF EXISTS predictions")
    cur.execute("DROP TABLE IF EXISTS input_data")

    cur.execute(
        """
        CREATE TABLE input_data (
            id   INTEGER PRIMARY KEY AUTOINCREMENT,
            text TEXT NOT NULL
        )
        """
    )

    cur.execute(
        """
        CREATE TABLE predictions (
            id                   INTEGER PRIMARY KEY AUTOINCREMENT,
            input_id             INTEGER UNIQUE NOT NULL,
            prediction           TEXT NOT NULL,
            prediction_timestamp DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (input_id) REFERENCES input_data(id)
        )
        """
    )

    # Seed the first 50 documents
    data = fetch_20newsgroups(subset="all", categories=CATEGORIES, remove=("headers", "footers", "quotes"))
    initial = [(doc,) for doc in data.data[:50]]
    cur.executemany("INSERT INTO input_data (text) VALUES (?)", initial)

    conn.commit()
    conn.close()
    print(f"Database ready — {len(initial)} initial documents loaded into input_data.")


if __name__ == "__main__":
    setup_database()
