import joblib
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

CATEGORIES = ["rec.sport.baseball", "sci.med", "talk.politics.misc", "comp.graphics"]


def train_and_save(path="model.pkl"):
    data = fetch_20newsgroups(subset="all", categories=CATEGORIES, remove=("headers", "footers", "quotes"))
    X, y = data.data, data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=10000, ngram_range=(1, 2))),
        ("clf", LogisticRegression(max_iter=1000, random_state=42)),
    ])
    pipeline.fit(X_train, y_train)

    acc = accuracy_score(y_test, pipeline.predict(X_test))
    print(f"Test accuracy: {acc:.4f}")
    print(f"Classes: {list(data.target_names)}")

    joblib.dump({"pipeline": pipeline, "target_names": data.target_names}, path)
    print(f"Model saved to {path}")


if __name__ == "__main__":
    train_and_save()
