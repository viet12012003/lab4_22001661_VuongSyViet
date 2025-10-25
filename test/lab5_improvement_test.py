import os
import sys
from typing import List
import re
import json
import csv

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models.text_classifier import TextClassifier

# =====================
# Constants
# =====================
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "sentiments.csv")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
RANDOM_STATE = 42
TEST_SIZE = 0.3

# TF-IDF configurations
TFIDF_BASE_KW = {}
TFIDF_IMPROVED_KW = {
    "ngram_range": (1, 2),
    "stop_words": "english",
    "max_df": 0.85,
}


def evaluate(y_true, y_pred):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def clean_text(s: str) -> str:
    s = s.lower()
    s = re.sub(r"https?://\S+|www\.\S+", " ", s)
    s = re.sub(r"<[^>]+>", " ", s)
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def main():
    texts: List[str] = []
    labels: List[int] = []
    if not os.path.isfile(DATA_PATH):
        raise FileNotFoundError(f"Data file not found: {DATA_PATH}")
    with open(DATA_PATH, "r", encoding="utf-8", errors="ignore") as f:
        reader = csv.reader(f)
        first_row = next(reader, None)
        if first_row is None:
            raise ValueError("CSV is empty")
        row_is_header = False
        if len(first_row) >= 2:
            h0 = first_row[0].strip().lower()
            h1 = first_row[1].strip().lower()
            if ("text" in h0) and ("sentiment" in h1 or "label" in h1):
                row_is_header = True
        if not row_is_header:
            try:
                txt, lab = first_row[0], first_row[1]
                texts.append(txt)
                labels.append(1 if str(lab).strip() == "1" else 0)
            except Exception:
                pass
        for row in reader:
            if not row or len(row) < 2:
                continue
            txt, lab = row[0], row[1]
            y = 1 if str(lab).strip() == "1" else 0
            texts.append(txt)
            labels.append(y)

    texts_clean = [clean_text(t) for t in texts]

    X_train, X_test, y_train, y_test = train_test_split(
        texts_clean, labels, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=labels
    )

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    results = {}

    # Baseline: TF-IDF unigram + Logistic Regression
    tfidf_base = TfidfVectorizer(**TFIDF_BASE_KW)
    baseline = TextClassifier(tfidf_base)
    baseline.fit(X_train, y_train)
    y_pred_lr_base = baseline.predict(X_test)
    results["tfidf_lr_base"] = baseline.evaluate(y_test, y_pred_lr_base)

    # Cải tiến 1: TF-IDF bigram + stopwords + giảm vocab (max_df)
    tfidf_improved = TfidfVectorizer(**TFIDF_IMPROVED_KW)
    improved = TextClassifier(tfidf_improved)
    improved.fit(X_train, y_train)
    y_pred_lr_imp = improved.predict(X_test)
    results["tfidf_lr_improved"] = improved.evaluate(y_test, y_pred_lr_imp)

    # Cải tiến 2: Thay mô hình sang Naive Bayes
    X_train_vec = tfidf_improved.fit_transform(X_train)
    X_test_vec = tfidf_improved.transform(X_test)
    nb = MultinomialNB()
    nb.fit(X_train_vec, y_train)
    y_pred_nb = nb.predict(X_test_vec)
    results["tfidf_nb"] = evaluate(y_test, y_pred_nb)

    print("=== Lab4 - Improvement Experiments ===")
    for k, v in results.items():
        print(k, v)

    with open(os.path.join(OUTPUT_DIR, "lab5_improvement_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
