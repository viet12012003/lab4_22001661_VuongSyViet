import os
import sys
from typing import List
from sklearn.model_selection import train_test_split
import json

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models.text_classifier import TextClassifier

LAB1_ROOT = os.path.join(os.path.dirname(PROJECT_ROOT), "lab1_22001661_VuongSyViet")
_orig_sys_path = list(sys.path)
try:
    if LAB1_ROOT not in sys.path:
        sys.path.insert(0, LAB1_ROOT)
    from src.representations.count_vectorizer import CountVectorizer
    from src.preprocessing.regex_tokenizer import RegexTokenizer
finally:
    sys.path = _orig_sys_path

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def main():
    texts: List[str] = [
        "This movie is fantastic and I love it!",
        "I hate this film, it's terrible.",
        "The acting was superb, a truly great experience.",
        "What a waste of time, absolutely boring.",
        "Highly recommend this, a masterpiece.",
        "Could not finish watching, so bad.",
    ]
    labels: List[int] = [1, 0, 1, 0, 1, 0]  # 1: positive, 0: negative

    # Tách tập train/test
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )

    vectorizer = CountVectorizer(RegexTokenizer())

    # Khởi tạo classifier và huấn luyện
    clf = TextClassifier(vectorizer)
    clf.fit(X_train, y_train)

    # Dự đoán & đánh giá
    y_pred = clf.predict(X_test)
    metrics = clf.evaluate(y_test, y_pred)

    print("=== Lab4 - Basic Test Case ===")
    print("Texts (test):", X_test)
    print("True labels:", y_test)
    print("Pred labels:", y_pred)
    print("Metrics:", metrics)

    # Lưu kết quả
    with open(os.path.join(OUTPUT_DIR, "lab5_test_predictions.txt"), "w", encoding="utf-8") as f:
        for t, yt, yp in zip(X_test, y_test, y_pred):
            f.write(f"{yt}\t{yp}\t{t}\n")
    with open(os.path.join(OUTPUT_DIR, "lab5_test_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
