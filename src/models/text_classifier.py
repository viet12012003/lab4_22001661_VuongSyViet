from typing import List, Dict, Any

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class TextClassifier:
    def __init__(self, vectorizer: Any):
        self._vectorizer = vectorizer
        self._model = LogisticRegression(solver="liblinear", random_state=42)
        self._is_fitted = False

    def fit(self, texts: List[str], labels: List[int]):
        """
        Huấn luyện bộ phân loại: vector hóa văn bản và train LogisticRegression.
        """
        X = self._vectorizer.fit_transform(texts)
        self._model.fit(X, labels)
        self._is_fitted = True

    def predict(self, texts: List[str]) -> List[int]:
        """
        Dự đoán nhãn cho danh sách văn bản đầu vào.
        """
        if not self._is_fitted:
            raise RuntimeError("Model is not fitted. Call fit() before predict().")
        X = self._vectorizer.transform(texts)
        y_pred = self._model.predict(X)
        return y_pred.tolist() if hasattr(y_pred, "tolist") else list(y_pred)

    def evaluate(self, y_true: List[int], y_pred: List[int]) -> Dict[str, float]:
        """
        Tính các chỉ số đánh giá: accuracy, precision, recall, f1 (binary/macro-safe).
        """
        metrics = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        }
        return metrics
