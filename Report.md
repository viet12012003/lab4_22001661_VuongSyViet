# Báo cáo Lab 4 – Text Classification

## 1) Các bước hiện thực
- **Tổ chức mã nguồn**
  - `src/models/text_classifier.py`: Lớp `TextClassifier` gói quy trình train/predict/evaluate dựa trên một vectorizer (TF‑IDF/Count) và một classifier (LogReg,...).
  - `test/lab5_test.py`: Bài kiểm thử cơ bản dùng `CountVectorizer` tự cài (Lab1) + `RegexTokenizer`, lưu kết quả ra thư mục `output/`.
  - `test/lab5_improvement_test.py`: Thực nghiệm cải tiến (tiền xử lý + TF‑IDF cấu hình khác + so sánh Naive Bayes).
  - `test/lab5_spark_sentiment_analysis.py`: Pipeline Spark (Tokenizer → StopWords → HashingTF → IDF → LogisticRegression), đọc `data/sentiments.csv`, chuẩn hóa nhãn và lưu output.
  - `data/sentiments.csv`: Bộ dữ liệu sentiment (text, label −1/1).
  - `output/`: Lưu các file kết quả, gồm metrics và dự đoán.

- **Tiền xử lý (Improved preprocessing)**
  - Hàm `clean_text()` trong `lab5_improvement_test.py` loại URL, HTML tags, ký tự đặc biệt; lowercasing và chuẩn hóa khoảng trắng.
  - Ở Spark script: chuẩn hóa cột, loại null, chuyển nhãn −1/1 → 0/1.

- **Đặc trưng và mô hình**
  - Baseline sklearn: `TfidfVectorizer()` (unigram) + `LogisticRegression` qua `TextClassifier`.
  - Cải tiến sklearn: `TfidfVectorizer(ngram_range=(1,2), stop_words="english", max_df=0.85)` + `LogisticRegression` và so sánh thêm `MultinomialNB`.
  - Spark: `Tokenizer` → `StopWordsRemover` → `HashingTF(numFeatures=2^18)` → `IDF` → `LogisticRegression`.

- **Lưu kết quả**
  - Sklearn: `output/lab5_improvement_metrics.json` chứa các metrics từng cấu hình.
  - Spark: `output/spark_predictions/` (CSV) và `output/spark_metrics.txt` (4 chỉ số: accuracy, f1, weighted_precision, weighted_recall).

## 2) Hướng dẫn chạy mã
- Chuẩn bị môi trường (Windows, Python 3.11):
  - Tạo venv và cài gói tối thiểu: `pip install scikit-learn numpy`.
  - Với Spark:
    - `pip install pyspark==3.5.1` (tương thích Python 3.11). Nếu lỗi `distutils`, cài thêm: `pip install setuptools`.
    - Cấu hình Hadoop/winutils khi cần: đặt `winutils.exe` tại `F:\Hadoop\hadoop-3.3.0\bin\winutils.exe`, đặt `HADOOP_HOME=F:\Hadoop\hadoop-3.3.0` và thêm `%HADOOP_HOME%\bin` vào PATH.

- Chạy các script:
  - Baseline CountVectorizer nhỏ:
    - `python F:\NLP\lab4_22001661_VuongSyViet\test\lab5_test.py`
    - Output: `output/lab5_test_predictions.txt`, `output/lab5_test_metrics.json`.
  - Thí nghiệm cải tiến sklearn:
    - `python F:\NLP\lab4_22001661_VuongSyViet\test\lab5_improvement_test.py`
    - Output: `output/lab5_improvement_metrics.json`.
  - Spark pipeline:
    - `python F:\NLP\lab4_22001661_VuongSyViet\test\lab5_spark_sentiment_analysis.py`
    - Output: `output/spark_predictions/` (CSV), `output/spark_metrics.txt`.

## 3) Phân tích kết quả
- Kết quả `lab5_test.py` (CountVectorizer + RegexTokenizer, từ `output/lab5_test_metrics.json`):
  - accuracy: 0.0000
  - precision: 0.0000
  - recall: 0.0000
  - f1: 0.0000
  - Ghi chú: đây là bài test demo rất nhỏ (6 câu) nên số liệu có thể không phản ánh hiệu năng thực tế; phần thực nghiệm chính được đánh giá ở mục sklearn và Spark ở trên với dữ liệu `sentiments.csv`.

- Kết quả Spark (từ `output/spark_metrics.txt`):
  - accuracy: 0.7529
  - f1: 0.7516
  - weighted_precision: 0.7509
  - weighted_recall: 0.7529

- Kết quả sklearn (từ `output/lab5_improvement_metrics.json`):
  - `tfidf_lr_base` (TF‑IDF unigram + LogisticRegression):
    - accuracy: 0.7756
    - precision: 0.7668
    - recall: 0.9304
    - f1: 0.8407
  - `tfidf_lr_improved` (TF‑IDF (1,2) + stopwords + max_df=0.85 + LogisticRegression):
    - accuracy: 0.7537
    - precision: 0.7394
    - recall: 0.9467
    - f1: 0.8303
  - `tfidf_nb` (TF‑IDF (1,2) + MultinomialNB):
    - accuracy: 0.7106
    - precision: 0.6914
    - recall: 0.9846
    - f1: 0.8124

- So sánh và giải thích hiệu quả cải tiến
  - So với Spark, cả ba cấu hình sklearn đều cho điểm F1 cao hơn đáng kể, nổi bật là baseline TF‑IDF unigram + LR (f1 ≈ 0.841).
  - Cấu hình `tfidf_lr_improved` có recall rất cao (≈ 0.947) nhưng precision thấp hơn, nên F1 kém nhẹ baseline. Bigram và loại stopwords giúp bắt nhiều mẫu dương (recall tăng), nhưng có thể kéo theo nhiễu (precision giảm) trên tập dữ liệu này.
  - Naive Bayes đạt recall cực cao (≈ 0.985) nhưng accuracy/precision thấp hơn, phù hợp khi ưu tiên không bỏ sót mẫu dương.
  - Spark dùng HashingTF (không có từ điển rõ ràng, có va chạm băm) nên chất lượng đặc trưng có thể thấp hơn TF‑IDF có từ vựng; do đó F1 thấp hơn các cấu hình sklearn dựa trên TF‑IDF.

- Kết luận
  - Cấu hình tốt nhất tổng thể ở đây là `tfidf_lr_base` (unigram + LR) với F1 ≈ 0.841.
  - Nếu mục tiêu tối đa hóa recall, `tfidf_nb` là đáng cân nhắc; nếu cân bằng tốt precision/recall, `tfidf_lr_improved` là lựa chọn trung gian.

## 4) Khó khăn và cách khắc phục
- Lỗi import PySpark trên Python 3.11 (ModuleNotFoundError: `typing.io`/`typing is not a package`):
  - Giải pháp: nâng PySpark lên `3.5.1` (tương thích 3.11).
- Lỗi `No module named distutils` khi import PySpark:
  - Giải pháp: `pip install setuptools`.
- Lỗi Hadoop Home trên Windows: `java.io.FileNotFoundException: Hadoop home directory F:\Hadoop\hadoop-3.3.0 does not exist`:
  - Giải pháp: cài `winutils.exe` vào `F:\Hadoop\hadoop-3.3.0\bin\winutils.exe`, đặt `HADOOP_HOME=F:\Hadoop\hadoop-3.3.0`, thêm `%HADOOP_HOME%\bin` vào PATH. Sau đó chạy Spark thành công.
- Log nhiều và dọn dẹp tiến trình trên Windows:
  - Các dòng `SUCCESS: The process with PID ... has been terminated.` là thông báo dọn dẹp tiến trình khi `spark.stop()`, không phải lỗi.

## 5) Tài liệu tham khảo
- Scikit-learn Documentation – Feature Extraction & Models:
  - https://scikit-learn.org/stable/modules/feature_extraction.html
  - https://scikit-learn.org/stable/modules/naive_bayes.html
  - https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
- Apache Spark ML Guide:
  - https://spark.apache.org/docs/latest/ml-guide.html
  - https://spark.apache.org/docs/latest/ml-features
- Về winutils trên Windows:
  - https://github.com/steveloughran/winutils

---

- Thư mục output minh họa:
  - `output/lab5_improvement_metrics.json`: tổng hợp metrics cho từng cấu hình sklearn.
  - `output/spark_metrics.txt`: 4 chỉ số của Spark (accuracy, f1, weighted_precision, weighted_recall).
  - `output/spark_predictions/`: CSV dự đoán từ Spark.
  - `output/lab5_test_predictions.txt`, `output/lab5_test_metrics.json`: bài test nhỏ với CountVectorizer (giá trị có thể dao động tùy dữ liệu rất nhỏ).