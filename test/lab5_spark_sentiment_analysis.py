import os
import sys

# =====================
# Constants
# =====================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "sentiments.csv")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
SPARK_APP_NAME = "Lab4-Spark-Sentiment"
HASHING_NUM_FEATURES = 1 << 18
RANDOM_SEED = 42

try:
    from pyspark.sql import SparkSession
    from pyspark.ml import Pipeline
    from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF
    from pyspark.ml.classification import LogisticRegression
    from pyspark.ml.evaluation import MulticlassClassificationEvaluator
    from pyspark.sql.functions import col, when
    _spark_available = True
except Exception as e:
    print("[Error loading PySpark]:", e)
    _spark_available = False


def main():
    if not _spark_available:
        print("[Info] PySpark is not available. Skipping Spark example.")
        return

    spark = SparkSession.builder.appName(SPARK_APP_NAME).getOrCreate()

    try:
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        if os.path.isfile(DATA_PATH):
            try:
                df = spark.read.csv(DATA_PATH, header=True, inferSchema=True)
                cols = [c.lower() for c in df.columns]
                if not ("text" in cols and ("sentiment" in cols or "label" in cols)):
                    df = spark.read.csv(DATA_PATH, header=False, inferSchema=True)
                    if len(df.columns) >= 2:
                        df = df.select(df.columns[0].alias("text"), df.columns[1].alias("sentiment"))
                    else:
                        raise ValueError("CSV must have at least 2 columns: text, sentiment")
                else:
                    rename = {}
                    for c in df.columns:
                        lc = c.lower()
                        if lc == "text":
                            rename[c] = "text"
                        elif lc in ("sentiment", "label"):
                            rename[c] = "sentiment"
                    for src, dst in rename.items():
                        if src != dst:
                            df = df.withColumnRenamed(src, dst)
            except Exception as e:
                raise e
        else:
            raise FileNotFoundError(f"Data file not found: {DATA_PATH}")

        # Loại bỏ dòng rỗng và chuẩn hóa nhãn -1/1 -> 0/1
        df = df.select("text", "sentiment").na.drop(subset=["text", "sentiment"]).withColumn(
            "label_raw", col("sentiment").cast("double")
        )
        df = df.withColumn(
            "label",
            when(col("label_raw") == -1, 0.0)
            .when(col("label_raw") == 1, 1.0)
            .otherwise(col("label_raw"))
            .cast("double"),
        ).drop("label_raw")

        # Tiền xử lý và đặc trưng
        tokenizer = Tokenizer(inputCol="text", outputCol="words")
        remover = StopWordsRemover(inputCol="words", outputCol="filtered_words")
        hashingTF = HashingTF(inputCol="filtered_words", outputCol="raw_features", numFeatures=HASHING_NUM_FEATURES)
        idf = IDF(inputCol="raw_features", outputCol="features")

        # Mô hình
        lr = LogisticRegression(maxIter=50, regParam=0.0, featuresCol="features", labelCol="label")

        pipeline = Pipeline(stages=[tokenizer, remover, hashingTF, idf, lr])
        train_df, test_df = df.randomSplit([0.8, 0.2], seed=RANDOM_SEED)

        model = pipeline.fit(train_df)
        preds = model.transform(test_df)

        evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction")
        acc = float(evaluator.setMetricName("accuracy").evaluate(preds))
        f1 = float(evaluator.setMetricName("f1").evaluate(preds))
        wprec = float(evaluator.setMetricName("weightedPrecision").evaluate(preds))
        wrec = float(evaluator.setMetricName("weightedRecall").evaluate(preds))

        print("=== Lab4 - Spark Sentiment Pipeline ===")
        preds.select("text", "label", "prediction", "probability").show(truncate=False)
        result = {
            "accuracy": acc,
            "f1": f1,
            "weighted_precision": wprec,
            "weighted_recall": wrec,
        }
        print(result)

        preds.select("text", "label", "prediction").coalesce(1).write.mode("overwrite").option("header", True).csv(
            os.path.join(OUTPUT_DIR, "spark_predictions")
        )
        with open(os.path.join(OUTPUT_DIR, "spark_metrics.txt"), "w", encoding="utf-8") as f:
            f.write(str(result))

    finally:
        print("Stopping Spark session...")
        spark.stop()
        print("Spark session stopped successfully.")

if __name__ == "__main__":
    main()
