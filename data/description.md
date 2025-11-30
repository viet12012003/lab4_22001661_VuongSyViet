# Mô tả bộ dữ liệu Sentiment

Bộ dữ liệu `sentiments.csv` chứa các đoạn văn bản ngắn bằng tiếng Anh, liên quan đến thảo luận về chứng khoán/tài chính, kèm theo nhãn cảm xúc.

## Thông tin file
- **Định dạng**: CSV, các trường được phân tách bằng dấu phẩy, có dòng tiêu đề (header) ở dòng đầu tiên.

## Các cột dữ liệu
- **`text`** (`string`)
  - Nội dung câu/bình luận ngắn (ví dụ: tweet, comment của người dùng) về cổ phiếu hoặc thị trường.
  - Có thể chứa tên người dùng, mã cổ phiếu (ticker), dấu câu và nhiều nhiễu (noise) trong văn bản.

- **`sentiment`** (`integer`)
  - Nhãn cảm xúc của câu/bình luận.
  - Các giá trị xuất hiện trong file:
    - `1`: Cảm xúc tích cực (lạc quan, đánh giá tốt, bullish).
    - `-1`: Cảm xúc tiêu cực (bi quan, đánh giá xấu, bearish).

## Cách sử dụng trong Lab4
- Được dùng làm dữ liệu đầu vào cho:
  - Các thí nghiệm mô hình học máy truyền thống (TF-IDF + Logistic Regression / Naive Bayes).
  - Pipeline phân tích cảm xúc sử dụng Spark (HashingTF + IDF + Logistic Regression).
- Các script xử lý dữ liệu này có thể:
  - Tự động nhận diện dòng header.
  - Chuẩn hóa lại nhãn (ví dụ: ánh xạ `-1` thành `0` trong một số thí nghiệm phân loại nhị phân).
