# Hướng dẫn Giải bài tập Dự báo Chứng khoán (Stock Return Forecasting)

Dưới đây là thông tin về một bài tập dự báo giá chứng khoán. Đóng vai trò là một chuyên gia khoa học dữ liệu (Data Scientist) và kỹ sư học máy (Machine Learning Engineer), hãy đọc kỹ yêu cầu và giúp tôi giải quyết bài toán này.

## 1. Đề bài gốc (Original Assignment)

**Assignment:**
Problem. Given data of daily close stocks' price and volume in some hidden interval of time. Use all knowledge / idea that you could find to forecast the following:

    P, V = price, volume
    f(P[: -(n - 1)], V[: -(n - 1)]) = Return[n]

**Evaluation:**
Performance will be tested on a different time interval on the same stocks.

**Materials:**
Research data and a coding template will be shared via LMS system. Please do your work by customizing the template, fill in the names of members of your group and submit them back.

---

## 2. Phân tích yêu cầu bài toán
- **Dữ liệu đầu vào (Input):** Chuỗi thời gian về giá đóng cửa (`P` - price) và khối lượng giao dịch (`V` - volume) hàng ngày của các cổ phiếu trong một khoảng thời gian.
- **Mục tiêu (Objective):** Xây dựng một hàm/mô hình dự báo `f`. Hàm này sử dụng dữ liệu lịch sử của giá và khối lượng (từ quá khứ cho đến thời điểm trước `n-1`) để dự báo tỷ suất sinh lời (Return) tại thời điểm `n` (`Return[n]`).
- **Đánh giá (Evaluation):** Mô hình sẽ được kiểm thử hiệu suất (backtest) trên một khoảng thời gian khác nhưng trên cùng danh sách các cổ phiếu đó (Out-of-sample testing).

---

## 3. Yêu cầu dành cho AI
Dựa trên đề bài và phân tích trên, hãy thực hiện các bước sau một cách chi tiết để tạo thành một lời giải hoàn chỉnh:

1. **Đề xuất phương pháp (Methodology):** Gợi ý các hướng tiếp cận học máy (Machine Learning) hoặc học sâu (Deep Learning) tốt nhất cho bài toán chuỗi thời gian tài chính này (ví dụ: Gradient Boosting Trees, LSTM, Transformer, v.v.). Giải thích lý do lựa chọn.
2. **Kỹ thuật trích xuất đặc trưng (Feature Engineering):** Từ 2 biến gốc là `Price` và `Volume`, hãy đề xuất các đặc trưng (features) và chỉ báo kỹ thuật (Technical Indicators) quan trọng cần tạo ra (VD: Moving Averages, RSI, MACD, Volume Profile...). Viết công thức hoặc cách tính toán cho chúng.
3. **Chiến lược xác thực chéo (Cross-Validation Strategy):** Đề xuất cách chia tập dữ liệu huấn luyện và kiểm định (Train/Validation split) phù hợp cho dữ liệu chuỗi thời gian để tránh hiện tượng rò rỉ dữ liệu (Data Leakage) theo tương lai.
4. **Code Python mẫu (Baseline Model):** Viết một đoạn code Python đầy đủ sử dụng các thư viện phổ biến (như `pandas`, `scikit-learn` hoặc `lightgbm`) để:
   - Xây dựng hàm tạo features.
   - Xử lý mảng `P` và `V` theo đúng công thức `f(P[: -(n - 1)], V[: -(n - 1)]) = Return[n]`.
   - Huấn luyện một mô hình Baseline và dự báo.
