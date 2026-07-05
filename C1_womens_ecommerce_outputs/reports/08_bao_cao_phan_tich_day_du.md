# Báo cáo phân tích dữ liệu Women's E-Commerce Clothing Reviews

## Thông tin dữ liệu và bài báo tham chiếu
- Dataset: Women's E-Commerce Clothing Reviews trên Kaggle.
- Bài báo tham chiếu: Statistical Analysis on E-Commerce Reviews, with Sentiment Classification using Bidirectional RNN (Agarap, A. F., 2018).
- Link arXiv: https://arxiv.org/abs/1805.03687
- Số dòng ban đầu: 23,486
- Số cột ban đầu: 11

## 1. Phân loại bản chất của biến
Chi tiết nằm trong `reports/01_phan_loai_ban_chat_bien.csv`.

Tóm tắt:
- Biến định danh: `Unnamed: 0`, `Clothing ID`.
- Biến định lượng: `Age`, `Rating`, `Positive Feedback Count`, các biến tạo thêm như `Review_Text_Length`, `Review_Word_Count`, `Title_Length`.
- Biến định tính: `Title`, `Review Text`, `Division Name`, `Department Name`, `Class Name`, `Age_Group`, `Sentiment_Label`, `Recommended_Label`.
- Biến mục tiêu/nhãn quan trọng: `Rating`, `Recommended IND`, `Sentiment_Label`.

## 2. Phân tích các biến định lượng
Các biến định lượng được mô tả bằng count, mean, std, min, Q1, median, Q3, max trong:
- `reports/02_mo_ta_bien_dinh_luong_before.csv`
- `reports/05_mo_ta_bien_dinh_luong_after.csv`

Nhận xét chính:
- `Rating` có xu hướng lệch về mức cao, cho thấy phần lớn khách hàng đánh giá tích cực.
- `Positive Feedback Count` lệch phải mạnh, đa số review ít phản hồi tích cực nhưng có một số review nhận nhiều phản hồi.
- `Age` tập trung nhiều ở nhóm người trưởng thành, phù hợp với dữ liệu mua sắm quần áo nữ.
- Các biến độ dài văn bản giúp bổ sung góc nhìn NLP: review dài có thể phản ánh trải nghiệm chi tiết hơn.

## 3. Phân tích các biến định tính
Báo cáo tần suất nằm trong:
- `reports/03_tan_suat_bien_dinh_tinh_before.csv`
- `reports/06_tan_suat_bien_dinh_tinh_after.csv`

Nhận xét chính:
- `Class Name`, `Department Name`, `Division Name` cho biết nhóm sản phẩm nào có nhiều review.
- `Recommended IND` cho biết khách hàng có khuyến nghị sản phẩm hay không.
- `Sentiment_Label` được suy ra từ `Rating`: 1-2 tiêu cực, 3 trung lập, 4-5 tích cực.
- Các cột văn bản `Title` và `Review Text` có giá trị thiếu đáng kể nên cần xử lý trước khi phân tích văn bản.

## 4. Trực quan hóa dữ liệu
Đã xuất ảnh PNG theo 2 giai đoạn:
- Trước tiền xử lý: `charts/before_preprocessing/` (39 biểu đồ)
- Sau tiền xử lý: `charts/after_preprocessing/` (41 biểu đồ)

Đủ các loại biểu đồ yêu cầu:
- Histogram: phân phối Age, Rating, Positive Feedback Count, Review length, Word count, Title length.
- Boxplot: phát hiện ngoại lệ ở Age, Positive Feedback Count, độ dài review.
- Scatter Plot: quan hệ giữa Age, Rating, Positive Feedback Count, Review length, Word count.
- Bar Chart: tần suất Rating, Recommended, Division, Department, Class, Sentiment, Age Group.
- Line Chart: số review theo tuổi, rating trung bình theo tuổi, recommended rate theo rating/nhóm tuổi.
- Heatmap: tương quan giữa các biến định lượng.

## 5. Python và Excel
- Code Python hoàn chỉnh: `phan_tich_womens_ecommerce.py`.
- Hướng dẫn làm bằng Excel: `reports/09_huong_dan_excel_step_by_step.txt`.
- Ưu/khuyết điểm Python và Excel: `reports/10_uu_nhuoc_diem_python_excel.txt`.

## 6. So sánh trước và sau tiền xử lý
Chi tiết: `reports/07_so_sanh_truoc_sau_tien_xu_ly.csv`.

Giá trị thiếu ban đầu:
- Title: 3810 giá trị thiếu
- Review Text: 845 giá trị thiếu
- Department Name: 14 giá trị thiếu
- Class Name: 14 giá trị thiếu
- Division Name: 14 giá trị thiếu

Các bước tiền xử lý đã thực hiện:
1. Xóa cột chỉ số `Unnamed: 0` vì không mang ý nghĩa phân tích.
2. Xóa dòng trùng lặp nếu có.
3. Điền thiếu `Title` bằng `No title`, `Review Text` bằng `No review text`.
4. Điền thiếu nhóm sản phẩm bằng mode.
5. Tạo biến mới: `Review_Text_Length`, `Review_Word_Count`, `Title_Length`, `Has_Title`, `Sentiment_Label`, `Age_Group`, `Recommended_Label`.
6. Capping outlier bằng IQR cho Age, Positive Feedback Count và các biến độ dài văn bản.

## 7. Nhận xét tổng hợp
Dataset phù hợp để phân tích thống kê mô tả, trực quan hóa hành vi đánh giá sản phẩm và làm bài toán phân loại cảm xúc/khuyến nghị. Sau tiền xử lý, dữ liệu sạch hơn, giảm ảnh hưởng của missing value và outlier, đồng thời có thêm biến đặc trưng văn bản để phục vụ mô hình sentiment classification như định hướng của bài báo Agarap (2018).
