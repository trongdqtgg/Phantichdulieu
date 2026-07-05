PHÂN TÍCH WOMEN'S E-COMMERCE CLOTHING REVIEWS - THEO MẪU OUTPUT

Nguồn dữ liệu: Kaggle Women's E-Commerce Clothing Reviews.
Bài báo tham khảo: Agarap, A. F. (2018), Statistical Analysis on E-Commerce Reviews, with Sentiment Classification using Bidirectional RNN, arXiv:1805.03687.

1. CẤU TRÚC THƯ MỤC
- charts/before_preprocessing: biểu đồ trước tiền xử lý
- charts/after_preprocessing: biểu đồ sau tiền xử lý
- reports: bảng thống kê CSV
- womens_clothing_reviews_cleaned.csv: dữ liệu đã tiền xử lý
- phan_tich_womens_ecommerce_reviews.py: code Python tái chạy toàn bộ phân tích

2. CÁC LOẠI BIỂU ĐỒ ĐÃ XUẤT
- Histogram: Age, Rating, Recommended IND, Positive Feedback Count, Review Text Length, Word Count.
- Boxplot: Age, Rating, Positive Feedback Count, Review/Text Length.
- Scatter Plot: Age vs Rating, Review Text Length vs Rating, Rating vs Positive Feedback Count.
- Bar Chart: Division Name, Department Name, Class Name, Recommended Label, Sentiment Label, Age Group.
- Line Chart: Rating trung bình theo tuổi, tỷ lệ recommended theo rating, thống kê theo Age Group.
- Heatmap: tương quan các biến định lượng.

3. TIỀN XỬ LÝ ĐÃ THỰC HIỆN
- Xóa cột Unnamed: 0 vì chỉ là chỉ mục dư thừa.
- Điền Title thiếu bằng 'No Title'.
- Điền Review Text thiếu bằng 'No Review'.
- Điền Division/Department/Class thiếu bằng 'Unknown'.
- Tạo Review Text Length, Title Length, Word Count.
- Tạo Sentiment Label: Rating >= 4 là Positive, Rating = 3 là Neutral, Rating <= 2 là Negative.
- Tạo Recommended Label từ Recommended IND.
- Tạo Age Group.
- Tạo bản capping 1%-99% cho một số biến lệch phải để biểu đồ dễ đọc hơn.

4. HƯỚNG DẪN EXCEL STEP BY STEP
Bước 1: Data -> Get Data -> From Text/CSV -> chọn Womens Clothing E-Commerce Reviews.csv.
Bước 2: Kiểm tra missing bằng COUNTBLANK, ví dụ =COUNTBLANK(D:D) cho Title.
Bước 3: Tạo cột độ dài Review Text: =LEN(E2). Tạo Word Count: =LEN(TRIM(E2))-LEN(SUBSTITUTE(TRIM(E2)," ",""))+1.
Bước 4: Tạo Sentiment Label: =IF(F2>=4,"Positive",IF(F2=3,"Neutral","Negative")).
Bước 5: Tạo Recommended Label: =IF(G2=1,"Recommended","Not Recommended").
Bước 6: Insert -> PivotTable để đếm Rating, Department Name, Class Name, Sentiment Label.
Bước 7: Insert -> Histogram cho Age, Rating, Positive Feedback Count, Review Text Length.
Bước 8: Insert -> Box & Whisker cho Age, Positive Feedback Count.
Bước 9: Insert -> Scatter cho Review Text Length và Rating.
Bước 10: Insert -> Line chart cho Rating trung bình theo Age hoặc Age Group.
Bước 11: Tạo ma trận CORREL giữa các biến số rồi dùng Conditional Formatting -> Color Scales để tạo heatmap.

5. ƯU ĐIỂM / KHUYẾT ĐIỂM
Python:
Ưu điểm: tự động hóa cao, xử lý lặp lại nhanh, xuất nhiều biểu đồ, dễ tái lập kết quả, phù hợp NLP.
Khuyết điểm: cần biết code, lỗi môi trường/thư viện có thể gây khó cho người mới.
Excel:
Ưu điểm: dễ thao tác, PivotTable trực quan, phù hợp báo cáo nhanh và trình bày với giáo viên.
Khuyết điểm: khó tự động hóa nhiều biểu đồ, xử lý văn bản/NLP hạn chế, dễ sai khi thao tác thủ công.

6. NHẬN XÉT TỔNG HỢP
Dữ liệu phù hợp cho phân tích hành vi đánh giá khách hàng và bài toán phân loại cảm xúc/khuyến nghị. Rating và Recommended IND là hai biến quan trọng nhất. Review Text và Title là nguồn dữ liệu văn bản quan trọng cho mô hình NLP như BiLSTM trong bài Agarap 2018. Sau tiền xử lý, dữ liệu đầy đủ hơn, có thêm biến đặc trưng độ dài văn bản, nhóm tuổi và nhãn cảm xúc, giúp trực quan hóa và mô hình hóa tốt hơn.