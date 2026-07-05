PHÂN TÍCH STORE SALES TIME SERIES FORECASTING - ĐÃ BỔ SUNG TRAIN.CSV

File này đã phân tích lại theo cấu trúc giống mau_outputs.zip.

Cấu trúc:
- charts/before_preprocessing/: Biểu đồ trước tiền xử lý
- charts/after_preprocessing/: Biểu đồ sau tiền xử lý / clip outlier để dễ quan sát
- reports/: Các bảng CSV thống kê và so sánh
- store_sales_train_merged_cleaned_sample_100k.csv: Mẫu 100.000 dòng đã merge + tiền xử lý để mở Excel nhẹ
- store_sales_daily_aggregated_cleaned.csv: Tổng hợp ngày từ toàn bộ train
- phan_tich_store_sales_with_train.py: Code Python tạo output

Đã bổ sung phân tích SALES thật từ train.csv:
- Histogram sales
- Boxplot sales
- Scatter sales với onpromotion / transactions / oil
- Line chart tổng sales theo ngày và tháng
- Bar chart doanh thu theo family và store
- Heatmap tương quan

Thông tin chính:
- train.csv: 3,000,888 dòng, 6 cột
- Giai đoạn: 2013-01-01 đến 2017-08-15
- Tổng sales: 1,073,644,952.20
- Sales trung bình: 357.7757

Ghi chú:
Do train.csv lớn, file cleaned chi tiết chỉ lưu mẫu 100.000 dòng để ZIP nhẹ và dễ mở bằng Excel. Các bảng tổng hợp theo ngày/tháng/family/store được tính từ toàn bộ train.
