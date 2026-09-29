# 🛒 Instacart Big Data Dashboard (Streamlit)

Web App minh họa tiểu luận **“Phân tích hành vi mua sắm, khai phá giỏ hàng và dự đoán mua lại trên nền tảng TMĐT tạp hóa trực tuyến Instacart với Apache Spark”**.

- Học viên: **Đào Thị Hồng Vân** (C25611268) – GVHD: **TS. Nguyễn Thôn Dã**
- Môn học: Nghiên cứu Dữ liệu lớn trong Thương mại Điện tử

## Dữ liệu
- `data/orders_sample.parquet`, `data/order_products_sample.parquet`: **mẫu 10% khách hàng** (lấy mẫu theo khách hàng bằng xxhash64) trích từ pipeline PySpark.
- Các tệp `*_full.*`: luật kết hợp FP-Growth, hồ sơ phân khúc K-Means, chỉ số & tham số mô hình được tính trên **100% dữ liệu** (3,4 triệu đơn hàng, 33,8 triệu dòng sản phẩm).

## Chạy local
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Triển khai
share.streamlit.io → Create app → chọn repository này → Main file: `app.py` → Deploy.
