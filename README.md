# home-credit-behavioral-scorecard

Mã nguồn bốn phần và demo chấm điểm tín dụng hành vi Home Credit.
Repo có sẵn dự đoán OOF và các bảng kết quả để chạy demo; dữ liệu thô,
hai bảng đặc trưng lớn và mô hình đã huấn luyện không được đưa lên Git.
Nguồn dữ liệu được ghi tại `data_set/link.txt`.

Để chạy lại cả bốn phần, cài `python -m pip install -r requirements.txt`,
chuẩn bị các bảng dữ liệu gốc và file cohort rồi chạy theo thứ tự bên dưới.

## Demo Streamlit

Chạy trong PowerShell tại thư mục dự án:

```powershell
python -m pip install -r requirements-demo.txt
python -m streamlit run streamlit_app.py
```

Mở địa chỉ Streamlit hiển thị trong terminal (mặc định http://localhost:8501).
Demo đọc các kết quả lịch sử đã lưu, không huấn luyện lại hoặc chấm hồ sơ mới.

Ứng dụng có ba màn hình:

- **Tổng quan:** ROC-AUC, Average Precision, khoảng tin cậy chênh lệch A/B và calibration.
- **Hồ sơ khách hàng:** chọn khách có SHAP, so sánh xác suất/điểm/hạng và xem đóng góp nhóm.
- **Điểm và hạng:** bảng 10 hạng, tỷ lệ xấu, PSI, chuyển hạng với t=2/3/4.

Nếu thiếu dữ liệu, chạy các Part theo thứ tự:

```powershell
$env:HC_QUICK_TEST = "0"
python .\part1_hieu_fe_and_models.py
python .\part2_hien_scoring_and_calibration.py
python .\part3_duy_shap_and_waterfalls.py
python .\part4_dung_bootstrap_policy_and_checks.py
```

File `frozen_cohort_labels_N12.csv` cần được chuẩn bị trước; bốn Part không tạo file này.
Ứng dụng dùng điểm/nhãn từ CSV hiện tại thay vì sao chép điểm từ JSON minh họa cũ.
Nếu SHAP không khớp điểm hiện tại, ứng dụng thông báo cần chạy lại Part 3.
Ứng dụng đọc lại file đã cập nhật khi tải lại trang dựa trên thời gian sửa file.
Ứng dụng không sửa hoặc ghi đè file đầu ra của bốn Part.

Gợi ý trình bày: tổng quan → khách phục hồi tăng điểm → khách phục hồi xuống hạng
→ ma trận chuyển hạng. CI chứa 0 không chứng minh hai phương án tương đương.

Kiểm tra chức năng các màn hình bằng [AppTest của Streamlit](https://docs.streamlit.io/develop/api-reference/app-testing/st.testing.v1.apptest):

```powershell
python -m unittest discover -s tests -p "test_streamlit_demo.py"
```
