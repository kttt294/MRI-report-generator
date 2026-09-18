# MRI report generation — V1 và V2

Dự án nghiên cứu sinh báo cáo MRI cột sống thắt lưng tiếng Việt theo hai hướng:

- **V1:** ảnh NIfTI → các lát sagittal chọn bằng quy tắc → Qwen2.5-VL + QLoRA → findings và impression. Mặc định một lát giữa; có thể cấu hình nhiều lát như một ablation riêng.
- **V2, hướng A:** JSON gồm tám nhãn grading của năm tầng → report engine. Hiện dùng annotation thay đầu ra vision; vision engine và calibration chưa được xây dựng.

Bắt đầu với [hướng dẫn Kaggle cho người mới](docs/README_KAGGLE.md), [trạng thái và lệnh chạy](docs/implementation_status.md), [implementation plan](docs/implementation_plan_v1_v2_report.md).

## Notebook Kaggle

- [V1 training / resume / inference](notebooks/Kaggle_V1_Training.ipynb)
- [V2 report: template hoặc LLM](notebooks/Kaggle_V2_Report.ipynb)
- [V2 LoRA: chỉ target đã duyệt](notebooks/Kaggle_V2_Training.ipynb)

Notebook tải code từ repo public và gọi `.py`. Dữ liệu ảnh/annotations ở hai Kaggle Dataset private; outputs và notebook chạy dữ liệu thật giữ private. Không đưa dữ liệu bệnh nhân hoặc model weights lên GitHub.

## V2 contract

`src/contracts/report_input.py` là contract suy luận có kiểm tra kiểu/domain/status/tầng. JSON schemas ở `schemas/`; kiểm tra logic liên trường vẫn phải dùng Pydantic. `reports`, demographic, tọa độ và folds không thuộc input của report engine. Giá trị thiếu là `null`, không phải `0`; không sinh confidence giả.

V2 có template baseline, backend Hugging Face, kiểm tra catalog, tối đa một lần repair và fallback được ghi rõ. Validator hiện dùng câu có kiểm soát, không phải bộ đánh giá ngôn ngữ y khoa tự do. `needs_review` phản ánh QC input chưa được xác nhận. Không dùng độ khớp grading để khẳng định chẩn đoán đúng trên MRI.

## Kiểm thử

```bash
python -m pytest -q
python scripts/check_environment.py
```

Giữ cặp torch/torchvision CUDA do Kaggle cung cấp; cài `requirements-kaggle.txt`. **33 test** là test hành vi của code, dùng fixture giả lập và mô hình rất nhỏ trên CPU; không phải 33 ca MRI. **247 ca V2** là 247 bản ghi grading local được chạy qua template deterministic trên CPU; không đọc NIfTI, không gọi VLM/LLM và không phải kết quả chẩn đoán đã được bác sĩ đánh giá. Kiểm thử CPU và template đã thực hiện; GPU Kaggle, ảnh thật, train V1, chất lượng LLM và đánh giá bác sĩ vẫn cần nghiệm thu. Xem bằng chứng cụ thể trong tài liệu trạng thái.

## Dữ liệu và nghiên cứu

SFT legacy được giữ để truy vết/thí nghiệm, không được coi là target đầy đủ phù hợp cho V2 chỉ có tám nhãn. Target V2 phải có review và liên kết đúng hash input. Các báo cáo đầy đủ của bác sĩ có thể chứa thông tin ngoài grading; đó không tự động là lỗi nhãn.

Dữ liệu PSPINES không phân phối công khai. Tuân thủ quyền sử dụng của nguồn dữ liệu; `.gitignore` loại ảnh, dữ liệu gốc, derived outputs và checkpoint khỏi repo. Không có kết quả calibration, tỷ lệ lỗi LLM hoặc hiệu quả lâm sàng được khẳng định khi chưa đo.
