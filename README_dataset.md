# PSPINES — dữ liệu dẫn xuất (đã khử định danh)

MRI cột sống thắt lưng, 250 ca chụp, kèm nhãn chẩn đoán theo từng tầng đĩa đệm.
Đây là **dữ liệu bệnh nhân đã khử định danh**, không phải dữ liệu công khai.
Xem [Điều kiện sử dụng](#điều-kiện-sử-dụng) ở cuối.

## Có gì trong này

| Thư mục | Nội dung | Dung lượng |
|---|---|---|
| `dicom/` | DICOM gốc, thư mục đặt theo `patient_id`, **header đã rửa** | 33,8 GB |
| `nifti/` | Sagittal T1/T2 đã chuyển sang NIfTI + mask SPINEPS | 9,9 GB |
| `grading/` | Nhãn chẩn đoán, tách theo tháng | 3 file |
| `folds/` | Chia 5-fold: `fold1/…fold5/` × `{train,val,test}.csv` | 17 file |
| `localize/` | Toạ độ tâm 5 đĩa đệm mỗi ca | 1 CSV |
| `reports_json/` | Báo cáo chẩn đoán (`report/`) + nhân khẩu (`metadata/`) | 480 file |
| `reports_text*/` | Báo cáo dạng bảng, đã chia sẵn train/val/test | 9 file |

`patient_id` là khoá chính nối mọi thứ lại với nhau.

**Bắt đầu từ đâu**: `folds/` để lấy cách chia, `grading/` để lấy nhãn,
`localize/disc_localization.csv` để biết đĩa đệm nằm ở đâu. Ba thứ đó đủ chạy
baseline mà chưa cần đụng tới ảnh. Mỗi thư mục có `README.md` riêng.

## Cohort

- **250** ca có ảnh · **249** ca có nhãn chẩn đoán · **247** ca vào cohort phân tích
- 2 ca bị loại vì thiếu chuỗi sagittal T2-like (`configs/excluded_patients.csv`)
- **248** ca có định vị đĩa đệm; 1 trong số đó ngoài cohort → lọc `in_cohort == 1`
- 12 `patient_id` đã sửa lỗi nhập liệu (`configs/patient_id_corrections.csv`)

## Đã khử định danh những gì

Chạy `DiscAnchor/preprocessing/deid_dicom_headers.py` và `deid_derived_text.py`,
theo DICOM PS3.15 Basic Application Level Confidentiality Profile.

**Header DICOM** — 16.813 file:

- Họ tên, ngày sinh, địa chỉ, điện thoại → xoá; `PatientName`/`PatientID` = `patient_id`
- Cơ sở khám, bác sĩ gửi/thực hiện, kỹ thuật viên, tên máy, số máy → xoá
- Số hồ sơ, `StudyID` → để rỗng
- Ngày → chỉ giữ **năm** (`YYYY0101`); giờ → xoá
- Tuổi >89 → gộp thành `090Y`
- Toàn bộ tag **private** (nhóm lẻ) → xoá — nhà sản xuất hay nhét PHI vào đó
- Overlay (60xx) / Curve (50xx) → xoá
- UID được ánh xạ lại một-một bằng muối bí mật giữ tại máy nguồn
- Mỗi file mang `PatientIdentityRemoved = YES`

**Giữ lại có chủ ý**: `SeriesDescription`, `ProtocolName`, `PatientSex`,
`PatientAge` — cần cho nghiên cứu, đã soi và không chứa họ tên.

**Văn bản**:

- `nifti/subject_map.csv`: bỏ cột `study_folder` (chính là họ tên)
- `nifti/axial_slabs.csv`: `dicom_relpath` đổi tên thư mục thành `patient_id`
- `reports_json/report/*`: `source_file` thay bằng tên vô hại
- `reports_json/metadata/*`: `study_date` chỉ còn **năm**; tuổi >89 → 90
- `reports_text*/train.csv`: 1 chỗ lọt họ tên → `[TEN DA XOA]`

**Pixel không bị đụng tới.** Ảnh nguyên vẹn bit-for-bit.

## Còn lại rủi ro gì

- **Chữ cháy trên ảnh (burned-in annotation) chưa được kiểm bằng máy.** Bộ này
  không có tag `BurnedInAnnotation` để đọc. Nếu máy chụp nung tên lên góc ảnh
  thì nó vẫn còn. Cần soi mắt một mẫu ảnh trước khi phát tán rộng hơn.
- **Năm chụp + giới tính + tuổi** vẫn còn. Đủ dùng cho nghiên cứu, nhưng trên
  cohort 247 người thì đây là định danh gián tiếp — đừng ghép với nguồn ngoài.
- Đây là **khử định danh, không phải ẩn danh**. Bên giữ dữ liệu gốc vẫn nối
  ngược được.

Kiểm lại bất cứ lúc nào:

```bash
python -m DiscAnchor.preprocessing.deid_audit
```

## Điều kiện sử dụng

Dữ liệu bệnh nhân, chia sẻ cho **cộng tác nghiên cứu**. Không phải dữ liệu mở.

- Không phát tán lại, không đăng công khai, không nạp vào dịch vụ bên thứ ba
- Không tìm cách định danh ngược, không ghép với nguồn dữ liệu khác
- Dùng cho nghiên cứu — **không dùng cho chẩn đoán lâm sàng**
- Trích dẫn và ghi nhận đóng góp: hỏi bên giữ dữ liệu trước khi công bố

## Dựng lại

Toàn bộ thư mục này sinh ra từ `raw/` bằng `DiscAnchor/preprocessing/`.
Xem `DiscAnchor/docs/PIPELINE.md`.
