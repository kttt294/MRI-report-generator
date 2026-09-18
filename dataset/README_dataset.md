# PSPINES — dữ liệu dẫn xuất (đã khử định danh)

MRI cột sống thắt lưng, 250 ca chụp, kèm nhãn chẩn đoán theo từng tầng đĩa đệm.
Đây là **dữ liệu bệnh nhân đã khử định danh**, không phải dữ liệu công khai.
Xem [Điều kiện sử dụng](#điều-kiện-sử-dụng) ở cuối.

## Có gì trong này

| Thư mục          | Nội dung                                                                  | Dung lượng |
| ------------------ | -------------------------------------------------------------------------- | ------------ |
| `dicom/`         | DICOM gốc, thư mục đặt theo`patient_id`, **header đã rửa** | 33,8 GB      |
| `nifti/`         | Sagittal T1/T2 đã chuyển sang NIfTI + mask SPINEPS                      | 9,9 GB       |
| `grading/`       | Nhãn chẩn đoán, tách theo tháng                                      | 3 file       |
| `folds/`         | Chia 5-fold:`fold1/…fold5/` × `{train,val,test}.csv`                 | 17 file      |
| `localize/`      | Toạ độ tâm 5 đĩa đệm mỗi ca                                       | 1 CSV        |
| `reports_json/`  | Báo cáo chẩn đoán (`report/`) + nhân khẩu (`metadata/`)         | 480 file     |
| `reports_text*/` | Báo cáo dạng bảng, đã chia sẵn train/val/test                       | 9 file       |

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


Dưới đây là **miền giá trị chuẩn xác, kiểu dữ liệu, ý nghĩa y khoa và phân bố thực tế** của toàn bộ 8 nhãn bệnh lý được trích xuất trực tiếp từ 1.235 tầng đĩa đệm trong tệp [`dataset/dataset_master.csv`](file:///c:/Users/trang/Desktop/healcare_a2i/dataset/dataset_master.csv):

---

### BẢNG TỔNG HỢP MIỀN GIÁ TRỊ VÀ PHÂN BỐ LÂM SÀNG

| Tên nhãn bệnh lý            |    Kiểu dữ liệu    |    **Miền giá trị**    | Ý nghĩa lâm sàng chi tiết                                                                                                                                                                                                                                                                                                                                                                                                          | Số lượng thực tế trong Dataset (1.235 tầng)                                                                                                         |
| :------------------------------ | :-------------------: | :-----------------------------: | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :-------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **`pfirrmann_grade`**   | Số nguyên rời rạc | **$\{1, 2, 3, 4, 5\}$** | **Phân độ thoái hóa đĩa đệm theo thang Pfirrmann quốc tế:**• **1:** Bình thường, đĩa đệm sáng trắng, đồng nhất.• **2:** Bình thường sinh lý, có vệt xám ngang.• **3:** Thoái hóa nhẹ/vừa, đĩa đệm xám, mờ ranh giới.• **4:** Thoái hóa nặng, xám sẫm/đen, giảm chiều cao.• **5:** Giai đoạn cuối, đĩa đệm đen kịt, xẹp hoàn toàn. | • Độ 1: 36 tầng (2.9%)• **Độ 2: 689 tầng (55.8%)**• **Độ 3: 418 tầng (33.9%)**• Độ 4: 88 tầng (7.1%)• Độ 5: 4 tầng (0.3%) |
| **`disc_herniation`**   |  Nhị phân (Binary)  |     **$\{0, 1\}$**     | **Thoát vị đĩa đệm thực thụ:**• **0:** Không có thoát vị.• **1:** Có thoát vị (nhân nhầy phá vỡ bao xơ).                                                                                                                                                                                                                                                                                        | •**0: 1.147 tầng (92.9%)**• **1: 88 tầng (7.1%)**                                                                                         |
| **`disc_bulging`**      |  Nhị phân (Binary)  |     **$\{0, 1\}$**     | **Phình / Lồi đĩa đệm:**• **0:** Không phình.• **1:** Có phình lồi (lan tỏa $>25\%$ chu vi).                                                                                                                                                                                                                                                                                                          | •**0: 906 tầng (73.4%)**• **1: 328 tầng (26.6%)**                                                                                         |
| **`disc_narrowing`**    |  Nhị phân (Binary)  |     **$\{0, 1\}$**     | **Hẹp khe đĩa đệm:**• **0:** Chiều cao khe đĩa đệm bình thường.• **1:** Khe đĩa đệm bị xẹp / hẹp lại.                                                                                                                                                                                                                                                                                          | •**0: 1.176 tầng (95.2%)**• **1: 59 tầng (4.8%)**                                                                                         |
| **`spondylolisthesis`** |  Nhị phân (Binary)  |     **$\{0, 1\}$**     | **Trượt thân đốt sống:**• **0:** Các thân đốt sống thẳng hàng, vững chắc.• **1:** Có trượt thân đốt sống (ra trước hoặc ra sau).                                                                                                                                                                                                                                                           | •**0: 1.194 tầng (96.7%)**• **1: 41 tầng (3.3%)**                                                                                         |
| **`modic`**             |  Phân loại 4 mức  |  **$\{0, 1, 2, 3\}$**  | **Thoái hóa bờ thân đốt sống theo thang Modic:**• **0:** Không có thoái hóa Modic (Bình thường).• **1 (Modic I):** Phù nề tủy xương dưới sụn (giai đoạn viêm cấp).• **2 (Modic II):** Thoái hóa mỡ tủy xương (mãn tính phổ biến).• **3 (Modic III):** Xơ hóa xương dưới sụn (giai đoạn xơ chai).                                                        | •**0: 1.058 tầng (85.7%)**• **1: 24 tầng (1.9%)**• **2: 149 tầng (12.1%)**• **3: 4 tầng (0.3%)**                          |
| **`up_endplate`**       |  Nhị phân (Binary)  |     **$\{0, 1\}$**     | **Tổn thương bản đệm trên (Upper Endplate):**• **0:** Bản đệm trên phẳng, nguyên vẹn.• **1:** Bị nứt, khuyết xương, nốt Schmorl hoặc tổn thương.                                                                                                                                                                                                                                            | •**0: 1.120 tầng (90.7%)**• **1: 115 tầng (9.3%)**                                                                                        |
| **`low_endplate`**      |  Nhị phân (Binary)  |     **$\{0, 1\}$**     | **Tổn thương bản đệm dưới (Lower Endplate):**• **0:** Bản đệm dưới nguyên vẹn.• **1:** Có tổn thương / khuyết xương bản đệm dưới.                                                                                                                                                                                                                                                        | •**0: 1.163 tầng (94.2%)**• **1: 72 tầng (5.8%)**                                                                                         |

---

### 💡 Ghi chú quan trọng cho việc Huấn Luyện AI (Machine Learning Insights):

1. **Hiện tượng mất cân bằng dữ liệu (Class Imbalance):**
   - Các tổn thương nặng như `disc_herniation` ($7.1\%$), `spondylolisthesis` ($3.3\%$), `modic` type 1 ($1.9\%$) và `pfirrmann_grade` độ 5 ($0.3\%$) có tỷ lệ xuất hiện khá thấp (chiếm $<10\%$).
   - Khi bạn làm **Phase V2** (huấn luyện bộ phân loại bệnh lý thị giác), cần chú ý dùng kỹ thuật **Focal Loss** hoặc **Weighted Cross-Entropy** để mô hình không bị thiên lệch (bias) về các ca bình thường (`0`).
2. **Đa số người bệnh nằm ở mức thoái hóa trung bình:**
   - Hơn **$89.7\%$** các tầng đĩa đệm rơi vào `pfirrmann_grade` độ 2 (sinh lý) và độ 3 (thoái hóa nhẹ/vừa), phản ánh rất đúng thực tế lâm sàng của các bệnh nhân đến viện khám vì đau lưng.
