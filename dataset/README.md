![1789829707485](image/README/1789829707485.png)

# BỘ DỮ LIỆU TINH GỌN PSPINES (CONSOLIDATED PSPINES DATASET)

Bộ dữ liệu này được tạo tự động bởi `scripts/consolidate_dataset.py` từ các tệp dữ liệu phân mảnh trong `dataset_local/`.

Toàn bộ thông tin về nhãn bệnh lý 5 tầng đĩa đệm, toạ độ không gian, phân chia 5-fold cross-validation và báo cáo chẩn đoán (Việt/Anh) được nối qua khóa `patient_id`; độ đúng của nhãn, mapping tầng và nội dung báo cáo cần kiểm tra riêng. Xem [trạng thái triển khai](../docs/implementation_status.md).

---

## 1. Cấu trúc tệp trong thư mục `dataset/`

```text
dataset/
├── dataset_master.csv          # Bảng tổng hợp phẳng chi tiết theo từng tầng (1.235 dòng, 42 cột)
├── dataset_patients.jsonl      # Dữ liệu phân tầng cấp bệnh nhân (247 dòng)
└── README.md                   # Tài liệu này
```

---

## 2. Mô tả chi tiết các tệp

### A. `dataset_master.csv` (1.235 dòng: 247 ca $\times$ 5 tầng)

Mỗi dòng đại diện cho một tầng đĩa đệm cụ thể của bệnh nhân:

- **Khóa & Nhân khẩu học:** `patient_id`, `sub_id`, `sex`, `birth_year`, `age_at_scan`, `study_date`.
- **Phân tầng:** `level` (`L1/L2` ... `L5/S1`), `ivd_label` ($1 \to 5$).
- **Nhãn bệnh lý lâm sàng:**
  - `pfirrmann_grade`: Phân độ thoái hóa ($1 \to 5$).
  - `modic`: Biến đổi Modic type 0–3; không phải nhãn nhị phân hay thang severity.
  - `disc_herniation`: Thoát vị đĩa đệm ($0$ hoặc $1$).
  - `disc_bulging`: Phình đĩa đệm ($0$ hoặc $1$).
  - `disc_narrowing`: Hẹp khe đĩa đệm ($0$ hoặc $1$).
  - `spondylolisthesis`: Trượt đốt sống ($0$ hoặc $1$).
  - `up_endplate`, `low_endplate`: Tổn thương bản đệm trên/dưới.
- **Toạ độ tâm đĩa (Localization):**
  - `voxel_i`, `voxel_j`, `voxel_k`: Toạ độ voxel trong file `volume` gốc; dùng đầy đủ affine khi đổi orientation, không coi `voxel_k` luôn là trục sagittal.
  - `x_lps`, `y_lps`, `z_lps`: Toạ độ không gian thực (LPS DICOM, đơn vị mm).
  - `volume`, `spacing_i`, `spacing_j`, `spacing_k`: Tên volume và voxel spacing theo trục của file nguồn; không giả định cùng spacing cho mọi ảnh.
- **Phân chia 5-Fold:** `fold1_split`, `fold2_split`, `fold3_split`, `fold4_split`, `fold5_split` (`train`, `val`, `test`).
- **Văn bản báo cáo:**
  - `report_vi_technique` (alias `report_vi_kythuat`): Kỹ thuật xung chụp (tiếng Việt).
  - `report_vi_findings` (alias `report_vi_mota`): Toàn bộ các câu mô tả chi tiết (tiếng Việt).
  - `report_vi_impression` (alias `report_vi_ketluan`): Phần kết luận lâm sàng (tiếng Việt).
  - `report_en`: Bản Clinician's Notes (tiếng Anh).

### B. `dataset_patients.jsonl` (247 dòng)

Mỗi dòng là một JSON độc lập biểu diễn một bệnh nhân hoàn chỉnh:

- Chứa mảng `levels`: bao gồm đầy đủ 5 tầng kèm toạ độ và nhãn.
- Chứa trường `reports`: chuẩn hóa theo chuẩn quốc tế:
  - `reports.vi`: chỉ xuất `technique`, `findings[]`, `impression[]`; bộ nhập vẫn đọc tên cũ trong nguồn. Alias cột CSV được giữ để tương thích. Nhãn thiếu là `null`, không phải `0`.
  - `reports.en`: `clinicians_notes`, `split`.
- Rất thuận tiện để load vào Python:
  ```python
  import json
  with open("dataset/dataset_patients.jsonl", encoding="utf-8") as f:
      patients = [json.loads(line) for line in f]
  ```

### C. Dữ liệu cho V1 và V2

V1 đọc CSV và ảnh NIfTI. V2 dùng `build_v2_inputs.py` để chuyển JSONL thành request chỉ có grading; không đưa báo cáo gốc vào prompt.

V2 fine-tune chỉ nhận target đã duyệt do `build_v2_targets.py` chuẩn bị, có scope và hash input hợp lệ. Split test chỉ dùng đánh giá, không dùng huấn luyện.

Đã bỏ export SFT VI/EN cũ và bản sao một bệnh nhân. Ví dụ contract giả lập nằm ở `examples/report_request.synthetic.json`. Dữ liệu tiếng Anh trong hai file chính được giữ để truy vết nguồn; pipeline mặc định dùng tiếng Việt.

---

## 3. Cách chạy lại script tái tạo dữ liệu

Nếu bạn muốn bổ sung logic hoặc cập nhật thêm trường dữ liệu, chỉ cần chạy:

```bash
python scripts/consolidate_dataset.py --source dataset_local --output output/derived/dataset
```
