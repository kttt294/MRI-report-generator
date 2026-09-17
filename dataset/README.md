# BỘ DỮ LIỆU TINH GỌN PSPINES (CONSOLIDATED PSPINES DATASET)

Bộ dữ liệu này được tạo tự động bởi `scripts/consolidate_dataset.py` từ các tệp dữ liệu phân mảnh trong `dataset_local/`.

Toàn bộ thông tin về nhãn bệnh lý 5 tầng đĩa đệm, toạ độ không gian, phân chia 5-fold cross-validation và báo cáo chẩn đoán (Việt/Anh) đã được liên kết chuẩn xác qua khóa `patient_id`.

---

## 1. Cấu trúc tệp trong thư mục `dataset/`

```text
dataset/
├── dataset_master.csv          # Bảng tổng hợp phẳng chi tiết theo từng tầng (1.235 dòng, 39 cột)
├── dataset_patients.jsonl      # Dữ liệu phân tầng cấp bệnh nhân (247 dòng)
├── sft_data/                   # Dữ liệu chuẩn Prompt-Response sẵn sàng cho Huấn luyện LoRA (Fold 1)
│   ├── sft_fold1_train_vi.jsonl (142 mẫu tiếng Việt)
│   ├── sft_fold1_val_vi.jsonl   (49 mẫu tiếng Việt)
│   ├── sft_fold1_test_vi.jsonl  (47 mẫu tiếng Việt)
│   ├── sft_fold1_train_en.jsonl (140 mẫu tiếng Anh)
│   ├── sft_fold1_val_en.jsonl   (49 mẫu tiếng Anh)
│   └── sft_fold1_test_en.jsonl  (47 mẫu tiếng Anh)
└── README.md                   # Tài liệu này
```

---

## 2. Mô tả chi tiết các tệp

### A. `dataset_master.csv` (1.235 dòng: 247 ca $\times$ 5 tầng)
Mỗi dòng đại diện cho một tầng đĩa đệm cụ thể của bệnh nhân:
* **Khóa & Nhân khẩu học:** `patient_id`, `sub_id`, `sex`, `birth_year`, `age_at_scan`, `study_date`.
* **Phân tầng:** `level` (`L1/L2` ... `L5/S1`), `ivd_label` ($1 \to 5$).
* **Nhãn bệnh lý lâm sàng:**
  * `pfirrmann_grade`: Phân độ thoái hóa ($1 \to 5$).
  * `modic`: Thoái hóa Modic ($0$ hoặc $1$).
  * `disc_herniation`: Thoát vị đĩa đệm ($0$ hoặc $1$).
  * `disc_bulging`: Phình đĩa đệm ($0$ hoặc $1$).
  * `disc_narrowing`: Hẹp khe đĩa đệm ($0$ hoặc $1$).
  * `spondylolisthesis`: Trượt đốt sống ($0$ hoặc $1$).
  * `up_endplate`, `low_endplate`: Tổn thương bản đệm trên/dưới.
* **Toạ độ tâm đĩa (Localization):**
  * `voxel_i`, `voxel_j`, `voxel_k`: Toạ độ voxel trên ảnh Sagittal T2.
  * `x_lps`, `y_lps`, `z_lps`: Toạ độ không gian thực (LPS DICOM, đơn vị mm).
  * `volume`, `spacing_i`, `spacing_j`, `spacing_k`: Kích thước và độ dày lát cắt ($4.4\text{ mm}$).
* **Phân chia 5-Fold:** `fold1_split`, `fold2_split`, `fold3_split`, `fold4_split`, `fold5_split` (`train`, `val`, `test`).
* **Văn bản báo cáo:**
  * `report_vi_technique` (alias `report_vi_kythuat`): Kỹ thuật xung chụp (tiếng Việt).
  * `report_vi_findings` (alias `report_vi_mota`): Toàn bộ các câu mô tả chi tiết (tiếng Việt).
  * `report_vi_impression` (alias `report_vi_ketluan`): Phần kết luận lâm sàng (tiếng Việt).
  * `report_en`: Bản Clinician's Notes (tiếng Anh).

### B. `dataset_patients.jsonl` (247 dòng) & `data_of_1patient.json`
Mỗi dòng là một JSON độc lập biểu diễn một bệnh nhân hoàn chỉnh:
* Chứa mảng `levels`: bao gồm đầy đủ 5 tầng kèm toạ độ và nhãn.
* Chứa trường `reports`: chuẩn hóa theo chuẩn quốc tế:
  * `reports.vi`: `technique`, `findings[]`, `impression[]` (kèm alias `ky_thuat`, `mo_ta`, `ket_luan`).
  * `reports.en`: `clinicians_notes`, `split`.
* Rất thuận tiện để load vào Python:
  ```python
  import json
  with open("dataset/dataset_patients.jsonl", encoding="utf-8") as f:
      patients = [json.loads(line) for line in f]
  ```

### C. Thư mục `sft_data/` (Sẵn sàng đưa vào Hugging Face SFTTrainer)
Định dạng Prompt/Response chuẩn để tinh chỉnh MedGemma / Qwen2.5:
```json
{
  "patient_id": "250002076",
  "prompt": "[THÔNG TIN BỆNH NHÂN]: Tuổi: 46, Giới tính: F\n[KẾT QUẢ KHẢO SÁT 5 TẦNG ĐĨA ĐỆM CỘT SỐNG THẮT LƯNG]:\n- Tầng L1/L2: Thoái hóa Pfirrmann độ 2...\n[YÊU CẦU]: Dựa trên các phát hiện bệnh lý trên, hãy viết phần MÔ TẢ và KẾT LUẬN báo cáo cộng hưởng từ cột sống thắt lưng.",
  "response": "[MÔ TẢ]:\n- Đường cong cột sống thắt lưng giảm.\n...\n[KẾT LUẬN]:\n- Thoái hóa đốt sống - đĩa đệm..."
}
```

---

## 3. Cách chạy lại script tái tạo dữ liệu
Nếu bạn muốn bổ sung logic hoặc cập nhật thêm trường dữ liệu, chỉ cần chạy:
```bash
python scripts/consolidate_dataset.py
```
