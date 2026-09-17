# TÀI LIỆU CHUYỂN GIAO TOÀN DIỆN DỰ ÁN
## Đề tài: "Sinh Báo Cáo MRI Cột Sống Thắt Lưng Theo Mức Có Hiệu Chuẩn Độ Bất Định"
*(Calibrated Level-wise Lumbar Spine MRI Report Generation via Medical VLM & LoRA)*

> **Dành cho:** Thành viên mới gia nhập nhóm nghiên cứu.  
> **Mục tiêu tài liệu:** Giúp bạn nắm bắt toàn bộ bối cảnh học thuật, cấu trúc dữ liệu, thiết kế hệ thống, mã nguồn đã triển khai và các bước tiếp theo trong vòng 15 phút.

---

## 1. BỐI CẢNH VÀ MỤC TIÊU ĐỀ TÀI

### 1.1. Vấn đề thực tiễn trong Y tế
- Chụp cộng hưởng từ (MRI) là tiêu chuẩn vàng để chẩn đoán các bệnh lý thoái hóa cột sống thắt lưng (thoát vị đĩa đệm, hẹp ống sống, hẹp lỗ liên hợp chèn ép rễ thần kinh).
- Mỗi ca chụp có hàng trăm lát cắt 3D (các chuỗi xung Sagittal T1, Sagittal T2, Axial T2). Bác sĩ chẩn đoán hình ảnh phải căng mắt kiểm tra tỉ mỉ **5 tầng đĩa đệm độc lập** (từ **L1/L2, L2/L3, L3/L4, L4/L5 đến L5/S1**) và gõ báo cáo thủ công. Việc này tốn từ 10 - 20 phút mỗi ca, dễ gây quá tải và sai sót do mỏi mắt.

### 1.2. Mục tiêu nghiên cứu
Xây dựng hệ thống Trí tuệ Nhân tạo đa phương thức (Multimodal AI) hỗ trợ bác sĩ tự động sinh văn bản báo cáo chẩn đoán lâm sàng bằng tiếng Việt chuẩn y khoa theo từng tầng đĩa đệm, đồng thời giải quyết triệt để vấn đề lớn nhất của AI y tế hiện nay: **Hiện tượng ảo giác (Hallucination)** bằng cách **Hiệu chuẩn độ bất định (Probability Calibration & Conformal Prediction)**.

### 1.3. Định dạng Báo cáo Mục tiêu
Không sinh cả văn bản hành chính rườm rà, mô hình tập trung sinh đoạn văn bản lâm sàng cô đọng (~200 - 250 từ) gồm 2 phần chuẩn:
- **`[MÔ TẢ]`**: Tường thuật chi tiết tình trạng đĩa đệm, ống sống, bao màng cứng và rễ thần kinh qua 5 tầng L1/L2 $\rightarrow$ L5/S1.
- **`[KẾT LUẬN]`**: Tóm tắt các chẩn đoán bệnh lý chính (ví dụ: Thoát vị đĩa đệm L4/L5 ra sau thể lồi, hẹp ống sống mức độ vừa).

---

## 2. KIẾN TRÚC DỮ LIỆU & BỘ DATASET ĐÃ CHUẨN HÓA

### 2.1. Dữ liệu gốc
- Bộ dữ liệu **PSPINES**: Gồm **247 bệnh nhân** chụp MRI cột sống thắt lưng đầy đủ các chuỗi xung NIfTI 3D (`.nii.gz`).
- Đi kèm: Nhãn tọa độ voxel định vị tầng đĩa đệm, nhãn phân loại độ thoái hóa (Pfirrmann grade 1-5), nhãn hẹp ống sống/lỗ liên hợp, và văn bản báo cáo y khoa của bác sĩ.

### 2.2. Dữ liệu đã hợp nhất trong mã nguồn (`dataset/`)
1. **`dataset/dataset_master.csv`**:
   - Gồm **1.235 dòng** (247 bệnh nhân $\times$ 5 tầng đĩa đệm L1/L2 đến L5/S1) và **39 cột thông tin**.
   - Tích hợp sẵn: Nhân khẩu học, tọa độ lát cắt voxel (`voxel_k`), phân loại bệnh lý, cột phân chia kiểm định chéo 5-fold (`fold1_split` đến `fold5_split` với tỷ lệ chuẩn 70% Train - 10% Val - 20% Test), và văn bản báo cáo tiếng Việt (`report_vi_mota`, `report_vi_ketluan`) cùng tiếng Anh (`report_en`).
2. **`dataset/sft_data/`**:
   - Đã tạo sẵn các file JSONL phân chia theo Fold 1 phục vụ tinh chỉnh mô hình:
     - `train_vi.jsonl` (173 ca), `val_vi.jsonl` (25 ca), `test_vi.jsonl` (49 ca).
     - Các file tương ứng bằng tiếng Anh (`train_en.jsonl`,...).
3. **Ảnh thể tích 3D NIfTI**:
   - Được lưu trữ trên Google Drive dự án: `https://drive.google.com/drive/u/1/folders/1uJuVbDIoNQP4Qrva9ZTgZSePtGfHj-Bg`. Đã có script tự động tải và liên kết.

---

## 3. LỘ TRÌNH 2 GIAI ĐOẠN (2-PHASE STRATEGY)

Để bài báo có đóng góp khoa học (Contribution) chặt chẽ và có tính phản biện so sánh (Ablation Study), dự án chia làm 2 giai đoạn:

```
                      LỘ TRÌNH NGHIÊN CỨU
                      
   ┌────────────────────────────────────────────────────────┐
   │  PHASE V1 (ĐÃ HOÀN THÀNH CODE & COLAB NOTEBOOK)        │
   │  • Kiến trúc: End-to-End Vision-Language Model (VLM)   │
   │  • Backbone: Qwen2.5-VL-3B-Instruct                    │
   │  • Kỹ thuật: 4-bit Quantization (NF4) + QLoRA (r=16)   │
   │  • Đầu vào: 1 Lát cắt dọc giữa Mid-sagittal T2 + Prompt│
   │  • Hàm Loss: Cross-Entropy (Label Masking trên Report) │
   │  • Vai trò: Làm mô hình Baseline đối chứng chuẩn mực   │
   └──────────────────────────┬─────────────────────────────┘
                              │
                              ▼ (Chỉ ra hạn chế: VLM End-to-End vẫn mắc lỗi ảo giác)
   ┌────────────────────────────────────────────────────────┐
   │  PHASE V2 (HƯỚNG ĐI TIẾP THEO & ĐÓNG GÓP CHÍNH CỦA BÀI) │
   │  • Kiến trúc: Modular Pipeline qua cầu nối JSON        │
   │  • Module 1: Vision Classifier / Detector từng tầng    │
   │  • Module 2: Conformal Prediction (Bảo hiểm xác suất)  │
   │  • Cầu nối: Schema JSON có nhãn và độ tin cậy toán học │
   │  • Module 3: Medical LLM Generator (Data-to-Text)      │
   │  • Lợi thế: Triệt tiêu ảo giác, cho phép bác sĩ can    │
   │             thiệp sửa JSON trước khi sinh báo cáo!     │
   └────────────────────────────────────────────────────────┘
```

---

## 4. NỀN TẢNG HỌC THUẬT & 2 BÀI BÁO QUỐC TẾ NÒNG CỐT

Chúng ta đã nghiên cứu và dịch chi tiết 2 bài báo quốc tế đỉnh cao để làm khung lý thuyết và trích dẫn:

1. **Bài báo 1 — C2M-DoT (*Medical Image Analysis*, 2023 - Q1, IF ~10.9):**
   - Đề xuất học tương phản đa góc nhìn (Frontal + Lateral) và Mạng chuyển dịch miền (**Domain Transfer Network - DoT** dùng Gumbel-Softmax) để mô hình train đa góc nhìn vẫn suy luận tốt khi chỉ có 1 góc nhìn.
   - **Liên hệ với đề tài của ta:** 
     - Cặp góc nhìn Sagittal - Axial trong cột sống hoàn toàn tương đương Frontal - Lateral.
     - Cơ chế DoT là cơ sở lý luận giải quyết triệt để bài toán **"Thiếu lát cắt Axial"** (các tầng L1/L2 thường không được chụp Axial trong thực tế).
2. **Bài báo 2 — ViMed-PET (*NeurIPS 2025* - Top 1 AI thế giới, Track Datasets & Benchmarks):**
   - Công bố bộ dữ liệu VLM y tế tiếng Việt đầu tiên (2.757 ca PET/CT toàn thân từ BV 108).
   - Chứng minh các VLM y tế tiếng Anh (LLaVA-Med, RadFM, M3D) bị "tê liệt" hoàn toàn khi sinh báo cáo tiếng Việt (BLEU-4 < 0.1); mô hình fine-tune trực tiếp trên tiếng Việt đạt kết quả vượt trội cả GPT-4o.
   - **Liên hệ với đề tài của ta:**
     - Ý tưởng phân chia 3 vùng cơ thể của ViMed-PET tương đồng với cấu trúc **phân chia 5 tầng đĩa đệm (Level-wise)** của ta.
     - Kế thừa hệ thống đo lường lâm sàng **Clinical F1** (đánh giá đúng tầng, đúng mức độ hẹp thay vì chỉ đo BLEU/ROUGE từ vựng).

---

## 5. BÀN LUẬN CHIẾN LƯỢC: TẠI SAO PHẢI TRAIN TIẾNG VIỆT MÀ KHÔNG DÙNG DỊCH MÁY?

Một câu hỏi rất hay mà các phản biện sẽ hỏi: *"Tại sao không train mô hình tiếng Anh rồi dùng Google Translate / DeepL dịch sang tiếng Việt?"*
Chúng ta đã làm rõ 3 sự thật đắt giá:
1. **Dữ liệu tiếng Anh về MRI cột sống có báo cáo văn bản gần như KHÔNG TỒN TẠI:**
   - Bộ RSNA Lumbar Spine 2024 chỉ có nhãn số phân loại rời rạc, không có free-text report. Tập PSPINES của chúng ta là một trong những bộ dữ liệu hiếm hoi có đủ cả ảnh 3D và báo cáo bác sĩ.
2. **Dịch máy trong Y tế chứa sai số lâm sàng nghiêm trọng (12% - 25%):**
   - Dịch máy thường lẫn lộn: *Disc bulge* (phồng đĩa đệm) thành *Disc herniation* (thoát vị); dịch *Neural Foraminal Stenosis* thành *"hẹp lỗ thần kinh"* (sai chuẩn y khoa, chuẩn là *"hẹp lỗ liên hợp"*); dịch nhầm *"ngách bên"* thành *"dưới khớp"*; nuốt mất từ phủ định hoặc đảo lộn bên trái/phải.
3. **Mô hình nền tảng (Qwen2.5-VL) vốn dĩ ĐÃ RẤT GIỎI TIẾNG VIỆT:**
   - Ta chỉ cần dùng QLoRA để căn chỉnh phong cách hành văn của bác sĩ Việt Nam trực tiếp trên ảnh MRI.

---

## 6. BẢN ĐỒ MÃ NGUỒN (REPOSITORY STRUCTURE)

Kho mã nguồn chính thức: `https://github.com/kttt294/MRI-report-generator.git` (branch `main`).

```text
healcare_a2i/
├── configs/
│   └── v1_config.yaml            # Siêu tham số V1 (Qwen2.5-VL-3B, LoRA r=16, lr=2e-4, epochs=10)
├── dataset/
│   ├── dataset_master.csv        # 1.235 mẫu cấp tầng đĩa đệm (39 cột)
│   ├── dataset_patients.jsonl    # 247 mẫu cấp bệnh nhân
│   └── sft_data/                 # Dữ liệu Train/Val/Test JSONL (Việt & Anh)
├── docs/
│   └── paper_translation_and_comparative_analysis.md  # Bản dịch & phân tích 2 bài báo quốc tế
├── notebooks/
│   └── V1_End_to_End_VLM_Training.ipynb  # Notebook Colab huấn luyện trọn gói V1 trên GPU T4
├── scripts/
│   └── consolidate_dataset.py    # Script ETL hợp nhất dữ liệu từ raw sang master CSV
└── src/
    ├── data/
    │   └── v1_dataset.py         # PyTorch Dataset: Trích xuất Mid-sagittal slice, tiền xử lý, chat template
    ├── models/
    │   └── v1_vlm.py             # Nạp Qwen2.5-VL 4-bit (NF4) và gắn LoRA adapter
    └── train_v1.py               # Script huấn luyện với VLMDataCollator và Trainer
```

---

## 7. CÔNG VIỆC CẦN LÀM TIẾP THEO CHO THÀNH VIÊN MỚI

Nếu bạn vừa nhận bàn giao, dưới đây là các bước hành động cụ thể:

1. **Bước 1: Chạy thử nghiệm Baseline V1 trên Google Colab:**
   - Mở file [`notebooks/V1_End_to_End_VLM_Training.ipynb`](file:///c:/Users/trang/Desktop/healcare_a2i/notebooks/V1_End_to_End_VLM_Training.ipynb) trên Google Colab (chọn môi trường T4 GPU miễn phí).
   - Mount Google Drive chứa ảnh NIfTI và bấm chạy toàn bộ notebook để quan sát mô hình hội tụ qua 10 epochs.
2. **Bước 2: Đánh giá & Phân tích lỗi lâm sàng của V1:**
   - Chạy inference trên tập `test_vi.jsonl` (49 ca kiểm thử).
   - Tính toán điểm BLEU-4, ROUGE-L và quan sát các trường hợp mô hình sinh sai vị trí hoặc "ảo giác" tổn thương.
3. **Bước 3: Bắt tay cùng xây dựng Phase V2 (Modular Pipeline):**
   - Huấn luyện mạng phân loại tổn thương (Pfirrmann, Stenosis) cho từng lát cắt.
   - Cài đặt thuật toán **Conformal Prediction** để gắn độ tin cậy vào JSON schema.
   - Viết prompt cho LLM sinh báo cáo từ JSON và so sánh trực tiếp kết quả với V1!
