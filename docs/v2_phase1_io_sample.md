# Mẫu Dữ Liệu Đầu Vào và Đầu Ra của Hệ Thống Phase 1 V2

Tài liệu này minh họa chi tiết cấu trúc dữ liệu đầu vào (Input) và đầu ra (Output) trong kiến trúc chẩn đoán MRI cột sống thắt lưng Giai đoạn 1 (Phase 1 V2).

---

## 1. Tổng Quan Luồng Dữ Liệu Phase 1 V2

Kiến trúc Phase 1 V2 vận hành theo mô hình 2 chặng có kiểm soát bằng chứng nghiêm ngặt (*Evidence-Grounded Cascaded Pipeline*):

```
[Ảnh MRI Cột sống]
       │
       ▼ (Mô hình Thị giác / Phân loại)
[40 Chỉ số Bệnh lý: 5 tầng x 8 nhóm bệnh]  <─── Input 1: ReportRequest (JSON)
       │
       ▼ (Bộ sinh Catalog & Prompt)
[Prompt hướng dẫn + Danh mục câu mẫu]     <─── Input 2: LLM Prompt (Text + JSON)
       │
       ▼ (Mô hình Ngôn ngữ Qwen2.5-3B)
[Bản nháp JSON gắn mã bằng chứng]          <─── Output 1: ReportDraft (JSON)
       │
       ▼ (Lớp Kiểm định 4 tầng & Ghép chuỗi)
[Báo cáo y khoa hoàn chỉnh cho bác sĩ]     <─── Output 2: Rendered Report (Text)
```

---

## 2. Dữ Liệu Đầu Vào (Input)

### 2.1. Đầu vào Cấp Hệ Thống: `ReportRequest`
Dữ liệu do mô hình thị giác phân tích ảnh MRI tạo ra, bao gồm 40 thông số được chuẩn hóa cho 5 tầng đĩa đệm (L1/L2 đến L5/S1) và 8 nhóm bệnh lý.

```json
{
  "schema_version": "report-input/1.0",
  "ontology_version": "pspines-grading/1.0",
  "scope": "lumbar_grading_8_fields",
  "case_id": "250002255",
  "quality": {
    "level_mapping": "verified",
    "ontology_review": "verified"
  },
  "levels": [
    {
      "level": "L1/L2",
      "gradings": {
        "pfirrmann_grade": {"value": 2, "status": "observed", "uncertainty": null},
        "modic": {"value": 0, "status": "observed", "uncertainty": null},
        "disc_herniation": {"value": 0, "status": "observed", "uncertainty": null},
        "disc_bulging": {"value": 0, "status": "observed", "uncertainty": null},
        "disc_narrowing": {"value": 0, "status": "observed", "uncertainty": null},
        "spondylolisthesis": {"value": 0, "status": "observed", "uncertainty": null},
        "up_endplate": {"value": 0, "status": "observed", "uncertainty": null},
        "low_endplate": {"value": 0, "status": "observed", "uncertainty": null}
      }
    },
    {
      "level": "L2/L3",
      "gradings": {
        "pfirrmann_grade": {"value": 3, "status": "observed", "uncertainty": null},
        "modic": {"value": 0, "status": "observed", "uncertainty": null},
        "disc_herniation": {"value": 0, "status": "observed", "uncertainty": null},
        "disc_bulging": {"value": 0, "status": "observed", "uncertainty": null},
        "disc_narrowing": {"value": 0, "status": "observed", "uncertainty": null},
        "spondylolisthesis": {"value": 0, "status": "observed", "uncertainty": null},
        "up_endplate": {"value": 0, "status": "observed", "uncertainty": null},
        "low_endplate": {"value": 0, "status": "observed", "uncertainty": null}
      }
    },
    {
      "level": "L3/L4",
      "gradings": {
        "pfirrmann_grade": {"value": 3, "status": "observed", "uncertainty": null},
        "modic": {"value": 0, "status": "observed", "uncertainty": null},
        "disc_herniation": {"value": 0, "status": "observed", "uncertainty": null},
        "disc_bulging": {"value": 1, "status": "observed", "uncertainty": null},
        "disc_narrowing": {"value": 0, "status": "observed", "uncertainty": null},
        "spondylolisthesis": {"value": 0, "status": "observed", "uncertainty": null},
        "up_endplate": {"value": 0, "status": "observed", "uncertainty": null},
        "low_endplate": {"value": 0, "status": "observed", "uncertainty": null}
      }
    },
    {
      "level": "L4/L5",
      "gradings": {
        "pfirrmann_grade": {"value": 3, "status": "observed", "uncertainty": null},
        "modic": {"value": 1, "status": "observed", "uncertainty": null},
        "disc_herniation": {"value": 0, "status": "observed", "uncertainty": null},
        "disc_bulging": {"value": 0, "status": "observed", "uncertainty": null},
        "disc_narrowing": {"value": 0, "status": "observed", "uncertainty": null},
        "spondylolisthesis": {"value": 0, "status": "observed", "uncertainty": null},
        "up_endplate": {"value": 1, "status": "observed", "uncertainty": null},
        "low_endplate": {"value": 0, "status": "observed", "uncertainty": null}
      }
    },
    {
      "level": "L5/S1",
      "gradings": {
        "pfirrmann_grade": {"value": 3, "status": "observed", "uncertainty": null},
        "modic": {"value": 1, "status": "observed", "uncertainty": null},
        "disc_herniation": {"value": 0, "status": "observed", "uncertainty": null},
        "disc_bulging": {"value": 1, "status": "observed", "uncertainty": null},
        "disc_narrowing": {"value": 0, "status": "observed", "uncertainty": null},
        "spondylolisthesis": {"value": 1, "status": "observed", "uncertainty": null},
        "up_endplate": {"value": 0, "status": "observed", "uncertainty": null},
        "low_endplate": {"value": 1, "status": "observed", "uncertainty": null}
      }
    }
  ]
}
```

---

### 2.2. Đầu vào Cấp Mô Hình Ngôn Ngữ: `LLM Prompt`
Văn bản câu lệnh (prompt) chính xác được đưa vào mô hình Qwen2.5-3B, bao gồm luật cấm suy diễn và danh mục các câu mẫu tương ứng với dữ kiện:

```text
Bạn diễn đạt báo cáo từ các nhãn đã cung cấp, không đọc ảnh và không chẩn đoán thêm.
Trả duy nhất JSON có hai key findings và impression, mỗi key là danh sách các object:
{"text": "câu văn", "evidence_ids": ["mã dữ kiện"]}.
Mỗi câu chỉ mang một evidence_id. Dùng đúng các câu và evidence_ids trong danh mục tương ứng.
Giữ đủ từng câu đúng một lần ở từng phần; sắp theo tầng L1/L2 đến L5/S1.
Không thêm bên trái/phải, mức độ, rễ thần kinh, hẹp ống sống hoặc những điều không có trong dữ kiện.
Không biến dữ liệu thiếu/chưa đánh giá/chưa chắc chắn thành âm tính.
Không thêm giải thích, markdown, câu giới thiệu hoặc nhận xét ngoài JSON.

{
  "scope": "lumbar_grading_8_fields",
  "policy_version": "grading-verbatim/1.0",
  "findings_catalog": [
    {"evidence_ids": ["L1/L2:pfirrmann_grade"], "text": "Tại tầng L1/L2: Pfirrmann độ 2."},
    {"evidence_ids": ["L1/L2:modic"], "text": "Tại tầng L1/L2: không có biến đổi Modic."},
    {"evidence_ids": ["L1/L2:disc_herniation"], "text": "Tại tầng L1/L2: không có thoát vị đĩa đệm."},
    {"evidence_ids": ["L1/L2:disc_bulging"], "text": "Tại tầng L1/L2: không có phình đĩa đệm."},
    {"evidence_ids": ["L1/L2:disc_narrowing"], "text": "Tại tầng L1/L2: không có giảm chiều cao khe đĩa đệm."},
    {"evidence_ids": ["L1/L2:spondylolisthesis"], "text": "Tại tầng L1/L2: không có trượt đốt sống."},
    {"evidence_ids": ["L1/L2:up_endplate"], "text": "Tại tầng L1/L2: không có bất thường bản đệm trên."},
    {"evidence_ids": ["L1/L2:low_endplate"], "text": "Tại tầng L1/L2: không có bất thường bản đệm dưới."},
    {"evidence_ids": ["L2/L3:pfirrmann_grade"], "text": "Tại tầng L2/L3: Pfirrmann độ 3."},
    ... (tiếp tục liệt kê đủ 40 câu tương ứng 40 mã dữ kiện từ L1/L2 đến L5/S1)
  ],
  "impression_catalog": [
    {"evidence_ids": ["L1/L2:pfirrmann_grade"], "text": "Tại tầng L1/L2: Pfirrmann độ 2."},
    {"evidence_ids": ["L2/L3:pfirrmann_grade"], "text": "Tại tầng L2/L3: Pfirrmann độ 3."},
    {"evidence_ids": ["L3/L4:pfirrmann_grade"], "text": "Tại tầng L3/L4: Pfirrmann độ 3."},
    {"evidence_ids": ["L3/L4:disc_bulging"], "text": "Tại tầng L3/L4: có phình đĩa đệm."},
    {"evidence_ids": ["L4/L5:pfirrmann_grade"], "text": "Tại tầng L4/L5: Pfirrmann độ 3."},
    {"evidence_ids": ["L4/L5:modic"], "text": "Tại tầng L4/L5: biến đổi Modic type I."},
    {"evidence_ids": ["L4/L5:up_endplate"], "text": "Tại tầng L4/L5: có bất thường bản đệm trên."},
    {"evidence_ids": ["L5/S1:pfirrmann_grade"], "text": "Tại tầng L5/S1: Pfirrmann độ 3."},
    {"evidence_ids": ["L5/S1:modic"], "text": "Tại tầng L5/S1: biến đổi Modic type I."},
    {"evidence_ids": ["L5/S1:disc_bulging"], "text": "Tại tầng L5/S1: có phình đĩa đệm."},
    {"evidence_ids": ["L5/S1:spondylolisthesis"], "text": "Tại tầng L5/S1: có trượt đốt sống."},
    {"evidence_ids": ["L5/S1:low_endplate"], "text": "Tại tầng L5/S1: có bất thường bản đệm dưới."}
  ]
}
```

---

## 3. Dữ Liệu Đầu Ra (Output)

### 3.1. Đầu ra Cấp Mô Hình: `ReportDraft` (JSON)
Kết quả trực tiếp sinh ra bởi LLM. Mỗi câu được gắn đúng 1 mã `evidence_ids` tương ứng trong danh mục:

```json
{
  "findings": [
    {"evidence_ids": ["L1/L2:pfirrmann_grade"], "text": "Tại tầng L1/L2: Pfirrmann độ 2."},
    {"evidence_ids": ["L1/L2:modic"], "text": "Tại tầng L1/L2: không có biến đổi Modic."},
    {"evidence_ids": ["L1/L2:disc_herniation"], "text": "Tại tầng L1/L2: không có thoát vị đĩa đệm."},
    {"evidence_ids": ["L1/L2:disc_bulging"], "text": "Tại tầng L1/L2: không có phình đĩa đệm."},
    {"evidence_ids": ["L1/L2:disc_narrowing"], "text": "Tại tầng L1/L2: không có giảm chiều cao khe đĩa đệm."},
    {"evidence_ids": ["L1/L2:spondylolisthesis"], "text": "Tại tầng L1/L2: không có trượt đốt sống."},
    {"evidence_ids": ["L1/L2:up_endplate"], "text": "Tại tầng L1/L2: không có bất thường bản đệm trên."},
    {"evidence_ids": ["L1/L2:low_endplate"], "text": "Tại tầng L1/L2: không có bất thường bản đệm dưới."},
    {"evidence_ids": ["L2/L3:pfirrmann_grade"], "text": "Tại tầng L2/L3: Pfirrmann độ 3."},
    {"evidence_ids": ["L2/L3:modic"], "text": "Tại tầng L2/L3: không có biến đổi Modic."},
    {"evidence_ids": ["L2/L3:disc_herniation"], "text": "Tại tầng L2/L3: không có thoát vị đĩa đệm."},
    {"evidence_ids": ["L2/L3:disc_bulging"], "text": "Tại tầng L2/L3: không có phình đĩa đệm."},
    {"evidence_ids": ["L2/L3:disc_narrowing"], "text": "Tại tầng L2/L3: không có giảm chiều cao khe đĩa đệm."},
    {"evidence_ids": ["L2/L3:spondylolisthesis"], "text": "Tại tầng L2/L3: không có trượt đốt sống."},
    {"evidence_ids": ["L2/L3:up_endplate"], "text": "Tại tầng L2/L3: không có bất thường bản đệm trên."},
    {"evidence_ids": ["L2/L3:low_endplate"], "text": "Tại tầng L2/L3: không có bất thường bản đệm dưới."},
    {"evidence_ids": ["L3/L4:pfirrmann_grade"], "text": "Tại tầng L3/L4: Pfirrmann độ 3."},
    {"evidence_ids": ["L3/L4:modic"], "text": "Tại tầng L3/L4: không có biến đổi Modic."},
    {"evidence_ids": ["L3/L4:disc_herniation"], "text": "Tại tầng L3/L4: không có thoát vị đĩa đệm."},
    {"evidence_ids": ["L3/L4:disc_bulging"], "text": "Tại tầng L3/L4: có phình đĩa đệm."},
    {"evidence_ids": ["L3/L4:disc_narrowing"], "text": "Tại tầng L3/L4: không có giảm chiều cao khe đĩa đệm."},
    {"evidence_ids": ["L3/L4:spondylolisthesis"], "text": "Tại tầng L3/L4: không có trượt đốt sống."},
    {"evidence_ids": ["L3/L4:up_endplate"], "text": "Tại tầng L3/L4: không có bất thường bản đệm trên."},
    {"evidence_ids": ["L3/L4:low_endplate"], "text": "Tại tầng L3/L4: không có bất thường bản đệm dưới."},
    {"evidence_ids": ["L4/L5:pfirrmann_grade"], "text": "Tại tầng L4/L5: Pfirrmann độ 3."},
    {"evidence_ids": ["L4/L5:modic"], "text": "Tại tầng L4/L5: biến đổi Modic type I."},
    {"evidence_ids": ["L4/L5:disc_herniation"], "text": "Tại tầng L4/L5: không có thoát vị đĩa đệm."},
    {"evidence_ids": ["L4/L5:disc_bulging"], "text": "Tại tầng L4/L5: không có phình đĩa đệm."},
    {"evidence_ids": ["L4/L5:disc_narrowing"], "text": "Tại tầng L4/L5: không có giảm chiều cao khe đĩa đệm."},
    {"evidence_ids": ["L4/L5:spondylolisthesis"], "text": "Tại tầng L4/L5: không có trượt đốt sống."},
    {"evidence_ids": ["L4/L5:up_endplate"], "text": "Tại tầng L4/L5: có bất thường bản đệm trên."},
    {"evidence_ids": ["L4/L5:low_endplate"], "text": "Tại tầng L4/L5: không có bất thường bản đệm dưới."},
    {"evidence_ids": ["L5/S1:pfirrmann_grade"], "text": "Tại tầng L5/S1: Pfirrmann độ 3."},
    {"evidence_ids": ["L5/S1:modic"], "text": "Tại tầng L5/S1: biến đổi Modic type I."},
    {"evidence_ids": ["L5/S1:disc_herniation"], "text": "Tại tầng L5/S1: không có thoát vị đĩa đệm."},
    {"evidence_ids": ["L5/S1:disc_bulging"], "text": "Tại tầng L5/S1: có phình đĩa đệm."},
    {"evidence_ids": ["L5/S1:disc_narrowing"], "text": "Tại tầng L5/S1: không có giảm chiều cao khe đĩa đệm."},
    {"evidence_ids": ["L5/S1:spondylolisthesis"], "text": "Tại tầng L5/S1: có trượt đốt sống."},
    {"evidence_ids": ["L5/S1:up_endplate"], "text": "Tại tầng L5/S1: không có bất thường bản đệm trên."},
    {"evidence_ids": ["L5/S1:low_endplate"], "text": "Tại tầng L5/S1: có bất thường bản đệm dưới."}
  ],
  "impression": [
    {"evidence_ids": ["L1/L2:pfirrmann_grade"], "text": "Tại tầng L1/L2: Pfirrmann độ 2."},
    {"evidence_ids": ["L2/L3:pfirrmann_grade"], "text": "Tại tầng L2/L3: Pfirrmann độ 3."},
    {"evidence_ids": ["L3/L4:pfirrmann_grade"], "text": "Tại tầng L3/L4: Pfirrmann độ 3."},
    {"evidence_ids": ["L3/L4:disc_bulging"], "text": "Tại tầng L3/L4: có phình đĩa đệm."},
    {"evidence_ids": ["L4/L5:pfirrmann_grade"], "text": "Tại tầng L4/L5: Pfirrmann độ 3."},
    {"evidence_ids": ["L4/L5:modic"], "text": "Tại tầng L4/L5: biến đổi Modic type I."},
    {"evidence_ids": ["L4/L5:up_endplate"], "text": "Tại tầng L4/L5: có bất thường bản đệm trên."},
    {"evidence_ids": ["L5/S1:pfirrmann_grade"], "text": "Tại tầng L5/S1: Pfirrmann độ 3."},
    {"evidence_ids": ["L5/S1:modic"], "text": "Tại tầng L5/S1: biến đổi Modic type I."},
    {"evidence_ids": ["L5/S1:disc_bulging"], "text": "Tại tầng L5/S1: có phình đĩa đệm."},
    {"evidence_ids": ["L5/S1:spondylolisthesis"], "text": "Tại tầng L5/S1: có trượt đốt sống."},
    {"evidence_ids": ["L5/S1:low_endplate"], "text": "Tại tầng L5/S1: có bất thường bản đệm dưới."}
  ]
}
```

---

### 3.2. Đầu ra Cấp Hiển Thị Lâm Sàng: `Rendered Text`
Báo cáo văn bản được bóc tách từ JSON sau khi đã vượt qua bộ kiểm định (Validate), hiển thị cho bác sĩ:

```text
=== PHẦN MÔ TẢ (FINDINGS) ===
Tại tầng L1/L2: Pfirrmann độ 2.
Tại tầng L1/L2: không có biến đổi Modic.
Tại tầng L1/L2: không có thoát vị đĩa đệm.
Tại tầng L1/L2: không có phình đĩa đệm.
Tại tầng L1/L2: không có giảm chiều cao khe đĩa đệm.
Tại tầng L1/L2: không có trượt đốt sống.
Tại tầng L1/L2: không có bất thường bản đệm trên.
Tại tầng L1/L2: không có bất thường bản đệm dưới.
Tại tầng L2/L3: Pfirrmann độ 3.
Tại tầng L2/L3: không có biến đổi Modic.
Tại tầng L2/L3: không có thoát vị đĩa đệm.
Tại tầng L2/L3: không có phình đĩa đệm.
Tại tầng L2/L3: không có giảm chiều cao khe đĩa đệm.
Tại tầng L2/L3: không có trượt đốt sống.
Tại tầng L2/L3: không có bất thường bản đệm trên.
Tại tầng L2/L3: không có bất thường bản đệm dưới.
Tại tầng L3/L4: Pfirrmann độ 3.
Tại tầng L3/L4: không có biến đổi Modic.
Tại tầng L3/L4: không có thoát vị đĩa đệm.
Tại tầng L3/L4: có phình đĩa đệm.
Tại tầng L3/L4: không có giảm chiều cao khe đĩa đệm.
Tại tầng L3/L4: không có trượt đốt sống.
Tại tầng L3/L4: không có bất thường bản đệm trên.
Tại tầng L3/L4: không có bất thường bản đệm dưới.
Tại tầng L4/L5: Pfirrmann độ 3.
Tại tầng L4/L5: biến đổi Modic type I.
Tại tầng L4/L5: không có thoát vị đĩa đệm.
Tại tầng L4/L5: không có phình đĩa đệm.
Tại tầng L4/L5: không có giảm chiều cao khe đĩa đệm.
Tại tầng L4/L5: không có trượt đốt sống.
Tại tầng L4/L5: có bất thường bản đệm trên.
Tại tầng L4/L5: không có bất thường bản đệm dưới.
Tại tầng L5/S1: Pfirrmann độ 3.
Tại tầng L5/S1: biến đổi Modic type I.
Tại tầng L5/S1: không có thoát vị đĩa đệm.
Tại tầng L5/S1: có phình đĩa đệm.
Tại tầng L5/S1: không có giảm chiều cao khe đĩa đệm.
Tại tầng L5/S1: có trượt đốt sống.
Tại tầng L5/S1: không có bất thường bản đệm trên.
Tại tầng L5/S1: có bất thường bản đệm dưới.

=== PHẦN KẾT LUẬN (IMPRESSION) ===
Tại tầng L1/L2: Pfirrmann độ 2.
Tại tầng L2/L3: Pfirrmann độ 3.
Tại tầng L3/L4: Pfirrmann độ 3.
Tại tầng L3/L4: có phình đĩa đệm.
Tại tầng L4/L5: Pfirrmann độ 3.
Tại tầng L4/L5: biến đổi Modic type I.
Tại tầng L4/L5: có bất thường bản đệm trên.
Tại tầng L5/S1: Pfirrmann độ 3.
Tại tầng L5/S1: biến đổi Modic type I.
Tại tầng L5/S1: có phình đĩa đệm.
Tại tầng L5/S1: có trượt đốt sống.
Tại tầng L5/S1: có bất thường bản đệm dưới.
```

---

## 4. Điểm Nhận Định & So Sánh Hướng Đến Phase 2

| Đặc điểm | Phase 1 V2 (Hiện tại) | Phase 2 V2 (Mục tiêu nâng cấp) |
| :--- | :--- | :--- |
| **Quy tắc câu** | 1 câu = 1 nhãn bệnh = 1 `evidence_id` | 1 câu có thể gộp nhiều tầng và nhiều `evidence_ids` |
| **Độ dài Findings** | Luôn cố định đủ 40 câu vụn vặt | Gom các tầng bình thường lại, chỉ khoảng 8–12 câu súc tích |
| **Độ dài Impression** | 6–12 dòng đơn lặp lại chữ *"Tại tầng..."* | Gom lại thành 2–4 câu kết luận y khoa tổng hợp |
| **Vai trò của LLM** | Gần như máy chép mẫu catalog | Trợ lý thư ký y khoa biết tổng hợp, bám sát bằng chứng |
| **Tiếp nhận chỉ dẫn bác sĩ** | Không hỗ trợ | Hỗ trợ ghi chú lâm sàng và điều chỉnh của bác sĩ |
| **Nguy cơ Ảo giác** | **0.0%** (do kiểm soát verbatim) | **0.0%** (do kiểm soát set-partition multi-evidence) |
