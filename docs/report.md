# TỔNG QUAN KIẾN TRÚC & PHƯƠNG PHÁP NGHIÊN CỨU MÔ HÌNH V2 (TEXT-TO-REPORT)

*(Tài liệu cô đọng dành cho Trưởng nhóm nghiên cứu & Tác giả bài báo khoa học)*

---

> [!IMPORTANT]
> **Tuyên ngôn an toàn cốt lõi (Core System Disclaimer):**
> Template fallback giúp trả về nội dung bám theo grading và policy đã định nghĩa khi LLM không đạt kiểm tra. Nó không bảo đảm grading đúng trên MRI hoặc báo cáo chính xác về mặt lâm sàng; các trường hợp nhãn thiếu hoặc chưa chắc chắn đều phải được ghi rõ và gắn cờ `needs_review`.

---

## 1. Bài toán & Phân tách trách nhiệm hệ thống (System Architecture)

Dự án triển khai kiến trúc 2 giai đoạn (Two-Stage Neuro-Symbolic Pipeline) nhằm triệt tiêu hoàn toàn rủi ro ảo giác (hallucination) trong sinh báo cáo MRI:

```
[Ảnh MRI 3D (NIfTI)] ──► [Vision Engine (V1)] ──► Bảng phân độ định lượng (8 nhãn x 5 tầng)
                                                           │
                                                           ▼
                                               [Report Engine (V2)] ──► Báo cáo JSON có cấu trúc
```

* **Nhánh V1 (Vision Engine):** Chịu trách nhiệm về cảm nhận thị giác trên các lát cắt 2D trích từ thể tích 3D (mặc định lát sagittal giữa). V1 chịu trách nhiệm lượng hóa độ không chắc chắn (Uncertainty Estimation) thông qua **Probability Calibration** và **Conformal Prediction**. Nếu độ tin cậy không đạt, Vision Engine xuất cờ `status = "uncertain"`.
* **Nhánh V2 (Report Engine):** Đóng vai trò là bộ chuyển dịch ngôn ngữ có kiểm soát (**Controlled Text-to-Report Generator**). V2 **hoàn toàn mù về mặt thị giác**, chỉ nhận bảng nhãn số định lượng và chuyển hóa thành văn bản JSON y khoa có cấu trúc, mỗi câu phát biểu đều gắn chặt 1-1 với mã bằng chứng (`evidence_ids`).

---

## 2. Đặc tả Hợp đồng Dữ liệu (Formal Data Contracts)

### A. Đầu vào (`ReportRequest`)

* **Không gian phân tích:** Đúng 5 tầng thắt lưng ($L_1/L_2 \dots L_5/S_1$) $\times$ 8 nhãn bệnh lý = 40 thuộc tính.
* **Phân định quan sát (`Observation`):**
  * `observed`: Có giá trị số nguyên ($v \in \mathbb{Z}$).
  * `missing`, `not_assessed`, `uncertain`: Giá trị bắt buộc là `null`.
* **Bộ lọc cách ly an toàn:** Toạ độ voxel, tuổi, giới tính, thông tin bệnh nhân và báo cáo tự do gốc của bác sĩ **bị loại bỏ 100% khỏi prompt** để ngăn chặn thiên kiến và suy diễn ngoài dữ kiện.

### B. Đầu ra (`ReportResult` bọc `ReportDraft`)

* **Bản dự thảo nội bộ (`ReportDraft`):**
  * `findings`: Đúng 40 câu mô tả tương ứng 40 thuộc tính, sắp theo tầng $L_1/L_2 \dots L_5/S_1$. Mỗi câu mang đúng 1 mã `evidence_id`.
  * `impression`: Tập con các câu chọn lọc từ `findings` (giữ lại Pfirrmann, nhãn dương tính $v > 0$, hoặc trạng thái chưa xác định).
* **Kết quả trả về chính thức (`ReportResult`):** Chứa văn bản hoàn chỉnh, trạng thái thực thi (`ok`, `fallback`, `needs_review`, `failed`), cảnh báo giới hạn (`limitations`), lịch sử các lần thử (`attempts`), và băm SHA-256 đối chiếu toàn vẹn.

---

## 3. Bản chất Thực nghiệm của Phase 1 hiện tại (`grading-verbatim/1.0`)

Đánh giá khách quan và trung thực về mặt kỹ thuật:

1. **Prompt chứa trọn vẹn đáp án:** Hàm `make_prompt()` cung cấp cho LLM: chỉ dẫn hệ thống + `findings_catalog` (40 câu) + `impression_catalog` (các câu kết luận).
2. **Thực chất của phần `impression`:** Ở Phase 1, `impression` **chính xác 100% là một phép lọc tập con có bảo thủ (Conservative Subset Filtering)**, nhặt y nguyên các câu từ `findings` sang, **chưa phải là bản tổng hợp lâm sàng tự do (Clinical Synthesis)**.
3. **Mục tiêu nghiên cứu (Ablation Study):**
   * Phase 1 **không nhằm mục đích vượt trội hơn Rule-based** về tốc độ hay độ hay của câu văn (Template Rule-based thuần túy đã sinh được kết quả này trong $0.0001$s).
   * Phase 1 là nghiên cứu kiểm chứng tính khả thi (**Feasibility / Ablation Study**): Chứng minh rằng một mô hình mã nguồn mở (`Qwen2.5-3B-Instruct`) có thể bị "cầm cương" tuyệt đối thông qua LoRA để tuân thủ 100% định dạng JSON và catalog y tế, triệt tiêu hoàn toàn ảo giác.

---

## 4. Giao thức Huấn luyện & Cơ chế Tính Loss

* **Mô hình & Cấu hình QLoRA:** `Qwen2.5-3B-Instruct` nén 4-bit (NF4). Adapter gắn vào 7 module chiếu tuyến tính (`q, k, v, o, gate, up, down_proj`), rank $r=16, \alpha=32$.
* **Đặc tính Tokenizer (Subword BPE):**
  * Qwen sử dụng Byte-level BPE ($151.643$ tokens). Các thuật ngữ y khoa (`Pfirrmann`, `Modic`, `thoát vị đĩa đệm`...) được phân rã thành các chuỗi mảnh ghép subword.
  * Nhờ bảng mã byte UTF-8 fallback, **tỷ lệ Out-of-Vocabulary (OOV / `<UNK>`) là 0% tuyệt đối**.
* **Công thức Loss (Completion-Only Cross-Entropy Loss):**
  $$
  \mathcal{L} = - \frac{1}{|\mathcal{M}|} \sum_{t \in \mathcal{M}} \log P(x_t \mid x_{<t})
  $$

  * `TextCollator` gán nhãn `-100` cho toàn bộ phần Prompt của User $\rightarrow$ Triệt tiêu hoàn toàn gradient của prompt.
  * Loss chỉ tính trên các token thuộc phần Assistant JSON Completion ($\mathcal{M}$) và token kết thúc `<|im_end|>`.
* **Kiểm soát tràn số (`FiniteLossTrainer`):** Kiểm tra tính hữu hạn của loss tại từng bước (`torch.isfinite`). Ngắt ngay lập tức nếu xuất hiện `NaN` hoặc `Inf`.

---

## 5. Cơ chế Kiểm định & An toàn Đóng (Fail-Closed Safety)

```
[Input: ReportRequest] ──► [Planner: Catalog] ──► [Prompt nạp vào Qwen]
                                                         │
                                                         ▼
                                                [Sinh JSON Dự thảo]
                                                         │
                                                         ▼
                                              [validate_draft() Check]
                                                 │               │
                                             (Hợp lệ)         (Lỗi)
                                                 │               │
                                                 │               ▼
                                                 │      [Thử sửa lỗi (max=1)]
                                                 │               │
                                                 │          (Vẫn lỗi)
                                                 │               │
                                                 ▼               ▼
                                          [ReportResult] ◄── [Template Fallback]
```

* **Bộ kiểm tra 4 tầng (`validate_draft`):**
  1. *Cú pháp:* Parse JSON nghiêm ngặt (`strict_loads`, không chấp nhận lỗi định dạng).
  2. *Schema:* Khớp chính xác `ReportDraft`, cấm thêm key ngoài (`extra="forbid"`).
  3. *Bằng chứng:* Mỗi câu đúng 1 `evidence_id`, không trùng lặp, đủ 40 dữ kiện.
  4. *Khớp chuỗi chính xác (Verbatim Match):* Khớp từng ký tự với catalog; mọi cách diễn đạt lại (paraphrase) đều bị loại bỏ.
* **Template Fallback:** Nếu LLM lỗi cú pháp hoặc ngoại lệ, thuật toán ghép chuỗi tất định tự động kích hoạt, bảo đảm hệ thống không bao giờ crash.

---

## 6. Tại sao dùng LLM thay vì thuần Rule-based? (Rationale for Paper Discussion)

1. **Chuẩn đối chứng (Baseline):** Rule-based (`template_draft`) đóng vai trò là mốc chuẩn tối ưu (Gold Standard) để đo lường mức độ tuân thủ của LLM.
2. **Ngăn chặn bùng nổ tổ hợp (Combinatorial Explosion):** Khi mở rộng thêm các nhãn phối hợp phức tạp, số lượng nhánh `if-else` sẽ bùng nổ theo cấp số nhân. LLM sở hữu năng lực tổng quát hóa ngữ nghĩa để kết hợp câu mà không cần viết luật thủ công.
3. **Mở đường cho dữ liệu phi cấu trúc (EMR Text):** Rule-based bất lực trước văn bản tự do. Việc tích hợp LLM là bước chuẩn bị năng lực để trong tương lai hệ thống tiếp nhận thêm tiền sử bệnh án (EMR) và chỉ định lâm sàng dạng text mở.
4. **Kiến trúc Neuro-Symbolic:** Kết hợp tính linh hoạt ngôn ngữ của LLM với "chiếc lồng sắt" an toàn của Rule-based.

---

## 7. Lộ trình Mở rộng Phase 2: Gom nhóm Lâm sàng có kiểm soát (Clinical Aggregation Roadmap)

Điểm nâng cấp lớn nhất của bài báo khoa học từ Phase 1 lên Phase 2:

* **Mục tiêu:** Rút gọn phần `findings` từ 40 câu máy móc xuống **8 – 12 câu súc tích** giống phong cách thực tế của bác sĩ.
* **Nguyên tắc an toàn kép:**
  * *Gom nhóm âm tính (Negative Aggregation):* Gộp các tầng có cùng trạng thái bình thường vào 1 câu:
    > *"Tại các tầng L1/L2, L2/L3: không có phình đĩa đệm."* $\rightarrow$ `evidence_ids: ["L1/L2:disc_bulging", "L2/L3:disc_bulging"]`
    >
  * *Tách biệt dương tính (Positive Isolation):* Bất kỳ tầng nào có tổn thương bệnh lý ($v > 0$) bắt buộc phải mô tả riêng từng tầng để **triệt tiêu hoàn toàn nguy cơ nhầm tầng (cross-level contamination)**.
* **Nâng cấp Validator:** Chuyển từ so sánh chuỗi đơn lẻ sang **kiểm tra phân hoạch tập hợp (Set Partition Check)**: bảo đảm hợp các `evidence_ids` phủ kín 100% 40 dữ kiện và không có 2 câu nào giao nhau về bằng chứng.
