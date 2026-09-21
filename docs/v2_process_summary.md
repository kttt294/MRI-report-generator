# BẢN TÓM TẮT QUY TRÌNH HOẠT ĐỘNG & HUẤN LUYỆN MÔ HÌNH V2 (TEXT-TO-REPORT)
*(Phiên bản đã chuẩn hóa và đính chính theo đối chiếu thực tế mã nguồn)*

---

> [!IMPORTANT]
> **Nguyên tắc cốt lõi của hệ thống:**
> Template fallback giúp trả về nội dung bám theo grading và policy đã định nghĩa khi LLM không đạt kiểm tra. Nó không bảo đảm grading đúng trên MRI hoặc báo cáo chính xác về mặt lâm sàng; các trường hợp cần duyệt vẫn phải được ghi rõ.

---

## 1. Bản chất & Phạm vi thực tế của V2 (Scope & Objective)
* **Vị trí & Nhiệm vụ:** V2 hiện tại là bài toán **sinh văn bản có kiểm soát theo danh mục (Catalog-Constrained Generation)** theo policy `grading-verbatim/1.0`. V2 không phải là mô hình viết báo cáo tự do, mà chủ yếu kiểm tra khả năng **xuất đúng nội dung theo catalog và đúng cấu trúc JSON**.
* **Đầu vào (Input - `ReportRequest`):**
  * Đúng 5 tầng đĩa đệm thắt lưng: `L1/L2`, `L2/L3`, `L3/L4`, `L4/L5`, `L5/S1`.
  * 8 trường bệnh lý cho mỗi tầng: `pfirrmann_grade` (1-5), `modic` (0-3), `disc_herniation` (0/1), `disc_bulging` (0/1), `disc_narrowing` (0/1), `spondylolisthesis` (0/1), `up_endplate` (0/1), `low_endplate` (0/1).
  * Phân biệt rõ ràng giữa giá trị đã ghi nhận (`observed` kèm số nguyên) và các trạng thái chưa chắc chắn (`missing`, `not_assessed`, `uncertain` kèm giá trị `null`).
  * `ReportRequest` lưu vết thêm phiên bản ontology, provenance và đánh giá chất lượng (QC: `level_mapping`, `ontology_review`). 
  * **Bộ lọc an toàn cho Prompt:** Toạ độ voxel/không gian, nhân khẩu học (tuổi, giới tính), mã ca bệnh (`case_id`), báo cáo gốc của bác sĩ và thông tin split đều **hoàn toàn bị loại bỏ khỏi prompt**.
* **Prompt đã chứa gần như toàn bộ đáp án:**
  * Hàm `make_prompt()` đưa trực tiếp vào prompt: chỉ dẫn hệ thống + `findings_catalog` (40 câu) + `impression_catalog` (các câu chọn lọc).
* **Đầu ra của hệ thống:**
  * **Bản dự thảo nội bộ (`ReportDraft`):** Gồm 2 key `findings` (40 câu mang `evidence_id`) và `impression` (chọn lại nguyên văn các câu từ `findings` có `pfirrmann_grade`, nhãn dương tính `value > 0`, hoặc trạng thái chưa xác định; chưa phải kết luận được tổng hợp hay diễn đạt cô đọng mới).
  * **Kết quả trả về chính thức (`ReportResult`):** Chứa văn bản hoàn chỉnh đã render, trạng thái (`ok`, `fallback`, `needs_review`, hoặc `failed`), danh sách hạn chế (`limitations`), lịch sử các lần thử (`attempts`), kết quả kiểm tra (`validation`), và các mã băm SHA-256 phục vụ truy vết.

---

## 2. Nguồn gốc Dữ liệu & Xử lý Target Huấn luyện (Data & Review Status)
* **Quy trình duyệt Target (`build_v2_targets.py`):**
  * Khi khởi tạo, script chỉ tạo bản nháp với trạng thái `review_status="pending"` và `target_kind="template_draft_not_clinical_ground_truth"`.
  * Chỉ khi có file duyệt thực tế với danh tính người duyệt (`reviewed_by`), mã băm `input_sha256` khớp tuyệt đối và target vượt qua bộ kiểm tra thì bản ghi mới được mang trạng thái `review_status="accepted"` và `target_kind="human_reviewed_scoped"`.
  * Việc chạy template trên 247 ca không đồng nghĩa là đã duyệt xong 247 target lâm sàng.
* **Về nhóm 43 ca trong báo cáo Audit:**
  * Đây là 43/238 ca được bộ quy tắc heuristic phân loại vào **Nhóm ưu tiên kiểm toán A** (gồm nghi vấn lệch nhãn giữa CSV và báo cáo chữ, HOẶC có mâu thuẫn nội bộ trong chính báo cáo). Đây là danh sách ưu tiên rà soát, **chưa phải 43 lỗi sai đã được bác sĩ xác nhận**.
* **Thuật ngữ & Bảng nhãn (`LABELS`):**
  * `LABELS` trong `src/report/planner.py` là bảng quy ước thuật ngữ nội bộ được định nghĩa trong code dự án.
  * Policy y tế trong mã nguồn có tên định danh chính thức là **`grading-verbatim/1.0`** (R3 là tên nhánh phát triển / thử nghiệm).
* **Vì sao không dùng báo cáo tự do gốc cho V2:**
  * Báo cáo tự do của bác sĩ mô tả toàn diện (thân đốt sống, tủy, cơ, rễ, khớp cùng chậu), trong khi V2 chỉ nhận 8 nhãn đĩa đệm. Ép LLM học từ báo cáo tự do khi input chỉ có 8 nhãn sẽ khiến mô hình bị phạt loss sai và học thói quen tự suy diễn ngoài dữ kiện.

---

## 3. Cấu hình Huấn luyện & Masking (Training Setup)
* **Mô hình & Kiến trúc:** 
  * Cấu hình triển khai mặc định: `Qwen/Qwen2.5-3B-Instruct` với QLoRA 4-bit (NF4, `compute_dtype=auto/bfloat16/float16`).
  * LoRA config: `r=16`, `lora_alpha=32`, `lora_dropout=0.05`, áp dụng trên 7 module tuyến tính (`q, k, v, o, gate, up, down_proj`).
* **Cơ chế Masking Token (`TextCollator`):**
  * Toàn bộ phần Prompt người dùng và padding được gán `labels = -100`.
  * Hàm mất mát (Cross-Entropy Loss) được tính trên phần completion của trợ lý (Assistant JSON) và các token kết thúc lượt hội thoại theo chat template.
* **Chiến lược Fold:**
  * Hệ thống hỗ trợ nạp và chạy trên **một fold được chỉ định** (từ Fold 1 đến Fold 5) trong mỗi lượt chạy (`config["fold"]`).
  * Pipeline chưa tự động chạy liên hoàn cả 5 fold và tổng hợp metric; việc chạy k-fold do người dùng điều phối qua cấu hình từng run.

---

## 4. Kiểm tra, Sửa chữa & Giới hạn Bảo đảm (Validation & Safety)
* **Bộ kiểm tra (`validate_draft`):**
  * Kiểm tra cú pháp JSON hợp lệ.
  * Kiểm tra độ phủ bằng chứng: Mỗi câu phải có đúng 1 `evidence_id`, không được trùng lặp, phải bao phủ đủ các dữ kiện yêu cầu.
  * **Khớp chuỗi chính xác (Exact String Match):** Validator yêu cầu câu chữ phải khớp từng ký tự với catalog; **kể cả cách diễn đạt khác nhưng cùng nghĩa về lâm sàng cũng bị tính là lỗi**.
  * **Lưu ý về thứ tự:** Validator hiện tại **không kiểm tra thứ tự từ L1/L2 đến L5/S1**; thứ tự giải phẫu này do planner và template sắp xếp sẵn trong catalog.
* **Cơ chế Repair & Fallback:**
  * Mặc định cho phép tự sửa lỗi tối đa 1 lần (`max_repairs=1`). Nếu LLM trả về lỗi cú pháp hoặc sai câu chữ, hệ thống nối thông báo lỗi vào prompt để LLM sinh lại.
  * Nếu sau lượt sửa vẫn không đạt (hoặc nếu xảy ra ngoại lệ backend) và cờ `fallback=True`, hệ thống kích hoạt `template_draft` (chế độ ghép chuỗi tất định).
* **Giới hạn bảo đảm:**
  * Hệ thống kiểm tra độ khớp logic giữa văn bản sinh ra và bảng nhãn/catalog đầu vào. **Hệ thống không xác minh được nhãn grading có đúng trên ảnh MRI hay không**.
  * Fallback có thể tắt (`fallback=False`); input sai contract sẽ báo lỗi; nếu bật `require_verified=True` mà dữ liệu đầu vào chưa đạt QC verified thì pipeline sẽ trả về trạng thái `failed` chứ không tự động fallback.

---

## 5. Đính chính về Mô hình Thị giác V1 (Multimodal Context)
* Báo cáo văn bản tự do của bác sĩ sẽ được dùng trong pipeline V1 (được trích xuất qua CSV và lọc các ca đạt chuẩn).
* **Mô hình V1 nhận dữ liệu 2D:** V1 trích xuất các lát cắt **2D** từ thể tích ảnh NIfTI 3D (mặc định là lát cắt sagittal chính giữa - mid-sagittal slice), **không phải nạp trực tiếp toàn bộ ma trận thể tích 3D vào mạng neural**.

---

## 6. Bảng tổng hợp các điểm đính chính kỹ thuật

| Thành phần | Hiểu nhầm / Diễn giải quá mạnh trước đây | Sự thật trong mã nguồn |
|---|---|---|
| **Bản chất V2** | LLM học viết báo cáo y khoa tự do | Thử nghiệm khả năng trích xuất catalog và xuất đúng cấu trúc JSON; template thuần đã sinh được kết quả này mà không cần LLM. |
| **Bảo đảm an toàn** | "Zero Hallucination", "Chính xác 100% lâm sàng" | Chỉ bảo đảm khớp chính xác văn bản với catalog; không bảo đảm grading đúng trên ảnh MRI. |
| **Duyệt Target** | 247 ca đã được đóng dấu accepted | Mặc định là `pending`. Phải có file duyệt thực tế có `reviewed_by` và khớp SHA-256 mới xuất được `accepted`. |
| **Báo cáo Audit** | 43 ca sai lệch nhãn do bác sĩ xác nhận | 43 ca thuộc nhóm ưu tiên kiểm toán A (nghi vấn lệch hoặc mâu thuẫn nội bộ theo bộ luật heuristic). |
| **Tên Policy** | Medical Policy R3 | Tên chính thức là `grading-verbatim/1.0`; R3 là tên nhánh thử nghiệm. |
| **Kiểm tra thứ tự** | Validator bắt buộc thứ tự L1/L2 đến L5/S1 | Validator chỉ kiểm tra đủ/không trùng evidence và khớp câu chữ; thứ tự do Planner/Template định hình. |
| **Đầu vào của V1** | Đọc trực tiếp ảnh MRI thể tích 3D | Trích xuất các lát cắt **2D** từ file NIfTI (mặc định lát giữa sagittal). |
