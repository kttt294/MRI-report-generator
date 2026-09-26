# So sánh V2-2 đã fine-tune với Qwen3-VL-4B-Instruct

Đây là so sánh hai hệ thống: Qwen2.5-3B-Instruct + adapter V2-2 đã học từ
báo cáo bác sĩ, và Qwen3-VL-4B-Instruct chưa fine-tune trên dữ liệu dự án.
Qwen3-VL chỉ nhận văn bản grading, không nhận ảnh MRI. Khác model nền nên
kết quả không tách riêng được tác động của fine-tune.

## Điều kiện cố định

- Cùng 47 ca fold 1 test có đủ findings và impression gốc.
- Cùng nội dung user prompt và grading; dùng chat template/tokenizer riêng của model.
- Một lần sinh, greedy, batch size 1, tối đa 1.536 token; không retry, sửa JSON hay fallback.
- `repetition_penalty=1.05`: giá trị lượt cũ kế thừa từ generation_config của Qwen2.5.
  Đây là phép so sánh dưới cấu hình cũ, không phải cấu hình sampling khuyến nghị của Qwen3-VL.
- Một GPU, NF4 với tính toán float16; bản mới dùng SDPA phù hợp T4.
- BLEU-4 (tokenize=none), ROUGE-1 F1, ROUGE-L F1, BERTScore F1
  (bert-base-multilingual-cased, không rescale). JSON lỗi chấm chuỗi rỗng với
  BLEU/ROUGE và 0 với BERTScore, giữ tất cả ca trong mẫu số.

Script dùng lại dự đoán cũ, kiểm tra SHA-256 của dữ liệu, dự đoán, metrics và
receipt trước khi chạy. Cả hai bộ dự đoán được chấm lại trong cùng runtime mới;
metrics cũ được giữ riêng để đối chiếu. Mã nguồn tạo prompt đã được đối chiếu
với commit 6434020 của lượt test cũ; manifest khóa từng input, target và prompt.
Thông số và phiên bản model cố định trong `configs/v2_2_qwen3_comparison.json`.

Thời gian Qwen2.5 là số đo lịch sử. Phiên bản Transformers mới hơn là cần thiết
để nạp Qwen3-VL, nên thời gian không phải benchmark phần cứng/phần mềm hoàn toàn
đồng nhất. Fold 1 test đã được xem trước đây; kết quả mang tính thăm dò.
Đúng JSON và điểm trùng văn bản không chứng minh đúng lâm sàng.

## Chạy trên Kaggle

Import `notebooks/Kaggle_V2_2_Qwen3_Comparison.ipynb` vào notebook private,
bật Internet và GPU T4. Trong Input, thêm Output của notebook private
`kieuthithutrang/mri-v2-2-evaluation` có lượt `v2-2-fold1-test-01`.
Output này cần có predictions.jsonl, metrics.json, receipt.json và thư mục
derived/dataset/ từ lần test đó. Không cần tải ảnh hoặc nạp lại adapter.
Script nhận đúng phiên bản bằng hash; nếu dữ liệu không đúng, dừng trước khi tải model.

Notebook tải code public theo commit đã ghim, cài requirements riêng cho
Qwen3 và gọi một script. Không cần chép logic inference vào các cell.

Ước lượng ban đầu: khoảng 1–3 giờ inference trên T4, tùy độ dài sinh;
chưa phải số đo Qwen3 trên dataset này. Tải model/cài môi trường/chấm metrics
là thời gian bổ sung. Lượt Qwen2.5 cũ mất 109,06 phút inference cho 47 ca;
tái sử dụng dự đoán giúp tránh chạy lại phần này. Log in từng ca và ước lượng
thời gian còn lại từ trung bình các ca đã hoàn thành, không thay đổi protocol
theo kết quả trên test.

## Kết quả lưu khi hoàn thành

Trong `/kaggle/working/runs/v2-2-qwen3-comparison-01/`:

- comparison.csv/json: bảng metrics của hai hệ thống.
- human_review.csv: tham chiếu, hai đầu ra đã parse, cả hai văn bản thô, trạng thái,
  thời gian; các cột nhận xét chuyên môn để trống cho người đánh giá.
- predictions.jsonl, per_patient.jsonl: đầu ra Qwen3 theo từng ca.
- inputs.jsonl, prompt_manifest.jsonl, receipt.json: dữ liệu tái lập và cấu hình thực tế.
- baseline_original_metrics.json: điểm cũ trước khi chấm lại trong runtime mới.
- progress.json: số ca đã sinh và thời gian còn lại dự kiến, chưa phải kết quả cuối.

Các file chứa văn bản bệnh nhân chỉ lưu trong Kaggle private và thư mục output/
bị Git bỏ qua. Repo public chỉ chứa code, notebook không có output và protocol.
