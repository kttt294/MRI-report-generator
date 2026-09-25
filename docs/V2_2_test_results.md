# V2-2: đánh giá lần đầu trên fold 1 test

Notebook Kaggle private: https://www.kaggle.com/code/kieuthithutrang/mri-v2-2-evaluation

Lượt chạy `v2-2-fold1-test-01` đã hoàn thành trên 47 ca có đủ `findings` và `impression` gốc. Notebook ghim code tại commit `6434020`, dùng adapter SHA-256 `2f81abc5d9dc0408b7d045515987737ba9a85a39b9e1b55e58762bc0e2bf4353` và protocol SHA-256 `b9be293cb37b4b87975fcce47185be7ec60aa2a6d1b44bc5f9f0b217208618db`. File `receipt.json` khớp các SHA và ghi đúng 47 ca; `predictions.jsonl` có 47 ID duy nhất. Kết quả chi tiết nằm trong Output của notebook và bản tải xuống cục bộ tại `output/v2_2_eval_test/runs/v2-2-fold1-test-01/` (không đưa báo cáo bệnh nhân lên GitHub).

| Chỉ số | Kết quả |
| --- | ---: |
| JSON hợp lệ, đủ hai phần | 7/47 (14,89%) |
| JSON lỗi | 40/47 (85,11%) |
| Chạm giới hạn 1.536 token | 40/47 |
| BLEU-4, thang 0–100 | 0,000711 |
| ROUGE-1 F1 | 0,086681 |
| ROUGE-L F1 | 0,052906 |
| BERTScore F1 | 0,107997 |
| Thời gian sinh trung bình | 139,23 giây/ca |

Protocol dùng đúng một lượt greedy generation mỗi ca, không retry hay sửa văn bản sau sinh. Với ca JSON lỗi, BLEU/ROUGE chấm chuỗi rỗng và BERTScore được tính bằng 0; do đó điểm trên là kết quả toàn bộ 47 ca, không phải điểm chỉ trên 7 ca hợp lệ. BLEU-4 dùng `sacrebleu` với `tokenize=none`; ROUGE dùng `rouge_score` không stemming; BERTScore dùng `bert-base-multilingual-cased` không rescale. Những cấu hình này cần được giữ nguyên khi so sánh.

Ở cả 40 ca lỗi, output bắt đầu bằng JSON và có key `findings`, nhưng chưa đến key `impression` trước khi hết 1.536 token. Đây là bằng chứng cụ thể của lỗi kéo dài phần `findings`, không phải chỉ lỗi định dạng của phần `impression`. Bảy ca hợp lệ sinh trung bình 27,8 giây/ca; 40 ca lỗi sinh trung bình 158,7 giây/ca. Tăng giới hạn token hoặc thêm một lượt gọi lại có thể làm tăng chi phí mà chưa chắc sửa được lỗi. Pilot retry trên 2 ca validation chỉ chấp nhận 1 ca ở lượt đầu; ca còn lại vẫn bị từ chối sau một lần retry với chỉ dẫn ngắn hơn.

Kết quả hiện tại chưa đủ để sử dụng V2-2 làm report engine. Điểm trùng văn bản không kiểm chứng độ đúng lâm sàng, mức độ bỏ sót hoặc nhận định thiếu bằng chứng; bảng `human_review.csv` đã được tạo để bác sĩ điền đánh giá. Đầu vào chỉ gồm 8 nhãn grading mỗi tầng nên không thể suy ra chắc chắn mọi nội dung của báo cáo gốc. Fold 1 test đã từng được xem ở Phase 1, vì vậy đây là phép so sánh thăm dò, không phải test hoàn toàn độc lập cho toàn bộ quá trình nghiên cứu.

Hướng thử tiếp theo: huấn luyện một mục tiêu chỉ sinh `findings` với giới hạn độ dài, rồi tạo `impression` từ dữ kiện có cấu trúc đã đối chiếu với grading; kiểm tra riêng tính hợp lệ, tính đầy đủ, lỗi tầng đốt sống và nhận định ngoài căn cứ trên validation trước khi chốt một protocol test mới. Không nên chỉ lấy regex chép các câu có chữ chỉ bệnh, vì impression của bác sĩ có thể gộp tầng, diễn giải mức độ quan trọng và không lặp nguyên văn findings.
