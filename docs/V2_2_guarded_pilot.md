# V2-2: thử giới hạn độ dài, chống lặp, sinh lại và gộp tầng

Thử nghiệm chỉ dùng fold 1 validation của bộ annotations private. Adapter giữ nguyên từ V2-2; fold 1 test đã được báo cáo riêng trong `V2_2_test_results.md`. Các văn bản bệnh nhân và output thô nằm trong Kaggle private và thư mục `output/` bị Git bỏ qua.

## Chẩn đoán lỗi của adapter hiện tại

V2-2 đang fine-tune trực tiếp từ năm tầng × tám nhãn grading sang hai chuỗi tự do `findings` và `impression` gốc của bác sĩ. Prompt huấn luyện/test gốc chỉ yêu cầu JSON hai phần, không có giới hạn ký tự hoặc quy tắc gộp câu. Token kết thúc `<|im_end|>` có nằm trong nhãn huấn luyện; không tìm thấy lỗi bỏ quên token kết thúc ở collator. Trên 47 ca test đã chạy, 40 ca chạm 1.536 token mà chưa đến key `impression`; trung bình 94,4% cụm năm từ của các ca lỗi là cụm lặp. Đây là lỗi lặp khi sinh tự do. Việc báo cáo bác sĩ chứa thông tin ngoài tám nhãn đầu vào là một nguyên nhân khả dĩ khiến bài toán học không xác định, nhưng chưa có thí nghiệm tách riêng để quy trách nhiệm nhân quả.

## Những cách đã thử trên validation

| Cách | Kết quả quan sát | Kết luận |
| --- | --- | --- |
| Một lần sinh lại với prompt yêu cầu `findings` ≤ 1.200 ký tự, `impression` ≤ 650 ký tự | Trong 2 ca: 1 đạt ngay; ca lỗi vẫn sai JSON sau retry 1.024 token. | Chỉ dẫn độ dài trong prompt không đảm bảo mô hình tuân thủ. |
| Prompt giới hạn + `repetition_penalty=1.15`, `no_repeat_ngram_size=6` | 2/2 JSON hợp lệ, không chạm giới hạn token, nhưng văn bản lẫn ngôn ngữ và có nhận định/tầng sai rõ ràng. | Không dùng cấu hình này làm báo cáo; đúng định dạng không đồng nghĩa đúng nội dung. |
| Prompt giới hạn và gộp có điều kiện + `no_repeat_ngram_size=6`, không thêm repetition penalty; kiểm tra rồi tối đa một lần sinh lại | Pilot 2 ca: 0/2 đạt bộ kiểm tra, cả hai bị chặn vì sai tầng hoặc nội dung ngoài grading. | Bộ kiểm tra phát hiện lỗi, nhưng chưa chữa được adapter. |
| Cùng cấu hình trên, chạy trọn 10 ca validation với fallback xác định | 0 ca đạt ở lượt đầu; 1 ca qua kiểm tra bề mặt sau retry; 9 ca chuyển fallback. Trung bình 116,15 giây/ca tính cả các lượt LLM. | Ca qua kiểm tra bề mặt vẫn có văn bản lủng củng và nhận định khó gắn với nhãn. Không tự động chấp nhận bất kỳ đầu ra LLM nào. |

Notebook private: https://www.kaggle.com/code/kieuthithutrang/mri-v2-2-bounded-validation và https://www.kaggle.com/code/kieuthithutrang/mri-v2-2-guarded-validation.

## Cách triển khai có kiểm tra

`src/report/v2_2_quality.py` thêm prompt giới hạn độ dài và quy tắc gộp: chỉ gộp nhiều tầng khi cùng loại tổn thương, cùng giá trị/mức độ, cùng trạng thái chắc chắn và thuộc tính liên quan; câu phải liệt kê rõ từng tầng. Những bệnh khác nhau ở cùng một tầng có thể đặt chung một câu nếu mỗi bệnh gắn rõ với tầng. Nếu quan hệ không rõ, tách câu.

`src/report/v2_2_guarded.py` kiểm tra JSON đủ hai phần, giới hạn 1.200/650 ký tự, vòng lặp rõ, ký tự CJK, tầng ngoài năm tầng chuẩn và một số nội dung chắc chắn ngoài phạm vi tám nhãn. Mô hình được sinh lại tối đa **một lần** với lỗi cụ thể trong prompt. Bộ kiểm tra này không thể phát hiện mọi nhận định sai hoặc xác nhận câu gộp đủ rõ về lâm sàng. Sau khi xem ca qua kiểm tra bề mặt ở lượt 10 ca, code được sửa để **mọi đầu ra LLM chỉ là candidate cần bác sĩ duyệt**; báo cáo tự động luôn lấy từ fallback có cấu trúc cho tới khi có bộ đánh giá nội dung được hiệu chuẩn.

Cho đầu ra tự động, `src/report/v2_2_grouped_template.py` tạo báo cáo dự phòng xác định từ đúng tám nhãn grading, ghi trạng thái `fallback` riêng và giữ nguyên mọi output LLM để audit. Mỗi trong 40 dữ kiện có mặt đúng một lần trong findings; các tầng chỉ được gộp nếu cùng field, status và value. Những dữ kiện còn đơn lẻ cùng tầng được viết chung dưới tên tầng đó. Impression chỉ giữ phân độ Pfirrmann, nhãn dương tính và nhãn chưa xác định theo chính sách Phase 1. Đây là một báo cáo **giới hạn theo grading**, không phải báo cáo MRI toàn diện.

Lượt 10 ca Kaggle chạy code trước thay đổi fail-closed cuối cùng: 9 status `fallback`, 1 status `ok` về mặt kiểm tra bề mặt. Đối chiếu lại **cùng 10 dự đoán đã lưu** dưới chính sách cuối: 10/10 báo cáo tự động sẽ lấy fallback, còn 1 bản nháp LLM được giữ riêng cho bác sĩ xem. Đây là đổi cách chấp nhận kết quả, không phải lượt train hoặc inference mới và không chứng minh fallback đúng trên ảnh.

Trên đúng 183 ca train/validation của nguồn đã khóa, template dự phòng qua tất cả kiểm tra tự động; findings dài nhất 886 ký tự, impression dài nhất 466 ký tự, không ca nào vượt ngưỡng 1.200/650. Findings còn trung bình 10,25 câu (khoảng 8–15) thay cho 40 câu tách từng nhãn; impression trung bình 3,31 câu. Kết quả này kiểm tra tính nhất quán với JSON đầu vào, không kiểm chứng grading có đúng trên ảnh MRI hay không.

Không nên coi adapter V2-2 hiện tại đã được sửa chỉ vì có fallback. Hướng nghiên cứu tiếp theo là thiết kế lại target sinh findings từ dữ kiện được hỗ trợ, đánh giá riêng khả năng gộp và sai lệch nội dung bằng bác sĩ, rồi mới so với impression có cấu trúc và các metric NLP trên một protocol mới.
