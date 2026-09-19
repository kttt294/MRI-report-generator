# Dữ liệu hiện có và chiến lược fine-tune report engine V2

## Kết luận đề xuất

Có thể dùng dữ liệu này cho nghiên cứu V2, nhưng chưa nên đưa nguyên cặp grading JSON → toàn văn báo cáo cũ vào fine-tune rồi coi đó là target đúng. Bước cần ưu tiên là xác định nội dung report engine có thể sinh từ tám trường JSON, tạo target tương ứng và có tập đánh giá được người có chuyên môn duyệt.

Không thể dự đoán tỷ lệ overfitting, tỷ lệ lỗi hoặc số epoch hội tụ chỉ từ EDA. Chưa có kết quả training GPU trong đánh giá này.

## Những gì đã đo và những gì chưa biết

- Có 247 ca, 1.235 dòng tầng đĩa đệm; chỉ có 230 ca đủ findings và impression tiếng Việt. Năm tầng không biến một bệnh nhân thành năm mẫu report độc lập. Fold 1 chỉ có 137 ca train đủ hai phần, 46 validation và 47 test.
- Có 131/238 ca có ít nhất một cờ heuristic grading–báo cáo ở cấp bệnh nhân. Đây không phải tỷ lệ sai nhãn hay tỷ lệ mâu thuẫn findings–impression được bác sĩ xác nhận.
- Sàng lọc riêng hai phần trên 230 ca: không phát hiện khẳng định/phủ định trái dấu ở sáu nhóm thuật ngữ xét cấp bệnh nhân. Bộ dò không bao phủ đầy đủ sai tầng, bên, mức độ và cách diễn đạt; không phát hiện không có nghĩa không tồn tại.
- Có 67 ca parser thấy giảm chiều cao trong findings nhưng không thấy phát biểu tương ứng trong impression; tương tự Modic là 7 ca. Đây là khác biệt về nội dung được nhắc, chưa phải mâu thuẫn. Impression có thể chỉ tóm tắt thông tin được chọn.
- Pfirrmann 5 và Modic III mỗi loại chỉ có 4 dòng tầng trong toàn cohort. Độ phủ bệnh hiếm hạn chế; không lấy điểm trung bình cao làm bằng chứng model xử lý tốt mọi lớp.

Nguồn số liệu: `outputs/eda_2026-09-18/eda_tables.json`, sinh bằng `scripts/build_complete_eda.py`. Bảng EDA có dấu vân tay dữ liệu và code.

## Các rủi ro khác nhau

| Rủi ro | Cơ chế | Cần đo / xử lý |
| --- | --- | --- |
| JSON không đủ thông tin cho target | Target nói bên, rễ, kích thước hoặc tổn thương khác mà input không có; model được thưởng khi đoán đúng văn phong | Chỉ giữ phát biểu có bằng chứng trong JSON, hoặc bổ sung trường input sau khi có nguồn đáng tin cậy |
| JSON và target trái nhau | Cùng sự kiện nhưng đầu vào và đầu ra bất đồng | Gắn cờ ở cấp phát biểu; duyệt, sửa hoặc bỏ phần không giải quyết được; không mặc định grading hay văn bản luôn đúng |
| Hai phần của report trái nhau | Findings và impression có thể dạy hai đầu ra không tương thích | Dùng cùng danh sách sự kiện cho cả hai phần; kiểm tra vị trí, phủ định, mức độ và phạm vi |
| Ít ca, văn bản lặp | Model có thể nhớ mẫu câu/ca thay vì dùng JSON | Split theo bệnh nhân, kiểm ảnh trùng khi có ảnh, theo dõi train/validation và đánh giá giữ lại |
| Nhãn hiếm | Model học thiên về các lớp phổ biến | Báo cáo kết quả theo trường, giá trị, tầng; không gộp/oversample mù quáng |

Training loss vẫn có thể giảm khi model ghi nhớ dữ liệu nhiễu. Công trình [Understanding deep learning requires rethinking generalization](https://arxiv.org/abs/1611.03530) minh họa khả năng fit nhãn ngẫu nhiên trong mạng phân loại ảnh; đây là cơ sở cảnh giác với loss, không phải ước lượng định lượng cho V2 này.

## Quy trình đề xuất, theo thứ tự ưu tiên

1. **Chốt phạm vi đầu ra.** Với schema tám trường hiện tại, mô tả các nhãn được cung cấp. Không coi đầu ra đó là báo cáo MRI toàn diện. Giữ null là chưa biết, không biến thành âm tính. Ánh xạ tầng và ontology còn cần xác nhận.
2. **Giữ nguyên bản gốc và tạo target dẫn xuất có truy vết.** Mỗi phát biểu được liên kết tới trường và tầng trong JSON. Có thể dùng template để tạo bản nháp cho bác sĩ duyệt, giảm công viết lại. Bất đồng thực sự cần quay lại bằng chứng ảnh/nguồn; không tự sửa nhãn để hai phía khớp nhau.
3. **Dành công duyệt cho tập đánh giá và các ca khó.** Giữ split theo bệnh nhân; kiểm tập test đủ điều kiện theo quy trình cố định rồi khóa lại. Trong train, ưu tiên nhóm A, nhãn hiếm, null và ca khác nhau về thuật ngữ; lấy thêm mẫu B/C để kiểm cờ bỏ sót. Không tự loại toàn bộ A/B hoặc coi C sạch, vì sẽ thay đổi phân bố và mất ca khó. Nếu chỉ đánh giá được một tập con, báo rõ tiêu chí và độ phủ.
4. **Chạy baseline trước.** So sánh template, model nền dùng prompt và model LoRA/QLoRA fine-tuned trên target đã duyệt. Nếu muốn đo ảnh hưởng nhiễu, thêm nhánh dùng target gốc như một ablation, trên cùng split và cùng tập đánh giá; không dùng nhánh đó làm chuẩn triển khai.
5. **Fine-tune nhỏ và theo dõi tổng quát hóa.** LoRA/QLoRA giúp giảm tài nguyên, không chữa nhãn sai. Tìm learning rate/rank trong phạm vi nhỏ bằng validation, đánh giá định kỳ, lưu checkpoint và dừng khi chất lượng validation không cải thiện. Không chọn cấu hình bằng test và không cam kết trước một số epoch là tối ưu. Khi tài nguyên cho phép, lặp nhiều seed/fold và báo độ biến thiên.
6. **Đánh giá đúng mục tiêu.** Đếm sai sự kiện so với JSON, tự thêm chi tiết không có bằng chứng, bỏ sót sự kiện bắt buộc, sai tầng, biến null thành âm tính và mâu thuẫn hai phần. Đánh giá độ đúng trên MRI là một phép đánh giá khác, cần ảnh và chuyên gia. BLEU/ROUGE/loss chỉ bổ trợ. [Nghiên cứu về factual correctness của báo cáo radiology](https://aclanthology.org/2022.findings-emnlp.319/) hỗ trợ việc đánh giá ngữ nghĩa; metric phát triển cho X-quang ngực không mặc nhiên hợp lệ cho MRI cột sống tiếng Việt.

Sinh nhiều cách diễn đạt từ cùng JSON có thể hỗ trợ học văn phong, nhưng không tạo thêm bệnh nhân hay đa dạng bệnh học. Mọi biến thể phải ở cùng split với ca nguồn và được kiểm tính đúng của sự kiện. Dữ liệu tổng hợp chỉ sinh từ phần train, không dùng target test để viết prompt hay sinh mẫu. Không dùng LLM tự chấm nhãn làm ground truth thay bác sĩ.

## Sinh findings và impression trong một lần

Một lần sinh cả hai phần vẫn là thiết kế có thể thử. Hai phần nên dùng chung một danh sách sự kiện từ JSON, đầu ra có cấu trúc và có bước kiểm sau sinh. Tách thành hai lần gọi không tự động giải quyết lỗi: impression có thể kế thừa lỗi của findings. Không có cơ sở để đưa ra tỷ lệ lỗi cho model/prompt chưa chạy; phải đo trên tập đã duyệt.

Trong repo hiện tại, `src/data/v2_dataset.py` đã yêu cầu target `human_reviewed_scoped`, trạng thái accepted, người duyệt và hash khớp input. `src/report/validate.py` hiện chỉ chấp nhận đúng câu trong catalog kiểm soát, chưa cho diễn đạt tự do. Với ràng buộc này, template là baseline rất mạnh và fine-tune có thể không đem lại thêm lợi ích. Nếu muốn nghiên cứu khả năng viết báo cáo tự nhiên, cần thiết kế một nhánh riêng với target tương ứng và phép kiểm ngữ nghĩa đã được đánh giá; đây là đề xuất, chưa sửa chính sách hiện có.
