Kế hoạch V2 Phase 2: Nâng cấp mô hình sinh báo cáo

Điểm nghẽn của Phase 1 và lý do cần lên Phase 2

Trong giai đoạn đầu (Phase 1), mô hình ngôn ngữ chỉ được dùng để chuyển đổi các kết quả phân loại từ ảnh MRI sang văn bản theo các mẫu câu có sẵn. Cách tiếp cận này bộc lộ ba điểm nghẽn lớn:

Thứ nhất, văn bản sinh ra rất máy móc và vụn vặt. Với mỗi phát hiện ở từng tầng đĩa đệm, hệ thống lại viết riêng một câu độc lập lặp đi lặp lại. Một báo cáo có thể dài hơn chục câu đơn lẻ, rời rạc, không giống với cách hành văn tổng hợp, súc tích và chuyên nghiệp của bác sĩ.

Thứ hai, chưa phát huy được giá trị thực sự của llm. Việc dịch từng nhãn phân loại sang từng câu định sẵn thực chất chỉ cần một đoạn mã quy tắc cơ bản (rule-based) là làm được trọn vẹn, chưa xứng tầm với việc phải vận hành một llm.

Thứ ba, hệ thống chưa biết tiếp nhận chỉ dẫn thực tế. Trong thực tế lâm sàng, bác sĩ đọc phim thường gộp các tầng có cùng tổn thương vào một câu ngắn gọn và có những ghi chú riêng. Giai đoạn 1 hoàn toàn bất lực trước các yêu cầu tổng hợp hoặc chỉ dẫn tùy biến này.

Vì vậy, Phase 2 sẽ được triển khai nhằm nâng cấp mô hình ngôn ngữ thành một trợ lý thư ký y khoa. Mô hình sẽ biết tổng hợp các tầng bệnh lý tương đồng thành các câu văn gãy gọn, mạch lạc, tiếp nhận linh hoạt các chỉ dẫn của bác sĩ, nhưng vẫn giữ nguyên tắc an toàn cốt lõi là bám sát bằng chứng hình ảnh và tuyệt đối không suy diễn thêm bệnh ngoài nhãn.

Mục tiêu cốt lõi: Xây dựng mô hình ngôn ngữ có khả năng kết hợp dữ liệu thị giác từ ảnh chụp cùng chỉ dẫn riêng của bác sĩ (ví dụ đưa thêm 1 vài biểu hiện mà bệnh nhân cảm nhận được vào làm input cho llm) để viết phần mô tả và kết luận ngắn gọn, tự nhiên. E thấy hướng này khá hay nhưng hiện tại mình chưa có data để finetune theo hướng này, hoặc do em chưa nghĩ ra cách tận dụng data hiện tại.

Đầu vào và đầu ra:

Đầu vào của hệ thống gồm các chỉ số bệnh lý do mô hình thị giác phân tích, trạng thái ghi nhận xem dữ liệu có bị thiếu hay nghi ngờ hay không, cùng với đoạn ghi chú hoặc chỉ dẫn nghiệp vụ của bác sĩ nếu có. Hệ thống sẽ phân biệt rõ đâu là yêu cầu về cách trình bày, đâu là thông tin bệnh sử lâm sàng, và đâu là nhận định y khoa mà bác sĩ muốn thêm hoặc sửa. Tuyệt đối không mớm sẵn câu chữ đáp án vào câu lệnh đầu vào.

Đầu ra của hệ thống là phần mô tả chi tiết, phần kết luận cô đọng, danh sách liên kết chỉ rõ mỗi câu dựa trên bằng chứng nào, và các cảnh báo nếu có điểm cần bác sĩ xác nhận lại. Hệ thống được phép tự do gộp các câu có cùng nội dung chứ không bị ép số lượng câu cứng nhắc như trước.

Giới hạn an toàn trong suy luận

Mô hình được phép tổng hợp, tóm tắt và sắp xếp mức độ ưu tiên của các thông tin, nhưng không được tự ý bịa thêm tổn thương hoặc biến các triệu chứng lâm sàng thành tổn thương nhìn thấy trên phim chụp. Nguồn thông tin từ mô hình phân tích ảnh và từ ghi chú của bác sĩ phải được quản lý riêng biệt. Nếu hai nguồn này mâu thuẫn nhau, hệ thống phải chủ động đưa ra yêu cầu nhờ bác sĩ xác nhận lại, thay vì tự ý âm thầm đè thông tin lên nhau. Dữ liệu chưa quan sát thấy sẽ không được tự ý coi là bình thường, và mọi thông số phân độ phải được diễn giải chính xác theo đúng từng tiêu chuẩn y khoa.

Dữ liệu huấn luyện: Xây dựng bộ dữ liệu mẫu gồm các thông số phân tích ảnh, chỉ dẫn thực tế của bác sĩ và báo cáo mẫu chuẩn đã được bác sĩ chuyên khoa phê duyệt. Mọi dữ liệu do máy tự sinh ra để mở rộng tập học bắt buộc phải đánh dấu nguồn rõ ràng, không được phép dán nhãn là dữ liệu đã qua bác sĩ duyệt. Không tạo chỉ dẫn giả bằng cách chép ngược lại câu chữ từ báo cáo đích. Việc chia tập huấn luyện và tập kiểm tra luôn được thực hiện theo từng bệnh nhân, bảo đảm mọi biến thể của một ca bệnh đều nằm chung trong một tập để tránh rò rỉ dữ liệu.

Quy trình triển khai kỹ thuật: Hệ thống sẽ mở rộng cấu trúc dữ liệu đầu vào, câu lệnh hướng dẫn, bộ nạp dữ liệu và lớp kiểm định tự động để phục vụ bài toán tổng hợp mới. Bỏ cơ chế kiểm tra bắt buộc phải trùng khớp từng chữ với mẫu câu cũ. Thay vào đó, bộ kiểm định sẽ tập trung soát lỗi cấu trúc dữ liệu, nguồn gốc bằng chứng, độ chính xác của vị trí tầng đốt sống và các thông số bệnh lý. Bộ kiểm định tự động chỉ đóng vai trò bảo vệ an toàn kỹ thuật, không thể thay thế cho đánh giá chuyên môn của bác sĩ. Kết quả sinh trực tiếp từ mô hình và kết quả sau khi qua lớp xử lý an toàn sẽ được lưu trữ riêng biệt để tiện theo dõi.

Thử nghiệm và huấn luyện: Thử nghiệm phương pháp hướng dẫn bằng câu lệnh trên tập kiểm định trước. Đặt lên bàn cân so sánh giữa bốn phương án: dùng mã quy tắc thông thường, mô hình ngôn ngữ chỉ nhận thông số ảnh, mô hình nhận thêm chỉ dẫn bác sĩ, và mô hình được tinh chỉnh chuyên sâu cho bài toán này. Chỉ tiến hành tinh chỉnh sâu khi nhận thấy mô hình gốc lặp lại các lỗi cụ thể và đã có sẵn dữ liệu chuẩn phù hợp. Thử nghiệm cả mô hình gốc lẫn mô hình kế thừa từ Giai đoạn 1, quyết định lựa chọn dựa trên hiệu quả thực tế trên tập kiểm định. Khi bắt đầu giai đoạn mới, các thông số tối ưu phải được cài đặt lại từ đầu, không tận dụng trạng thái dang dở của giai đoạn trước. Chạy kiểm tra kỹ thuật nhanh trước khi huấn luyện chính thức, áp dụng cơ chế lưu lại điểm số tốt nhất và dừng sớm khi mô hình không còn tiến bộ, không bắt buộc phải chạy đủ số lượt học định sẵn.

Tiêu chí đánh giá: Đo lường độ tương đồng văn bản qua các chỉ số ngôn ngữ phổ biến, khả năng tuân thủ đúng chỉ dẫn của bác sĩ, tỷ lệ sót hoặc sai lệch dữ kiện, mức độ nhầm lẫn tầng đốt sống, các nhận định thiếu căn cứ, khả năng phát hiện mâu thuẫn, số lần phải kích hoạt cơ chế bảo vệ an toàn, độ dài văn bản và tốc độ sinh phản hồi. Các chuyên gia y tế sẽ trực tiếp thẩm định các lỗi liên quan đến ngữ nghĩa và chuyên môn. Kết quả trực tiếp từ mô hình và kết quả sau lớp bảo vệ an toàn được chấm điểm độc lập. Toàn bộ cấu hình phải được chốt cứng trước khi bước vào bài thi cuối cùng trên tập kiểm tra độc lập, tuyệt đối không dùng tập kiểm tra này để tinh chỉnh hay chọn mô hình.
