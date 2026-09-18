# Kế hoạch triển khai V1 và V2 Report Engine

Ngày lập: 17/09/2026; cập nhật hướng Kaggle: 18/09/2026. Trạng thái: **đã triển khai code theo hướng A và luồng Kaggle; đã kiểm thử CPU, ETL và template trên cohort local. Chưa nghiệm thu GPU/ảnh thật/LLM trên Kaggle.** Xem [bảng nghiệm thu và lệnh chạy](implementation_status.md). Những mục “chưa có” ở phần hiện trạng dưới đây mô tả thời điểm lập kế hoạch, không thay cho trạng thái phát hành.

Phạm vi: sửa V1 làm baseline có thể tái lập; xây report engine V2 từ dữ liệu grading hiện có. Chưa xây vision engine, chưa huấn luyện calibration, chưa thay nhãn của hai bác sĩ.

## 1. Quyết định kiến trúc đề xuất

1. **Giữ V1, sửa có mục tiêu.** Khung Dataset → processor → QLoRA → Trainer có thể tái sử dụng. Viết lại phần chuẩn bị mẫu, đọc ảnh và collator có lỗi; không cần xóa toàn bộ repo. V1 là baseline end-to-end nghiêm túc, không chủ động làm yếu để V2 thắng.
2. **Chưa dùng nguyên JSON bệnh nhân làm giao diện giữa hai engine.** Giữ JSON hiện tại làm bản ghi dataset; đề xuất adapter tạo payload suy luận riêng, có version, trạng thái thiếu dữ liệu và nguồn thông tin.
3. **V2 giai đoạn đầu chỉ sinh những nội dung mà tám nhãn hiện tại hỗ trợ.** Đây là báo cáo có phạm vi giới hạn về grading năm tầng, chưa phải báo cáo MRI thắt lưng toàn diện.
4. **Giữ một lần gọi LLM để sinh cả findings và impression**, thêm kế hoạch nội dung từ dữ kiện, đầu ra có cấu trúc và kiểm tra tính nhất quán. Hai lần gọi là một nhánh thực nghiệm, chưa phải thiết kế mặc định.
5. **Chưa fine-tune V2 bằng toàn bộ SFT hiện tại.** Trước tiên dựng baseline bằng template và prompting, đo lỗi, rồi mới quyết định LoRA trên cặp input–target đã được rà soát.
6. **Kaggle Free là môi trường train chính.** Mọi logic nằm trong `.py` trên repo GitHub public; notebook chỉ chọn tham số và gọi CLI. Dữ liệu được gắn từ hai Kaggle Dataset private (ảnh và annotations), không mount Drive trong mỗi phiên train. Colab là phương án dự phòng hoặc trung chuyển dữ liệu một lần. Chi tiết ở phần 12; hướng dẫn người mới tại `docs/README_KAGGLE.md`.

**Quyết định của chủ dự án:** đã chọn hướng A sau khi xem kế hoạch. Điểm chờ quyết định phạm vi JSON đã được giải quyết; triển khai tiếp theo theo hướng A, giữ nguyên dữ liệu nguồn và không tự bổ sung nhãn ngoài phạm vi.

## 2. Căn cứ từ working tree hiện tại

### 2.1. Dữ liệu đã kiểm tra

| Hạng mục | Hiện trạng | Hệ quả đối với kế hoạch |
|---|---|---|
| Master | 1.235 dòng, **42 cột**, 247 bệnh nhân | Số 39 cột trong tài liệu cũ đã lỗi thời sau chuẩn hóa |
| Patient JSONL | 247 bệnh nhân, mỗi ca 5 tầng | Dùng làm nguồn adapter, không đưa nguyên record vào LLM |
| Có findings Việt | 238 ca | 9 ca không có target findings Việt |
| Có cả findings và impression Việt | **230 ca** | Fold 1: **137 train / 46 val / 47 test** trước kiểm tra ảnh và nội dung |
| Có findings Việt theo fold 1 | 142 / 49 / 47 | 8 ca thiếu impression cần chính sách riêng; không biến thiếu thành kết luận bình thường |
| `disc_bulging` thiếu | 1 ô trong CSV, bị ETL chuyển thành 0 trong JSON | Adapter không thể khôi phục chỉ bằng JSON; phải đối chiếu CSV/nguồn |
| Mapping grading → tầng | README grading ghi `IVD_TO_LEVEL_VERIFIED = False` | Kết quả theo tầng là tạm thời cho đến khi kiểm tra quy ước nguồn |
| NIfTI/checkpoint | Chưa thấy trong working tree được kiểm tra | Chưa thể xác nhận chất lượng ảnh hoặc chạy huấn luyện V1 thật |
| V2 | Chưa có report engine hoặc đánh giá thực nghiệm | Chưa có tỷ lệ lỗi sinh báo cáo của dự án |

Số 137/46/47 chỉ là số ca có đủ hai phần văn bản, **không phải số cặp đã được bác sĩ xác nhận nhất quán**, cũng chưa xác nhận có ảnh đọc được.

### 2.2. Chênh lệch giữa grading và văn bản phải được xử lý trong thiết kế

Kết quả sàng lọc trong `output/annotation_audit/summary.json` khớp SHA-256 hiện tại của patient JSONL: `cc89ee34e115330e02d841848afae3c4cfd4a4bb3f0881de0e01265bc2b5418a`.

- 131/238 ca có ít nhất một dấu hiệu chênh lệch ở mức bệnh nhân theo bộ quy tắc hiện có. Đây là **tỷ lệ bị gắn cờ sàng lọc**, không phải tỷ lệ sai nhãn và không phải tỷ lệ hallucination của LLM.
- 43 ca thuộc nhóm ưu tiên rà soát A: chênh lệch hiện diện tổn thương trên toàn bệnh nhân hoặc mâu thuẫn nội bộ văn bản. Nhóm này ít phụ thuộc vào mapping tầng hơn các đối chiếu cấp tầng.
- 112 ca nhóm B cần làm rõ tầng/định nghĩa; 83 ca nhóm C chưa bị gắn cờ, **không mặc nhiên là ground truth sạch**.
- Quy ước làm việc từ README: “phình/lồi” đối chiếu với `disc_bulging`; thoát vị nội xốp không đồng nhất với `disc_herniation`. Cần xác nhận với người gán nhãn trước khi đóng băng ontology.
- Bộ quy tắc chưa được thẩm định như một bộ trích xuất lâm sàng hoàn chỉnh. Không coi việc báo cáo không đề cập là âm tính; không suy ra Pfirrmann chính xác từ câu “thoái hóa”.

Kế hoạch tiết kiệm chi phí là **rà soát có mục tiêu**, trước hết xác minh mapping/định nghĩa và nhóm A, đồng thời kiểm tra một mẫu nhóm B/C để tìm lỗi bị bỏ sót. Chưa có căn cứ để yêu cầu gán nhãn lại toàn bộ 247 ca. Không loại cả bệnh nhân khỏi mọi bài toán chỉ vì grading và report chưa khớp: từng nguồn vẫn có thể phục vụ mục tiêu riêng.

## 3. JSON hiện tại phù hợp đến đâu?

### 3.1. Những phần giữ được

- `levels[].level`: năm tầng giải phẫu với thứ tự chuẩn.
- Tám trường trong `gradings`, giữ nguyên giá trị nguồn và tên tiếng Anh.
- Tọa độ phục vụ truy vết khi cần, không dùng làm bằng chứng rằng một bệnh lý tồn tại.
- `technique`, `findings`, `impression` là tên section hợp lý; alias cũ có thể tiếp tục tồn tại trong dataset để tương thích.

### 3.2. Những vấn đề cần sửa trước khi coi đây là contract suy luận

| Vấn đề | Rủi ro | Đề xuất |
|---|---|---|
| Có `reports.vi/en` ngay trong record | Nếu serialize toàn bộ vào prompt, LLM nhìn thấy đáp án | Tách inference payload khỏi targets; xây input bằng danh sách trường cho phép |
| Có folds, split, metadata không cần thiết | Lẫn dữ liệu quản lý thí nghiệm vào input | Giữ trong manifest bên ngoài report request |
| Nhãn thiếu bị đổi thành 0 | Sinh “không có” từ dữ liệu chưa biết | Giá trị nullable và trạng thái quan sát rõ ràng; sửa từ nguồn, không đoán |
| Chỉ có nhãn cứng | Không biết nguồn là bác sĩ hay mô hình, có được kiểm tra chưa | Ghi provenance và QC; uncertainty có thể để null |
| Chưa có schema version | Vision và report có thể diễn giải khác nhau sau cập nhật | Schema version + ontology version + kiểm tra tương thích |
| Mapping `ivd_label` chưa được xác minh | Đúng bệnh nhưng gắn nhầm tầng | Ghi trạng thái mapping; cho phép thử nghiệm có cờ, chưa chứng nhận kết quả cấp tầng |
| Tọa độ thiếu `volume`, hệ quy chiếu/QC trong patient JSON | Không biết tọa độ thuộc ảnh nào | Nếu giữ evidence hình học: thêm reference ảnh, quy ước voxel/LPS và QC từ nguồn |
| Các nhãn hiện có không mô tả toàn bộ report | LLM phải suy đoán phần còn thiếu nếu bị yêu cầu viết đủ | Giới hạn phạm vi report hoặc bổ sung nhãn bằng quy trình riêng |

Không dùng `reports.vi.technique` làm bằng chứng ảnh một cách mặc định. Kỹ thuật chụp chỉ được đưa vào input nếu lấy từ acquisition metadata đã xác minh, hoặc qua bước trích xuất được khai báo rõ là dùng thông tin từ báo cáo. MVP chỉ cần findings/impression.

### 3.3. Ranh giới diễn đạt từ tám nhãn

| Nhãn đầu vào | Có thể diễn đạt | Không tự suy ra |
|---|---|---|
| `pfirrmann_grade` | Phân độ được cung cấp, theo từ điển đã duyệt | Thêm đầy đủ mọi dấu hiệu MRI của một grade như thể đã quan sát riêng từng dấu hiệu |
| `disc_herniation` | Có/không thoát vị theo quy ước bộ nhãn | Lệch trái/phải, thể trung tâm, extrusion, kích thước, rách vòng xơ, chèn ép rễ |
| `disc_bulging` | Có/không phình/lồi theo quy ước đã xác nhận | Hẹp ống sống hoặc chèn ép rễ chỉ vì phình đĩa |
| `disc_narrowing` | Có/không giảm chiều cao khe đĩa đệm | Hẹp ống sống hoặc lỗ liên hợp; mức nhẹ/vừa/nặng không có trong nhãn |
| `spondylolisthesis` | Có/không trượt theo đơn vị giải phẫu đã xác nhận | Hướng trượt, độ trượt, mất vững hoặc tên thân đốt nếu quy ước chưa rõ |
| `modic` | Type 0/I/II/III | Diễn giải I→III thành thang mức độ tăng dần |
| `up_endplate`, `low_endplate` | Có/không bất thường bản đệm theo định nghĩa nguồn | Khẳng định riêng nốt Schmorl, gãy hoặc nhiễm trùng từ một nhãn tổng hợp |

Nhị phân 0 chỉ phủ định đúng hạng mục mà nhãn định nghĩa. Không từ tám nhãn âm tính kết luận “MRI bình thường hoàn toàn”, “không u”, “không nhiễm trùng” hoặc “rễ thần kinh bình thường”.

### 3.4. Hai lựa chọn cần chủ dự án chốt

**A — Khuyến nghị: V2 phạm vi tám nhãn trước.** Tạo contract mới bằng adapter, giữ dataset gốc; report chỉ mô tả thông tin được hỗ trợ. Các mục ngoài phạm vi được ghi trong metadata giới hạn, không tự thêm câu bình thường. Có thể bắt đầu mà chưa mua thêm nhãn. Đổi lại, không gọi đầu ra là báo cáo MRI toàn diện tương đương nguyên bản bác sĩ.

**B — V2 hướng tới đầy đủ nội dung report bác sĩ ngay.** Mở rộng contract cho bên/vị trí, hình thái, hẹp ống sống/lỗ liên hợp, rễ thần kinh, vòng xơ, thân đốt và các phát hiện khác đã chọn. Việc thêm key chưa tạo ra dữ liệu: các field mới phải để `not_assessed` cho đến khi được bổ sung/xác minh. Trích thông tin từ report chỉ tạo dữ liệu silver để khảo sát, không biến thành output của vision hay ground truth độc lập; không dùng target report của test để xây input rồi tuyên bố hiệu quả chẩn đoán.

**Đã chốt A theo quyết định của chủ dự án.** Lựa chọn B được giữ trong tài liệu để giải thích phạm vi chưa triển khai.

## 4. Contract đề xuất cho lựa chọn A

### 4.1. Ba đối tượng tách biệt

| Đối tượng | Nội dung | LLM có được đọc? |
|---|---|---|
| Dataset record | JSON hiện tại, báo cáo, folds, provenance nguồn | Không đọc nguyên record |
| Report request | Schema/ontology version, scope, grading theo tầng, trạng thái quan sát | Chỉ phần dữ kiện cần sinh văn bản |
| Training/evaluation target | Văn bản và claims đã được rà soát, liên kết bằng mã ca | Chỉ dùng làm completion lúc train hoặc reference lúc đánh giá |

Dataset có thể vẫn lưu aliases để tương thích; contract mới chỉ có tên canonical. Nếu field mới và alias cùng có giá trị nhưng khác nhau, ghi lỗi đối chiếu thay vì âm thầm chọn một. Nếu field canonical thiếu/rỗng, adapter có thể đọc alias có nội dung hợp lệ.

### 4.2. Cấu trúc logic, chưa phải schema đã phê duyệt

| Trường | Quy tắc đề xuất |
|---|---|
| `schema_version` | Phiên bản contract, ví dụ `report-input/1.0` |
| `ontology_version` | Phiên bản miền giá trị, định nghĩa thuật ngữ và quy tắc diễn đạt |
| `case_id` | Mã ca phục vụ ghép kết quả; không cần đưa vào prompt |
| `scope` | `lumbar_grading_8_fields`; phạm vi này quyết định điều được phép nói |
| `provenance` | Nguồn `dataset_grading` hiện tại; sau này `vision_prediction`, có phiên bản và dấu vết nguồn |
| `quality.level_mapping` | `unverified` hiện tại; chỉ đổi thành `verified` khi có bằng chứng |
| `levels` | Đúng năm tầng, không trùng, sắp xếp canonical |
| `levels[].gradings.<field>.value` | Integer đúng miền hoặc `null`; không chấp nhận NaN/Infinity |
| `levels[].gradings.<field>.status` | `observed`, `missing`, `not_assessed`, `uncertain` |
| `levels[].gradings.<field>.uncertainty` | `null` hiện tại; sau này đối tượng có phương pháp, probabilities/prediction set và tham chiếu calibration |
| `levels[].evidence` | Optional, tọa độ gắn với volume/reference và QC; không phải nội dung tự do để LLM thêm bệnh |

Quy tắc trạng thái: `observed` phải có giá trị hợp lệ; `missing/not_assessed` có `value=null`; `uncertain` không được diễn đạt như chắc chắn. Provenance bác sĩ không đồng nghĩa confidence=1. Không tự tạo xác suất từ nhãn 0/1, không suy ra độ tin cậy lâm sàng từ xác suất token của LLM.

Ví dụ một field giả lập, không phải dữ liệu bệnh nhân:

```json
{
  "disc_bulging": {
    "value": null,
    "status": "missing",
    "uncertainty": null
  }
}
```

MVP không bắt buộc tạo field bệnh học mới ngoài tám nhãn. Scope khai báo rõ những gì không được đánh giá; có thể mở rộng ở phiên bản sau. Mapping chưa xác minh được phép chạy ở chế độ `research_preview`, kết quả luôn mang cờ; chế độ xuất kết quả đã kiểm tra phải từ chối hoặc yêu cầu review.

### 4.3. Output contract

- Kết quả ứng dụng có `findings` và `impression` dạng chuỗi, cùng `status`, `limitations`, `validation`, phiên bản model/prompt và trace nguồn.
- Cấu trúc nội bộ có thể giữ danh sách câu với `evidence_ids`; renderer nối thành hai đoạn văn cuối cùng.
- `status` phân biệt `ok`, `needs_review`, `fallback`, `failed`; không giấu việc fallback dưới một báo cáo trông như đã thành công.
- Evidence ID phải tồn tại và đúng nội dung câu. Việc LLM tự đính một ID không đủ chứng minh câu được hỗ trợ.
- Impression chỉ tổng hợp dữ kiện có trong request và đã được findings hỗ trợ; không thêm bệnh, tầng, bên hay mức độ mới.
- Với dữ liệu thiếu hoặc ngoài scope, không viết kết luận bình thường. Nếu không đủ dữ kiện để kết luận, trả trạng thái và lời giới hạn tương ứng.

## 5. Sinh findings và impression một lần: có ổn không?

**Có, đây là một lựa chọn triển khai và baseline hợp lý.** Một lần generate vẫn là giải mã tự hồi quy; khi findings xuất hiện trước, các token impression có thể dựa trên phần findings đã sinh. Điều đó giúp hai phần có chung ngữ cảnh nhưng không bảo đảm tính nhất quán.

Vấn đề là chỉ dùng lời nhắc để kiểm soát toàn bộ nội dung. Các lỗi cần đo riêng:

| Loại lỗi | Ví dụ khái quát | Cách kiểm soát |
|---|---|---|
| Định dạng | Thiếu impression, JSON sai, câu trả lời bị cắt | Schema, parse, kiểm tra kết thúc và giới hạn token |
| Thêm nội dung | JSON chỉ có phình nhưng viết thêm chèn ép rễ trái | Đối chiếu claims với dữ kiện nguồn |
| Nhầm thuộc tính | Đúng bệnh nhưng nhầm tầng, cực tính, Modic type | So sánh tuple `(level, finding, value, certainty)` |
| Bỏ sót | Tổn thương có trong request không xuất hiện trong findings | Kiểm tra coverage theo kế hoạch nội dung |
| Hai section mâu thuẫn | Findings phủ định, impression khẳng định | Kiểm tra liên section dựa trên cùng facts |
| Nâng độ chắc chắn | Missing/uncertain trở thành “không có”/“chắc chắn có” | Chính sách trạng thái và kiểm tra bất định |

Structured decoding ràng buộc cấu trúc JSON, không tự chứng minh chuỗi văn bản bên trong đúng bệnh học. vLLM hỗ trợ ràng buộc theo JSON schema/grammar; việc kiểm tra nội dung vẫn là một bước riêng trong thiết kế này. [Tài liệu vLLM](https://docs.vllm.ai/en/latest/features/structured_outputs/)

**Chưa thể trả lời tỷ lệ lỗi cao hay thấp bằng một con số.** Nó phụ thuộc checkpoint, input/target, prompt, cách giải mã, độ dài, tiêu chí lỗi và phạm vi bệnh học. Chưa có lần chạy V2 nào để đo. Tỷ lệ 131/238 từ audit là chênh lệch nguồn dữ liệu, tuyệt đối không dùng làm tỷ lệ lỗi LLM.

Với input hiện tại, nguy cơ thêm thông tin có cơ sở để lo ngại nếu bắt LLM tái tạo toàn bộ report: nhiều nội dung đích không nằm trong input. Đây là nhận định về thiếu thông tin, chưa phải ước lượng xác suất thực nghiệm. Temperature thấp/greedy giúp ổn định kết quả nhưng không sửa được thông tin bị thiếu.

**Thiết kế mặc định:** request → validate → facts/content plan bằng quy tắc → một lần LLM sinh hai section → kiểm tra cấu trúc và nội dung → render. Khi lỗi, cho phép tối đa một lần sửa có kèm danh sách vi phạm; còn lỗi thì template fallback hoặc `needs_review`. Lưu cả lỗi ban đầu, lần sửa và kết quả cuối để đánh giá trung thực.

Một biến thể hai lần gọi chỉ được thêm để đo: facts → findings; sau đó facts **cùng findings đã kiểm tra** → impression. Không cho lần hai chỉ đọc findings chưa kiểm tra, vì có thể khuếch đại lỗi của lần một. Đánh đổi là thêm độ trễ/token; chưa có cơ sở khẳng định luôn tốt hơn một lần gọi.

## 6. Workstream V1: sửa thành baseline có thể tin cậy

### V1-01 — Đóng băng hợp đồng dữ liệu và môi trường

**Tệp:** `.gitignore`, `requirements.txt` hoặc lockfile riêng, `configs/v1_config.yaml`, mới `src/data/report_targets.py`, `scripts/preflight_v1.py`.

- Sửa quy tắc ignore `models/` để code `src/models/` được theo dõi; tiếp tục bỏ qua weights/checkpoints. Không xóa loader đang có.
- Khóa bộ phiên bản thư viện sau smoke test trên môi trường GPU mục tiêu, ghi model revision và processor revision. Không giả định lower bound hiện tại đủ để tái lập.
- Canonical hóa văn bản: xử lý None/NaN/rỗng trước khi ép string; kiểm tra cả alias và canonical; báo xung đột nếu hai bản khác nhau.
- Mặc định thí nghiệm đủ hai section dùng cohort 230 ca trước kiểm tra ảnh. Tám ca chỉ có findings giữ lại cho nhánh train findings-only riêng nếu cần; không điền impression giả.
- Đọc patient_id dạng chuỗi; kiểm tra năm tầng, split nhất quán và không chồng bệnh nhân; lưu manifest bao gồm lý do loại mẫu.

**Nghiệm thu:** không có target `nan/None`; không rò split; số mẫu và lý do loại giải thích được; clone sạch import được loader.

### V1-02 — Viết lại đường đọc ảnh và cache

**Tệp:** `src/data/v1_dataset.py`, mới `src/data/mri_images.py`, cấu hình V1 và preflight.

- Dùng `volume`/manifest để tìm đúng ảnh; không ngầm giả định mọi ảnh cùng tên/path/contrast. Xác minh trên nơi thật sự chứa NIfTI.
- Bỏ fallback ảnh xám trong train/val/test. Ảnh giả chỉ tồn tại trong test fixture có nhãn rõ ràng và không lọt vào huấn luyện.
- Đọc affine/orientation, xác định mặt phẳng sagittal, có xem ảnh mẫu. Không coi `shape[2]` và `rot90` cố định là chứng minh đúng hướng.
- Nếu reorient/resample, biến đổi tọa độ theo đúng không gian. Không dùng `voxel_k` cũ trực tiếp trên volume đã đổi trục. Nếu tọa độ ngoài biên, fail/QC thay vì clip âm thầm.
- Bảo toàn tỷ lệ hình học hoặc resample theo spacing rồi resize/pad; percentile normalization có kiểm tra volume rỗng/NaN.
- Cache key gồm source fingerprint, preprocessing version, plane/slice selection, orientation và kích thước; đổi cấu hình phải invalid cache.
- Giữ baseline một lát sagittal làm nhánh chi phí thấp; ghi rõ nếu chọn lát bằng tọa độ annotation thì có hỗ trợ định vị từ dataset. Có cấu hình chọn giữa volume không dùng annotation cho baseline tự động thuần túy.
- Khi có ảnh, thêm nhánh vài lát sagittal/axial được chọn bằng quy tắc cố định và không dùng grading/report. Đây vẫn là V1 ảnh→text vì không có JSON bệnh học trung gian; ghi rõ giới hạn bộ nhớ và chuỗi ảnh thiếu.

**Nghiệm thu:** missing image fail trước khi nạp model nặng; có QC ảnh thật; kiểm thử orientation/tọa độ bằng phantom; thay preprocessing không dùng lại cache cũ.

### V1-03 — Viết lại collator và loss masking

**Tệp:** tách `VLMDataCollator` khỏi `src/train_v1.py` sang `src/data/vlm_collator.py`.

- Mã hiện tại chỉ mask pad và có thử mask token ảnh, **chưa mask user prompt** dù docstring nói đã làm.
- Dùng assistant mask của template nếu template/revision đã kiểm tra hỗ trợ; nếu không, xác định boundary assistant từ chuỗi token đã serialize và kiểm chứng chính xác. Không dò token theo một chuỗi header có thể xuất hiện trong nội dung.
- Mask toàn bộ system/user, padding và token đặc biệt không thuộc completion; giữ EOS kết thúc answer nếu cần học. Không mask mọi token bằng pad ID nếu pad và EOS chung ID; dựa vào attention mask/vị trí.
- Giữ đủ `pixel_values`, `image_grid_thw` và cấu trúc đa phương thức processor yêu cầu. Kiểm thử batch >1 và độ dài khác nhau.
- Bỏ `truncation=True,max_length=512` vô điều kiện. Đo token sau mở rộng token ảnh; điều khiển số ảnh/resolution và ngân sách văn bản. Không cắt target mà vẫn báo mẫu hợp lệ; vượt context thì báo/loại có thống kê hoặc xử lý theo chính sách đã chốt.
- Assert mỗi mẫu có token assistant được supervise; log tỷ lệ và số token target được giữ.

TRL có hỗ trợ assistant/completion-only loss, nhưng assistant mask phụ thuộc chat template. Tài liệu cũng cảnh báo truncation có thể cắt image tokens ở VLM; không chỉ đổi tên trainer rồi mặc định coi masking đã đúng. [Tài liệu TRL](https://huggingface.co/docs/trl/sft_trainer)

**Nghiệm thu:** test nhìn trực tiếp nhãn -100; prompt không nhận loss, target đủ, EOS/pad đúng; một forward/backward batch ảnh thật có loss hữu hạn.

### V1-04 — Loader, training và resume

**Tệp:** `src/models/v1_vlm.py`, `src/train_v1.py`, `configs/v1_config.yaml`.

- Đọc và áp dụng dtype từ cấu hình theo khả năng thiết bị; T4 ưu tiên FP16 phù hợp, BF16 chỉ khi thiết bị hỗ trợ. Sửa comment đánh đồng T4 và A100.
- Bật gradient checkpointing khi cần, kiểm tra `use_cache` và tương thích PEFT/quantization.
- Chốt phạm vi LoRA bằng full module names/đếm tham số: language decoder mặc định; vision tower/projector đóng băng hoặc mở theo cấu hình. Không để tên `q_proj` chung vô tình mở cả vision rồi mô tả là chỉ fine-tune LLM.
- Chạy preflight trước khi tải model; log VRAM và thông lượng, không cam kết T4 chạy được mọi cấu hình.
- Hỗ trợ resume checkpoint, lưu optimizer/scheduler/trainer state, cấu hình, seed, revision, hashes và manifest dữ liệu; lưu adapter + processor đủ để inference lại.
- Chọn checkpoint theo validation đã định trước. Test không dùng chọn prompt, checkpoint hoặc số epoch.

**Nghiệm thu:** overfit được 2–4 mẫu train như kiểm tra dây chuyền; adapter save/reload được; resume tiếp tục đúng state; nghiệm thu chất lượng cần thí nghiệm riêng, loss giảm chưa đủ.

### V1-05 — Inference và notebook tái lập

**Tệp mới:** `src/infer_v1.py`, `src/evaluation/` dùng chung, `scripts/make_cloud_notebooks.py` và `notebooks/Kaggle_V1_Training.ipynb`. Notebook Colab cũ được đánh dấu legacy hoặc sinh lại từ cùng generator, không duy trì một implementation train riêng.

- Inference chỉ nhận ảnh + prompt, không có target/gradings trong messages; chỉ decode token mới.
- Cấu hình `max_new_tokens`, stop tokens và sampling rõ ràng; ghi nhận bị cắt/thiếu section.
- Notebook gọi cùng CLI, không có loader/dtype/preprocessing bản riêng. Chọn commit/revision, fail khi thiếu data; không tự pull code mới giữa một run.
- Notebook tải code từ repo public, đọc hai Kaggle Dataset private đã Add Input rồi gọi CLI theo phần 12. Checkpoint ghi ở `/kaggle/working/runs/`; phiên train có ngân sách thời gian để chủ động save và kết thúc trước giới hạn. Resume đọc từ outputs của phiên đã lưu, không giả định ổ làm việc của phiên trước còn tồn tại.
- Lưu predictions có case_id, split, model/prompt/input hash và generation settings để đánh giá lại.

**Nghiệm thu:** chạy lại notebook từ đầu trên môi trường sạch và có NIfTI thật; có một artifact dự đoán đủ tập test đã khóa. Hiện tại chưa thể nghiệm thu phần GPU/ảnh ngay trong working tree này.

## 7. Workstream V2: report engine từ grading JSON

Các task V2-01 trở đi thực hiện theo **hướng A đã được chọn ở phần 3.4**.

### V2-01 — Schema, adapter và fixtures

**Tệp mới:** `src/contracts/report_input.py`, `src/contracts/report_output.py`, `schemas/report_input.schema.json`, `schemas/report_output.schema.json`, `src/data/v2_adapter.py`, `scripts/build_v2_inputs.py`.

- Schema là nguồn định nghĩa duy nhất; export JSON Schema từ model dữ liệu hoặc kiểm thử hai bản không lệch.
- Adapter đọc JSON hiện tại và CSV nguồn cần thiết để khôi phục missing. Nếu không có nguồn phục hồi, báo không đủ dữ liệu thay vì giả định các số 0 đều đúng.
- Không sửa/xóa reports hoặc aliases trong bản ghi gốc. Xuất request/target/manifest sang thư mục derived riêng bị ignore; ghi input hash và adapter version.
- Validation strict miền giá trị, năm tầng, thiếu/trùng tầng, status/value, provenance và các trường cấm trong inference. Extension chưa hỗ trợ phải báo rõ.
- Chỉ dùng fixtures giả lập trong Git; không commit report/ID bệnh nhân thật.

**Nghiệm thu:** 247 bản ghi đều có kết quả chuyển đổi hoặc lỗi rõ ràng; đúng một ô missing được giữ lại khi nguồn không đổi; không có bất kỳ nội dung report hoặc split nào trong prompt payload.

### V2-02 — Fact planner và template baseline

**Tệp mới:** `src/report/facts.py`, `src/report/ontology.py`, `src/report/planner.py`, `src/report/templates_vi.py`.

- Chuyển từng observation thành fact có ID, tầng, loại bệnh, giá trị, trạng thái và nguồn; sắp xếp ổn định.
- Định nghĩa bằng quy tắc nội dung bắt buộc trong findings, cách gom tầng và những gì được chọn vào impression. Chính sách summary cần bác sĩ duyệt; chưa tự đặt thứ hạng nghiêm trọng từ Modic type hoặc suy ra bệnh khác.
- Không tự ép `disc_bulging` và `disc_herniation` loại trừ nhau nếu quy ước nguồn chưa nói vậy. Các tổ hợp cần làm rõ được flag, không tự đổi nhãn.
- Xây renderer câu mẫu theo fact, bao gồm negative/unknown/uncertain đúng nghĩa; đây là baseline và fallback có thể kiểm chứng.
- Trace mỗi câu về facts; không lấy câu report nguồn để quyết định fact nào được phép sinh lúc inference.

**Nghiệm thu:** các tổ hợp dữ kiện biên cho ra nội dung đúng rule, không thêm thuộc tính; dữ kiện không đổi thì template output không đổi; vẫn ghi giới hạn mapping nếu chưa xác minh.

### V2-03 — LLM sinh một lần hai section

**Tệp mới:** `src/report/backends/base.py`, `src/report/backends/hf_local.py`, `src/report/prompts/vi_v1.txt`, `src/report/generator.py`, `configs/v2_report.yaml`.

- Dùng text LLM chạy local làm backend; không cần vision encoder khi đầu vào chỉ có facts. Bắt đầu bằng một checkpoint instruction-tuned đủ tài nguyên, đánh giá tiếng Việt thực tế; ghi model/license/revision. Không mặc định model mang nhãn medical sẽ tốt hơn.
- Backend interface tách tải model, generate và usage; có thể thêm backend constrained decoding tương thích phần cứng sau smoke test. Không phụ thuộc vLLM bắt buộc trên máy hiện tại.
- Prompt gồm scope, facts, section policy, cách diễn đạt missing và yêu cầu hai phần; không gồm original reports, patient identifiers hoặc folds. Examples chỉ lấy từ train đã duyệt hoặc fixture tổng hợp.
- Một lần generate trả hai section và trace; structured decoding nếu backend hỗ trợ, còn không thì parse/validate bắt buộc.
- Greedy làm cấu hình chính để so sánh; cấu hình sampling riêng phục vụ stress test, không trộn kết quả.

**Nghiệm thu:** cùng input/config cho đầu ra có thể truy vết; timeout/context overflow/parse failure đều có trạng thái; không gọi dịch vụ ngoài để gửi dữ liệu bệnh nhân.

### V2-04 — Kiểm tra đầu ra, sửa có giới hạn và fallback

**Tệp mới:** `src/report/validate.py`, `src/report/pipeline.py`.

- Kiểm tra cấu trúc và section không rỗng theo policy; phát hiện generation bị cắt.
- So sánh claims: bệnh, tầng, polarity, type/grade, certainty; không thêm bên, mức độ hoặc nguyên nhân không có trong facts.
- Kiểm tra completeness theo planner và consistency findings↔impression. Không bắt impression lặp hết findings; chỉ bắt đủ các facts được policy chọn.
- Không xem evidence_ids hợp lệ là đủ: kiểm tra nội dung câu có thực sự khớp fact được trỏ tới.
- Rule-based checker cho phạm vi tám nhãn là điểm khởi đầu. Với văn phong tự do, checker có thể bỏ sót/hiểu nhầm; cần tập bác sĩ rà soát để đo chính checker. LLM-as-judge chỉ là chỉ báo phụ, không là nguồn sự thật duy nhất.
- Tối đa một lần repair từ facts gốc + lỗi cụ thể; nếu vẫn không đạt thì template fallback/needs_review. Không loop sinh lại đến khi che mất tỷ lệ lỗi thật.
- Nếu yêu cầu đảm bảo chặt rằng không thêm facts, dùng câu mẫu hoặc lựa chọn trong danh sách câu đã kiểm tra. Free-text có validator không có bảo đảm tuyệt đối tương đương.

**Nghiệm thu:** bộ test lỗi chủ đích bị phát hiện; không xuất kết quả `ok` khi vi phạm đã biết; trace ghi số lần gọi, lỗi thô và fallback. “Qua validator” chỉ là đạt kiểm tra của phần mềm, không phải chứng nhận lâm sàng.

### V2-05 — Chuẩn bị target và quyết định fine-tuning

**Tệp mới:** `scripts/build_v2_targets.py`, `src/data/v2_dataset.py`; nếu cần mới thêm `src/train_v2_report.py` và cấu hình LoRA.

- Giữ `dataset/sft_data/`: sáu file có ích để tái lập thí nghiệm cũ, khảo sát văn phong và baseline prompt→report; chưa thỏa điều kiện “vô ích thì xóa”.
- Không coi chúng là cặp input–target sạch: prompt mất thông tin, response có thông tin ngoài schema. Không ghép trực tiếp cả response vào train cho lựa chọn A.
- Tạo target theo từng claim có provenance: được input hỗ trợ, ngoài scope, mâu thuẫn, chưa rõ. Có thể trích xuất tự động để giảm công bác sĩ, nhưng target cuối phải qua chính sách kiểm duyệt.
- Câu chứa cả phần hỗ trợ và không hỗ trợ phải được tách/viết lại có kiểm tra; không giữ nguyên chỉ vì chứa một từ khóa khớp.
- Target từ template hợp lệ để kiểm tra kỹ thuật/style distillation, không chứng minh tương đương văn bản bác sĩ. Evaluation không chỉ dựa trên bộ target được sinh bởi cùng template/rule.
- Tách ca/claim cần adjudication; không ép grading “thắng” report hoặc ngược lại. Dùng bản gốc và bản adjudicated song song để truy vết.
- Sau baseline prompt/template, chỉ chạy LoRA nếu còn lỗi diễn đạt/tuân thủ đáng kể và có đủ target được hỗ trợ. So sánh cùng backbone trước/sau LoRA; completion-only loss, theo dõi overfit vì tập train nhỏ.

**Nghiệm thu:** mỗi target dùng train có nguồn, scope và trạng thái review; test không dùng làm few-shot hoặc sinh dữ liệu train; script có thống kê số ca/claim bị loại cùng lý do.

### V2-06 — CLI và khả năng gắn vision engine sau này

**Tệp mới:** `src/generate_report.py`, `notebooks/Kaggle_V2_Report.ipynb`, hướng dẫn chạy và run manifests.

- CLI nhận một request JSON hoặc JSONL, chọn config/output; chạy validate→plan→generate→verify→render.
- Batch có lỗi từng ca, tiếp tục các ca hợp lệ và trả tổng kết; retry theo policy, không âm thầm bỏ ca lỗi.
- Lưu text, structured output, validation, limitations, thời gian/token/cost và hash. Tách log vận hành khỏi văn bản báo cáo đưa người đọc xem.
- Contract không phụ thuộc `dataset_master.csv` ở inference. CSV chỉ phục vụ adapter dữ liệu hiện tại; vision tương lai tạo đúng request trực tiếp.
- Backend tương lai có calibrated probabilities/prediction sets, còn hiện tại uncertainty để null. Không tuyên bố đã có calibration/conformal từ grading bác sĩ.

**Nghiệm thu:** một fixture JSON chạy độc lập không cần dataset/report target; malformed JSON bị từ chối; thay nguồn dataset bằng mock vision request cùng schema không đổi report engine.

## 8. Thực nghiệm: đo lỗi thay vì ước lượng bằng cảm giác

### 8.1. Hai câu hỏi đánh giá riêng

**Bám dữ kiện:** report engine có diễn đạt đúng JSON không? Có thể đánh giá bằng facts hiện tại dù chúng chưa được chứng minh đúng trên MRI. Kết quả chỉ nói về độ trung thành với input.

**Đúng lâm sàng:** báo cáo có đúng với bệnh nhân/ảnh không? Cần nguồn đối chiếu được rà soát hoặc đọc MRI. Việc dùng grading bác sĩ giả làm output vision là thí nghiệm **V2 với input annotation**, không phải kết quả end-to-end của một vision engine đã hoạt động và không phải trần hiệu năng tuyệt đối do nhãn vẫn có thể sai.

### 8.2. Ma trận ablation

| ID | Nhánh | Mục tiêu |
|---|---|---|
| V1-S | Ảnh một lát → VLM → hai section | Baseline ảnh hạn chế, ghi rõ cách chọn lát |
| V1-M | Nhiều ảnh chọn bằng quy tắc → VLM | Kiểm tra tác động coverage ảnh, nếu có đủ NIfTI/tài nguyên |
| R0 | JSON → template | Kiểm soát độ trung thành và mốc độ trễ |
| R1 | JSON → LLM, prompt-only, một lần | Đo đúng phương án đang hỏi |
| R2 | JSON → LLM, một lần, structured format | Tách tác động ràng buộc cấu trúc |
| R3 | Facts/planner → LLM, một lần + validator | Đo kiểm soát nội dung, là cấu hình đề xuất |
| R4 | Hai lần gọi với cùng facts/planner/validator | Đo riêng tác động chia findings/impression |
| R5 | R3 + LoRA trên target đã kiểm tra | Đo lợi ích fine-tune, chỉ chạy khi đủ dữ liệu |

R3 phải báo cả kết quả thô và kết quả sau repair/fallback; báo template fallback riêng để tránh nhầm thành năng lực LLM. R4 giữ cùng checkpoint/facts/policy để không trộn nhiều thay đổi. Khi so V1 và V2, báo rõ input V2 là annotation; không kết luận V2 end-to-end tốt hơn V1 từ thí nghiệm này.

### 8.3. Split và quy trình

1. Giữ patient-level folds hiện có. Không random split 1.235 tầng hoặc các câu báo cáo.
2. Dùng train cho few-shot/LoRA; validation để chốt prompt, rule, hyperparameter và threshold. Nhóm nghiên cứu đã xem một số dữ liệu test khi audit: phải ghi nhận tiếp xúc này, khóa thiết kế tiếp theo và coi kết quả cohort này là nội bộ/exploratory nếu ảnh hưởng lựa chọn phương pháp; test ngoài cohort sẽ cần cho xác nhận mạnh hơn.
3. Chốt tiêu chí đủ target trước khi chạy. Report fidelity có thể đo trên input không có report gốc; so với văn bản bác sĩ dùng tập giao đủ target, cùng cohort giữa các nhánh, công bố số lượng và coverage.
4. Không tự loại mọi ca khó/mâu thuẫn khỏi test rồi chỉ báo tập sạch. Báo toàn bộ cohort phù hợp và phân tầng trạng thái review; tập adjudicated là phân tích riêng có mẫu số rõ.
5. Nhờ bác sĩ kiểm tra mẫu lỗi và mẫu không bị flag, làm mù tên phương pháp nếu khả thi; chọn mẫu theo quy tắc trước khi xem outputs. Nếu khả năng cho phép, có người thứ hai xử lý bất đồng.
6. Fold 1 để phát triển; sau khi đóng băng quy trình có thể chạy 5 fold cho out-of-fold predictions. Mỗi outer test phải độc lập với tuning của fold đó; không gọi kết quả đã dùng để sửa prompt là test cuối.

### 8.4. Metrics cần lưu

| Metric | Định nghĩa |
|---|---|
| Lỗi định dạng | Số request có output thô không parse/không đúng schema/thiếu section chia số request hợp lệ được gọi generate |
| Lỗi theo báo cáo | Số báo cáo có ≥1 lỗi nội dung theo thẩm định chia số báo cáo được thẩm định; nếu lấy mẫu có chọn lọc, không ngoại suy trực tiếp cho toàn cohort |
| Unsupported claim rate | Số claims không có bằng chứng trong input chia tổng claims được trích/duyệt; công bố sai số bộ trích xuất |
| Fact precision/recall/F1 | So tuple bệnh–tầng–giá trị–certainty với facts và content policy; tách findings/impression và nhãn |
| Critical error rate | Sai tầng, polarity hoặc thêm bệnh quan trọng theo rubric chốt với bác sĩ; mẫu số ở mức bệnh nhân |
| Section contradiction | Báo cáo có xung đột findings/impression chia báo cáo đánh giá được |
| Omission | Required facts bị bỏ sót chia required facts; impression dùng tập facts được planner chọn |
| Unknown/uncertain violation | Output biến chưa biết/bất định thành khẳng định không có căn cứ |
| Operational coverage | Các ca thành công, needs_review, fallback, failed trên toàn bộ request; thêm latency và token |
| NLP metrics | ROUGE/BERTScore và metric từ vựng chỉ bổ trợ trên target tương ứng scope; không thay factual accuracy |

Không gọi một parser schema failure là sai lâm sàng đã được chứng minh; không gộp request thất bại thành “không có hallucination”. Luôn báo raw và final, denominator và số lượng tuyệt đối.

Tính khoảng tin cậy theo bệnh nhân, không coi năm tầng/các câu như mẫu độc lập. Dùng bootstrap ghép cặp theo bệnh nhân để so phương pháp khi phù hợp; tỷ lệ lỗi có thể dùng Wilson hoặc exact interval. Ví dụ **giả định**: nếu không gặp lỗi trong 47 bệnh nhân test độc lập, cận trên một phía 95% theo nhị thức vẫn khoảng 6,2%; không có nghĩa tỷ lệ lỗi thật bằng 0. Khoảng 300 ca không lỗi mới đưa cận trên tương tự về khoảng 1%, dưới giả định sampling và tiêu chí lỗi phù hợp. Đây không phải số đã đo của dự án.

## 9. Thứ tự triển khai và điều kiện hoàn thành

| Mốc | Công việc | Phụ thuộc | Bằng chứng hoàn thành |
|---|---|---|---|
| M0 — đã chốt | Chủ dự án chọn A, report giới hạn tám nhãn và contract riêng | Tài liệu này | Quyết định đã được ghi vào thiết kế |
| M1 | Manifest, canonical text, môi trường, tests dữ liệu | Có thể chuẩn bị độc lập schema V2 | Không leak/sai split/NaN; snapshot nguồn |
| M2 | V1-01…05 | NIfTI/GPU cho kiểm thử thật | Baseline train/infer tái lập, output test |
| M3 | V2-01…02 | M0 | Adapter + template + schema fixtures đạt kiểm tra |
| M4 | V2-03…04 và V2-06 | M3 | Một request end-to-end, raw/final logs, repair/fallback |
| M5 | Rà soát target + R0…R4 | M4, protocol evaluation | Bảng lỗi/coverage/chi phí có CI và kiểm tra bác sĩ |
| M6 | LoRA V2 nếu có lợi ích cần kiểm chứng | Target đã kiểm tra + M5 | Ablation R5, không chỉ train loss |
| MC | Notebook Kaggle mỏng, hai dataset private, checkpoint/resume và README cho người mới | Các CLI đã kiểm thử; dataset/GPU thật để nghiệm thu | Clone code đã push, chạy từ runtime sạch và resume từ outputs phiên trước |
| M7 | Vision/calibration thật | Ngoài phạm vi hiện tại | Thay source request, đánh giá lại toàn pipeline |

M2 và M3–M4 có thể phát triển độc lập sau các quyết định chung; V2 report không cần chờ vision engine. Mapping grading→tầng chưa xác minh không ngăn viết fixtures và kiểm thử phần mềm, nhưng ngăn khẳng định hiệu quả chẩn đoán đúng tầng trên cohort thật.

## 10. Các test quan trọng cần viết khi triển khai

- **Data leakage:** gắn chuỗi sentinel vào reports/aliases/folds rồi chứng minh input/tokenized prompt không chứa nó; train completion vẫn có target đúng chỗ.
- **Missingness:** ô thiếu bulging giữ null; âm tính thật giữ 0; cả hai sinh nội dung khác nhau.
- **Domain:** Pfirrmann 1–5, Modic 0–3, nhị phân 0/1; giữ grade 5 gốc trong contract. Nếu vision sau này gộp 4+5 để học, đó là lớp dẫn xuất có version, không sửa dữ liệu gốc.
- **Anatomy:** sai/trùng tầng bị từ chối; orientation/tọa độ kiểm bằng phantom và ảnh thật; không tự đổi mapping để tối đa khớp report.
- **Training:** prompt bị mask, answer không mất; pad/EOS đúng; image tokens và grid tương thích; save/reload/resume.
- **Semantic:** input không có bên thì output thêm bên bị flag; Modic type không biến thành severity; không có nhãn rễ thì không khẳng định rễ bình thường/bị chèn ép.
- **Counterfactual:** đổi đúng một fact phải đổi nội dung tương ứng, không gây đổi bệnh/tầng khác; đổi thứ tự keys không làm đổi nghĩa.
- **Robustness:** toàn âm tính, tất cả missing, nhiều tầng dương, hai nhãn cùng dương, value/status xung đột, context dài, backend timeout, output bị cắt.
- **Fallback:** lỗi lặp lại kết thúc hữu hạn, có trạng thái và trace; không mất ca trong batch.
- **Evaluation:** metric mẫu số đúng, không đếm năm tầng thành năm bệnh nhân độc lập; report các ca bị từ chối và fallback.

Các test này xác minh hành vi có rủi ro thực chất; không cần viết test cho mọi thay đổi văn bản tài liệu. Không công bố số lỗi thấp hơn ngưỡng nào cho đến khi có dữ liệu thực nghiệm đủ và protocol đã khóa.

## 11. Tài liệu và nguồn đối chiếu

- Nguồn trong repo: `src/data/v1_dataset.py`, `src/train_v1.py`, `src/models/v1_vlm.py`, `configs/v1_config.yaml`, `.gitignore`, `requirements.txt`, `scripts/consolidate_dataset.py`, `scripts/make_colab_notebook.py`.
- Định nghĩa dữ liệu: `dataset/README_dataset.md`, `dataset_local/grading/README.md`, `dataset_local/localize/README.md`.
- Dữ liệu được đọc: `dataset/dataset_master.csv`, `dataset/dataset_patients.jsonl`, `dataset/data_of_1patient.json`; các output audit ở `output/annotation_audit/`.
- Audit trước: `docs/repository_audit_2026-09-17.md`; một số số liệu trước chuẩn hóa trong tài liệu đó đã được cập nhật lại ở phần 2 của kế hoạch này.
- [Qwen2.5-VL / Transformers](https://huggingface.co/docs/transformers/model_doc/qwen2_5_vl): processor và input đa phương thức.
- [TRL SFTTrainer](https://huggingface.co/docs/trl/sft_trainer): mask assistant/completion và rủi ro truncation VLM.
- [vLLM structured outputs](https://docs.vllm.ai/en/latest/features/structured_outputs/): ràng buộc output theo schema/grammar.

Tài liệu này phân biệt ba mức: **đã quan sát trong repo**, **đề xuất triển khai**, **cần đo thực nghiệm**. Nó không xác nhận đã train V1, đã xây V2 hoặc đã sửa nhãn dữ liệu.

## 12. Kaggle Free là môi trường train chính

### 12.1. Quyết định và phạm vi

Chủ dự án đã chọn **Kaggle Free**. Code `.py` và notebook được giao cùng một phiên bản trên repo public `https://github.com/kttt294/MRI-report-generator`. Kaggle chỉ thực thi code; dữ liệu và notebook chạy thực nghiệm giữ private. [README cho người mới](README_KAGGLE.md) là hướng dẫn thao tác đi kèm.

- Kaggle là đường chạy bắt buộc phải nghiệm thu trước; Colab là đường dự phòng, không còn là điều kiện để train.
- Google Drive giữ bản gốc. Nếu ảnh chỉ có ở Drive, có thể dùng Colab CPU để chuyển một lần sang Kaggle sau khi xác thực hai tài khoản; không cần làm vậy trong từng run.
- Không coi tài nguyên miễn phí là được bảo đảm. Preflight kiểm tra GPU, VRAM và dung lượng thực tế; người dùng đọc quota còn lại trên giao diện Kaggle để đặt time budget. Không giả định script có thể tự đọc quota tài khoản, không hard-code một quota cố định.
- Theo thông báo Kaggle, P100 ngừng từ 15/09/2026; cấu hình GPU mục tiêu ban đầu là T4, được cấp qua lựa chọn T4 ×2 nếu có. MVP dùng **một GPU** đã smoke-test; dùng hai GPU là cấu hình phân tán riêng, không cộng VRAM một cách tự động. [Thông báo chính thức](https://www.kaggle.com/discussions/product-announcements/735239)

### 12.2. Hai dataset private, không upload ảnh vào GitHub

| Dataset đề xuất | Nội dung | Được dùng bởi |
|---|---|---|
| `lumbar-mri-images` | `nifti/`, khoảng 9 GB, giữ cấu trúc ảnh nguồn; thêm manifest file/checksum khi đóng gói | V1; vision engine sau này |
| `lumbar-mri-annotations` | `grading/`, `localize/`, `folds/`, `reports_json/`, `README.md`; `reports_text/` chỉ khi cần tái lập export legacy | V1 và V2 report |

Ảnh ít đổi, annotations có thể cập nhật độc lập. Mỗi run phải lưu owner/slug/version của cả hai dataset và hashes của manifest/code/config, không chỉ lưu tên dataset. Kiểm tra tập patient_id và volume giữa hai phiên bản; mismatch phải fail trước train.

- Không cần upload `dicom/` và `reports_text_v1_reference/` cho hướng triển khai hiện tại.
- `dataset/` là dữ liệu derived do code tái tạo trong ổ ghi được, khác với các folder nguồn trên Drive/Kaggle.
- Ảnh thư mục Drive chỉ xác nhận có `nifti/`, chưa xác nhận cấu trúc con. Resolver phải tìm đúng `volume`; nhiều file ứng viên thì báo lỗi thay vì chọn đầu tiên.
- Upload qua UI bằng ZIP hoàn chỉnh hoặc Kaggle CLI; không tự chia một archive thành `.001/.002` trừ khi có công cụ ghép lại. Giữ `.nii.gz` và tên file nguồn; kiểm tra những gì Kaggle thực sự đã giải nén trước khi dùng.
- Quy định bộ dữ liệu gốc có điều khoản về dịch vụ bên thứ ba. Việc triển khai Kaggle phải nằm trong phạm vi quyền sử dụng đã được chủ dữ liệu cho phép; private không tự thay đổi điều khoản đó. Không có bước upload tự động trong việc viết tài liệu này.

Kaggle cho tạo dataset private và nhận upload archive; giới hạn hiện tại được đối chiếu trong [tài liệu Datasets](https://www.kaggle.com/docs/datasets). Không dùng public như một cách xử lý thiếu dung lượng.

### 12.3. Đường dẫn và storage contract

| Vị trí | Vai trò và quy tắc |
|---|---|
| `/kaggle/input/...` | Hai dataset nguồn và outputs cũ được Add Input; chỉ đọc, lấy path thực tế từ giao diện |
| `/kaggle/working/repo` | Checkout code đúng ref; không tự pull giữa run |
| `/kaggle/working/derived` | CSV/JSON/manifest cần giữ để tái lập, được sinh từ nguồn |
| `/kaggle/working/runs/<run_name>` | Checkpoint hoàn chỉnh, adapter, processor, config, environment receipt, logs và predictions |
| Thư mục tạm ngoài `/kaggle/working` | Cache model/base weights và ảnh trung gian có thể tái tạo; không trông đợi tồn tại ở phiên mới |
| Saved Notebook Version → Output | Bản kết quả đã được Kaggle lưu; phải mở kiểm tra trước khi coi là có thể resume |

Không chép toàn bộ 9 GB ảnh sang working một cách mặc định. Đọc NIfTI từ input; cache cục bộ chọn lọc nếu đo thấy cần, kiểm tra dung lượng và fingerprint. Chỉ để artifact cần lưu trong working, tránh đưa base weights/cache lớn vào output quota.

`/kaggle/working` không phải checkpoint remote được ghi bền liên tục. Auto-save code notebook không đồng nghĩa checkpoint đã được giữ qua mất phiên. Kaggle hiện nêu tối đa 20 GB outputs và phiên CPU/GPU tối đa 12 giờ; phải xem lại khi chạy vì giới hạn có thể đổi. [Kaggle Notebooks](https://www.kaggle.com/docs/notebooks)

### 12.4. Các tệp phải giao và entry point dự kiến

**Các tên dưới đây là deliverable triển khai; chưa coi tồn tại/chạy được chỉ vì được liệt kê trong plan.**

| Tệp | Nhiệm vụ |
|---|---|
| `notebooks/Kaggle_V1_Training.ipynb` | Clone/install/config/preflight/smoke/train/infer; không chứa implementation model |
| `notebooks/Kaggle_V2_Report.ipynb` | Tạo request JSON, chạy report template/LLM và đánh giá; không ép train khi chưa có target hợp lệ |
| `scripts/make_cloud_notebooks.py` | Sinh notebook mỏng cho Kaggle; tùy chọn sinh Colab từ cùng nguồn |
| `scripts/cloud_prepare.py` | Nhận images_root + annotations_root riêng, đọc input và xuất derived; resolve ảnh, đối chiếu hashes, không ghi vào input |
| `scripts/cloud_run.py` | Điều phối task, kiểm tra GPU, thời gian, checkpoint/resume và gọi module `.py` dùng chung |
| `configs/cloud_kaggle.yaml` | Đường dẫn, task, fold, run_name, smoke/full, max_runtime_minutes, resume_from, chính sách một GPU |
| `requirements-kaggle.txt` | Bộ dependencies khóa sau smoke test; không thay CUDA/PyTorch Kaggle tùy tiện |
| `docs/README_KAGGLE.md` | Hướng dẫn từ tài khoản đã verified đến upload, Add Input, chạy và lấy kết quả |

Cần sửa ETL để nhận đường dẫn nguồn/đích qua CLI; không dựa vào symlink duy nhất `dataset_local` khi ảnh và annotations nằm ở hai dataset khác nhau. Các CLI phải fail rõ nếu path/field không hợp lệ. Sinh các notebook cùng commit với `.py` và kiểm tra imports từ clone sạch, bao gồm mã trong `src/models/` hiện đang bị ignore nhầm.

Notebook chỉ có các cell ngắn: tham số → clone checkout → cài dependencies → chuẩn bị dữ liệu → smoke/train hoặc report → tóm tắt artifact. Mọi exception ở subprocess phải làm cell thất bại; không chạy tiếp khi prepare/install lỗi.

### 12.5. Luồng dùng Kaggle và quản lý phiên

1. Người dùng upload hai dataset ở chế độ private, ghi slug/version, kiểm tra file và quyền truy cập.
2. Import notebook Kaggle đã phát hành vào một notebook private; Add Input hai dataset, bật Internet cho clone/cài thư viện/tải model.
3. Chọn Accelerator GPU T4 ×2 nếu được cấp. Wrapper một GPU phải thiết lập device trước khi import torch; không khởi tạo huấn luyện đa GPU ngoài ý muốn.
4. Điền input paths thực tế, code ref, fold, run_name, chế độ smoke/full và resume_from. Không cần viết lại vòng lặp train trong cell.
5. Smoke test interactive bằng một số mẫu nhỏ: validate dữ liệu, đọc ảnh, forward/backward, save/reload. Xem VRAM/token và kiểm tra mọi bước đúng.
6. Job chính dùng Save Version → Save & Run All, chạy từ trạng thái sạch. Không giả định biến/file của phiên interactive có sẵn; cell bootstrap phải tự tái tạo môi trường từ code và inputs đã gắn.
7. Kết thúc có kiểm soát rồi mở version hoàn thành, kiểm tra Output và checkpoint. Tải backup hoặc gắn outputs đó vào phiên kế tiếp để resume.

Save & Run All có thể chạy nền, nhưng không vượt quota/giới hạn runtime. Quick Save lưu snapshot notebook; nếu dùng để giữ output interactive phải kiểm tra tùy chọn lưu outputs và xác nhận file trong version, không suy ra từ việc code đã được lưu. [Kaggle về chạy nền](https://www.kaggle.com/general/232625)

### 12.6. Checkpoint/resume phù hợp Kaggle Free

- Lưu checkpoint sau một số optimizer steps được cấu hình; giữ số checkpoint hữu hạn, kiểm tra ước lượng dung lượng trước train.
- Mỗi checkpoint có completion marker và danh sách file/checksum. Trainer ghi xong mới đánh dấu hợp lệ, không nhận checkpoint copy dở.
- Lưu optimizer/scheduler/RNG/trainer state cùng adapter/config/processor. Adapter cuối dùng inference, không thay thế trạng thái resume đầy đủ.
- Callback time budget chủ động save và kết thúc khi gần giới hạn đã đặt; chừa thời gian chạy các cell cuối và lưu outputs. Ngưỡng là cấu hình người dùng căn theo quota hiện tại, không hứa phiên luôn được đủ 12 giờ.
- Mất runtime đột ngột trước khi outputs được lưu có thể làm mất checkpoint chưa được lưu bền. Chạy theo chặng ngắn và kiểm tra Output sau mỗi chặng; không ghi README rằng checkpoint chắc chắn được cứu sau mọi interruption.
- Resume nhận path checkpoint thuộc saved outputs đã Add Input. Kiểm tra code/config/model/data versions và copy phần cần ghi sang run directory mới; không ghi vào input cũ.
- Cell inference/evaluation chỉ chạy khi checkpoint cần thiết tồn tại; test không được dùng chọn epoch/prompt. Hoàn thành train một chặng khác với đã hoàn thành toàn bộ thí nghiệm.

### 12.7. Điều kiện phát hành và nghiệm thu

| Bước | Bằng chứng bắt buộc |
|---|---|
| Code + notebook local | Test phần dữ liệu, masking, schema, checkpoint policy; notebook không có class/train loop riêng |
| Push GitHub public | Chỉ code/config/docs/fixtures giả lập; notebook cleared outputs; có ref cụ thể và clone sạch đủ mã |
| Dữ liệu Kaggle | Hai private dataset truy cập được, phiên bản/manifest khớp, không cần Drive trong run |
| Smoke test Kaggle | GPU/ảnh thật; loss hữu hạn, save/reload được; code không sửa tay trong cell |
| Saved version | Job chạy từ runtime sạch, có logs và artifact thực sự trong Output |
| Resume | Add Input output cũ và tiếp tục đúng global step/trạng thái, không bắt đầu lại từ 0 |
| V2 | Notebook template chạy không cần NIfTI/GPU; LLM/LoRA được ghi nhận riêng, không nhầm template thành train thành công |

Thử lỗi thiếu ảnh, path lồng, input read-only, Internet tắt, thiếu GPU, OOM, hết disk, code chưa push và checkpoint chưa hoàn chỉnh. Chưa có phiên Kaggle thật thì trạng thái chỉ là **đã chuẩn bị code/notebook**, không phải **đã nghiệm thu cloud**.


## 13. Đối chiếu sau triển khai

Các module đã được giao theo nhóm: V1 Dataset/ảnh/collator/model/trainer/inference; V2 contract/adapter/planner/template/backend/validator/CLI; target review và LoRA có điều kiện; notebook/config/CLI Kaggle; evaluation R0–R5 và V1; kiểm thử CPU. Bằng chứng, lệnh chạy và các bước còn phải nghiệm thu nằm ở [implementation_status.md](implementation_status.md).

Điều chỉnh triển khai cụ thể:

- Ontology, fact rendering và planner được gom trong `src/report/planner.py`, không tách nhiều file ngắn như tên gợi ý ban đầu.
- Validator MVP dùng catalog câu chính xác. Đây là policy bảo thủ, có thể từ chối paraphrase đúng; tỷ lệ từ chối không phải tỷ lệ lỗi lâm sàng. LoRA R5 cũng phải tuân thủ policy này.
- Ablation mặc định không repair để so sánh raw output; R3/R4/R5 vẫn ghi fallback riêng. Luồng production/research CLI thông thường cho tối đa một repair. R1/R2 cần bác sĩ đánh giá nội dung; chưa có NLP judge được xác nhận.
- R4 dùng cùng facts/catalog nhưng chia hai lần gọi; findings vừa sinh được giữ nguyên trong trace rồi kiểm tra toàn bộ output. Lỗi của stage trước không được che bởi stage sau.
- Đã thêm V2 Training notebook riêng; chưa train V2 từ SFT cũ hay tự đánh dấu target đã duyệt.
- Template/CPU tests hoàn thành không đóng M5/M6 thực nghiệm: cần chạy GPU, xem ảnh thật, duyệt target, khóa protocol, thực hiện blinded review và đo tỷ lệ lỗi. Không tự làm hoặc bịa số liệu cho các bước đó.
