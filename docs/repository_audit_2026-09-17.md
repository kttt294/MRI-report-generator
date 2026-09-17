# Đối chiếu repository với tài liệu bàn giao

Ngày kiểm tra: 17/09/2026. Phạm vi: working tree tại `healcare_a2i`, nhánh `main`, commit nền `2a38724`, bao gồm tệp chưa commit và tệp bị Git bỏ qua. Không kiểm tra trực tiếp Google Drive hoặc trạng thái mới nhất của remote.

## 1. Kết luận về trạng thái dự án

**Dự án đã có dữ liệu bảng hợp nhất có thể tái tạo, mã khung V1 và notebook Colab. Chưa có bằng chứng trong repo rằng V1 đã huấn luyện thành công; chưa đủ điều kiện coi pipeline là sẵn sàng bàn giao để chạy ngay. V2 vẫn là thiết kế nghiên cứu.**

Các phát hiện chắc chắn nhất:

- 1.235 dòng, 39 cột, 247 bệnh nhân và 5 tầng mỗi bệnh nhân là đúng.
- Cách chia thực tế xấp xỉ 60/20/20, không phải 70/10/20.
- Fold 1 có 147/50/50 bệnh nhân; SFT Việt có 142/49/47 báo cáo, SFT Anh có 140/49/47.
- Tệp nạp mô hình có trên máy nhưng bị `.gitignore` loại khỏi Git; người clone commit hiện tại sẽ thiếu mã này.
- Loader có thể huấn luyện bằng ảnh trống và đưa chuỗi `nan` vào văn bản đích; collator chưa mask phần prompt.
- Không có ảnh NIfTI, checkpoint, log huấn luyện hay kết quả đánh giá được lưu trong working tree được kiểm tra.
- Các CSV tiếng Anh đang sử dụng giống từng byte với bộ tham chiếu cũ được ghi chú không dùng huấn luyện.
- Một số kết quả C2M-DoT trong tài liệu không khớp PDF đi kèm.

## 2. Kiểm kê phần thực sự tồn tại

| Thành phần | Trên máy | Trong Git tại commit nền | Nhận xét |
|---|---|---|---|
| `README.md`, `requirements.txt`, `.gitignore` | Có | Có | README tập trung kiến trúc V2; chưa mô tả đầy đủ V1 hiện có |
| `configs/v1_config.yaml` | Có | Có | Qwen2.5-VL-3B, NF4, LoRA r=16, alpha=32, dropout=0.05, lr=2e-4, **5 epochs** |
| `src/data/v1_dataset.py` | Có | Có | Một ảnh 2D sagittal mỗi bệnh nhân; đọc master CSV |
| `src/models/v1_vlm.py`, `__init__.py` | Có | **Không** | Bị quy tắc `models/` ở `.gitignore:52` loại trừ |
| `src/train_v1.py` | Có | Có | Trainer, train/val, lưu adapter; chưa có đánh giá chất lượng báo cáo |
| `scripts/consolidate_dataset.py` | Có | Có | Hợp nhất grading, localization, folds, metadata, báo cáo; xuất 8 tệp dữ liệu |
| `scripts/make_colab_notebook.py` | Có | Có | Nội dung notebook sinh ra khớp notebook đang lưu |
| Notebook V1 | Có | Có | 17 cell, 8 cell code; tất cả chưa có execution count và output được lưu |
| Master CSV, patient JSONL, 6 SFT JSONL | Có | Không, có chủ ý | Dữ liệu được bỏ qua bởi Git |
| `dataset/data_of_1patient.json` | Có | Không | Mẫu minh họa một bệnh nhân; không được pipeline V1 tham chiếu |
| `dataset_local/` | Có | Không, có chủ ý | 31 CSV, 480 JSON báo cáo/metadata, một JSON metadata cohort, 4 README |
| NIfTI, cache ảnh V1 | **Không thấy** | Không | Không có cả thư mục `dataset_local/nifti` và `dataset/v1_cache_images` |
| Checkpoint/trọng số/log/kết quả thực nghiệm | **Không thấy** | Không | Không thể suy ra dự án đã có mô hình được huấn luyện ở nơi khác |
| Calibration, detector/classifier V2, generator JSON, evaluation | **Chưa có mã** | Không | Nội dung README là kế hoạch |
| Hai bài báo PDF | Có | Không | Bị `*.pdf` loại khỏi Git |
| Tài liệu phân tích bài báo, bản bàn giao | Có | Có | Tài liệu phân tích đang có thay đổi chưa commit từ trước |
| `docs/lora.md`, ảnh minh họa, `dataset/README_dataset.md` | Có | Chưa được theo dõi | Có nội dung mới/di chuyển chưa commit |
| `docs/Document.docx` | Có | Không | Ghi chú ngắn hỏi cách đo tính phù hợp của cụm từ bất định; không phải tài liệu thiết kế hoàn chỉnh |

Working tree trước khi kiểm tra đã có thao tác di chuyển/xóa `README_dataset.md`, hai ảnh trong `documents/`, thay đổi tài liệu phân tích và các tệp mới. Các thay đổi đó được giữ nguyên.

## 3. Đối chiếu dữ liệu

### 3.1. Những kiểm tra đã đạt

- Master: **1.235 dòng × 39 cột**, **247 patient_id**, không trùng khóa `(patient_id, level)`.
- Mỗi bệnh nhân có đúng năm tầng L1/L2, L2/L3, L3/L4, L4/L5, L5/S1.
- Nhãn grading và tọa độ voxel trong master khớp các bảng nguồn khi đối chiếu theo khóa.
- Patient JSONL: 247 đối tượng, tập patient_id khớp master, mỗi đối tượng có 5 tầng.
- Cả 5 fold không chồng patient_id giữa train/val/test; không có bệnh nhân bị chia tầng sang các tập khác nhau.
- 5 tập test rời nhau và phủ đủ 247 bệnh nhân, mỗi bệnh nhân xuất hiện trong test đúng một lần.
- Sáu tệp SFT không trùng bệnh nhân trong tệp, không sai split so với master và không có response rỗng.
- Chạy lại script hợp nhất với **đích đầu ra tạm**, giữ nguyên dữ liệu đang dùng: cả **8/8 tệp đầu ra giống từng byte** với bản hiện có.

Đây là kiểm tra tính nhất quán và tái tạo dữ liệu; không xác nhận tính đúng lâm sàng của nhãn, định vị hoặc bản dịch.

### 3.2. Số lượng chia tập thực tế

| Fold | Train | Validation | Test |
|---|---:|---:|---:|
| 1 | 147 | 50 | 50 |
| 2 | 147 | 50 | 50 |
| 3 | 148 | 50 | 49 |
| 4 | 148 | 50 | 49 |
| 5 | 148 | 50 | 49 |

| Fold 1 | Train | Validation | Test | Tổng |
|---|---:|---:|---:|---:|
| Toàn cohort | 147 | 50 | 50 | 247 |
| Có báo cáo Việt để xuất SFT | 142 | 49 | 47 | 238 |
| Có báo cáo Anh để xuất SFT | 140 | 49 | 47 | 236 |

Tên thật là `sft_fold1_train_vi.jsonl`, `sft_fold1_val_vi.jsonl`, `sft_fold1_test_vi.jsonl` và các tệp `_en`; không phải `train_vi.jsonl`, `val_vi.jsonl`, `test_vi.jsonl` như bản bàn giao.

### 3.3. Thiếu dữ liệu và sai mô tả schema

| Hạng mục | Đếm trực tiếp | Ảnh hưởng |
|---|---:|---|
| Có mô tả tiếng Việt | 238/247 bệnh nhân | 9 ca thiếu báo cáo Việt |
| Có kết luận tiếng Việt | 230/247 | 17 ca thiếu kết luận, trong đó 8 ca vẫn có mô tả |
| Có báo cáo tiếng Anh | 236/247 | 11 ca không có target Anh |
| Thiếu metadata cơ bản | 9/247 | Không thể gọi toàn bộ 247 ca là đầy đủ metadata/báo cáo |
| `disc_bulging` thiếu | 1/1.235 dòng | ETL đổi missing thành 0 trong patient JSONL; mất phân biệt “chưa biết” và “âm tính” |
| Modic | 0: 1.058; 1: 24; 2: 149; 3: 4 dòng | `dataset/README.md` ghi nhị phân 0/1 là sai; prompt SFT cũng không giữ subtype |
| Pfirrmann 1–5 | 36 / 689 / 418 / 88 / 4 dòng | Mất cân bằng mạnh, độ 5 chỉ có 4 mẫu |
| Localization được đánh dấu reviewed/verified | 195/1.235 tầng | 1.040 tầng còn là auto/auto_ok; không được mô tả tất cả là bác sĩ xác nhận |

**Không tìm thấy cột nhãn hẹp ống sống hoặc hẹp lỗ liên hợp trong master hay schema các CSV local.** `disc_narrowing` là hẹp khe đĩa đệm, không thay thế hai nhãn đó. Văn bản báo cáo có thể mô tả các bệnh lý này, nhưng chưa có bước trích xuất/thẩm định nhãn có cấu trúc cho mục tiêu V2.

README grading nguồn còn ghi `IVD_TO_LEVEL_VERIFIED = False`: ánh xạ nhãn grading sang tầng chưa được đối chiếu bằng mắt với ảnh theo tài liệu nguồn. Kiểm tra thứ tự tọa độ localization là một kiểm tra khác, không tự xác nhận được ánh xạ grading.

### 3.4. Nguồn báo cáo tiếng Anh

Cả `train.csv`, `val.csv`, `test.csv`, `missax2.csv` trong `dataset_local/reports_text` giống từng byte với các tệp cùng tên trong `reports_text_v1_reference`.

README của bộ reference ghi đây là bản dịch cũ và **“Không dùng để huấn luyện nữa.”** Trong khi đó ETL đọc chính `reports_text` để sinh master và SFT Anh. Cần xác nhận lại nguồn được chấp nhận trước khi chạy thí nghiệm tiếng Anh hoặc so sánh song ngữ; repo chưa chứa bằng chứng bộ dịch mới đã thay thế bộ cũ.

## 4. Các lỗi và thiếu sót của pipeline V1

### A. Mã model bị loại khỏi Git — chặn bàn giao

**Bằng chứng:** `.gitignore:52` dùng `models/`, khớp cả `src/models/`; `git ls-files src/models` không trả tệp. `src/train_v1.py:25` lại import `src.models.v1_vlm`.

**Hệ quả:** local có implementation NF4/LoRA, nhưng bản clone từ commit kiểm tra sẽ không có module cần thiết, kể cả đã cài đủ dependencies.

**Cách xử lý:** giới hạn ignore vào thư mục trọng số ở gốc hoặc thêm ngoại lệ cho mã `src/models`, sau đó đưa hai tệp mã vào Git.

### B. Thiếu MRI bị thay bằng ảnh xám — làm sai thí nghiệm

**Bằng chứng:** `src/data/v1_dataset.py:172` trả ảnh RGB màu `(30,30,30)` khi không tìm thấy NIfTI. Smoke check với dữ liệu hiện tại xác nhận ảnh 384×384 có cả ba kênh đồng nhất giá trị 30.

Working tree không có NIfTI hoặc cache ảnh. Vì vậy, với đường dẫn local hiện tại, loader có thể cấp ảnh placeholder cho toàn bộ mẫu mà không báo lỗi.

**Cách xử lý:** kiểm tra đầy đủ đường dẫn ảnh trước train; dừng khi ảnh thiếu. Chỉ cho phép placeholder trong chế độ kiểm thử được bật rõ ràng.

### C. Báo cáo thiếu biến thành chuỗi `nan` — làm bẩn target

**Bằng chứng:** `src/data/v1_dataset.py:119–132` chuyển giá trị thiếu từ pandas thành chuỗi trước khi kiểm tra rỗng. `str(NaN)` trở thành chuỗi không rỗng nên ca thiếu không bị loại.

Smoke check cho thấy loader thực tế vẫn nạp **147/50/50 ca** cho cả hai ngôn ngữ. Số target chứa nguyên một dòng `nan`:

| Ngôn ngữ | Train | Validation | Test |
|---|---:|---:|---:|
| Việt | 10 | 4 | 3 |
| Anh | 7 | 1 | 3 |

Ở Việt, số trên gồm cả ca thiếu riêng kết luận. Cần xử lý missing trước khi chuyển chuỗi và thống nhất tiêu chí target hợp lệ giữa ETL và Dataset.

### D. Loss chưa chỉ tính trên báo cáo

**Bằng chứng:** `src/train_v1.py:61–69` clone toàn bộ input_ids, chỉ mask padding và token ảnh nếu tokenizer có thuộc tính tương ứng. Không xác định đoạn assistant để mask prompt.

Probe collator độc lập với processor giả lập: input `[11,99,12,21,22,0]`, trong đó 11/12 là prompt, 99 là ảnh, 21/22 là answer, 0 là pad. Labels thực tế `[11,-100,12,21,22,-100]`: prompt vẫn được tính loss.

**Cách xử lý:** tạo assistant-only labels theo template/model; kiểm tra rằng còn token target sau cắt độ dài. Probe này xác nhận logic mask; chưa phải chạy forward/backward trên Qwen thật.

### E. Giới hạn độ dài và xử lý ảnh chưa được kiểm chứng

- Collator cố định `max_length=512` cho cả ảnh, prompt và report; không thống kê số report bị cắt hoặc token trả lời còn lại.
- Notebook sinh tối đa 300 token, trong khi mục tiêu bàn giao là 200–250 từ. Hai đơn vị không tương đương; chưa có kiểm tra completeness.
- Dataset lấy trung bình `voxel_k` của 5 tầng rồi cắt `vol[:,:,k]`, xoay 90 độ và ép ảnh về 384×384. Không kiểm tra affine/orientation và không dùng trực tiếp `volume` để tìm tệp gốc.
- Đây là V1 cấp bệnh nhân, một ảnh và một báo cáo toàn ca; chưa có crop, classifier hoặc loss độc lập cho từng tầng.

Những mục này là rủi ro cần kiểm chứng với ảnh/processor thật; chưa đo tỷ lệ report bị cắt hoặc xác nhận ảnh bị sai hướng vì local không có NIfTI.

### F. Cấu hình T4 và khả năng tái lập

- Config và notebook đều dùng **5 epochs**; bản bàn giao ghi 10.
- Config ghi T4 hỗ trợ BF16 tốt; notebook inference cố định `torch.bfloat16`. T4 không có hỗ trợ BF16 native như Ampere. Tham khảo [CUDA về BF16](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/mathematical-functions.html) và [thông số T4](https://www.nvidia.com/en-gb/data-center/tesla-t4/).
- Loader chọn NF4 compute dtype theo GPU, nhưng model dtype rơi về float32 khi không có BF16; `model.torch_dtype` trong YAML không được đọc. Trainer tắt BF16 khi không hỗ trợ nhưng `fp16` vẫn mặc định false. Cần kiểm tra nhất quán dtype trên GPU đích.
- Inference nạp lại base model không có cấu hình lượng tử hóa 4-bit. Vì vậy không thể suy ra chi phí inference giống cấu hình train.
- Requirements chỉ đặt cận dưới, không khóa bộ phiên bản đã chạy; `transformers>=4.45.0` không tự chứng minh mọi phiên bản được chấp nhận đều hỗ trợ Qwen2.5-VL. Có [tài liệu chính thức Qwen2.5-VL ở Transformers 4.49.0](https://huggingface.co/docs/transformers/v4.49.0/model_doc/qwen2_5_vl).
- Notebook clone/pull nhánh hiện tại thay vì pin commit; kết quả có thể thay đổi theo thời điểm chạy.
- Máy kiểm tra dùng Torch 2.10.0+cpu, không có CUDA và thiếu nibabel; chưa thực hiện huấn luyện GPU hoặc đo VRAM.

### G. Chưa có đánh giá và bằng chứng hoàn thành baseline

Không có script inference trên toàn test set, lưu predictions, BLEU/ROUGE/Clinical F1, calibration metrics, confidence intervals hay tổng hợp out-of-fold. Trainer có thể tính validation loss mặc định; điều đó không thay thế đánh giá báo cáo.

Notebook chỉ minh họa inference một ca và copy adapter cuối cùng sang Drive. Tất cả cell code không có output/execution count được lưu. Không thấy checkpoint, log hay bảng kết quả. Kết luận phù hợp là **“đã viết mã V1 ban đầu, chưa xác nhận chạy thành công”**.

## 5. V2 và khoảng cách với mục tiêu nghiên cứu

Chưa có các thành phần: định vị/classifier huấn luyện được; temperature scaling; conformal score/quantile/prediction set; calibration split/protocol; schema JSON bất định; kiểm tra JSON; generator bị ràng buộc bởi findings; giao diện bác sĩ sửa; Clinical F1 theo tầng/mức độ; cơ chế đa góc nhìn/thiếu axial; thực nghiệm ablation.

Patient JSONL và SFT hiện có là tiền xử lý nhãn thật sang text. Chúng chưa phải JSON dự đoán đã hiệu chuẩn. V1 đọc master trực tiếp, không đọc các tệp SFT tabular-to-text đó.

Cần sửa các câu “triệt tiêu/giải quyết triệt để ảo giác”: conformal prediction bảo đảm coverage của tập dự đoán dưới các giả định phù hợp, không bảo đảm mọi chẩn đoán hoặc văn bản LLM đều đúng. Việc hiệu chuẩn theo tầng cũng cần xét sự phụ thuộc của năm tầng trong cùng bệnh nhân. [Nguồn phương pháp](https://arxiv.org/abs/2107.07511).

## 6. Đối chiếu tài liệu và nguồn học thuật

### 6.1. C2M-DoT

Đã đối chiếu bảng 2, trang 9 của `2310.05355v1.pdf`, bao gồm kiểm tra hình render của trang:

| Metric | Tài liệu phân tích đang ghi | PDF trong repo |
|---|---:|---:|
| IU X-Ray BLEU-4 | 0.198 | **0.1593** |
| IU X-Ray ROUGE-L | 0.402 | **0.3803** |
| IU X-Ray METEOR | 0.201 | **0.2037** |
| MIMIC-CXR BLEU-4 | 0.126 | **0.1925** |
| MIMIC-CXR ROUGE-L | 0.298 | **0.3850** |

Các ý về đa góc nhìn, DoT, Gumbel-Softmax và nhất quán xuyên phương thức có trong PDF. Tuy nhiên, số liệu nêu trên không khớp bản v1 đi kèm. Nếu lấy từ một phiên bản khác, cần ghi rõ nguồn và phiên bản đó. Chưa tìm thấy dẫn chứng trực tiếp cho câu “bỏ DoT giảm hơn 28%” trong phần đã đối chiếu.

Ứng dụng frontal/lateral X-ray sang sagittal/axial MRI là giả thuyết chuyển giao phương pháp, chưa được repo thực nghiệm chứng minh và không bảo đảm khôi phục thông tin axial bị thiếu.

### 6.2. ViMed-PET

Các số 2.757 ca, 8.271 cặp phân vùng, bộ đánh giá 80 bệnh nhân/398 tổn thương, điểm baseline và cấu hình CT-ViT + Mistral-7B BLEU-4 58.07, ROUGE-L 72.74, F1-TPF 22.65 khớp các mục/bảng được kiểm tra trong PDF.

Nhưng 58.07 là kết quả một cấu hình, không phải BLEU-4 cao nhất toàn bảng: cấu hình Cosmos + Mistral O-G-C có 58.87. Kết quả PET/CT không tự chứng minh hiệu quả trên MRI cột sống hoặc trên bộ dữ liệu của dự án.

### 6.3. Các tài liệu còn lại

- `dataset/README.md` ghi đúng tên/số SFT, nhưng sai thang Modic và mô tả `volume` như kích thước; `volume` là định danh tệp ảnh.
- `docs/paper_translation_and_comparative_analysis.md` gọi `reports_json`, `reports_text` là cột master; đó là thư mục nguồn, các cột thực là `report_vi_*`, `report_en`.
- Chưa có trích dẫn cụ thể trong repo cho tỷ lệ lỗi dịch 12–25%, các khẳng định tuyệt đối về dữ liệu MRI tiếng Anh, hoặc khẳng định Qwen rất giỏi báo cáo MRI tiếng Việt.
- `docs/lora.md` bàn về RL/DPO/GRPO, merge, GGUF/AWQ/GPTQ và tốc độ CPU/GPU. Repo chưa có implementation hoặc benchmark cho các tuyên bố này; không nên trình bày như năng lực đã đạt.
- `docs/T1 vs T2.png` có **dấu hiệu nhãn T1/T2 bị đảo**: hình trái có dịch sáng, hình phải có dịch tối. Đây là nhận xét về minh họa, cần xác nhận bằng nguồn ảnh/sequence gốc. Quy luật thường dùng: dịch tối ở T1, sáng ở T2; [tài liệu MRI của Cambridge](https://assets.cambridge.org/97811070/14046/excerpt/9781107014046_excerpt.pdf).
- Ảnh JPG là bảng minh họa T1/T2 vùng não; ảnh trong `docs/image/` là sơ đồ chữ về pipeline ViMed-PET, không phải kết quả thực nghiệm của dự án.
- Các lệnh kiểm tra, đường dẫn `DiscAnchor/...`, `tests/test_preprocessing.py` trong README nguồn trỏ tới dự án tiền xử lý bên ngoài; không tồn tại trong repo này. Cần ghi rõ phụ thuộc bàn giao.
- Notebook mount Drive và tạo symlink; không có bước tự tải đầy đủ NIfTI từ URL chia sẻ như cách diễn đạt trong bản bàn giao.

## 7. Thứ tự xử lý đề xuất

1. Đưa mã `src/models` vào Git; sửa loader để báo lỗi khi thiếu ảnh, loại/chuẩn hóa target thiếu, và mask loss đúng phần assistant.
2. Xác nhận bộ ảnh khả dụng, mapping ảnh–bệnh nhân–tọa độ, nguồn tiếng Anh hợp lệ; bảo toàn missing label thay vì đổi thành âm tính.
3. Đồng bộ bản bàn giao và README về số mẫu, tên tệp, epochs, nhãn hiện có và trạng thái “chưa xác nhận huấn luyện”.
4. Pin môi trường/commit; kiểm tra token budget, dtype và forward/backward bằng ảnh thật trên GPU đích; sau đó mới chạy V1 đầy đủ.
5. Xây dựng inference toàn tập và đánh giá lâm sàng; lưu predictions, cấu hình, checkpoint và kết quả tái lập.
6. Chốt nhãn V2 và protocol calibration trước khi phát triển classifier/JSON/generator và so sánh V1–V2.

## 8. Bằng chứng và giới hạn kiểm tra

Đã đọc toàn bộ các tệp Python, config, requirements, notebook và tài liệu Markdown của repo; kiểm kê tệp Git/ignored; kiểm tra dữ liệu ở mức khóa/schema/số lượng; chạy loader và collator probe trên CPU; tái tạo dữ liệu vào thư mục tạm; đối chiếu các phần liên quan trong hai PDF và xem ba ảnh minh họa.

Không đọc để chẩn đoán từng bệnh nhân; không xác nhận chất lượng lâm sàng của toàn bộ báo cáo; không kiểm tra DICOM/NIfTI, GPU, Drive, trọng số từ xa hoặc lịch sử thực nghiệm ngoài working tree.

Bằng chứng cục bộ và các phép kiểm tra có thể chạy lại ở `scratch/repo_audit/`: `check_repo.py`, `smoke_checks.py`, `check_rebuild.py`, `evidence.json`, `smoke.json`, `rebuild_check.json`, `integrity.json`. Thư mục này nằm trong vùng Git ignore, không đưa dữ liệu bệnh nhân vào báo cáo công khai.

Đợt đối chiếu chỉ bổ sung báo cáo này và tệp kiểm tra tạm; chưa sửa mã huấn luyện, bản bàn giao hay dữ liệu gốc.
