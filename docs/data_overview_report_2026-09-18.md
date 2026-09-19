# Báo cáo tổng quan và chất lượng dữ liệu PSPINES

Ngày lập: 18/09/2026
Phạm vi: working tree local của `healcare_a2i`; báo cáo này không đọc Google Drive/Kaggle và không có NIfTI trong repo.
Notebook tái lập: [`notebooks/Data_Overview_Audit.ipynb`](../notebooks/Data_Overview_Audit.ipynb)

## Tóm tắt điều hành

Bộ dữ liệu hiện có là một cohort bảng ở **grain một dòng cho một bệnh nhân–một tầng đĩa đệm**, gồm **1.235 dòng, 247 bệnh nhân và 5 tầng mỗi bệnh nhân**. Đây là nền tảng tốt để kiểm thử adapter và report engine V2, nhưng chưa đủ để kết luận rằng một report sinh ra là đúng trên ảnh MRI.

Có bốn điểm cần giữ rõ khi dùng dữ liệu:

1. Có **một ô `disc_bulging` bị thiếu** trong CSV nguồn. JSON legacy đã từng biến ô này thành `0`; adapter hiện tại phục hồi thành `null` khi đối chiếu CSV.
2. Chỉ **238/247** bệnh nhân có findings tiếng Việt và **230/247** có đủ cả findings lẫn impression. Không được biến phần văn bản thiếu thành câu âm tính.
3. Audit đối chiếu grading với văn bản đánh dấu **131/247 bệnh nhân có ít nhất một khác biệt diễn đạt/nhãn**, trong đó 43 ca ưu tiên A và 112 ca ưu tiên B. Đây là tín hiệu cần rà soát, chưa phải bằng chứng 131 ca gán nhãn sai; văn bản bác sĩ có thể chứa thông tin ngoài tám nhãn grading.
4. `IVD_TO_LEVEL_VERIFIED = False` trong tài liệu grading nguồn. Vì vậy các kết luận theo tầng hiện phải ghi `needs_review` cho đến khi xác nhận quy ước mapping.

Kết luận sử dụng: dữ liệu **đủ để chạy ETL, kiểm thử phần mềm, baseline template V2 và chuẩn bị thí nghiệm**; **chưa đủ để train report engine tự do hoặc tuyên bố chất lượng lâm sàng**. V1 còn cần ảnh NIfTI thật và nghiệm thu trên Kaggle.

## 1. Nguồn, grain và lineage

Nguồn được đối chiếu:

| Nguồn                             | Vai trò                                                        |
| ---------------------------------- | --------------------------------------------------------------- |
| `dataset/dataset_master.csv`     | Bảng dẫn xuất 1.235 dòng, dùng cho profile và V1 metadata |
| `dataset/dataset_patients.jsonl` | Record cấp bệnh nhân, 247 dòng, dùng adapter V2            |
| `dataset_local/grading/`         | Grading nguồn theo tầng                                       |
| `dataset_local/localize/`        | Tọa độ và volume nguồn                                     |
| `dataset_local/folds/`           | Chia train/val/test theo bệnh nhân                            |
| `dataset_local/reports_json/`    | Báo cáo Việt và metadata                                    |
| `dataset_local/reports_text/`    | Báo cáo tiếng Anh được ETL đọc hiện tại               |
| `output/annotation_audit/`       | Kết quả audit đối chiếu grading–văn bản trước đó    |

Khóa hợp lý là `(patient_id, level)`. Kiểm tra hiện tại cho thấy không có duplicate ở khóa này và mỗi bệnh nhân có đúng năm level: `L1/L2`, `L2/L3`, `L3/L4`, `L4/L5`, `L5/S1`. Có 247 volume khác nhau trong master; điều này mới xác nhận metadata volume, chưa xác nhận file NIfTI tồn tại vì ảnh không có trong working tree.

ETL tạo dữ liệu dẫn xuất từ `dataset_local` sang một thư mục đích riêng. Nguồn không được ghi đè. JSON request V2 được tách khỏi reports, folds, demographics và tọa độ; đó là contract trung gian cho vision engine sau này.

## 2. Quy mô và chia tập

| Hạng mục                                   | Số lượng |
| -------------------------------------------- | ----------: |
| Bệnh nhân                                  |         247 |
| Dòng patient–level                         |       1.235 |
| Mỗi level                                   |         247 |
| Volume được tham chiếu                   |         247 |
| Cả findings Việt                           |         230 |
| Findings Việt, kể cả ca thiếu impression |         238 |
| Impression Việt                             |         230 |
| Báo cáo tiếng Anh                         |         236 |
| Kỹ thuật chụp Việt                       |         228 |

Chia Fold 1 ở grain bệnh nhân là **147 train / 50 val / 50 test**. Các fold còn lại lần lượt là 147/50/50, 148/50/49, 148/50/49 và 148/50/49. Trong pipeline V1, điều kiện có đủ hai section Việt loại còn 137 train / 46 val / 47 test; đây là số ca đủ target, không phải số ca đã được bác sĩ adjudicate.

Các fold không chồng bệnh nhân; không phát hiện patient bị chia các level sang nhiều split trong cùng một fold. Đây là điểm mạnh cho đánh giá patient-level.

## 3. Schema và missingness

Master hiện có 42 cột, gồm grading, localization, fold, metadata và report aliases. Các trường report chuẩn hóa gồm `report_vi_technique`, `report_vi_findings`, `report_vi_impression`; alias cũ vẫn được giữ để tương thích.

Missingness quan sát được ở master:

| Trường                                                             | Thiếu theo dòng | Diễn giải                                                                   |
| -------------------------------------------------------------------- | ----------------: | ----------------------------------------------------------------------------- |
| `disc_bulging`                                                     |           1/1.235 | Missing grading thật;`null`, không phải âm tính                        |
| `sub_id`, `sex`, `age_at_scan`, `birth_year`, `study_date` |             9/247 | Thiếu theo ca bệnh nhân, lặp lại trên 5 dòng -> 45 dòng thiếu        |
| `report_vi_findings`                                               |             9/247 | Tương ứng 9 bệnh nhân thiếu findings                                    |
| `report_vi_impression`                                             |            17/247 | Tương ứng 17 bệnh nhân thiếu impression                                 |
| `report_en`                                                        |            11/247 | Tương ứng 11 bệnh nhân thiếu tiếng Anh                                 |
| `report_vi_technique`                                              |          95/1.235 | Không nên dùng làm input vision nếu chưa xác minh acquisition metadata |

Các alias report mới/cũ phải đồng nhất. Adapter sẽ báo lỗi nếu canonical và alias cùng có giá trị nhưng khác nhau. Giá trị `null`, chuỗi rỗng, `nan`, `none` và `null` không được suy ra thành `0`.

## 4. Phân bố tám trường grading

| Trường              | Phân bố quan sát                |
| --------------------- | ---------------------------------- |
| `pfirrmann_grade`   | 1: 36; 2: 689; 3: 418; 4: 88; 5: 4 |
| `modic`             | 0: 1.058; 1: 24; 2: 149; 3: 4      |
| `disc_herniation`   | 0: 1.147; 1: 88                    |
| `disc_bulging`      | 0: 906; 1: 328; 1 missing          |
| `disc_narrowing`    | 0: 1.176; 1: 59                    |
| `spondylolisthesis` | 0: 1.194; 1: 41                    |
| `up_endplate`       | 0: 1.120; 1: 115                   |
| `low_endplate`      | 0: 1.163; 1: 72                    |

Mất cân bằng lớp là rõ ràng: Pfirrmann 5 có 4/1.235 tầng; Modic 3 có 4; spondylolisthesis dương 41; disc narrowing dương 59. Accuracy tổng thể sẽ gây hiểu nhầm. V1/Vision cần báo cáo sensitivity/recall theo nhãn, patient-level split và khoảng tin cậy; V2 report engine không nên biến các nhãn này thành severity mới.

Tài liệu cũ từng mô tả Modic là nhị phân; dữ liệu thực tế có type 0–3. Contract hiện tại đã sửa theo miền 0–3. `disc_narrowing` là hẹp khe đĩa đệm, không phải nhãn hẹp ống sống hoặc hẹp lỗ liên hợp; các tình trạng đó chỉ xuất hiện trong văn bản report và chưa có target có cấu trúc.

## 5. Grading so với report và vấn đề hai người gán nhãn

Audit hiện có cho thấy các mức độ khác biệt sau:

| Chỉ báo                                                 | Số lượng |
| --------------------------------------------------------- | ----------: |
| Bệnh nhân có ít nhất một conflict                   |     131/247 |
| Conflict không phải chỉ khác chiều cao/kích thước |      86/247 |
| Conflict “absence” rộng cần xem lại                  |      21/247 |
| Ưu tiên A: lỗi bệnh nhân hoặc mâu thuẫn nội bộ  |          43 |
| Ưu tiên B: khác mapping/định nghĩa/diễn đạt      |         112 |
| Ưu tiên C: chưa có cờ xung đột                     |          83 |

### Ý nghĩa của A và B

Đây là nhãn **ưu tiên của hàng đợi audit**, không phải nhãn đúng/sai và không phải hai bác sĩ khác nhau:

- **A — `A_patient_or_internal`**: cần xem trước vì có dấu hiệu mâu thuẫn mạnh ở cấp bệnh nhân hoặc ngay trong report. Cụ thể gồm: report mô tả tổn thương dạng displacement nhưng toàn bộ `disc_herniation` và `disc_bulging` đều bằng 0; mismatch ở `spondylolisthesis` hoặc `modic`; hoặc cùng report tự chứa tín hiệu trái ngược. Đây là nhóm 43/238 bệnh nhân có report được đưa vào queue.
- **B — `B_level_or_definition`**: chưa có dấu hiệu mâu thuẫn mạnh như A, nhưng có khác biệt cần xác minh về tầng, định nghĩa hoặc cách diễn đạt. Ví dụ report nói thoát vị/phình ở một tầng nhưng grading dương ở tầng khác; khác nhau giữa bulging và herniation; hoặc mismatch `disc_narrowing`. Đây là nhóm 112/238 bệnh nhân.
- **C — `C_no_flag_not_adjudicated`**: không bị parser gắn cờ trong lần audit đó, nhưng chưa có nghĩa là đã được bác sĩ xác nhận. Có 83/238 bệnh nhân.

Mẫu số của A+B+C là **238**, tức số bệnh nhân có report tiếng Việt để đối chiếu; 9 bệnh nhân thiếu findings nên không được xem là “không có conflict”.

Theo feature, số bệnh nhân có mismatch được audit ghi nhận là: disc herniation 18, disc bulging 39, spondylolisthesis 17, Modic 5, disc narrowing 69 và endplate-any 15. Disc narrowing có chênh lệch cao nhất trong bộ parser hiện tại. Tuy nhiên các số này là **patient có ít nhất một khác biệt**, không phải tỷ lệ lỗi trên từng nhãn và không chứng minh ai đúng.

Audit parser đã kiểm tra các trường hợp dễ nhầm như phủ định toàn câu, Schmorl, thành đốt sống, fissure, laterality, root level và từ “lồi/phình”. Dù vậy parser không thay thế việc bác sĩ adjudicate. Không nên gán lại toàn bộ dataset chỉ vì 131 ca có khác biệt. Nên:

1. adjudicate 43 ca ưu tiên A trước;
2. lấy mẫu có phân tầng trong 112 ca ưu tiên B để xác định phần lớn là khác ontology hay sai label;
3. khóa glossary và quy ước `disc_herniation`/`disc_bulging`/`Modic` trước khi train;
4. báo cáo kết quả theo “đồng thuận grading–report đã rà soát”, không gọi mọi mismatch là annotation error.

## 6. Localization và khả năng dùng cho V1

Trong 1.235 dòng localization, 195 được đánh dấu `reviewed/verified` và 1.040 là `auto/auto_ok`. Vì vậy tọa độ đủ để làm input thử nghiệm, nhưng không nên coi toàn bộ là ground truth giải phẫu đã được bác sĩ xác nhận.

`voxel_i/j/k` gắn với `volume` nguồn và affine của file NIfTI. Không được mặc định `voxel_k` là trục sagittal sau khi đổi orientation. Loader V1 hiện canonicalize RAS, xử lý obliquity và dùng tọa độ annotation chỉ ở chế độ được chọn; ảnh thiếu sẽ làm pipeline dừng, không dùng ảnh xám thay thế.

Working tree không có NIfTI, cache ảnh hay checkpoint. Do đó chưa có kết luận nào về chất lượng lát MRI, orientation thực tế, coverage chuỗi T1/T2, VRAM hay loss của Qwen2.5-VL.

## 7. Giá trị của thư mục SFT legacy

`dataset/sft_data` có giá trị cho:

- tái lập pipeline cũ;
- khảo sát văn phong và cấu trúc report;
- baseline prompt/report overlap;
- audit xem response có khớp report nguồn.

Nó chưa phù hợp làm target V2 mặc định vì report chứa nhiều thông tin ngoài tám grading, có thể bao gồm hẹp ống sống, rễ, laterality và diễn đạt không được input JSON hỗ trợ. Trong audit, các report có mentions ngoài schema như canal stenosis 217, root effect 220, annular fissure 114, laterality 238 và foraminal stenosis 57 bệnh nhân. Vì vậy không xóa SFT legacy, nhưng phải gắn nhãn rõ là legacy/unreviewed và không tự đưa vào LoRA V2.

## 8. Đánh giá mức sẵn sàng theo mục tiêu

| Mục tiêu                       | Mức sẵn sàng                      | Điều kiện còn thiếu                                                  |
| -------------------------------- | ------------------------------------ | ------------------------------------------------------------------------- |
| ETL và adapter giữ missingness | Sẵn sàng kiểm thử                | Tiếp tục hash nguồn và kiểm tra mỗi bản dữ liệu mới             |
| V2 template từ JSON grading     | Sẵn sàng làm baseline             | Xác nhận mapping/ontology trước khi bỏ`needs_review`               |
| V2 LLM prompt-only               | Có khung chạy                      | Cần benchmark R1–R4 và bác sĩ đánh giá mù                        |
| V2 LoRA                          | Chưa sẵn sàng nghiên cứu chính | Cần target human-reviewed, hash-bound, đủ train/val                    |
| V1 vision → report              | Chưa nghiệm thu                    | Cần NIfTI thật, smoke GPU Kaggle, orientation review, checkpoint/resume |
| Clinical accuracy                | Chưa thể kết luận                | Cần reference/adjudication độc lập và protocol khóa trước test    |

## 9. Quyết định và kiểm tra tiếp theo

Không cần gán nhãn lại toàn bộ ngay. Ưu tiên nhỏ nhất nhưng có giá trị là: rà soát 43 ca A, khóa glossary, rà soát mẫu B theo từng feature, giữ missing là `null`, và tạo một bảng adjudication có người duyệt/timestamp/reason. Chỉ sau bước đó mới dùng target đã accepted cho R5.

Các kiểm tra có thể chạy lại:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe scripts/cloud_prepare.py --annotations-root dataset_local --output output/profile/derived
.\.venv\Scripts\python.exe -m src.generate_report --input output/profile/derived/v2/requests.jsonl --output output/profile/reports.jsonl
```

Các con số trong báo cáo này là profile của snapshot local hiện tại. Mỗi bản cập nhật dữ liệu phải ghi hash nguồn, ngày profile và thay đổi schema; không so sánh hai snapshot chỉ bằng số dòng.

## Giới hạn

Báo cáo này không xác nhận de-identification hoàn hảo, không đọc pixel MRI, không adjudicate nhãn và không ước lượng clinical error. Các mismatch parser là tín hiệu ưu tiên review. Missing không phải negative. Output template không phải report của LLM và không phải chẩn đoán lâm sàng.
