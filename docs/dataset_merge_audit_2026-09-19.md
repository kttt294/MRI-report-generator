# Kiểm tra gộp dataset_local → dataset

> Báo cáo lịch sử trước đợt dọn repo 20/09/2026. Các tham chiếu SFT/file mẫu/output cũ mô tả lần kiểm tra đó; các file này đã được chuyển ra bản phục hồi. ETL hiện chỉ xuất CSV và JSONL; JSONL dùng key VI canonical và giữ nhãn thiếu là null.

Ngày kiểm tra: 19/09/2026. Phạm vi: toàn bộ các bảng nguồn local và các export CSV/JSON/SFT hiện có; không có ảnh MRI local để kiểm pixel/header.

## Kết luận

Các trường đã chọn đưa vào `dataset_master.csv` được nối đúng theo nguồn đang có: 247 bệnh nhân, 1.235 tầng, không thiếu khóa, không nhân bản khóa. Grading, tọa độ được chọn, splits và nội dung báo cáo khớp nguồn sau chuẩn hóa khoảng trắng đầu/cuối; số so với sai số tuyệt đối 1e-9.

Tuy nhiên, `dataset/` không phải bản gộp giữ toàn bộ thông tin: có một ô thiếu bị đổi thành 0 trong JSON legacy và nhiều trường/bảng nguồn không được xuất. Những thiếu hụt đó khác với việc nối sai bệnh nhân.

## Bằng chứng đối chiếu

| Phép kiểm | Số ô / phép so sánh | Sai khác |
|---|---:|---:|
| Grading_all → master: 8 nhãn + IVD + tháng | 12.350 | 0 |
| Grading từng tháng → grading_all | 12.350 | 0 |
| Cohort_frozen → master: 8 nhãn + IVD | 11.115 | 0 |
| Localization → các trường đã giữ trong master | 14.820 | 0 |
| 15 file split → master | 6.175 | 0 |
| 15 file split → fold_assignment | 6.175 | 0 |
| Nhãn trong 15 file split → master | 55.575 | 0 |
| 5 trường nhân khẩu → master | 1.235 | 0 |
| Kỹ thuật, mô tả, kết luận VI → master | 741 | 0 |
| Nội dung EN → master | 247 | 0 |
| Master → 8 nhãn trong JSONL | 9.880 | **1** |
| Master → tọa độ trong JSONL | 7.410 | 0 |
| Master → 3 phần báo cáo VI trong JSONL | 741 | 0 |
| Response của sáu file SFT → JSONL | 474 | 0 |

Các alias canonical/legacy trong master và JSONL khớp nhau. File `data_of_1patient.json` khớp bản ghi bệnh nhân tương ứng trong JSONL. Không phát hiện khóa trùng trong các bảng/index đã kiểm.

## Một lỗi dữ liệu xác nhận được

Một ô `disc_bulging` tại L4/L5 trống trong grading nguồn và master nhưng bằng 0 trong JSONL. Đây là mất thông tin missingness, cần xuất `null`. Mã bệnh nhân và giá trị hai phía nằm trong `output/merge_audit_2026-09-19/differences.private.json`; không đưa vào báo cáo tổng hợp public.

Script `consolidate_dataset.py` hiện đã giữ nhãn thiếu thành null. Đã chạy lại vào thư mục riêng `output/merge_audit_2026-09-19/rebuilt`, không ghi đè dataset. Ngoài sửa ô này, bản dựng lại chỉ khác khoảng trắng cuối một báo cáo EN, lặp ở năm dòng master, một bản ghi JSONL và một response SFT EN. File mẫu không đổi; cả sáu file SFT có prompt giữ nguyên khi đối chiếu lần dựng lại.

## Những phần chưa được đưa vào bản gộp

| Nguồn | Phần bị lược bỏ | Tác động / đề xuất |
|---|---|---|
| `grading/cohort_frozen.csv` | `modic_mixed_type`, `level_mapping_verified`, `source_file`, `cohort`, và các nhãn dẫn xuất | Nên giữ cờ gốc và provenance. Ba tầng có `modic_mixed_type=True`; chỉ giữ `modic=2` không phân biệt được hỗn hợp I+II với type II thuần. Không cần thay nhãn đang có để lưu thêm cờ. |
| `localize/disc_localization.csv` | `shape_i/j/k`, `spineps_label`, `mask_edited`, `reviewer`, `reviewed_at`, `in_cohort` | Chủ yếu là geometry, QC và truy vết; hữu ích cho vision engine. `in_cohort` có thể hằng 1 trong bản export cohort. |
| Master → patient JSONL | `volume`, `spacing_i/j/k`, `loc_source`, `loc_qc_status`, `month` | JSONL có tọa độ nhưng thiếu tham chiếu ảnh/spacing/QC để sử dụng độc lập cho vision. Những thông tin này vẫn còn trong CSV. Report engine chỉ dùng grading có thể không cần chúng. |
| `localize/axial_level_assignment.csv` | Toàn bộ bảng | Có 967 dòng của 247 ID, trong đó 244 ID thuộc cohort. Đây là bảng slab–tầng riêng; không nên join trực tiếp vào master gây nhân dòng. Nên giữ thành bảng liên kết hoặc danh sách theo ca nếu làm đa mặt phẳng. |
| `reports_json/report` | `source_file`, `source_type`, `has_conclusion`, `needs_review` | Mất provenance và cờ xử lý nguồn. Trong 238 ca khớp cohort: `needs_review=False` ở cả 238; `has_conclusion=False` ở 8 ca. Cờ này không chứng minh bác sĩ đã duyệt target V2. |
| `reports_json/metadata` | `age_now`, `age_now_computed_on` | Có thể lược bỏ có chủ đích vì model cần tuổi lúc chụp, đã giữ `age_at_scan`. |
| `reports_text` | `case_id`, `image_path` | Mất khóa/path truy vết nguồn EN; không nên mặc định path EN là ảnh đúng cho V1. |

README grading ghi mapping tầng chưa xác minh, nhưng `cohort_frozen.csv` ghi `level_mapping_verified=True` ở cả 1.235 dòng. Đây là bất nhất giữa các nguồn; chưa có bằng chứng đọc ảnh trong audit này để chọn một trạng thái. Cần xác nhận phiên bản nguồn trước khi chuyển trạng thái sang verified trong V2.

## Thiếu sẵn trong nguồn và lọc cohort

- Metadata/report nguồn có 240 ID, gồm 238 trong cohort và 2 ngoài cohort. Chín ca thiếu metadata/findings đã không có file nguồn tương ứng, không phải mất do join.
- Impression thiếu tổng cộng 17 ca: 9 ca không có report nguồn và 8 ca có report nhưng không có kết luận.
- Localization có 1.240 dòng; 5 dòng ngoài cohort được lọc có chủ đích. 1.235 khóa cohort còn lại đều được giữ.
- `fold_assignment.csv` có thêm 2 ID ngoài cohort; chúng cũng là hai ID report ngoài cohort. Cả 15 file split chỉ phủ cohort 247 ca và khớp master.
- Report EN có 238 bản ghi nguồn: 236 trong cohort, 2 ngoài cohort, không có sub_id không nối được. Vì vậy thiếu 11 EN trong cohort không phải lỗi join.

## Nội dung tiếng Anh và SFT

`reports_text` và `reports_text_v1_reference` có cùng 238 sub_id, nội dung Clinician's Notes giống nhau sau strip. README của bản reference gọi đây là bản dịch cũ và ghi không dùng huấn luyện nữa; không nên suy luận `report_en` đang gộp là bản dịch mới đã thẩm định. Con số 236 trong README phù hợp số ca được dùng trong cohort, còn số bản ghi CSV thực tế là 238.

SFT VI có 238 mẫu, không phải cả 238 đều đủ hai phần: 8 ca có findings nhưng thiếu impression vẫn được export. SFT EN có 236 mẫu. Đây là export có lọc theo báo cáo hiện có, không mất mẫu ngoài điều kiện đã định nghĩa. Prompt SFT legacy lược bỏ nhãn bản đệm và type Modic, cũng không biểu diễn rõ missingness/âm tính; đó là mất thông tin khi biểu diễn prompt, không phải sai ghép report. Không dùng sự khớp source/export để suy ra tập này đủ điều kiện V2 fine-tune.

## Cách kiểm và tái lập

Audit này so sánh giá trị trực tiếp theo khóa, không dùng regex đọc bệnh lý và không đánh giá tính đúng lâm sàng của nhãn.

```powershell
python scripts/audit_dataset_merge.py --output output/merge_audit_2026-09-19
```

Chỉ cần Python chuẩn, CPU, không phụ thuộc torch. Kết quả gồm `summary.json`, `differences.private.json`, notebook `merge_audit.ipynb` với code kiểm tra và dấu vân tay các nguồn đã đọc. Script không ghi lại dữ liệu nguồn. Bản dựng lại bằng ETL chỉ là phép kiểm bổ sung, không thay thế đối chiếu độc lập.

Ưu tiên xử lý: sửa export null; giữ cờ Modic hỗn hợp và provenance/QC cần thiết; giữ bảng axial riêng nếu định dùng; làm rõ mapping flag và nguồn dịch EN. Chưa có căn cứ phải gộp lại toàn bộ nhãn hay sửa nội dung báo cáo bác sĩ.
