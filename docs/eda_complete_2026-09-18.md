# EDA toàn bộ bảng dữ liệu hiện có

Tạo tự động: 2026-09-18T10:49:54.888712+00:00

Các số liệu dưới đây được tính bằng code trên toàn bộ dữ liệu local, không phải đọc từng dòng rồi ước lượng. Diễn giải dựa trên schema và quy tắc audit. Không có GPU hay model được chạy.

## Phạm vi và giới hạn

- Một dòng master là một tầng đĩa đệm; nhân khẩu học và báo cáo được khử lặp theo patient_id.
- Bao gồm missingness mọi cột, phân bố nhãn, tầng, fold, nhân khẩu, độ dài/lặp văn bản, localization, cấu trúc và cờ bất nhất.
- Không kiểm pixel ảnh, chất lượng chuỗi MRI, ảnh trùng, orientation hoặc tính đúng của chẩn đoán; ảnh không có local. Không coi cờ heuristic là tỷ lệ gán nhãn sai.
- Cờ cấp bệnh nhân: 131/238 (55.0%) ca có báo cáo được xét. A/B/C theo tiêu chí khác, không phải các phần của số ca có cờ này. Không phải tỷ lệ sai nhãn đã xác nhận.
- Null/ô trống không được đổi thành 0. Tuổi và ngày theo dữ liệu đã khử định danh; không suy luận dịch tễ đại diện dân số.
- Tỷ lệ chỉ là mô tả mẫu nghiên cứu, không có kiểm định thống kê hay suy luận quần thể.

## Tái lập

```powershell
python scripts/build_complete_eda.py --output outputs/eda_2026-09-18
```

Cần pandas. Script chỉ đọc nguồn, xuất CSV tổng hợp và JSON; audit theo bệnh nhân nằm trong output bị gitignore. File Excel trình bày lại chính các bảng tổng hợp này.

## Quy mô dữ liệu đang có

Snapshot bảng local; không suy ra số ảnh MRI thực có trên cloud.

| Chỉ tiêu | Số lượng | Đơn vị |
| --- | --- | --- |
| Bệnh nhân / ca trong master | 247 | bệnh nhân |
| Bản ghi patient–level | 1235 | tầng đĩa đệm |
| Trường dữ liệu | 42 | cột |
| Tầng chuẩn mỗi bệnh nhân | 5 | tầng |
| Bệnh nhân có findings và impression VI | 230 | bệnh nhân |

## Thiếu dữ liệu: toàn bộ các cột

Dòng = một tầng đĩa đệm; thông tin bệnh nhân/báo cáo lặp trên 5 dòng. Ô trống là thiếu; số 0 là giá trị.

| Trường | Dòng thiếu | Tổng dòng | Tỷ lệ dòng thiếu | BN có ít nhất 1 dòng thiếu | Tổng BN | Giá trị khác nhau (có dữ liệu) |
| --- | --- | --- | --- | --- | --- | --- |
| patient_id | 0 | 1235 | 0.0% | 0 | 247 | 247 |
| level | 0 | 1235 | 0.0% | 0 | 247 | 5 |
| ivd_label | 0 | 1235 | 0.0% | 0 | 247 | 5 |
| modic | 0 | 1235 | 0.0% | 0 | 247 | 4 |
| up_endplate | 0 | 1235 | 0.0% | 0 | 247 | 2 |
| low_endplate | 0 | 1235 | 0.0% | 0 | 247 | 2 |
| spondylolisthesis | 0 | 1235 | 0.0% | 0 | 247 | 2 |
| disc_herniation | 0 | 1235 | 0.0% | 0 | 247 | 2 |
| disc_narrowing | 0 | 1235 | 0.0% | 0 | 247 | 2 |
| disc_bulging | 1 | 1235 | 0.1% | 1 | 247 | 2 |
| pfirrmann_grade | 0 | 1235 | 0.0% | 0 | 247 | 5 |
| month | 0 | 1235 | 0.0% | 0 | 247 | 3 |
| voxel_i | 0 | 1235 | 0.0% | 0 | 247 | 1170 |
| voxel_j | 0 | 1235 | 0.0% | 0 | 247 | 1219 |
| voxel_k | 0 | 1235 | 0.0% | 0 | 247 | 320 |
| x_lps | 0 | 1235 | 0.0% | 0 | 247 | 1219 |
| y_lps | 0 | 1235 | 0.0% | 0 | 247 | 1219 |
| z_lps | 0 | 1235 | 0.0% | 0 | 247 | 1231 |
| volume | 0 | 1235 | 0.0% | 0 | 247 | 247 |
| spacing_i | 0 | 1235 | 0.0% | 0 | 247 | 4 |
| spacing_j | 0 | 1235 | 0.0% | 0 | 247 | 4 |
| spacing_k | 0 | 1235 | 0.0% | 0 | 247 | 3 |
| loc_source | 0 | 1235 | 0.0% | 0 | 247 | 2 |
| loc_qc_status | 0 | 1235 | 0.0% | 0 | 247 | 2 |
| fold1_split | 0 | 1235 | 0.0% | 0 | 247 | 3 |
| fold2_split | 0 | 1235 | 0.0% | 0 | 247 | 3 |
| fold3_split | 0 | 1235 | 0.0% | 0 | 247 | 3 |
| fold4_split | 0 | 1235 | 0.0% | 0 | 247 | 3 |
| fold5_split | 0 | 1235 | 0.0% | 0 | 247 | 3 |
| sub_id | 45 | 1235 | 3.6% | 9 | 247 | 238 |
| sex | 45 | 1235 | 3.6% | 9 | 247 | 2 |
| age_at_scan | 45 | 1235 | 3.6% | 9 | 247 | 62 |
| birth_year | 45 | 1235 | 3.6% | 9 | 247 | 63 |
| study_date | 45 | 1235 | 3.6% | 9 | 247 | 1 |
| report_vi_technique | 95 | 1235 | 7.7% | 19 | 247 | 16 |
| report_vi_findings | 45 | 1235 | 3.6% | 9 | 247 | 228 |
| report_vi_impression | 85 | 1235 | 6.9% | 17 | 247 | 216 |
| report_vi_kythuat | 95 | 1235 | 7.7% | 19 | 247 | 16 |
| report_vi_mota | 45 | 1235 | 3.6% | 9 | 247 | 228 |
| report_vi_ketluan | 85 | 1235 | 6.9% | 17 | 247 | 216 |
| report_en | 55 | 1235 | 4.5% | 11 | 247 | 233 |
| reports_text_split | 55 | 1235 | 4.5% | 11 | 247 | 3 |

## Nhân khẩu học và phân vùng nguồn

Đếm mỗi bệnh nhân một lần. Nhóm tuổi chỉ để mô tả; month là phân vùng nguồn, không dùng suy luận xu hướng thời gian.

| Biến | Nhóm | Số BN | Tổng BN | Tỷ lệ |
| --- | --- | --- | --- | --- |
| sex | F | 129 | 247 | 52.2% |
| sex | M | 109 | 247 | 44.1% |
| sex | Thiếu | 9 | 247 | 3.6% |
| month | 7 | 51 | 247 | 20.6% |
| month | 8 | 69 | 247 | 27.9% |
| month | 9 | 127 | 247 | 51.4% |
| Nhóm tuổi mô tả | <20 | 5 | 247 | 2.0% |
| Nhóm tuổi mô tả | 20–39 | 65 | 247 | 26.3% |
| Nhóm tuổi mô tả | 40–59 | 108 | 247 | 43.7% |
| Nhóm tuổi mô tả | 60–79 | 55 | 247 | 22.3% |
| Nhóm tuổi mô tả | 80+ | 5 | 247 | 2.0% |
| Nhóm tuổi mô tả | Thiếu | 9 | 247 | 3.6% |

## Thống kê biến số

Tuổi: cấp bệnh nhân. Tọa độ/spacing: cấp tầng, có lặp spacing theo ảnh; chưa kiểm tra bounds hay orientation trên NIfTI. Quantile nội suy tuyến tính; SD ddof=1.

| Biến | Đơn vị | N hợp lệ | Thiếu / không số | Min | Q1 | Median | Mean | Q3 | Max | SD mẫu |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| age_at_scan | tuổi ghi trong nguồn | 238 | 9 | 14.0 | 37.0 | 48.0 | 48.265 | 59.75 | 90.0 | 15.184 |
| voxel_i | voxel | 1235 | 0 | 404.15 | 462.43 | 484.51 | 489.322 | 514.19 | 620.17 | 35.18 |
| voxel_j | voxel | 1235 | 0 | 124.56 | 348.465 | 492.36 | 484.084 | 622.95 | 841.19 | 162.708 |
| voxel_k | voxel | 1235 | 0 | 2.0 | 6.12 | 6.68 | 6.681 | 7.07 | 14.57 | 0.952 |
| x_lps | mm | 1235 | 0 | -45.535 | -10.549 | -2.155 | -2.286 | 5.594 | 30.522 | 12.191 |
| y_lps | mm | 1235 | 0 | 0.525 | 26.465 | 33.13 | 33.471 | 40.06 | 66.976 | 10.301 |
| z_lps | mm | 1235 | 0 | -670.673 | -26.514 | 19.133 | 10.207 | 61.587 | 195.451 | 93.245 |
| spacing_i | mm | 1235 | 0 | 0.27 | 0.305 | 0.305 | 0.305 | 0.305 | 0.316 | 0.002 |
| spacing_j | mm | 1235 | 0 | 0.27 | 0.305 | 0.305 | 0.305 | 0.305 | 0.316 | 0.002 |
| spacing_k | mm | 1235 | 0 | 4.4 | 4.4 | 4.4 | 4.403 | 4.4 | 4.9 | 0.034 |

## Phân bố từng nhãn, toàn bộ và theo tầng

Mẫu số gồm cả ô thiếu. Modic là loại, không phải thang mức độ. Pfirrmann giữ nguyên 1–5, không gộp lớp hiếm.

| Phạm vi | Nhãn | Giá trị | Số tầng | Tổng tầng | Tỷ lệ |
| --- | --- | --- | --- | --- | --- |
| Tất cả | pfirrmann_grade | 1 | 36 | 1235 | 2.9% |
| Tất cả | pfirrmann_grade | 2 | 689 | 1235 | 55.8% |
| Tất cả | pfirrmann_grade | 3 | 418 | 1235 | 33.8% |
| Tất cả | pfirrmann_grade | 4 | 88 | 1235 | 7.1% |
| Tất cả | pfirrmann_grade | 5 | 4 | 1235 | 0.3% |
| Tất cả | pfirrmann_grade | Thiếu | 0 | 1235 | 0.0% |
| Tất cả | pfirrmann_grade | Ngoài miền | 0 | 1235 | 0.0% |
| Tất cả | modic | 0 | 1058 | 1235 | 85.7% |
| Tất cả | modic | 1 | 24 | 1235 | 1.9% |
| Tất cả | modic | 2 | 149 | 1235 | 12.1% |
| Tất cả | modic | 3 | 4 | 1235 | 0.3% |
| Tất cả | modic | Thiếu | 0 | 1235 | 0.0% |
| Tất cả | modic | Ngoài miền | 0 | 1235 | 0.0% |
| Tất cả | disc_herniation | 0 | 1147 | 1235 | 92.9% |
| Tất cả | disc_herniation | 1 | 88 | 1235 | 7.1% |
| Tất cả | disc_herniation | Thiếu | 0 | 1235 | 0.0% |
| Tất cả | disc_herniation | Ngoài miền | 0 | 1235 | 0.0% |
| Tất cả | disc_bulging | 0 | 906 | 1235 | 73.4% |
| Tất cả | disc_bulging | 1 | 328 | 1235 | 26.6% |
| Tất cả | disc_bulging | Thiếu | 1 | 1235 | 0.1% |
| Tất cả | disc_bulging | Ngoài miền | 0 | 1235 | 0.0% |
| Tất cả | disc_narrowing | 0 | 1176 | 1235 | 95.2% |
| Tất cả | disc_narrowing | 1 | 59 | 1235 | 4.8% |
| Tất cả | disc_narrowing | Thiếu | 0 | 1235 | 0.0% |
| Tất cả | disc_narrowing | Ngoài miền | 0 | 1235 | 0.0% |
| Tất cả | spondylolisthesis | 0 | 1194 | 1235 | 96.7% |
| Tất cả | spondylolisthesis | 1 | 41 | 1235 | 3.3% |
| Tất cả | spondylolisthesis | Thiếu | 0 | 1235 | 0.0% |
| Tất cả | spondylolisthesis | Ngoài miền | 0 | 1235 | 0.0% |
| Tất cả | up_endplate | 0 | 1120 | 1235 | 90.7% |
| Tất cả | up_endplate | 1 | 115 | 1235 | 9.3% |
| Tất cả | up_endplate | Thiếu | 0 | 1235 | 0.0% |
| Tất cả | up_endplate | Ngoài miền | 0 | 1235 | 0.0% |
| Tất cả | low_endplate | 0 | 1163 | 1235 | 94.2% |
| Tất cả | low_endplate | 1 | 72 | 1235 | 5.8% |
| Tất cả | low_endplate | Thiếu | 0 | 1235 | 0.0% |
| Tất cả | low_endplate | Ngoài miền | 0 | 1235 | 0.0% |
| L1/L2 | pfirrmann_grade | 1 | 4 | 247 | 1.6% |
| L1/L2 | pfirrmann_grade | 2 | 194 | 247 | 78.5% |
| L1/L2 | pfirrmann_grade | 3 | 37 | 247 | 15.0% |
| L1/L2 | pfirrmann_grade | 4 | 11 | 247 | 4.5% |
| L1/L2 | pfirrmann_grade | 5 | 1 | 247 | 0.4% |
| L1/L2 | pfirrmann_grade | Thiếu | 0 | 247 | 0.0% |
| L1/L2 | pfirrmann_grade | Ngoài miền | 0 | 247 | 0.0% |
| L1/L2 | modic | 0 | 234 | 247 | 94.7% |
| L1/L2 | modic | 1 | 3 | 247 | 1.2% |
| L1/L2 | modic | 2 | 10 | 247 | 4.0% |
| L1/L2 | modic | 3 | 0 | 247 | 0.0% |
| L1/L2 | modic | Thiếu | 0 | 247 | 0.0% |
| L1/L2 | modic | Ngoài miền | 0 | 247 | 0.0% |
| L1/L2 | disc_herniation | 0 | 244 | 247 | 98.8% |
| L1/L2 | disc_herniation | 1 | 3 | 247 | 1.2% |
| L1/L2 | disc_herniation | Thiếu | 0 | 247 | 0.0% |
| L1/L2 | disc_herniation | Ngoài miền | 0 | 247 | 0.0% |
| L1/L2 | disc_bulging | 0 | 237 | 247 | 96.0% |
| L1/L2 | disc_bulging | 1 | 10 | 247 | 4.0% |
| L1/L2 | disc_bulging | Thiếu | 0 | 247 | 0.0% |
| L1/L2 | disc_bulging | Ngoài miền | 0 | 247 | 0.0% |
| L1/L2 | disc_narrowing | 0 | 241 | 247 | 97.6% |
| L1/L2 | disc_narrowing | 1 | 6 | 247 | 2.4% |
| L1/L2 | disc_narrowing | Thiếu | 0 | 247 | 0.0% |
| L1/L2 | disc_narrowing | Ngoài miền | 0 | 247 | 0.0% |
| L1/L2 | spondylolisthesis | 0 | 245 | 247 | 99.2% |
| L1/L2 | spondylolisthesis | 1 | 2 | 247 | 0.8% |
| L1/L2 | spondylolisthesis | Thiếu | 0 | 247 | 0.0% |
| L1/L2 | spondylolisthesis | Ngoài miền | 0 | 247 | 0.0% |
| L1/L2 | up_endplate | 0 | 243 | 247 | 98.4% |
| L1/L2 | up_endplate | 1 | 4 | 247 | 1.6% |
| L1/L2 | up_endplate | Thiếu | 0 | 247 | 0.0% |
| L1/L2 | up_endplate | Ngoài miền | 0 | 247 | 0.0% |
| L1/L2 | low_endplate | 0 | 240 | 247 | 97.2% |
| L1/L2 | low_endplate | 1 | 7 | 247 | 2.8% |
| L1/L2 | low_endplate | Thiếu | 0 | 247 | 0.0% |
| L1/L2 | low_endplate | Ngoài miền | 0 | 247 | 0.0% |
| L2/L3 | pfirrmann_grade | 1 | 6 | 247 | 2.4% |
| L2/L3 | pfirrmann_grade | 2 | 173 | 247 | 70.0% |
| L2/L3 | pfirrmann_grade | 3 | 54 | 247 | 21.9% |
| L2/L3 | pfirrmann_grade | 4 | 13 | 247 | 5.3% |
| L2/L3 | pfirrmann_grade | 5 | 1 | 247 | 0.4% |
| L2/L3 | pfirrmann_grade | Thiếu | 0 | 247 | 0.0% |
| L2/L3 | pfirrmann_grade | Ngoài miền | 0 | 247 | 0.0% |
| L2/L3 | modic | 0 | 227 | 247 | 91.9% |
| L2/L3 | modic | 1 | 2 | 247 | 0.8% |
| L2/L3 | modic | 2 | 17 | 247 | 6.9% |
| L2/L3 | modic | 3 | 1 | 247 | 0.4% |
| L2/L3 | modic | Thiếu | 0 | 247 | 0.0% |
| L2/L3 | modic | Ngoài miền | 0 | 247 | 0.0% |
| L2/L3 | disc_herniation | 0 | 244 | 247 | 98.8% |
| L2/L3 | disc_herniation | 1 | 3 | 247 | 1.2% |
| L2/L3 | disc_herniation | Thiếu | 0 | 247 | 0.0% |
| L2/L3 | disc_herniation | Ngoài miền | 0 | 247 | 0.0% |
| L2/L3 | disc_bulging | 0 | 216 | 247 | 87.4% |
| L2/L3 | disc_bulging | 1 | 31 | 247 | 12.6% |
| L2/L3 | disc_bulging | Thiếu | 0 | 247 | 0.0% |
| L2/L3 | disc_bulging | Ngoài miền | 0 | 247 | 0.0% |
| L2/L3 | disc_narrowing | 0 | 241 | 247 | 97.6% |
| L2/L3 | disc_narrowing | 1 | 6 | 247 | 2.4% |
| L2/L3 | disc_narrowing | Thiếu | 0 | 247 | 0.0% |
| L2/L3 | disc_narrowing | Ngoài miền | 0 | 247 | 0.0% |
| L2/L3 | spondylolisthesis | 0 | 245 | 247 | 99.2% |
| L2/L3 | spondylolisthesis | 1 | 2 | 247 | 0.8% |
| L2/L3 | spondylolisthesis | Thiếu | 0 | 247 | 0.0% |
| L2/L3 | spondylolisthesis | Ngoài miền | 0 | 247 | 0.0% |
| L2/L3 | up_endplate | 0 | 239 | 247 | 96.8% |
| L2/L3 | up_endplate | 1 | 8 | 247 | 3.2% |
| L2/L3 | up_endplate | Thiếu | 0 | 247 | 0.0% |
| L2/L3 | up_endplate | Ngoài miền | 0 | 247 | 0.0% |
| L2/L3 | low_endplate | 0 | 237 | 247 | 96.0% |
| L2/L3 | low_endplate | 1 | 10 | 247 | 4.0% |
| L2/L3 | low_endplate | Thiếu | 0 | 247 | 0.0% |
| L2/L3 | low_endplate | Ngoài miền | 0 | 247 | 0.0% |
| L3/L4 | pfirrmann_grade | 1 | 7 | 247 | 2.8% |
| L3/L4 | pfirrmann_grade | 2 | 138 | 247 | 55.9% |
| L3/L4 | pfirrmann_grade | 3 | 87 | 247 | 35.2% |
| L3/L4 | pfirrmann_grade | 4 | 15 | 247 | 6.1% |
| L3/L4 | pfirrmann_grade | 5 | 0 | 247 | 0.0% |
| L3/L4 | pfirrmann_grade | Thiếu | 0 | 247 | 0.0% |
| L3/L4 | pfirrmann_grade | Ngoài miền | 0 | 247 | 0.0% |
| L3/L4 | modic | 0 | 203 | 247 | 82.2% |
| L3/L4 | modic | 1 | 5 | 247 | 2.0% |
| L3/L4 | modic | 2 | 37 | 247 | 15.0% |
| L3/L4 | modic | 3 | 2 | 247 | 0.8% |
| L3/L4 | modic | Thiếu | 0 | 247 | 0.0% |
| L3/L4 | modic | Ngoài miền | 0 | 247 | 0.0% |
| L3/L4 | disc_herniation | 0 | 240 | 247 | 97.2% |
| L3/L4 | disc_herniation | 1 | 7 | 247 | 2.8% |
| L3/L4 | disc_herniation | Thiếu | 0 | 247 | 0.0% |
| L3/L4 | disc_herniation | Ngoài miền | 0 | 247 | 0.0% |
| L3/L4 | disc_bulging | 0 | 180 | 247 | 72.9% |
| L3/L4 | disc_bulging | 1 | 67 | 247 | 27.1% |
| L3/L4 | disc_bulging | Thiếu | 0 | 247 | 0.0% |
| L3/L4 | disc_bulging | Ngoài miền | 0 | 247 | 0.0% |
| L3/L4 | disc_narrowing | 0 | 242 | 247 | 98.0% |
| L3/L4 | disc_narrowing | 1 | 5 | 247 | 2.0% |
| L3/L4 | disc_narrowing | Thiếu | 0 | 247 | 0.0% |
| L3/L4 | disc_narrowing | Ngoài miền | 0 | 247 | 0.0% |
| L3/L4 | spondylolisthesis | 0 | 244 | 247 | 98.8% |
| L3/L4 | spondylolisthesis | 1 | 3 | 247 | 1.2% |
| L3/L4 | spondylolisthesis | Thiếu | 0 | 247 | 0.0% |
| L3/L4 | spondylolisthesis | Ngoài miền | 0 | 247 | 0.0% |
| L3/L4 | up_endplate | 0 | 215 | 247 | 87.0% |
| L3/L4 | up_endplate | 1 | 32 | 247 | 13.0% |
| L3/L4 | up_endplate | Thiếu | 0 | 247 | 0.0% |
| L3/L4 | up_endplate | Ngoài miền | 0 | 247 | 0.0% |
| L3/L4 | low_endplate | 0 | 233 | 247 | 94.3% |
| L3/L4 | low_endplate | 1 | 14 | 247 | 5.7% |
| L3/L4 | low_endplate | Thiếu | 0 | 247 | 0.0% |
| L3/L4 | low_endplate | Ngoài miền | 0 | 247 | 0.0% |
| L4/L5 | pfirrmann_grade | 1 | 10 | 247 | 4.0% |
| L4/L5 | pfirrmann_grade | 2 | 83 | 247 | 33.6% |
| L4/L5 | pfirrmann_grade | 3 | 127 | 247 | 51.4% |
| L4/L5 | pfirrmann_grade | 4 | 26 | 247 | 10.5% |
| L4/L5 | pfirrmann_grade | 5 | 1 | 247 | 0.4% |
| L4/L5 | pfirrmann_grade | Thiếu | 0 | 247 | 0.0% |
| L4/L5 | pfirrmann_grade | Ngoài miền | 0 | 247 | 0.0% |
| L4/L5 | modic | 0 | 203 | 247 | 82.2% |
| L4/L5 | modic | 1 | 6 | 247 | 2.4% |
| L4/L5 | modic | 2 | 38 | 247 | 15.4% |
| L4/L5 | modic | 3 | 0 | 247 | 0.0% |
| L4/L5 | modic | Thiếu | 0 | 247 | 0.0% |
| L4/L5 | modic | Ngoài miền | 0 | 247 | 0.0% |
| L4/L5 | disc_herniation | 0 | 221 | 247 | 89.5% |
| L4/L5 | disc_herniation | 1 | 26 | 247 | 10.5% |
| L4/L5 | disc_herniation | Thiếu | 0 | 247 | 0.0% |
| L4/L5 | disc_herniation | Ngoài miền | 0 | 247 | 0.0% |
| L4/L5 | disc_bulging | 0 | 110 | 247 | 44.5% |
| L4/L5 | disc_bulging | 1 | 136 | 247 | 55.1% |
| L4/L5 | disc_bulging | Thiếu | 1 | 247 | 0.4% |
| L4/L5 | disc_bulging | Ngoài miền | 0 | 247 | 0.0% |
| L4/L5 | disc_narrowing | 0 | 226 | 247 | 91.5% |
| L4/L5 | disc_narrowing | 1 | 21 | 247 | 8.5% |
| L4/L5 | disc_narrowing | Thiếu | 0 | 247 | 0.0% |
| L4/L5 | disc_narrowing | Ngoài miền | 0 | 247 | 0.0% |
| L4/L5 | spondylolisthesis | 0 | 226 | 247 | 91.5% |
| L4/L5 | spondylolisthesis | 1 | 21 | 247 | 8.5% |
| L4/L5 | spondylolisthesis | Thiếu | 0 | 247 | 0.0% |
| L4/L5 | spondylolisthesis | Ngoài miền | 0 | 247 | 0.0% |
| L4/L5 | up_endplate | 0 | 216 | 247 | 87.4% |
| L4/L5 | up_endplate | 1 | 31 | 247 | 12.6% |
| L4/L5 | up_endplate | Thiếu | 0 | 247 | 0.0% |
| L4/L5 | up_endplate | Ngoài miền | 0 | 247 | 0.0% |
| L4/L5 | low_endplate | 0 | 229 | 247 | 92.7% |
| L4/L5 | low_endplate | 1 | 18 | 247 | 7.3% |
| L4/L5 | low_endplate | Thiếu | 0 | 247 | 0.0% |
| L4/L5 | low_endplate | Ngoài miền | 0 | 247 | 0.0% |
| L5/S1 | pfirrmann_grade | 1 | 9 | 247 | 3.6% |
| L5/S1 | pfirrmann_grade | 2 | 101 | 247 | 40.9% |
| L5/S1 | pfirrmann_grade | 3 | 113 | 247 | 45.7% |
| L5/S1 | pfirrmann_grade | 4 | 23 | 247 | 9.3% |
| L5/S1 | pfirrmann_grade | 5 | 1 | 247 | 0.4% |
| L5/S1 | pfirrmann_grade | Thiếu | 0 | 247 | 0.0% |
| L5/S1 | pfirrmann_grade | Ngoài miền | 0 | 247 | 0.0% |
| L5/S1 | modic | 0 | 191 | 247 | 77.3% |
| L5/S1 | modic | 1 | 8 | 247 | 3.2% |
| L5/S1 | modic | 2 | 47 | 247 | 19.0% |
| L5/S1 | modic | 3 | 1 | 247 | 0.4% |
| L5/S1 | modic | Thiếu | 0 | 247 | 0.0% |
| L5/S1 | modic | Ngoài miền | 0 | 247 | 0.0% |
| L5/S1 | disc_herniation | 0 | 198 | 247 | 80.2% |
| L5/S1 | disc_herniation | 1 | 49 | 247 | 19.8% |
| L5/S1 | disc_herniation | Thiếu | 0 | 247 | 0.0% |
| L5/S1 | disc_herniation | Ngoài miền | 0 | 247 | 0.0% |
| L5/S1 | disc_bulging | 0 | 163 | 247 | 66.0% |
| L5/S1 | disc_bulging | 1 | 84 | 247 | 34.0% |
| L5/S1 | disc_bulging | Thiếu | 0 | 247 | 0.0% |
| L5/S1 | disc_bulging | Ngoài miền | 0 | 247 | 0.0% |
| L5/S1 | disc_narrowing | 0 | 226 | 247 | 91.5% |
| L5/S1 | disc_narrowing | 1 | 21 | 247 | 8.5% |
| L5/S1 | disc_narrowing | Thiếu | 0 | 247 | 0.0% |
| L5/S1 | disc_narrowing | Ngoài miền | 0 | 247 | 0.0% |
| L5/S1 | spondylolisthesis | 0 | 234 | 247 | 94.7% |
| L5/S1 | spondylolisthesis | 1 | 13 | 247 | 5.3% |
| L5/S1 | spondylolisthesis | Thiếu | 0 | 247 | 0.0% |
| L5/S1 | spondylolisthesis | Ngoài miền | 0 | 247 | 0.0% |
| L5/S1 | up_endplate | 0 | 207 | 247 | 83.8% |
| L5/S1 | up_endplate | 1 | 40 | 247 | 16.2% |
| L5/S1 | up_endplate | Thiếu | 0 | 247 | 0.0% |
| L5/S1 | up_endplate | Ngoài miền | 0 | 247 | 0.0% |
| L5/S1 | low_endplate | 0 | 224 | 247 | 90.7% |
| L5/S1 | low_endplate | 1 | 23 | 247 | 9.3% |
| L5/S1 | low_endplate | Thiếu | 0 | 247 | 0.0% |
| L5/S1 | low_endplate | Ngoài miền | 0 | 247 | 0.0% |

## Nhãn nhị phân ở cấp bệnh nhân

Một tầng dương tính đủ để xếp có nhãn; chỉ xếp âm tính khi toàn bộ tầng bằng 0.

| Nhãn | Trạng thái | Số BN | Tổng BN | Tỷ lệ |
| --- | --- | --- | --- | --- |
| disc_herniation | Có ít nhất 1 tầng = 1 | 70 | 247 | 28.3% |
| disc_herniation | Tất cả tầng = 0 | 177 | 247 | 71.7% |
| disc_herniation | Chưa xác định | 0 | 247 | 0.0% |
| disc_bulging | Có ít nhất 1 tầng = 1 | 166 | 247 | 67.2% |
| disc_bulging | Tất cả tầng = 0 | 80 | 247 | 32.4% |
| disc_bulging | Chưa xác định | 1 | 247 | 0.4% |
| disc_narrowing | Có ít nhất 1 tầng = 1 | 35 | 247 | 14.2% |
| disc_narrowing | Tất cả tầng = 0 | 212 | 247 | 85.8% |
| disc_narrowing | Chưa xác định | 0 | 247 | 0.0% |
| spondylolisthesis | Có ít nhất 1 tầng = 1 | 34 | 247 | 13.8% |
| spondylolisthesis | Tất cả tầng = 0 | 213 | 247 | 86.2% |
| spondylolisthesis | Chưa xác định | 0 | 247 | 0.0% |
| up_endplate | Có ít nhất 1 tầng = 1 | 68 | 247 | 27.5% |
| up_endplate | Tất cả tầng = 0 | 179 | 247 | 72.5% |
| up_endplate | Chưa xác định | 0 | 247 | 0.0% |
| low_endplate | Có ít nhất 1 tầng = 1 | 49 | 247 | 19.8% |
| low_endplate | Tất cả tầng = 0 | 198 | 247 | 80.2% |
| low_endplate | Chưa xác định | 0 | 247 | 0.0% |

## Đồng xuất hiện nhãn tại cùng một tầng

Thống kê mô tả, không chứng minh quan hệ nhân quả hoặc tính độc lập.

| Nhãn A | Nhãn B | Cả hai = 1 | Số tầng biết cả hai | Tỷ lệ |
| --- | --- | --- | --- | --- |
| disc_herniation | disc_bulging | 6 | 1234 | 0.5% |
| disc_herniation | disc_narrowing | 11 | 1235 | 0.9% |
| disc_herniation | spondylolisthesis | 7 | 1235 | 0.6% |
| disc_herniation | up_endplate | 11 | 1235 | 0.9% |
| disc_herniation | low_endplate | 9 | 1235 | 0.7% |
| disc_bulging | disc_narrowing | 43 | 1234 | 3.5% |
| disc_bulging | spondylolisthesis | 33 | 1234 | 2.7% |
| disc_bulging | up_endplate | 57 | 1234 | 4.6% |
| disc_bulging | low_endplate | 43 | 1234 | 3.5% |
| disc_narrowing | spondylolisthesis | 14 | 1235 | 1.1% |
| disc_narrowing | up_endplate | 20 | 1235 | 1.6% |
| disc_narrowing | low_endplate | 22 | 1235 | 1.8% |
| spondylolisthesis | up_endplate | 7 | 1235 | 0.6% |
| spondylolisthesis | low_endplate | 8 | 1235 | 0.6% |
| up_endplate | low_endplate | 19 | 1235 | 1.5% |

## Quy mô từng fold theo điều kiện báo cáo

Các điều kiện là các tập con có thể chồng lấp. Số tầng = số BN × 5 nếu đủ cấu trúc.

| Fold | Split | Toàn bộ BN | Đủ findings + impression VI | Có findings VI | Có EN |
| --- | --- | --- | --- | --- | --- |
| 1 | train | 147 | 137 | 142 | 140 |
| 1 | val | 50 | 46 | 49 | 49 |
| 1 | test | 50 | 47 | 47 | 47 |
| 2 | train | 147 | 140 | 142 | 142 |
| 2 | val | 50 | 47 | 48 | 46 |
| 2 | test | 50 | 43 | 48 | 48 |
| 3 | train | 148 | 136 | 142 | 140 |
| 3 | val | 50 | 46 | 48 | 48 |
| 3 | test | 49 | 48 | 48 | 48 |
| 4 | train | 148 | 140 | 143 | 143 |
| 4 | val | 50 | 46 | 48 | 48 |
| 4 | test | 49 | 44 | 47 | 45 |
| 5 | train | 148 | 136 | 142 | 141 |
| 5 | val | 50 | 46 | 48 | 47 |
| 5 | test | 49 | 48 | 48 | 48 |

## Độ phủ lớp của từng fold/split

Có cả lớp đếm bằng 0 để thấy lớp vắng ở validation/test. Tỷ lệ dùng tổng tầng, gồm cả nhãn thiếu.

| Fold | Split | Nhãn | Giá trị | Số tầng | Tổng tầng | Tỷ lệ |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | train | pfirrmann_grade | 1 | 20 | 735 | 2.7% |
| 1 | train | pfirrmann_grade | 2 | 404 | 735 | 55.0% |
| 1 | train | pfirrmann_grade | 3 | 253 | 735 | 34.4% |
| 1 | train | pfirrmann_grade | 4 | 55 | 735 | 7.5% |
| 1 | train | pfirrmann_grade | 5 | 3 | 735 | 0.4% |
| 1 | train | modic | 0 | 620 | 735 | 84.4% |
| 1 | train | modic | 1 | 18 | 735 | 2.4% |
| 1 | train | modic | 2 | 93 | 735 | 12.7% |
| 1 | train | modic | 3 | 4 | 735 | 0.5% |
| 1 | train | disc_herniation | 0 | 692 | 735 | 94.1% |
| 1 | train | disc_herniation | 1 | 43 | 735 | 5.9% |
| 1 | train | disc_bulging | 0 | 533 | 735 | 72.5% |
| 1 | train | disc_bulging | 1 | 202 | 735 | 27.5% |
| 1 | train | disc_narrowing | 0 | 706 | 735 | 96.1% |
| 1 | train | disc_narrowing | 1 | 29 | 735 | 3.9% |
| 1 | train | spondylolisthesis | 0 | 711 | 735 | 96.7% |
| 1 | train | spondylolisthesis | 1 | 24 | 735 | 3.3% |
| 1 | train | up_endplate | 0 | 663 | 735 | 90.2% |
| 1 | train | up_endplate | 1 | 72 | 735 | 9.8% |
| 1 | train | low_endplate | 0 | 689 | 735 | 93.7% |
| 1 | train | low_endplate | 1 | 46 | 735 | 6.3% |
| 1 | val | pfirrmann_grade | 1 | 11 | 250 | 4.4% |
| 1 | val | pfirrmann_grade | 2 | 141 | 250 | 56.4% |
| 1 | val | pfirrmann_grade | 3 | 79 | 250 | 31.6% |
| 1 | val | pfirrmann_grade | 4 | 18 | 250 | 7.2% |
| 1 | val | pfirrmann_grade | 5 | 1 | 250 | 0.4% |
| 1 | val | modic | 0 | 218 | 250 | 87.2% |
| 1 | val | modic | 1 | 3 | 250 | 1.2% |
| 1 | val | modic | 2 | 29 | 250 | 11.6% |
| 1 | val | modic | 3 | 0 | 250 | 0.0% |
| 1 | val | disc_herniation | 0 | 231 | 250 | 92.4% |
| 1 | val | disc_herniation | 1 | 19 | 250 | 7.6% |
| 1 | val | disc_bulging | 0 | 181 | 250 | 72.4% |
| 1 | val | disc_bulging | 1 | 68 | 250 | 27.2% |
| 1 | val | disc_narrowing | 0 | 233 | 250 | 93.2% |
| 1 | val | disc_narrowing | 1 | 17 | 250 | 6.8% |
| 1 | val | spondylolisthesis | 0 | 240 | 250 | 96.0% |
| 1 | val | spondylolisthesis | 1 | 10 | 250 | 4.0% |
| 1 | val | up_endplate | 0 | 225 | 250 | 90.0% |
| 1 | val | up_endplate | 1 | 25 | 250 | 10.0% |
| 1 | val | low_endplate | 0 | 239 | 250 | 95.6% |
| 1 | val | low_endplate | 1 | 11 | 250 | 4.4% |
| 1 | test | pfirrmann_grade | 1 | 5 | 250 | 2.0% |
| 1 | test | pfirrmann_grade | 2 | 144 | 250 | 57.6% |
| 1 | test | pfirrmann_grade | 3 | 86 | 250 | 34.4% |
| 1 | test | pfirrmann_grade | 4 | 15 | 250 | 6.0% |
| 1 | test | pfirrmann_grade | 5 | 0 | 250 | 0.0% |
| 1 | test | modic | 0 | 220 | 250 | 88.0% |
| 1 | test | modic | 1 | 3 | 250 | 1.2% |
| 1 | test | modic | 2 | 27 | 250 | 10.8% |
| 1 | test | modic | 3 | 0 | 250 | 0.0% |
| 1 | test | disc_herniation | 0 | 224 | 250 | 89.6% |
| 1 | test | disc_herniation | 1 | 26 | 250 | 10.4% |
| 1 | test | disc_bulging | 0 | 192 | 250 | 76.8% |
| 1 | test | disc_bulging | 1 | 58 | 250 | 23.2% |
| 1 | test | disc_narrowing | 0 | 237 | 250 | 94.8% |
| 1 | test | disc_narrowing | 1 | 13 | 250 | 5.2% |
| 1 | test | spondylolisthesis | 0 | 243 | 250 | 97.2% |
| 1 | test | spondylolisthesis | 1 | 7 | 250 | 2.8% |
| 1 | test | up_endplate | 0 | 232 | 250 | 92.8% |
| 1 | test | up_endplate | 1 | 18 | 250 | 7.2% |
| 1 | test | low_endplate | 0 | 235 | 250 | 94.0% |
| 1 | test | low_endplate | 1 | 15 | 250 | 6.0% |
| 2 | train | pfirrmann_grade | 1 | 16 | 735 | 2.2% |
| 2 | train | pfirrmann_grade | 2 | 415 | 735 | 56.5% |
| 2 | train | pfirrmann_grade | 3 | 250 | 735 | 34.0% |
| 2 | train | pfirrmann_grade | 4 | 53 | 735 | 7.2% |
| 2 | train | pfirrmann_grade | 5 | 1 | 735 | 0.1% |
| 2 | train | modic | 0 | 628 | 735 | 85.4% |
| 2 | train | modic | 1 | 15 | 735 | 2.0% |
| 2 | train | modic | 2 | 91 | 735 | 12.4% |
| 2 | train | modic | 3 | 1 | 735 | 0.1% |
| 2 | train | disc_herniation | 0 | 686 | 735 | 93.3% |
| 2 | train | disc_herniation | 1 | 49 | 735 | 6.7% |
| 2 | train | disc_bulging | 0 | 530 | 735 | 72.1% |
| 2 | train | disc_bulging | 1 | 204 | 735 | 27.8% |
| 2 | train | disc_narrowing | 0 | 694 | 735 | 94.4% |
| 2 | train | disc_narrowing | 1 | 41 | 735 | 5.6% |
| 2 | train | spondylolisthesis | 0 | 714 | 735 | 97.1% |
| 2 | train | spondylolisthesis | 1 | 21 | 735 | 2.9% |
| 2 | train | up_endplate | 0 | 672 | 735 | 91.4% |
| 2 | train | up_endplate | 1 | 63 | 735 | 8.6% |
| 2 | train | low_endplate | 0 | 684 | 735 | 93.1% |
| 2 | train | low_endplate | 1 | 51 | 735 | 6.9% |
| 2 | val | pfirrmann_grade | 1 | 2 | 250 | 0.8% |
| 2 | val | pfirrmann_grade | 2 | 142 | 250 | 56.8% |
| 2 | val | pfirrmann_grade | 3 | 88 | 250 | 35.2% |
| 2 | val | pfirrmann_grade | 4 | 18 | 250 | 7.2% |
| 2 | val | pfirrmann_grade | 5 | 0 | 250 | 0.0% |
| 2 | val | modic | 0 | 218 | 250 | 87.2% |
| 2 | val | modic | 1 | 7 | 250 | 2.8% |
| 2 | val | modic | 2 | 25 | 250 | 10.0% |
| 2 | val | modic | 3 | 0 | 250 | 0.0% |
| 2 | val | disc_herniation | 0 | 226 | 250 | 90.4% |
| 2 | val | disc_herniation | 1 | 24 | 250 | 9.6% |
| 2 | val | disc_bulging | 0 | 185 | 250 | 74.0% |
| 2 | val | disc_bulging | 1 | 65 | 250 | 26.0% |
| 2 | val | disc_narrowing | 0 | 244 | 250 | 97.6% |
| 2 | val | disc_narrowing | 1 | 6 | 250 | 2.4% |
| 2 | val | spondylolisthesis | 0 | 242 | 250 | 96.8% |
| 2 | val | spondylolisthesis | 1 | 8 | 250 | 3.2% |
| 2 | val | up_endplate | 0 | 228 | 250 | 91.2% |
| 2 | val | up_endplate | 1 | 22 | 250 | 8.8% |
| 2 | val | low_endplate | 0 | 236 | 250 | 94.4% |
| 2 | val | low_endplate | 1 | 14 | 250 | 5.6% |
| 2 | test | pfirrmann_grade | 1 | 18 | 250 | 7.2% |
| 2 | test | pfirrmann_grade | 2 | 132 | 250 | 52.8% |
| 2 | test | pfirrmann_grade | 3 | 80 | 250 | 32.0% |
| 2 | test | pfirrmann_grade | 4 | 17 | 250 | 6.8% |
| 2 | test | pfirrmann_grade | 5 | 3 | 250 | 1.2% |
| 2 | test | modic | 0 | 212 | 250 | 84.8% |
| 2 | test | modic | 1 | 2 | 250 | 0.8% |
| 2 | test | modic | 2 | 33 | 250 | 13.2% |
| 2 | test | modic | 3 | 3 | 250 | 1.2% |
| 2 | test | disc_herniation | 0 | 235 | 250 | 94.0% |
| 2 | test | disc_herniation | 1 | 15 | 250 | 6.0% |
| 2 | test | disc_bulging | 0 | 191 | 250 | 76.4% |
| 2 | test | disc_bulging | 1 | 59 | 250 | 23.6% |
| 2 | test | disc_narrowing | 0 | 238 | 250 | 95.2% |
| 2 | test | disc_narrowing | 1 | 12 | 250 | 4.8% |
| 2 | test | spondylolisthesis | 0 | 238 | 250 | 95.2% |
| 2 | test | spondylolisthesis | 1 | 12 | 250 | 4.8% |
| 2 | test | up_endplate | 0 | 220 | 250 | 88.0% |
| 2 | test | up_endplate | 1 | 30 | 250 | 12.0% |
| 2 | test | low_endplate | 0 | 243 | 250 | 97.2% |
| 2 | test | low_endplate | 1 | 7 | 250 | 2.8% |
| 3 | train | pfirrmann_grade | 1 | 14 | 740 | 1.9% |
| 3 | train | pfirrmann_grade | 2 | 432 | 740 | 58.4% |
| 3 | train | pfirrmann_grade | 3 | 244 | 740 | 33.0% |
| 3 | train | pfirrmann_grade | 4 | 48 | 740 | 6.5% |
| 3 | train | pfirrmann_grade | 5 | 2 | 740 | 0.3% |
| 3 | train | modic | 0 | 633 | 740 | 85.5% |
| 3 | train | modic | 1 | 15 | 740 | 2.0% |
| 3 | train | modic | 2 | 89 | 740 | 12.0% |
| 3 | train | modic | 3 | 3 | 740 | 0.4% |
| 3 | train | disc_herniation | 0 | 684 | 740 | 92.4% |
| 3 | train | disc_herniation | 1 | 56 | 740 | 7.6% |
| 3 | train | disc_bulging | 0 | 549 | 740 | 74.2% |
| 3 | train | disc_bulging | 1 | 190 | 740 | 25.7% |
| 3 | train | disc_narrowing | 0 | 708 | 740 | 95.7% |
| 3 | train | disc_narrowing | 1 | 32 | 740 | 4.3% |
| 3 | train | spondylolisthesis | 0 | 712 | 740 | 96.2% |
| 3 | train | spondylolisthesis | 1 | 28 | 740 | 3.8% |
| 3 | train | up_endplate | 0 | 661 | 740 | 89.3% |
| 3 | train | up_endplate | 1 | 79 | 740 | 10.7% |
| 3 | train | low_endplate | 0 | 696 | 740 | 94.1% |
| 3 | train | low_endplate | 1 | 44 | 740 | 5.9% |
| 3 | val | pfirrmann_grade | 1 | 17 | 250 | 6.8% |
| 3 | val | pfirrmann_grade | 2 | 122 | 250 | 48.8% |
| 3 | val | pfirrmann_grade | 3 | 91 | 250 | 36.4% |
| 3 | val | pfirrmann_grade | 4 | 18 | 250 | 7.2% |
| 3 | val | pfirrmann_grade | 5 | 2 | 250 | 0.8% |
| 3 | val | modic | 0 | 211 | 250 | 84.4% |
| 3 | val | modic | 1 | 1 | 250 | 0.4% |
| 3 | val | modic | 2 | 38 | 250 | 15.2% |
| 3 | val | modic | 3 | 0 | 250 | 0.0% |
| 3 | val | disc_herniation | 0 | 230 | 250 | 92.0% |
| 3 | val | disc_herniation | 1 | 20 | 250 | 8.0% |
| 3 | val | disc_bulging | 0 | 187 | 250 | 74.8% |
| 3 | val | disc_bulging | 1 | 63 | 250 | 25.2% |
| 3 | val | disc_narrowing | 0 | 234 | 250 | 93.6% |
| 3 | val | disc_narrowing | 1 | 16 | 250 | 6.4% |
| 3 | val | spondylolisthesis | 0 | 245 | 250 | 98.0% |
| 3 | val | spondylolisthesis | 1 | 5 | 250 | 2.0% |
| 3 | val | up_endplate | 0 | 226 | 250 | 90.4% |
| 3 | val | up_endplate | 1 | 24 | 250 | 9.6% |
| 3 | val | low_endplate | 0 | 238 | 250 | 95.2% |
| 3 | val | low_endplate | 1 | 12 | 250 | 4.8% |
| 3 | test | pfirrmann_grade | 1 | 5 | 245 | 2.0% |
| 3 | test | pfirrmann_grade | 2 | 135 | 245 | 55.1% |
| 3 | test | pfirrmann_grade | 3 | 83 | 245 | 33.9% |
| 3 | test | pfirrmann_grade | 4 | 22 | 245 | 9.0% |
| 3 | test | pfirrmann_grade | 5 | 0 | 245 | 0.0% |
| 3 | test | modic | 0 | 214 | 245 | 87.3% |
| 3 | test | modic | 1 | 8 | 245 | 3.3% |
| 3 | test | modic | 2 | 22 | 245 | 9.0% |
| 3 | test | modic | 3 | 1 | 245 | 0.4% |
| 3 | test | disc_herniation | 0 | 233 | 245 | 95.1% |
| 3 | test | disc_herniation | 1 | 12 | 245 | 4.9% |
| 3 | test | disc_bulging | 0 | 170 | 245 | 69.4% |
| 3 | test | disc_bulging | 1 | 75 | 245 | 30.6% |
| 3 | test | disc_narrowing | 0 | 234 | 245 | 95.5% |
| 3 | test | disc_narrowing | 1 | 11 | 245 | 4.5% |
| 3 | test | spondylolisthesis | 0 | 237 | 245 | 96.7% |
| 3 | test | spondylolisthesis | 1 | 8 | 245 | 3.3% |
| 3 | test | up_endplate | 0 | 233 | 245 | 95.1% |
| 3 | test | up_endplate | 1 | 12 | 245 | 4.9% |
| 3 | test | low_endplate | 0 | 229 | 245 | 93.5% |
| 3 | test | low_endplate | 1 | 16 | 245 | 6.5% |
| 4 | train | pfirrmann_grade | 1 | 27 | 740 | 3.6% |
| 4 | train | pfirrmann_grade | 2 | 402 | 740 | 54.3% |
| 4 | train | pfirrmann_grade | 3 | 252 | 740 | 34.1% |
| 4 | train | pfirrmann_grade | 4 | 58 | 740 | 7.8% |
| 4 | train | pfirrmann_grade | 5 | 1 | 740 | 0.1% |
| 4 | train | modic | 0 | 643 | 740 | 86.9% |
| 4 | train | modic | 1 | 14 | 740 | 1.9% |
| 4 | train | modic | 2 | 82 | 740 | 11.1% |
| 4 | train | modic | 3 | 1 | 740 | 0.1% |
| 4 | train | disc_herniation | 0 | 698 | 740 | 94.3% |
| 4 | train | disc_herniation | 1 | 42 | 740 | 5.7% |
| 4 | train | disc_bulging | 0 | 542 | 740 | 73.2% |
| 4 | train | disc_bulging | 1 | 198 | 740 | 26.8% |
| 4 | train | disc_narrowing | 0 | 698 | 740 | 94.3% |
| 4 | train | disc_narrowing | 1 | 42 | 740 | 5.7% |
| 4 | train | spondylolisthesis | 0 | 715 | 740 | 96.6% |
| 4 | train | spondylolisthesis | 1 | 25 | 740 | 3.4% |
| 4 | train | up_endplate | 0 | 680 | 740 | 91.9% |
| 4 | train | up_endplate | 1 | 60 | 740 | 8.1% |
| 4 | train | low_endplate | 0 | 697 | 740 | 94.2% |
| 4 | train | low_endplate | 1 | 43 | 740 | 5.8% |
| 4 | val | pfirrmann_grade | 1 | 9 | 250 | 3.6% |
| 4 | val | pfirrmann_grade | 2 | 141 | 250 | 56.4% |
| 4 | val | pfirrmann_grade | 3 | 82 | 250 | 32.8% |
| 4 | val | pfirrmann_grade | 4 | 16 | 250 | 6.4% |
| 4 | val | pfirrmann_grade | 5 | 2 | 250 | 0.8% |
| 4 | val | modic | 0 | 207 | 250 | 82.8% |
| 4 | val | modic | 1 | 5 | 250 | 2.0% |
| 4 | val | modic | 2 | 35 | 250 | 14.0% |
| 4 | val | modic | 3 | 3 | 250 | 1.2% |
| 4 | val | disc_herniation | 0 | 225 | 250 | 90.0% |
| 4 | val | disc_herniation | 1 | 25 | 250 | 10.0% |
| 4 | val | disc_bulging | 0 | 195 | 250 | 78.0% |
| 4 | val | disc_bulging | 1 | 55 | 250 | 22.0% |
| 4 | val | disc_narrowing | 0 | 239 | 250 | 95.6% |
| 4 | val | disc_narrowing | 1 | 11 | 250 | 4.4% |
| 4 | val | spondylolisthesis | 0 | 242 | 250 | 96.8% |
| 4 | val | spondylolisthesis | 1 | 8 | 250 | 3.2% |
| 4 | val | up_endplate | 0 | 217 | 250 | 86.8% |
| 4 | val | up_endplate | 1 | 33 | 250 | 13.2% |
| 4 | val | low_endplate | 0 | 237 | 250 | 94.8% |
| 4 | val | low_endplate | 1 | 13 | 250 | 5.2% |
| 4 | test | pfirrmann_grade | 1 | 0 | 245 | 0.0% |
| 4 | test | pfirrmann_grade | 2 | 146 | 245 | 59.6% |
| 4 | test | pfirrmann_grade | 3 | 84 | 245 | 34.3% |
| 4 | test | pfirrmann_grade | 4 | 14 | 245 | 5.7% |
| 4 | test | pfirrmann_grade | 5 | 1 | 245 | 0.4% |
| 4 | test | modic | 0 | 208 | 245 | 84.9% |
| 4 | test | modic | 1 | 5 | 245 | 2.0% |
| 4 | test | modic | 2 | 32 | 245 | 13.1% |
| 4 | test | modic | 3 | 0 | 245 | 0.0% |
| 4 | test | disc_herniation | 0 | 224 | 245 | 91.4% |
| 4 | test | disc_herniation | 1 | 21 | 245 | 8.6% |
| 4 | test | disc_bulging | 0 | 169 | 245 | 69.0% |
| 4 | test | disc_bulging | 1 | 75 | 245 | 30.6% |
| 4 | test | disc_narrowing | 0 | 239 | 245 | 97.6% |
| 4 | test | disc_narrowing | 1 | 6 | 245 | 2.4% |
| 4 | test | spondylolisthesis | 0 | 237 | 245 | 96.7% |
| 4 | test | spondylolisthesis | 1 | 8 | 245 | 3.3% |
| 4 | test | up_endplate | 0 | 223 | 245 | 91.0% |
| 4 | test | up_endplate | 1 | 22 | 245 | 9.0% |
| 4 | test | low_endplate | 0 | 229 | 245 | 93.5% |
| 4 | test | low_endplate | 1 | 16 | 245 | 6.5% |
| 5 | train | pfirrmann_grade | 1 | 22 | 740 | 3.0% |
| 5 | train | pfirrmann_grade | 2 | 398 | 740 | 53.8% |
| 5 | train | pfirrmann_grade | 3 | 260 | 740 | 35.1% |
| 5 | train | pfirrmann_grade | 4 | 56 | 740 | 7.6% |
| 5 | train | pfirrmann_grade | 5 | 4 | 740 | 0.5% |
| 5 | train | modic | 0 | 636 | 740 | 85.9% |
| 5 | train | modic | 1 | 14 | 740 | 1.9% |
| 5 | train | modic | 2 | 86 | 740 | 11.6% |
| 5 | train | modic | 3 | 4 | 740 | 0.5% |
| 5 | train | disc_herniation | 0 | 680 | 740 | 91.9% |
| 5 | train | disc_herniation | 1 | 60 | 740 | 8.1% |
| 5 | train | disc_bulging | 0 | 537 | 740 | 72.6% |
| 5 | train | disc_bulging | 1 | 202 | 740 | 27.3% |
| 5 | train | disc_narrowing | 0 | 708 | 740 | 95.7% |
| 5 | train | disc_narrowing | 1 | 32 | 740 | 4.3% |
| 5 | train | spondylolisthesis | 0 | 714 | 740 | 96.5% |
| 5 | train | spondylolisthesis | 1 | 26 | 740 | 3.5% |
| 5 | train | up_endplate | 0 | 678 | 740 | 91.6% |
| 5 | train | up_endplate | 1 | 62 | 740 | 8.4% |
| 5 | train | low_endplate | 0 | 702 | 740 | 94.9% |
| 5 | train | low_endplate | 1 | 38 | 740 | 5.1% |
| 5 | val | pfirrmann_grade | 1 | 6 | 250 | 2.4% |
| 5 | val | pfirrmann_grade | 2 | 159 | 250 | 63.6% |
| 5 | val | pfirrmann_grade | 3 | 73 | 250 | 29.2% |
| 5 | val | pfirrmann_grade | 4 | 12 | 250 | 4.8% |
| 5 | val | pfirrmann_grade | 5 | 0 | 250 | 0.0% |
| 5 | val | modic | 0 | 218 | 250 | 87.2% |
| 5 | val | modic | 1 | 4 | 250 | 1.6% |
| 5 | val | modic | 2 | 28 | 250 | 11.2% |
| 5 | val | modic | 3 | 0 | 250 | 0.0% |
| 5 | val | disc_herniation | 0 | 236 | 250 | 94.4% |
| 5 | val | disc_herniation | 1 | 14 | 250 | 5.6% |
| 5 | val | disc_bulging | 0 | 185 | 250 | 74.0% |
| 5 | val | disc_bulging | 1 | 65 | 250 | 26.0% |
| 5 | val | disc_narrowing | 0 | 240 | 250 | 96.0% |
| 5 | val | disc_narrowing | 1 | 10 | 250 | 4.0% |
| 5 | val | spondylolisthesis | 0 | 241 | 250 | 96.4% |
| 5 | val | spondylolisthesis | 1 | 9 | 250 | 3.6% |
| 5 | val | up_endplate | 0 | 230 | 250 | 92.0% |
| 5 | val | up_endplate | 1 | 20 | 250 | 8.0% |
| 5 | val | low_endplate | 0 | 234 | 250 | 93.6% |
| 5 | val | low_endplate | 1 | 16 | 250 | 6.4% |
| 5 | test | pfirrmann_grade | 1 | 8 | 245 | 3.3% |
| 5 | test | pfirrmann_grade | 2 | 132 | 245 | 53.9% |
| 5 | test | pfirrmann_grade | 3 | 85 | 245 | 34.7% |
| 5 | test | pfirrmann_grade | 4 | 20 | 245 | 8.2% |
| 5 | test | pfirrmann_grade | 5 | 0 | 245 | 0.0% |
| 5 | test | modic | 0 | 204 | 245 | 83.3% |
| 5 | test | modic | 1 | 6 | 245 | 2.4% |
| 5 | test | modic | 2 | 35 | 245 | 14.3% |
| 5 | test | modic | 3 | 0 | 245 | 0.0% |
| 5 | test | disc_herniation | 0 | 231 | 245 | 94.3% |
| 5 | test | disc_herniation | 1 | 14 | 245 | 5.7% |
| 5 | test | disc_bulging | 0 | 184 | 245 | 75.1% |
| 5 | test | disc_bulging | 1 | 61 | 245 | 24.9% |
| 5 | test | disc_narrowing | 0 | 228 | 245 | 93.1% |
| 5 | test | disc_narrowing | 1 | 17 | 245 | 6.9% |
| 5 | test | spondylolisthesis | 0 | 239 | 245 | 97.6% |
| 5 | test | spondylolisthesis | 1 | 6 | 245 | 2.4% |
| 5 | test | up_endplate | 0 | 212 | 245 | 86.5% |
| 5 | test | up_endplate | 1 | 33 | 245 | 13.5% |
| 5 | test | low_endplate | 0 | 227 | 245 | 92.7% |
| 5 | test | low_endplate | 1 | 18 | 245 | 7.3% |

## Độ phủ báo cáo

Không coi findings và impression là các bộ bệnh nhân độc lập.

| Trường | Có dữ liệu | Tổng BN | Tỷ lệ có | Thiếu |
| --- | --- | --- | --- | --- |
| report_vi_technique | 228 | 247 | 92.3% | 19 |
| report_vi_findings | 238 | 247 | 96.4% | 9 |
| report_vi_impression | 230 | 247 | 93.1% | 17 |
| report_en | 236 | 247 | 95.5% | 11 |

## Độ dài văn bản

Đếm bằng code. Đơn vị tách khoảng trắng KHÔNG phải token của LLM; không có nội dung báo cáo cá nhân trong bảng này.

| Trường | Đơn vị | N | Min | Q1 | Median | Mean | Q3 | Max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| report_vi_technique | ký tự | 228 | 119.0 | 119.0 | 126.0 | 125.1 | 127.0 | 134.0 |
| report_vi_technique | đơn vị tách khoảng trắng | 228 | 23.0 | 23.0 | 24.0 | 24.1 | 24.0 | 28.0 |
| report_vi_findings | ký tự | 238 | 503.0 | 642.0 | 738.0 | 748.8 | 846.8 | 1158.0 |
| report_vi_findings | đơn vị tách khoảng trắng | 238 | 103.0 | 134.2 | 155.5 | 158.0 | 179.0 | 251.0 |
| report_vi_impression | ký tự | 230 | 37.0 | 156.0 | 209.5 | 224.2 | 279.5 | 605.0 |
| report_vi_impression | đơn vị tách khoảng trắng | 230 | 8.0 | 35.0 | 47.0 | 49.9 | 62.8 | 135.0 |
| report_en | ký tự | 236 | 549.0 | 763.5 | 891.0 | 901.6 | 1009.8 | 1406.0 |
| report_en | đơn vị tách khoảng trắng | 236 | 80.0 | 116.8 | 136.0 | 137.2 | 155.0 | 232.0 |

## Văn bản lặp sau chuẩn hóa khoảng trắng

Chỉ chuẩn hóa khoảng trắng, giữ hoa/thường. Văn bản giống nhau có thể do mẫu báo cáo; chưa đủ để kết luận rò rỉ bệnh nhân/ảnh.

| Trường | Nhóm văn bản lặp | BN thuộc nhóm lặp | Nhóm qua split F1 | F2 | F3 | F4 | F5 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| report_vi_technique | 11 | 223 | 10 | 11 | 9 | 11 | 10 |
| report_vi_findings | 3 | 13 | 3 | 3 | 3 | 2 | 3 |
| report_vi_impression | 5 | 19 | 4 | 4 | 3 | 5 | 4 |
| report_en | 3 | 6 | 2 | 2 | 2 | 0 | 2 |

## Kiểm tra cấu trúc và tính nhất quán

0 chỉ nghĩa không thấy vi phạm theo phép kiểm này, không phải chứng nhận chất lượng lâm sàng.

| Kiểm tra | Số vi phạm | Đơn vị | Cách hiểu |
| --- | --- | --- | --- |
| Dòng trùng hoàn toàn (phần dư) | 0 | dòng | Không tính dòng đầu mỗi nhóm |
| Khóa patient_id + level trùng (phần dư) | 0 | dòng | Khóa kỳ vọng duy nhất |
| Bệnh nhân không đủ đúng 5 tầng chuẩn | 0 | BN | Kiểm cả tập tên tầng và số dòng |
| Không nhất quán giữa 5 dòng: sex | 0 | BN | Kiểm tra trước khi lấy 1 dòng/BN |
| Không nhất quán giữa 5 dòng: age_at_scan | 0 | BN | Kiểm tra trước khi lấy 1 dòng/BN |
| Không nhất quán giữa 5 dòng: birth_year | 0 | BN | Kiểm tra trước khi lấy 1 dòng/BN |
| Không nhất quán giữa 5 dòng: study_date | 0 | BN | Kiểm tra trước khi lấy 1 dòng/BN |
| Không nhất quán giữa 5 dòng: sub_id | 0 | BN | Kiểm tra trước khi lấy 1 dòng/BN |
| Không nhất quán giữa 5 dòng: volume | 0 | BN | Kiểm tra trước khi lấy 1 dòng/BN |
| Không nhất quán giữa 5 dòng: month | 0 | BN | Kiểm tra trước khi lấy 1 dòng/BN |
| Không nhất quán giữa 5 dòng: fold1_split | 0 | BN | Kiểm tra trước khi lấy 1 dòng/BN |
| Không nhất quán giữa 5 dòng: fold2_split | 0 | BN | Kiểm tra trước khi lấy 1 dòng/BN |
| Không nhất quán giữa 5 dòng: fold3_split | 0 | BN | Kiểm tra trước khi lấy 1 dòng/BN |
| Không nhất quán giữa 5 dòng: fold4_split | 0 | BN | Kiểm tra trước khi lấy 1 dòng/BN |
| Không nhất quán giữa 5 dòng: fold5_split | 0 | BN | Kiểm tra trước khi lấy 1 dòng/BN |
| Không nhất quán giữa 5 dòng: report_vi_technique | 0 | BN | Kiểm tra trước khi lấy 1 dòng/BN |
| Không nhất quán giữa 5 dòng: report_vi_findings | 0 | BN | Kiểm tra trước khi lấy 1 dòng/BN |
| Không nhất quán giữa 5 dòng: report_vi_impression | 0 | BN | Kiểm tra trước khi lấy 1 dòng/BN |
| Không nhất quán giữa 5 dòng: report_vi_kythuat | 0 | BN | Kiểm tra trước khi lấy 1 dòng/BN |
| Không nhất quán giữa 5 dòng: report_vi_mota | 0 | BN | Kiểm tra trước khi lấy 1 dòng/BN |
| Không nhất quán giữa 5 dòng: report_vi_ketluan | 0 | BN | Kiểm tra trước khi lấy 1 dòng/BN |
| Không nhất quán giữa 5 dòng: report_en | 0 | BN | Kiểm tra trước khi lấy 1 dòng/BN |
| Giá trị không hợp lệ: pfirrmann_grade | 0 | ô | Miền cho phép [1, 2, 3, 4, 5]; loại ô trống |
| Giá trị không hợp lệ: modic | 0 | ô | Miền cho phép [0, 1, 2, 3]; loại ô trống |
| Giá trị không hợp lệ: disc_herniation | 0 | ô | Miền cho phép [0, 1]; loại ô trống |
| Giá trị không hợp lệ: disc_bulging | 0 | ô | Miền cho phép [0, 1]; loại ô trống |
| Giá trị không hợp lệ: disc_narrowing | 0 | ô | Miền cho phép [0, 1]; loại ô trống |
| Giá trị không hợp lệ: spondylolisthesis | 0 | ô | Miền cho phép [0, 1]; loại ô trống |
| Giá trị không hợp lệ: up_endplate | 0 | ô | Miền cho phép [0, 1]; loại ô trống |
| Giá trị không hợp lệ: low_endplate | 0 | ô | Miền cho phép [0, 1]; loại ô trống |
| Alias VI khác nhau: technique | 0 | BN | So sánh chuỗi sau strip |
| Alias VI khác nhau: findings | 0 | BN | So sánh chuỗi sau strip |
| Alias VI khác nhau: impression | 0 | BN | So sánh chuỗi sau strip |
| Fold 1: split không hợp lệ | 0 | BN | Miền train/val/test; kiểm split giữa 5 dòng ở trên |
| Fold 2: split không hợp lệ | 0 | BN | Miền train/val/test; kiểm split giữa 5 dòng ở trên |
| Fold 3: split không hợp lệ | 0 | BN | Miền train/val/test; kiểm split giữa 5 dòng ở trên |
| Fold 4: split không hợp lệ | 0 | BN | Miền train/val/test; kiểm split giữa 5 dòng ở trên |
| Fold 5: split không hợp lệ | 0 | BN | Miền train/val/test; kiểm split giữa 5 dòng ở trên |
| BN không xuất hiện đúng 1 lần ở test trong 5 fold | 0 | BN | Không đồng nghĩa đã kiểm tra ảnh trùng giữa bệnh nhân |
| spacing_i thiếu/không số/không dương | 0 | dòng | Chỉ kiểm bảng, chưa kiểm header ảnh |
| spacing_j thiếu/không số/không dương | 0 | dòng | Chỉ kiểm bảng, chưa kiểm header ảnh |
| spacing_k thiếu/không số/không dương | 0 | dòng | Chỉ kiểm bảng, chưa kiểm header ảnh |
| Patient JSONL trùng ID (phần dư) | 0 | bản ghi | Kiểm toàn bộ JSONL |
| ID chỉ có một phía CSV/JSONL | 0 | ID | Đối chiếu hai nguồn |
| Khóa tầng JSON không có trong CSV | 0 | khóa | Đối chiếu cấu trúc |
| Grading CSV và JSONL khác nhau | 1 | ô | Phân biệt null và 0 |
| CSV thiếu nhưng JSONL = 0 | 1 | ô | JSONL legacy mất phân biệt chưa biết/âm tính; không sửa nguồn khi EDA |

## Nguồn và trạng thái localization

auto_ok là trạng thái pipeline, không đồng nghĩa đã được bác sĩ xác nhận.

| Nguồn | QC | Số tầng | Tổng tầng | Tỷ lệ |
| --- | --- | --- | --- | --- |
| auto | auto_ok | 1040 | 1235 | 84.2% |
| reviewed | verified | 195 | 1235 | 15.8% |

## Cờ sàng lọc grading – báo cáo

Heuristic chưa được thẩm định lâm sàng. A/B/C phân nhóm theo tiêu chí rộng hơn 'cờ cấp BN'; A+B không phải tập con của 131. Không nêu ≠ âm tính. 9 BN không có báo cáo không được đánh giá.

| Tiêu chí | Số BN | BN có báo cáo được xét | Tỷ lệ |
| --- | --- | --- | --- |
| Ít nhất một cờ ở cấp BN | 131 | 238 | 55.0% |
| Cờ BN, bỏ disc_narrowing | 86 | 238 | 36.1% |
| Báo cáo dịch chuyển nhưng grading bulging/herniation đều 0 | 21 | 238 | 8.8% |
| Cờ BN: disc_herniation | 18 | 238 | 7.6% |
| Cờ BN: disc_bulging | 39 | 238 | 16.4% |
| Cờ BN: spondylolisthesis | 17 | 238 | 7.1% |
| Cờ BN: modic | 5 | 238 | 2.1% |
| Cờ BN: disc_narrowing | 69 | 238 | 29.0% |
| Cờ BN: endplate_any | 15 | 238 | 6.3% |
| A_patient_or_internal | 43 | 238 | 18.1% |
| B_level_or_definition | 112 | 238 | 47.1% |
| C_no_flag_not_adjudicated | 83 | 238 | 34.9% |

## Ma trận trạng thái grading / báo cáo

unspecified = không phát hiện phát biểu; internal_conflict = parser tìm thấy cả khẳng định/phủ định, cần đọc lại. Không dùng bảng này tính độ chính xác bác sĩ.

| Nhãn | Grading / report | Số BN | Tổng BN được xét | Tỷ lệ |
| --- | --- | --- | --- | --- |
| disc_herniation | negative/positive | 18 | 238 | 7.6% |
| disc_herniation | negative/unspecified | 150 | 238 | 63.0% |
| disc_herniation | positive/unspecified | 27 | 238 | 11.3% |
| disc_herniation | positive/positive | 41 | 238 | 17.2% |
| disc_herniation | negative/negative | 2 | 238 | 0.8% |
| disc_bulging | positive/unspecified | 11 | 238 | 4.6% |
| disc_bulging | positive/positive | 149 | 238 | 62.6% |
| disc_bulging | negative/positive | 39 | 238 | 16.4% |
| disc_bulging | negative/unspecified | 36 | 238 | 15.1% |
| disc_bulging | positive/internal_conflict | 1 | 238 | 0.4% |
| disc_bulging | negative/negative | 1 | 238 | 0.4% |
| disc_bulging | unknown/positive | 1 | 238 | 0.4% |
| spondylolisthesis | negative/negative | 169 | 238 | 71.0% |
| spondylolisthesis | positive/negative | 9 | 238 | 3.8% |
| spondylolisthesis | positive/positive | 24 | 238 | 10.1% |
| spondylolisthesis | negative/positive | 8 | 238 | 3.4% |
| spondylolisthesis | negative/unspecified | 27 | 238 | 11.3% |
| spondylolisthesis | positive/unspecified | 1 | 238 | 0.4% |
| modic | negative/unspecified | 151 | 238 | 63.4% |
| modic | positive/positive | 39 | 238 | 16.4% |
| modic | positive/unspecified | 43 | 238 | 18.1% |
| modic | negative/positive | 5 | 238 | 2.1% |
| disc_narrowing | negative/positive | 54 | 238 | 22.7% |
| disc_narrowing | negative/negative | 119 | 238 | 50.0% |
| disc_narrowing | positive/negative | 15 | 238 | 6.3% |
| disc_narrowing | negative/unspecified | 29 | 238 | 12.2% |
| disc_narrowing | negative/internal_conflict | 1 | 238 | 0.4% |
| disc_narrowing | positive/unspecified | 2 | 238 | 0.8% |
| disc_narrowing | positive/positive | 18 | 238 | 7.6% |
| endplate_any | negative/unspecified | 144 | 238 | 60.5% |
| endplate_any | positive/unspecified | 67 | 238 | 28.2% |
| endplate_any | positive/positive | 12 | 238 | 5.0% |
| endplate_any | negative/positive | 15 | 238 | 6.3% |

## Sàng lọc riêng findings – impression

Chỉ 6 nhóm thuật ngữ, xét cấp BN, chưa kiểm toàn bộ vị trí/mức độ. Không nêu trong impression có thể là tóm tắt hợp lệ, KHÔNG tự coi là lỗi. Trái dấu vẫn cần bác sĩ đọc xác nhận; 0 không chứng minh không có mâu thuẫn.

| Nhãn | Trái dấu rõ theo parser | Findings dương / impression không nêu | Findings không nêu / impression dương | Ít nhất 1 phần có cờ nội bộ | BN có cả 2 phần |
| --- | --- | --- | --- | --- | --- |
| disc_herniation | 0 | 1 | 0 | 0 | 230 |
| disc_bulging | 0 | 2 | 0 | 1 | 230 |
| spondylolisthesis | 0 | 1 | 0 | 0 | 230 |
| modic | 0 | 7 | 0 | 0 | 230 |
| disc_narrowing | 0 | 67 | 0 | 1 | 230 |
| endplate_any | 0 | 1 | 0 | 0 | 230 |

## Danh mục nguồn local

Đếm file gồm cả README. Hash tập file = SHA256 chuỗi tên tương đối + SHA256 nội dung từng file theo thứ tự; không công khai tên file bệnh nhân.

| Thư mục | Hiện diện | Số file | Bytes | SHA256 tập file |
| --- | --- | --- | --- | --- |
| dataset_local/grading | Có | 7 | 229782 | de7704253e786776806b2392a560ddb693d133b04d6cdb431ea9c1fd8da314c8 |
| dataset_local/localize | Có | 3 | 260281 | 178b12460598bc1576ac57ac7483705001458c9ae863662ff76a9e97bc96b64e |
| dataset_local/folds | Có | 17 | 231952 | 26fbac36da026ba177efdec7309ee0cb0ce1891ca8c50b5e0b42887f99934e00 |
| dataset_local/reports_json | Có | 480 | 469187 | 210151c3dde439ae0e7bc2293f253e3a8fa691736fd41bedf09ee45b7e012622 |
| dataset_local/reports_text | Có | 4 | 234517 | 2f4c7e2ec206767b821812608a13b70477e597b33b7cd057de45f583b29f495c |
| dataset_local/reports_text_v1_reference | Có | 5 | 235351 | af8cd35aa59b4d366ec2857ce31053f740829bec7b24603f1944f8de67788beb |
| dataset/sft_data | Có | 6 | 792896 | d0e54f59f7d0317c831c1a19220bde91bf26a82715f8524815e728756876796b |
| dataset_local/nifti | Không có local | 0 | 0 |  |
| dataset_local/dicom | Không có local | 0 | 0 |  |

## Dấu vân tay đầu vào và code



| Nguồn | Bytes | SHA256 |
| --- | --- | --- |
| dataset/dataset_master.csv | 4686552 | 593003695ae128145789f4b9d8ca0f7af272d8f6d7b1844afe6127783797a693 |
| dataset/dataset_patients.jsonl | 1412913 | cc89ee34e115330e02d841848afae3c4cfd4a4bb3f0881de0e01265bc2b5418a |
| scripts/audit_annotation_consistency.py | 16917 | 9f8d4dcf3b57c50c6698ddb7913b1420d4651f86e069e661ec655d03023c38d9 |
| scripts/build_complete_eda.py | 24231 | a2aeb1b861403a33f02f68ef1830232af3f615c9773481c61e2919ef43af1fb5 |

## Các file SFT legacy

EDA kiểm số dòng JSON hợp lệ; không suy ra tập này là đầu vào chuẩn V2 hoặc đủ findings + impression.

| File | Số bản ghi | Đơn vị / lưu ý |
| --- | --- | --- |
| sft_fold1_test_en.jsonl | 47 | bản ghi SFT; không mặc định là số BN duy nhất |
| sft_fold1_test_vi.jsonl | 47 | bản ghi SFT; không mặc định là số BN duy nhất |
| sft_fold1_train_en.jsonl | 140 | bản ghi SFT; không mặc định là số BN duy nhất |
| sft_fold1_train_vi.jsonl | 142 | bản ghi SFT; không mặc định là số BN duy nhất |
| sft_fold1_val_en.jsonl | 49 | bản ghi SFT; không mặc định là số BN duy nhất |
| sft_fold1_val_vi.jsonl | 49 | bản ghi SFT; không mặc định là số BN duy nhất |
