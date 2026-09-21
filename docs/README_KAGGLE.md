# Hướng dẫn Kaggle cho người mới — dự án MRI V1/V2

Cập nhật: 18/09/2026

**Mục tiêu:** ảnh và nhãn ở Kaggle private; code ở GitHub public; mở notebook, chọn vài tham số rồi gọi các script `.py` để chạy. Không viết Dataset, model hay vòng lặp train trong cell.

## 0. Trạng thái bản code

Đã triển khai V1, V2 report theo hướng A, ba notebook Kaggle và CLI dùng chung. Xem [bằng chứng kiểm thử và giới hạn](implementation_status.md). Code được chuẩn bị trên nhánh `codex/kaggle-v1-v2`; lấy commit SHA trên GitHub để điền `CODE_REF` khi khóa thí nghiệm.

- V1 đã có đọc ảnh theo affine, masking assistant, QLoRA, checkpoint/resume và inference. Chưa nghiệm thu ảnh thật/GPU Kaggle.
- V2 template đã chạy trên 247 ca local. V2 LLM chưa có số đo chất lượng; không coi template thành công là LLM thành công.
- V2 LoRA có entry point nhưng cần target được người chuyên môn duyệt. SFT cũ không tự được chấp nhận.
- Notebook cũ chỉ còn chỉ dẫn chuyển sang bản mới.

## 1. Hiểu ba thành phần cần dùng

| Tên              | Hiểu đơn giản                                  | Dự án này dùng thế nào?                                             |
| ----------------- | -------------------------------------------------- | ------------------------------------------------------------------------- |
| GitHub repository | Nơi lưu code                                     | Public, chỉ code/config/docs và dữ liệu giả lập phục vụ test      |
| Kaggle Dataset    | Nơi lưu file để notebook đọc                 | Private, gồm ảnh và nhãn của dự án                                 |
| Kaggle Notebook   | Trang chứa các ô lệnh, chạy trên máy Kaggle | Private; tải code GitHub, đọc Dataset, chạy script và lưu kết quả |

**Public code không có nghĩa public dữ liệu.** Notebook chạy dữ liệu thật cũng nên giữ private vì outputs có thể chứa báo cáo hoặc mã bệnh nhân.

Một số từ bạn sẽ gặp:

- **Cell:** một ô văn bản hoặc lệnh. Nút tam giác bên cạnh chạy ô đó.
- **GPU / Accelerator:** phần cứng giúp train nhanh hơn; chỉ bật khi cần chạy model.
- **Run:** một lần chạy thí nghiệm.
- **Checkpoint:** bản lưu trạng thái train để tiếp tục sau đó.
- **Resume:** tiếp tục từ checkpoint; khác với train lại từ đầu.
- **Version:** một phiên bản đã lưu của Dataset hoặc Notebook.
- **Slug:** tên ngắn nằm trong URL, ví dụ `lumbar-mri-images`.

Trước khi upload, bảo đảm việc dùng Kaggle nằm trong quyền sử dụng dữ liệu của nhóm. README đi kèm dữ liệu nguồn hiện có điều khoản không nạp dịch vụ bên thứ ba; lựa chọn Private không tự thay thế quyền sử dụng đó. Hướng dẫn này không tự upload file của bạn.

## 2. Chuẩn bị hai bộ dữ liệu riêng

**Tách ảnh và annotations để cập nhật nhãn không phải upload lại 9 GB ảnh.**

### Bộ A: ảnh MRI

Tên gợi ý: `lumbar-mri-images`, đặt **Private**.

```text
nifti/
└── ... giữ nguyên cây thư mục và file ảnh nguồn ...
```

- Dùng thư mục `nifti/` khoảng 9 GB. Giữ tên file và cấu trúc bên trong; không đổi tên bệnh nhân hoặc volume bằng tay.
- Chưa cần đưa `dicom/` lên cho V1 hiện tại.
- Nếu file là `.nii.gz`, giữ nguyên; không giải nén thành `.nii` chỉ để upload. ZIP ngoài cùng giúp gom folder, không nhất thiết giảm nhiều dung lượng vì `.gz` đã nén.
- Khi tạo ZIP, giữ thư mục `nifti` bên trong để dễ nhận diện. Dùng một ZIP hoàn chỉnh; đừng chọn kiểu chia archive thành `.zip.001`, `.zip.002`.

### Bộ B: nhãn và báo cáo

Tên gợi ý: `lumbar-mri-annotations`, đặt **Private**.

```text
grading/          # Giữ grading_all.csv và các file nguồn đang có
localize/         # Có disc_localization.csv
folds/            # fold1 ... fold5, giữ train/val/test
reports_json/     # Giữ cả report/ và metadata/
README.md         # Điều kiện sử dụng và mô tả nguồn
```

`reports_text/` có thể thêm nếu cần tái lập export tiếng Anh legacy. `reports_text_v1_reference/` chưa cần cho hướng A. Upload toàn bộ annotations của cohort, không dùng JSON của một bệnh nhân thay cho bộ dữ liệu.

Không tự sửa nhãn thiếu thành 0. Code sẽ đọc nguồn và giữ missingness. Dữ liệu hợp nhất trong `dataset/` sẽ được script tạo vào thư mục ghi được, không ghi đè hai bộ nguồn này.

### Nếu dữ liệu mới chỉ có trên Google Drive

Đường dễ thao tác nhất cho lần đầu:

1. Mở Drive bằng tài khoản có quyền.
2. Tải `nifti/` và các folder annotations cần thiết về máy.
3. Kiểm tra tải đã hoàn tất. Nếu Drive trả nhiều ZIP độc lập, giải nén tất cả và đối chiếu cây file; không upload thiếu một phần.
4. Gom lại thành hai bộ A/B như trên và tạo ZIP hoàn chỉnh cho từng bộ nếu muốn.

Máy cần đủ dung lượng cho bản tải, bản giải nén và archive tạm. Đừng xóa nguồn Drive. Nếu tải 9 GB xuống máy bất tiện, có thể dùng Colab **CPU** làm bước trung chuyển một lần: mount Drive có quyền → stage file → xác thực Kaggle API → tạo private dataset. Cách trung chuyển cần thông tin xác thực Kaggle riêng; không phải dán link Drive private vào Kaggle là tự tải được.

## 3. Upload lên Kaggle và kiểm tra

Tên nút có thể thay đổi nhẹ theo giao diện. Thao tác chính là tạo Dataset, upload file và giữ quyền truy cập private.

1. Đăng nhập Kaggle, mở **Datasets** ở thanh điều hướng.
2. Chọn **New Dataset**.
3. Chọn upload từ máy, kéo ZIP của bộ A vào cửa sổ hoặc bấm chọn file.
4. Đặt tên `Lumbar MRI Images`; ghi lại slug/URL Kaggle tạo ra.
5. Kiểm tra visibility là **Private** trước khi tạo. Không chọn Make Public hoặc Publish Publicly.
6. Chờ upload và xử lý hoàn tất; với 9 GB đừng đóng cửa sổ khi vẫn đang upload.
7. Mở trang Dataset vừa tạo, xem danh sách file: phải thấy đủ cây ảnh, không chỉ vài mảnh archive.
8. Lặp lại cho bộ B, đặt tên `Lumbar MRI Annotations`, cũng **Private**.

Kaggle có thể giải nén archive khi nhập dữ liệu; hãy kiểm tra cấu trúc sau xử lý thay vì đoán path từ tên ZIP. Theo tài liệu hiện tại, 9 GB nằm dưới giới hạn dataset/private storage nếu tài khoản còn đủ quota. [Tài liệu Kaggle Datasets](https://www.kaggle.com/docs/datasets)

Sau upload, ghi lại:

| Thông tin                  | Ảnh                | Annotations         |
| --------------------------- | ------------------- | ------------------- |
| URL hoặc`owner/slug`     | Điền URL bộ A    | Điền URL bộ B    |
| Version dùng cho run đầu | Điền số version  | Điền số version  |
| Visibility                  | Private             | Private             |
| Số file và dung lượng   | So với bản nguồn | So với bản nguồn |

Số file ảnh **không nhất thiết bằng 247**: một người có thể có nhiều chuỗi hoặc file phụ. Kiểm tra theo danh sách volume thực sự cần cho cohort; script preflight sẽ làm phần này. Khi cập nhật nhãn, tạo version mới của bộ B và ghi lại version, không đổi cả ảnh nếu ảnh không thay đổi.

## 4. Tạo notebook và gắn dữ liệu

Bạn có thể làm quen bước này ngay bằng một notebook trống, chưa cần bật GPU.

1. Vào **Code / Notebooks**, chọn **New Notebook**.
2. Đặt tên dễ nhớ, ví dụ `MRI V1 fold1 thử nghiệm`.
3. Mở **Share / Sharing**, kiểm tra notebook là **Private**.
4. Trong vùng **Input / Data**, chọn **Add Input** (một số giao diện ghi Add Data).
5. Tìm trong dữ liệu của bạn và thêm `lumbar-mri-images`, rồi `lumbar-mri-annotations`.
6. Mở cây file của từng input; dùng nút sao chép đường dẫn nếu có để ghi lại path thật của `nifti/` và folder chứa `grading/`.

Không gõ cứng một đường dẫn lấy từ ví dụ trên mạng. Dataset có thể có thêm folder bọc ngoài sau upload; path đúng là path đang hiển thị trong notebook của bạn.

Hai vùng file cần phân biệt:

| Vùng                   | Dùng để làm gì?                                                                    |
| ----------------------- | --------------------------------------------------------------------------------------- |
| `/kaggle/input/...`   | Đọc dataset đã gắn. Không lưu CSV mới, cache hoặc checkpoint vào đây        |
| `/kaggle/working/...` | Code checkout và kết quả của run; các file cần lưu thành outputs đặt ở đây |

V1 cần cả hai inputs. Notebook V2 report chỉ cần annotations; không cần đọc 9 GB ảnh.

## 5. Mở notebook dự án và lấy code GitHub

Dùng notebook trong nhánh `codex/kaggle-v1-v2`, không dùng vòng lặp train của bản Colab cũ.

Các notebook:

- `notebooks/Kaggle_V1_Training.ipynb`: train/resume V1.
- `notebooks/Kaggle_V2_Report.ipynb`: chuẩn bị JSON và chạy V2 report (template dùng CPU).
- `notebooks/Kaggle_V2_Training.ipynb`: train V2 sau khi có target đã duyệt.

Thực hiện:

1. Mở nhánh [codex/kaggle-v1-v2](https://github.com/kttt294/MRI-report-generator/tree/codex/kaggle-v1-v2), tải notebook tương ứng.
2. Dùng chức năng **Import/Upload Notebook** trong Kaggle; nếu giao diện hỗ trợ import URL GitHub thì có thể dùng URL notebook được cung cấp.
3. Kiểm tra lại tên, chế độ Private và hai inputs sau khi import.
4. Trong Settings/Session options, bật **Internet** để notebook tải code, dependencies và model.
5. Cell đầu tiên sẽ có chỗ điền code ref/commit, đường dẫn dữ liệu và tên run. Dùng ref được giao, không tự dùng `main` mới nhất cho tất cả thí nghiệm rồi quên ghi phiên bản.
6. Cell clone/install gọi script ngắn; không cần copy hàng trăm dòng Python vào notebook. Đọc thông báo cuối cell: nếu có lỗi, xử lý trước khi bấm cell tiếp theo.

GitHub public không cần GitHub token để clone. Nếu model cần quyền riêng thì dùng thông tin xác thực qua Secrets theo hướng dẫn của model; không dán token vào code sẽ công khai. Bộ code phát hành phải ghi rõ model nào được dùng.

## 6. Chọn GPU và cấu hình lần chạy đầu

Sau khi đã chuẩn bị xong dữ liệu và notebook:

1. Mở **Settings / Session options**.
2. Ở **Accelerator**, chọn **GPU T4 ×2** nếu tài khoản được cấp lựa chọn này.
3. Kiểm tra quota còn lại trong giao diện. Verified tài khoản giúp đáp ứng điều kiện truy cập, nhưng không đảm bảo luôn còn quota hoặc GPU trống.
4. Khi không dùng model, dừng session GPU để tránh tiêu quota.

P100 không còn là lựa chọn nên dựa vào: Kaggle thông báo ngừng P100 từ 15/09/2026. [Thông báo Kaggle](https://www.kaggle.com/discussions/product-announcements/735239)

**T4 ×2 không có nghĩa code mặc định dùng cả hai GPU.** Code hiện tại chọn một GPU; đa GPU cần cấu hình riêng. Hai GPU cũng không tự cộng bộ nhớ thành một vùng VRAM chung.

Các tham số người dùng sẽ điền trong notebook:

| Tham số         | Cách chọn lần đầu                                                 |
| ---------------- | ---------------------------------------------------------------------- |
| Code ref         | Commit/tag đã được giao cùng notebook                            |
| Images root      | Path thật của folder NIfTI từ Add Input                             |
| Annotations root | Path thật chứa`grading`, `folds`, `localize`, `reports_json` |
| Fold             | `1`                                                                  |
| Run name         | Tên mới, ví dụ`v1_fold1_smoke_01`                                |
| Mode             | `smoke` cho lần đầu; chưa chọn `full`                         |
| Resume from      | Để trống cho run đầu                                              |
| Time budget      | Theo quota/giới hạn hiện tại, chừa thời gian lưu output         |

Tên tham số chính xác nằm trong dictionary `CONFIG`: `annotations_root`, `images_root`, `annotations_dataset`, `images_dataset`, `fold`, `run_name`, `mode`, `resume_from`, `max_runtime_minutes`. Hai trường dataset ghi dạng `owner/slug/version`, ví dụ `tenban/lumbar-mri-images/1`. Dùng path thực tế từ Add Input, không chép nguyên `CHANGE-ME`.

`max_runtime_minutes` giới hạn vòng train, tính từ lúc bắt đầu train; tải model, kiểm tra toàn bộ ảnh/token và đánh giá/lưu checkpoint mất thêm thời gian. Bước dừng diễn ra ở ranh giới optimizer step, không phải ngắt cứng theo giây. Đặt 30 phút cho lần đầu, chừa dư thời gian theo quota thực tế.

Nếu cell bootstrap báo repo đã tồn tại, không clone đè. Với cùng phiên chỉ đổi CONFIG/run_name rồi chạy lại cell CONFIG và cell gọi cloud_run; với Save & Run All dùng runtime sạch. Không thay code trong một run.

## 7. Chạy kiểm tra nhỏ trước khi train đầy đủ

**Smoke test** là một lượt kiểm tra nhỏ để phát hiện sai đường dẫn, thiếu ảnh hoặc lỗi model trước khi tốn nhiều GPU giờ.

Trong notebook đã phát hành, chạy lần lượt các cell từ trên xuống:

1. **Clone/install:** tải đúng phiên bản code và dependencies.
2. **Prepare/preflight:** kiểm tra dữ liệu, tạo CSV/JSON ở thư mục derived có quyền ghi.
3. **Smoke test:** đọc một ít ảnh, chạy forward/backward, lưu checkpoint và nạp lại adapter để kiểm tra loss. Resume đầy đủ được kiểm tra riêng ở bước 10.
4. **Summary:** xem kết quả từng kiểm tra và vị trí file đã lưu.

Chỉ chuyển sang full khi:

- Đọc được volume thật; không có ảnh xám thay thế.
- Các bệnh nhân và split khớp; thiếu target/ảnh được báo rõ.
- Có loss hữu hạn; không xuất hiện `nan`, CUDA OOM hoặc lỗi token ảnh.
- `checkpoint-2/COMPLETE.json` và `smoke_reload.json` xuất hiện; reload adapter có loss hữu hạn.
- Preflight xác nhận không cắt mất target hoặc token ảnh.

Đọc được 9 GB dataset không có nghĩa phải nạp cả 9 GB vào GPU: loader đọc từng mẫu/batch. GPU có đủ VRAM hay không phụ thuộc model, số ảnh/token và batch size. Vì vậy cần đo bằng smoke test thực tế.

## 8. Train bằng Save & Run All

Sau khi smoke test đạt:

1. Đổi chế độ thành `full`, đặt run name mới, chọn thời gian chạy phù hợp.
2. Kiểm tra cell cuối lưu/tổng kết kết quả có trong notebook.
3. Chọn **Save Version → Save & Run All** và đặt mô tả phiên bản dễ nhận biết.
4. Xác nhận job bắt đầu; theo dõi trạng thái và logs ở version đang chạy.
5. Dừng phiên GPU interactive không còn cần thiết để tránh chạy hai job trùng nhau.

Save & Run All tạo phiên sạch và chạy notebook từ đầu. Nó **không tiếp tục nguyên biến hoặc file tạm của phiên bạn vừa bấm từng cell**. Do đó mọi bước clone/config/prepare phải có trong notebook. Quick Save chủ yếu chụp lại notebook hiện tại; nếu muốn giữ file output interactive cần kiểm tra tùy chọn lưu outputs và xác nhận file thực tế. [Tài liệu Notebook](https://www.kaggle.com/docs/notebooks)

Job Save & Run All có thể chạy nền sau khi bạn đóng tab, nhưng vẫn chịu quota và giới hạn phiên. Đóng tab không giống bấm Stop/Cancel. [Giải thích của Kaggle về chạy nền](https://www.kaggle.com/general/232625)

Không đặt toàn bộ train vừa khít giới hạn tối đa. Code cần dừng có kiểm soát theo time budget và lưu checkpoint để run sau tiếp tục; nếu runtime bị cắt đột ngột, phần chưa lưu bền có thể mất.

## 9. Kiểm tra và tải kết quả về

Khi job kết thúc:

1. Mở đúng **version** vừa chạy, xem trạng thái hoàn thành và logs; chưa có lỗi không đồng nghĩa chất lượng mô hình đã tốt.
2. Mở vùng **Output / Files**, kiểm tra thư mục run.
3. Tải bản kết quả cần giữ về máy hoặc tạo bản lưu riêng theo quy trình nhóm.
4. Ghi code commit, versions hai datasets và tên run vào nhật ký thí nghiệm.

Các artifact chính (tùy task):

```text
runs/<run_name>/
├── run_manifest.json      # Code/model/config/data versions và môi trường
├── checkpoint-.../        # Dùng resume, phải là checkpoint hoàn chỉnh
├── final_adapter/        # Có khi train hoàn thành; dùng inference
├── logs/                 # Tiến độ và metrics
└── predictions...        # Nếu đã chạy inference
```

Tên chi tiết có thể được điều chỉnh khi giao code, nhưng phải có manifest và thông tin resume. Adapter thường phải được nạp cùng đúng base model; không coi adapter là toàn bộ trọng số model.

Giữ working gọn: không copy 9 GB ảnh và toàn bộ cache base model vào thư mục kết quả. Output storage khác với input dataset storage. Chỉ có file trong ổ làm việc của phiên hiện tại chưa chứng minh file đã tồn tại trong Saved Version.

## 10. Tiếp tục train ở phiên sau — resume

1. Mở lại notebook dự án.
2. Trong **Add Input**, gắn hai dataset đúng phiên bản đã dùng và **outputs của notebook version trước**. Nếu giao diện không tìm thấy output, có thể dùng bản checkpoint đã tải về làm một dataset private riêng.
3. Mở cây file input mới, copy path checkpoint hoàn chỉnh.
4. Điền vào `resume_from`, giữ code/model/config/data tương thích và đặt tên run mới.
5. Chạy preflight resume rồi Save & Run All.
6. Xem logs: global step phải tiếp tục từ checkpoint; nếu về 0 thì chưa resume đúng.

Không cần có `final_adapter` mới được resume. Checkpoint giữa chừng phải có đầy đủ trạng thái optimizer/scheduler/trainer/RNG theo cấu hình code. Script sẽ copy phần cần ghi ra working; input chứa checkpoint cũ vẫn chỉ đọc.

Nếu phiên cũ mất trước khi checkpoint được lưu ra outputs hoặc backup thì không thể khôi phục chỉ từ notebook code. Đó là lý do nên train theo chặng và xác nhận artifact sau từng chặng.

## 11. V2 khác V1 ở đâu khi dùng Kaggle?

| Công việc                           | Input                                         | GPU                             |
| ------------------------------------- | --------------------------------------------- | ------------------------------- |
| Train V1 ảnh → report               | Ảnh + annotations                            | Có                             |
| Chuyển dữ liệu sang JSON hướng A | Annotations                                   | Không cần                     |
| V2 template baseline                  | JSON hướng A                                | Không cần                     |
| V2 LLM sinh report                    | JSON + model text                             | Thường nên có               |
| Fine-tune V2 LoRA                     | Cặp JSON–report được kiểm tra và model | Có; chưa tự bật từ SFT cũ |

JSON hiện tại là annotation dùng thay đầu ra vision trong thí nghiệm, không phải kết quả một vision engine đã được huấn luyện. V2 chạy template thành công không có nghĩa LLM đã được train hoặc đã đo xong chất lượng lâm sàng.

## 12. Các lỗi thường gặp

| Bạn thấy gì?                                      | Cách xử lý                                                                                                                |
| ---------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------- |
| Không tìm thấy dataset private                    | Đúng tài khoản chủ sở hữu/cộng tác viên chưa? Đã Add Input chưa?                                               |
| Thiếu`nifti` hoặc file ảnh                      | Mở cây Input và lấy đúng path; kiểm tra upload đủ các ZIP/phần download                                           |
| Read-only file system                                | Đang ghi vào input; output phải trỏ sang working                                                                         |
| Không chọn được GPU                             | Xem verification, quota và tài nguyên đang có; chờ nếu chưa được cấp, không cần đổi dữ liệu sang public    |
| Clone/pip/model download lỗi                        | Bật Internet; đọc lỗi cụ thể, không chạy tiếp cell train                                                            |
| Không có script/notebook trên GitHub              | Code mới chưa push hoặc checkout sai ref; không tự chép implementation vào cell                                       |
| Thiếu`src.models` hoặc lỗi collator của V1 cũ | Dùng nhầm bản chưa sửa; kiểm tra release ref theo bước 0                                                             |
| CUDA out of memory                                   | Dừng run; giảm batch/ảnh/token theo config đã hỗ trợ, giữ train/val policy nhất quán; đừng xóa target âm thầm |
| GPU thứ hai không hoạt động                     | Có thể đúng thiết kế một GPU của bản đầu; không tự đổi sang đa GPU giữa run                                 |
| Hết dung lượng output                             | Kiểm tra checkpoints giữ quá nhiều hoặc cache/ảnh bị copy vào working; giữ backup trước khi dọn                  |
| Save & Run All chạy lại từ đầu                  | Bình thường nếu chưa cấu hình resume; đây là một phiên sạch                                                     |
| Đã Quick Save nhưng không thấy checkpoint       | Kiểm tra tùy chọn lưu outputs và nội dung Saved Version; code autosave không đủ                                     |

## 13. Checklist ngắn cho lần đầu

- [ ] Có quyền dùng dữ liệu trên Kaggle; hai dataset và notebook đều private.
- [ ] Upload ảnh và annotations xong, thấy đủ file và ghi lại versions.
- [ ] Đã có notebook Kaggle và code phát hành được kiểm thử; không dùng bản Colab legacy.
- [ ] Add Input, lấy đúng hai root paths; Internet bật khi cần tải code/model.
- [ ] Chọn GPU khi bắt đầu smoke test/model, kiểm tra quota.
- [ ] Smoke test đạt rồi mới chạy full.
- [ ] Save & Run All hoàn thành, mở Output xác nhận checkpoint.
- [ ] Có backup và đã hiểu cách Add Input checkpoint để resume.

**Việc bạn nên làm trước tiên:** chuẩn bị hai bộ A/B, upload private và ghi lại hai URL dataset. Chưa cần bật GPU hoặc chạy notebook V1 cũ.

## 14. Chạy V2 LLM và chuẩn bị LoRA

Lần đầu chọn notebook V2 Report, giữ `report_overrides={"backend": "template"}`. Không cần gắn dataset ảnh và không cần GPU. Tất cả 247 ca hiện mang `needs_review` do ánh xạ tầng/ontology chưa được xác nhận; đây là trạng thái dự kiến, không phải lỗi cài đặt.

Để thử LLM, bật GPU, đặt `report_overrides`:

```python
"report_overrides": {
    "backend": "hf_local",
    "model_path": "Qwen/Qwen2.5-3B-Instruct",
    "local_files_only": False,
    "use_4bit": True,
    "max_new_tokens": 4096,
    "max_input_tokens": 8192,
}
```

Chế độ smoke của LLM chạy hai ca. Xem `reports.jsonl` và file `.summary.json`: attempts lưu câu trả lời thô, lỗi, số token; fallback được tách riêng. Validator hiện kiểm câu trong catalog rất chặt, có thể loại cả cách diễn đạt tương đương. Không diễn giải tỷ lệ bị loại là tỷ lệ sai lâm sàng. Mỗi output chỉ diễn đạt tám nhãn, không phải báo cáo MRI toàn diện.

V2 Training chỉ dùng khi có `reviewed_targets` gắn qua private dataset. Tạo review packet bằng `scripts/build_v2_targets.py`, duyệt từng ca rồi assemble bằng `--reviewed`. Code kiểm input hash, người duyệt, scope, đủ năm folds và tuân thủ catalog. Xem [các lệnh](implementation_status.md). LoRA hiện phục vụ ablation controlled generation; chưa chứng minh có ích hơn template, không phải fine-tune văn phong tự do.

## 15. Chọn adapter và đánh giá

V1 mặc định `Qwen/Qwen2.5-VL-3B-Instruct`, V2 text `Qwen/Qwen2.5-3B-Instruct`; ghi model revision thực tế trong manifest. Với thí nghiệm chính nên khóa `revision` trong config theo commit model. T4 dùng FP16; code không ép BF16 khi GPU không hỗ trợ.

Sau full train, `final_adapter` là model được chọn qua validation loss nếu có checkpoint được đánh giá; `checkpoint-N` là trạng thái train tại bước N. Khi hết time budget, `latest_adapter` chỉ dùng inference/kiểm tra, luôn resume bằng checkpoint đầy đủ. Smoke và full khác cấu hình nên không resume chéo.

Để inference V1, trong cùng notebook đặt task `v1-infer`, tên run mới, thêm `adapter_path` và `split: "val"`. File kết quả có `prediction`, `truncated`, `has_both_sections`; đánh giá nội dung cần bác sĩ. Đánh giá V2 R0–R5 dùng `scripts/evaluate_reports.py`; test split yêu cầu file protocol đã khóa, không dùng test để chọn prompt/epoch.
