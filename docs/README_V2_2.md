# V2-2: thử nghiệm grading → báo cáo gốc của bác sĩ

V2-2 dùng 5 tầng × 8 nhãn grading làm đầu vào và hai phần `reports.vi.findings`, `reports.vi.impression` gốc làm target. Prompt không chứa câu mẫu, báo cáo đích, mã bệnh nhân hoặc split. Model chỉ được tính loss trên câu trả lời. Đây là thí nghiệm nghiên cứu; báo cáo gốc có thể chứa phát hiện không biểu diễn được trong tám nhãn, vì vậy cần bác sĩ đánh giá các nhận định không có căn cứ từ đầu vào.

Fold 1 có 137 ca train, 46 ca validation và 47 ca test có đủ hai phần báo cáo. Chọn checkpoint bằng validation loss, không dùng test để chọn model. Fold 1 test đã được xem ở thí nghiệm Phase 1 nên kết quả V2-2 trên tập này chỉ là so sánh thăm dò. Grading ở đây là annotation sẵn có, chưa phải dự đoán của vision engine.

Mở `notebooks/Kaggle_V2_2_Training.ipynb` trên Kaggle và giữ notebook ở chế độ Private. Gắn Dataset private `kieuthithutrang/lumbar-mri-annotations`, bật Internet và GPU T4. Notebook tải code từ GitHub theo commit đã ghim rồi gọi `scripts/run_v2_2_cloud.py`; không cần chép vòng train vào cell.

Chạy đầu tiên với `MODE = "smoke"` và `RUN_NAME` mới. Save & Run All, xác nhận Saved Version có `runs/<RUN_NAME>/preflight.json`, `training/checkpoint-2/COMPLETE.json`, `training/smoke_adapter/` và `training/train_result.json`. Smoke chỉ kiểm tra pipeline, chưa tạo model nghiên cứu.

Sau khi smoke đạt, đổi `MODE = "full"`, đặt `RUN_NAME` mới và Save & Run All. Full mặc định 2 epochs tối đa, đánh giá validation và lưu checkpoint mỗi 10 optimizer steps; early stopping sau hai lượt đánh giá không cải thiện. `training/best_adapter/` là adapter được chọn, `training/train_result.json` ghi `best_metric`, checkpoint và bước huấn luyện. Khi job dừng vì thời gian, phải đọc `completed_schedule` và checkpoint; không coi việc có file adapter là đã hoàn tất lịch train.

Lệnh preflight local không cần GPU:

`python scripts/v2_2_preflight.py --patients dataset/lumbar-mri-annotations/dataset_patients.jsonl --master dataset/lumbar-mri-annotations/dataset_master.csv --fold 1 --output output/v2_2_preflight.json`

Đánh giá kết quả phải tách điểm NLP so với báo cáo bác sĩ khỏi độ đúng dữ kiện, sai tầng và nhận định không có trong grading. BLEU, ROUGE hay BERTScore cao không tự xác nhận tính đúng lâm sàng.

Kết quả lần chạy đầu tiên được ghi tại `docs/V2_2_first_run_results.md`.

Để đánh giá adapter, dùng notebook private `notebooks/Kaggle_V2_2_Evaluation.ipynb`. Gắn hai Dataset private `kieuthithutrang/lumbar-mri-annotations/1` và `kieuthithutrang/mri-v2-2-fold1-adapter/1`; chạy `MODE = "smoke"` trên 2 ca validation trước, rồi `MODE = "test"` trên đủ 47 ca test. Protocol `configs/v2_2_test_protocol.json` khóa nguồn dữ liệu, SHA-256 adapter, cách sinh và cách tính metrics. Script `scripts/evaluate_v2_2.py` ghi `metrics.json`, `predictions.jsonl`, `per_patient.jsonl` và `human_review.csv` vào Output private. Không đưa các file chứa văn bản bệnh nhân lên GitHub.

BLEU-4, ROUGE-1, ROUGE-L và BERTScore ở đây so với báo cáo bác sĩ gốc; do bộ tham chiếu khác Phase 1, không so sánh trực tiếp trị số hai thí nghiệm. JSON không hợp lệ được tính là lỗi, không sinh lại hay thay bằng rule. Bảng `human_review.csv` cần bác sĩ điền các lỗi sai tầng, bỏ sót và nhận định không có căn cứ.

Biến thể thử nghiệm `scripts/v2_2_retry_validation.py` dùng `src/report/v2_2_quality.py` kiểm tra JSON và giới hạn độ dài (`findings` 1.200 ký tự, `impression` 650 ký tự; đều trên giá trị lớn nhất của train/validation). Nếu lần đầu lỗi, nó sinh lại tối đa một lần với chỉ dẫn ngắn hơn. Trước mắt script chỉ chạy validation; kết quả của nó không được trộn với protocol test baseline. Bộ kiểm tra này không thể xác nhận một nhận định y khoa có căn cứ hay không.
