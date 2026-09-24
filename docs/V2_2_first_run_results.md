# V2-2: lần huấn luyện đầu tiên

Notebook Kaggle private: https://www.kaggle.com/code/kieuthithutrang/mri-v2-2-training

V2-2 học ánh xạ từ 5 tầng × 8 nhãn grading được gán sẵn sang `findings` và `impression` tiếng Việt gốc của bác sĩ. Đây chưa phải đầu ra dự đoán từ vision engine. Model nền là `Qwen/Qwen2.5-3B-Instruct`, fine-tune bằng QLoRA trên một GPU T4. Code chạy trên Kaggle được ghim tại commit `ecf1c3a`; dữ liệu annotations lấy từ Dataset private `kieuthithutrang/lumbar-mri-annotations/1`.

Fold 1 có 137 ca train và 46 ca validation đủ cả hai phần báo cáo. Có 47 ca test đủ báo cáo nhưng lần chạy này không dùng để train, chọn checkpoint hay tính metric. Đầu vào prompt không chứa báo cáo gốc, mã bệnh nhân hoặc split.

Smoke run hoàn tất 2/2 bước. Full run `v2-2-fold1-full-01` hoàn tất 34/34 bước (`completed_schedule=true`) sau khoảng 85 phút train. Validation loss giảm theo các checkpoint:

| Bước | Validation loss |
| ---: | ---: |
| 10 | 1,470641 |
| 20 | 1,194613 |
| 30 | 1,057668 |

`best_adapter` được chọn từ checkpoint 30. Chín file của adapter đã được tải về `output/v2_2_full_kaggle/runs/v2-2-fold1-full-01/training/best_adapter/` và kiểm tra SHA-256, khớp checkpoint 30. Bản gốc cũng nằm trong Output của notebook Kaggle. `train_result.json` và `trainer_state.json` được lưu cùng thư mục output local.

Validation loss cho thấy mô hình học được dạng báo cáo trong tập validation; nó không đo độ đúng lâm sàng hoặc mức độ bám sát grading. Báo cáo gốc có thể chứa thông tin không nằm trong 8 nhãn đầu vào. Chưa chạy sinh văn bản hoặc BLEU, ROUGE, BERTScore trên test trong lần này. Fold 1 test từng được xem trong thí nghiệm Phase 1, nên nếu dùng tiếp chỉ là tập so sánh thăm dò, không phải holdout hoàn toàn độc lập.
