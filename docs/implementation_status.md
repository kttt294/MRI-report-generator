# Trạng thái triển khai V1/V2 và cách chạy

Cập nhật 18/09/2026. Nhánh: `codex/kaggle-v1-v2`. Đây là bản code đã kiểm thử CPU; chưa có nghiệm thu GPU Kaggle, ảnh MRI thật hay chất lượng LLM.

## Đã triển khai

| Phần | Nội dung |
|---|---|
| V1 | Dataset cấp bệnh nhân, alias canonical, fail khi thiếu ảnh/sai split; NIfTI canonical RAS và resample oblique; cache theo fingerprint; một/nhiều sagittal views |
| Training V1 | Qwen2.5-VL 3B, LoRA chỉ language decoder, QLoRA, dtype theo GPU; loss chỉ trên assistant; không cắt âm thầm; validation loss; finite-loss guard |
| Checkpoint | Đầy đủ optimizer/scheduler/RNG/trainer state; marker/checksum; dừng theo time budget; xác minh identity trước resume, copy khỏi input read-only và sửa path best checkpoint |
| Inference V1 | Cùng preprocessing, chỉ gửi ảnh và prompt; lưu prediction, truncation và kiểm tra hai section |
| V2 contract | Tám nhãn, năm tầng, domain/status/version/provenance/QC; null khác 0; không đưa reports/folds/demographic vào prompt; JSON schemas + Pydantic |
| V2 report | Template, HF local hoặc model Hub, findings + impression một lần, validator catalog, tối đa một repair, fallback và raw attempts riêng |
| V2 LoRA | Review packet và hash-bound accepted targets, kiểm scope/catalog/folds, text-only QLoRA, checkpoint/resume/smoke reload |
| Evaluation | R0–R5, trace từng call; V1 word-F1/ROUGE-L theo whitespace; Wilson CI cho tỷ lệ và patient bootstrap cho overlap; phiếu bác sĩ, missing review không thành 0 |
| Kaggle | Ba notebook mỏng, CLI prepare/run, config overrides, dependencies, environment/version/hash receipts; ảnh và annotations tách riêng |

`examples/report_request.synthetic.json` là dữ liệu giả hoàn toàn, dùng minh họa giao diện vision → report. Không có báo cáo bệnh nhân trong repo public.

## Bằng chứng đã chạy local

- **33 tests tự động đã pass** (18/09/2026), kiểm schema/missing/aliases/split, affine phantom, prompt masking với pad bằng EOS, lỗi backend/repair/fallback, checkpoint/time budget, review gate, ETL không sửa nguồn, notebook sạch và evaluation. Không dùng dữ liệu bệnh nhân trong tests.
- Tiny GPT-2 + LoRA trên CPU: dừng ở step 2, lưu đầy đủ, nạp lại đến step 4; trọng số LoRA khớp run liên tục trong tolerance. Reload adapter có loss hữu hạn. Đây không phải Qwen GPU smoke test.
- ETL chạy dữ liệu local: 1.235 dòng, 42 cột, 247 ca. JSON derived giữ ô grading thiếu là null. Adapter cho JSON legacy phục hồi đúng một ô đã bị đổi thành zero. Hash JSON nguồn không đổi.
- Cloud orchestrator V2 template chạy local qua đường chuẩn bị dữ liệu đến receipt: 247/247 outputs; tất cả `needs_review` do mapping/ontology nguồn chưa xác nhận, không phải do lỗi thực thi.
- R0 trên validation fold 1: 50 ca annotation; không có lỗi format/catalog. Không suy ra tỷ lệ sai lâm sàng bằng 0. V1 có 46 ca validation đủ hai section; dùng `--matched-v1-cohort` khi cần cùng cohort.
- Processor Qwen2.5-VL thật, ảnh phantom và 230 target local: 137 train / 46 val / 47 test; tối đa 974 token, ít nhất 174 token target, mask đúng. Chỉ dùng ảnh giả cho kiểm tra này.
- Tokenizer Qwen2.5 3B thật với 247 input/target controlled V2: tối đa 2.659 prompt tokens, 2.170 target tokens, 4.831 full tokens; không vượt cấu hình 8.192. Đây là độ dài template, không đảm bảo LLM luôn dừng trước giới hạn.
- LM Format Enforcer khởi tạo được schema constraints với tokenizer Qwen; chưa đo R2 generation trên model weights thật.

Nhật ký/receipts riêng tư được lưu dưới `output/implementation_final`, `output/implementation_check`, `output/cloud_acceptance`; không commit lên public GitHub. Môi trường local: Python 3.12, PyTorch CPU 2.10.0, torchvision CPU 0.25.0; các version khác xem requirements/receipts. Kaggle giữ torch/torchvision CUDA được cài sẵn và kiểm tra lại bằng `check_environment.py`.

## Chạy nhanh

Chạy lệnh từ thư mục gốc repo. Trên Kaggle dùng notebook và [README từng bước](README_KAGGLE.md); phần này dành cho CLI.

### V2 template độc lập

```bash
python -m src.generate_report --input examples/report_request.synthetic.json --output output/demo/report.jsonl
```

### Chuẩn bị annotations và ảnh tách riêng

```bash
python scripts/cloud_prepare.py --annotations-root /kaggle/input/ANNOTATIONS --images-root /kaggle/input/IMAGES/nifti --output /kaggle/working/derived/run01
```

V2 không cần `--images-root`. Annotations root trực tiếp chứa grading/localize/folds/reports_json. Derived directory không được nằm trong nguồn; lần code/data đổi cần tên output mới.

### Train và inference V1

```bash
python -m src.train_v1 --config configs/v1_config.yaml --master_csv PATH/dataset_master.csv --nifti_dir PATH/nifti --output_dir output/v1-smoke --smoke --max-runtime-minutes 30
python -m src.infer_v1 --config PATH/effective_config.json --adapter PATH/final_adapter --output output/v1-val.jsonl --split val
```

YAML parser cũng đọc JSON effective config. Train chính bỏ `--smoke`, chọn output mới. Resume thêm `--resume-from PATH/checkpoint-N`; giữ code/model/config/data identity. Model load/preflight không nằm trong time budget vòng train; cần chừa thời gian save/eval và Saved Version. Callback chỉ kiểm ở ranh giới optimizer step.

Smoke kiểm hai optimizer steps, save checkpoint, reload adapter và loss hữu hạn. Full run kiểm token/image của mọi mẫu train/val trước bước train đầu. Các ca thiếu section bị loại với lý do trong manifest; ảnh thiếu hoặc lỗi không được thay bằng ảnh giả.

Trong notebook có thể thêm overrides vào `CONFIG` mà không viết implementation trong cell:

```python
"v1_overrides": {
    "model": {"revision": "MODEL_COMMIT_SHA"},
    "training": {"num_train_epochs": 5},
    "data": {"slice_offsets": [-1, 0, 1]},
}
```

Đây là run nhiều lát riêng. Không đổi overrides khi resume checkpoint của run một lát. `v2_train_overrides` tương tự với cấu trúc `configs/v2_train.yaml`.

### Chuẩn bị target V2 để duyệt

```bash
python scripts/build_v2_targets.py --requests PATH/v2/requests.jsonl --manifest PATH/v2/manifest.jsonl --output output/review_packet.jsonl
python scripts/build_v2_targets.py --requests PATH/v2/requests.jsonl --manifest PATH/v2/manifest.jsonl --reviewed PATH/reviewed.jsonl --output output/accepted_targets.jsonl
```

Packet ban đầu là template draft, `pending`, không phải clinical ground truth. Chuyên gia phải xem dữ kiện/nguồn phù hợp, ghi `reviewed_by`, `review_notes` và quyết định từng ca. File `--reviewed` chỉ chứa các ca đã accepted; code không tự phê duyệt. Cần cả train và val trong fold đã chọn. Nếu muốn văn phong ngoài catalog phải version lại policy/validator/targets, không chỉ tắt validation.

Đặt `reviewed_targets` vào Kaggle V2 Training notebook sau khi gắn file private. SFT legacy vẫn giữ cho truy vết và ablation, không được tự đưa vào nhánh này.

### Đánh giá

```bash
python scripts/evaluate_reports.py --requests PATH/v2/requests.jsonl --manifest PATH/v2/manifest.jsonl --output output/eval-r0 --arm R0 --split val --matched-v1-cohort
python scripts/evaluate_reports.py --requests PATH/v2/requests.jsonl --manifest PATH/v2/manifest.jsonl --output output/eval-r3 --arm R3 --config PATH/llm.yaml --split val
python scripts/evaluate_v1.py --predictions PATH/predictions.jsonl --master PATH/dataset_master.csv --output output/eval-v1 --split val
python scripts/summarize_review.py --input PATH/human_review.csv --output output/review_metrics.json
```

R1 prompt-only, R2 cùng prompt nhưng constrained JSON, R3 catalog một lần, R4 catalog hai lần, R5 có adapter. Mặc định evaluation không repair; fallback ghi riêng. R1/R2 chưa có bộ phân loại lỗi nội dung tự động được xác nhận; phải dùng human review. R3/R4/R5 đo **catalog nonconformance**, có thể từ chối câu paraphrase đúng. Các metric này không thay cho đọc ảnh và đánh giá bác sĩ.

`human_review.csv` để trống các cột chuyên môn; chỉ điền 0/1 sau khi review. Tổng hợp nhiều người đọc cần adjudicate thành một quyết định cấp bệnh nhân trước khi chạy script. Protocol file cho test là tài liệu nhóm khóa trước thí nghiệm (cohort, arms, model/ref, decoding, metrics, review/blinding); script ghi hash, không thể tự bảo đảm nhóm không xem test trước đó.

V1 và V2 có đầu vào khác nhau; so sánh phải báo rõ V2 dùng annotation. Không gọi V2 hiện tại là pipeline end-to-end đã có vision engine.

## Việc còn cần nghiệm thu bằng thực nghiệm

1. Chủ dự án tạo hai private Kaggle Dataset, xác nhận versions/paths và cấp GPU cho notebook.
2. V1 smoke với ảnh thật: orientation bằng mắt, coverage lát cắt, image tokens, VRAM, finite loss, adapter reload.
3. Saved Version từ runtime sạch; gắn Output lại và xác nhận full-state resume Qwen GPU.
4. Xác nhận mapping tầng và ontology với người gán nhãn; duyệt target V2 khi muốn LoRA.
5. Chạy R1–R5 theo protocol, đo latency/token/fallback và review lâm sàng. Chưa có bằng chứng để đưa ra tỷ lệ lỗi LLM hay kết luận ưu thế LoRA.

Chưa có NIfTI/model checkpoint/GPU trong máy local, nên các bước này không được ghi là đã đạt. Code không upload dữ liệu hoặc tự tạo phiên train trên tài khoản Kaggle.
