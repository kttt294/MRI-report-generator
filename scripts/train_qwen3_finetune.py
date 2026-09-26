"""Fine-tune Qwen3-VL-4B-Instruct with QLoRA on Fold 1 Vietnamese MRI reports, then evaluate on test set."""
import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
import yaml

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.evaluate_v2_2 import parse_report, join_sections, compute_metrics
from src.data.v2_2_dataset import load_samples
from src.io_utils import strict_loads, write_json, write_jsonl, file_hash, environment_receipt


class Qwen3TextCollator:
    def __init__(self, tokenizer, max_length=4096):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch):
        import torch
        conversations = [[{"role": "user", "content": r["prompt"]},
                          {"role": "assistant", "content": r["completion"]}] for r in batch]
        texts = [self.tokenizer.apply_chat_template(c, tokenize=False, add_generation_prompt=False)
                 for c in conversations]
        prefixes = [self.tokenizer.apply_chat_template(c[:1], tokenize=False, add_generation_prompt=True)
                    for c in conversations]
        inputs = self.tokenizer(texts, padding=True, truncation=False, add_special_tokens=False, return_tensors="pt")
        prompt_ids = self.tokenizer(prefixes, add_special_tokens=False)["input_ids"]
        labels = torch.full_like(inputs["input_ids"], -100)
        for i, prefix in enumerate(prompt_ids):
            positions = inputs["attention_mask"][i].nonzero(as_tuple=True)[0]
            sequence = inputs["input_ids"][i][positions]
            if len(sequence) > self.max_length or sequence[:len(prefix)].tolist() != prefix or len(sequence) <= len(prefix):
                # Fallback truncate if sequence exceeds max_length
                target_start = len(prefix)
                labels[i, positions[target_start:self.max_length]] = sequence[target_start:self.max_length]
            else:
                labels[i, positions[len(prefix):]] = sequence[len(prefix):]
        inputs["labels"] = labels
        return inputs


def find_dataset_files(input_root):
    input_root = Path(input_root)
    # Search for dataset_patients.jsonl and dataset_master.csv
    patients = list(input_root.rglob("dataset_patients.jsonl"))
    master = list(input_root.rglob("dataset_master.csv"))
    if patients and master:
        return patients[0], master[0]
    
    # If not found directly, run cloud_prepare on annotations
    from scripts.cloud_prepare import prepare
    derived = Path("/kaggle/working/derived") if Path("/kaggle/working").exists() else Path("output/derived")
    prepare(input_root, derived)
    return derived / "dataset/dataset_patients.jsonl", derived / "dataset/dataset_master.csv"


def train_and_eval(config_path, input_root, output_dir, smoke=False):
    import torch
    from transformers import (AutoTokenizer, BitsAndBytesConfig,
                              TrainingArguments, Trainer,
                              Qwen3VLForConditionalGeneration, set_seed)
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    set_seed(int(config.get("seed", 42)))

    patients_path, master_path = find_dataset_files(input_root)
    fold = int(config.get("fold", 1))
    train_rows, train_counts = load_samples(patients_path, master_path, fold, "train")
    val_rows, val_counts = load_samples(patients_path, master_path, fold, "val")
    test_rows, test_counts = load_samples(patients_path, master_path, fold, "test")

    if smoke:
        train_rows, val_rows, test_rows = train_rows[:2], val_rows[:2], test_rows[:2]

    print(f"Loaded samples: {len(train_rows)} train, {len(val_rows)} val, {len(test_rows)} test")

    model_cfg = config["model"]
    tokenizer = AutoTokenizer.from_pretrained(model_cfg["name_or_path"],
                                              revision=model_cfg.get("revision"),
                                              trust_remote_code=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    collator = Qwen3TextCollator(tokenizer, max_length=int(config.get("max_sequence_length", 4096)))

    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                               bnb_4bit_compute_dtype=torch.float16)

    print("Loading Qwen3-VL base model in 4-bit...")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_cfg["name_or_path"],
        revision=model_cfg.get("revision"),
        torch_dtype=torch.float16,
        quantization_config=quant,
        device_map={"": 0},
        attn_implementation="sdpa",
        trust_remote_code=False
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)

    lora_cfg = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    training_dir = output / "training"
    training_dir.mkdir(parents=True, exist_ok=True)

    save_eval_interval = 2 if smoke else int(config.get("save_steps", 34))
    args = TrainingArguments(
        output_dir=str(training_dir),
        num_train_epochs=2 if smoke else float(config.get("num_train_epochs", 8)),
        max_steps=2 if smoke else -1,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=1 if smoke else int(config.get("gradient_accumulation_steps", 4)),
        learning_rate=float(config.get("learning_rate", 1.5e-4)),
        lr_scheduler_type=config.get("lr_scheduler_type", "cosine"),
        warmup_ratio=float(config.get("warmup_ratio", 0.05)),
        eval_strategy="steps",
        save_strategy="steps",
        eval_steps=save_eval_interval,
        save_steps=save_eval_interval,
        save_total_limit=2,
        logging_steps=1 if smoke else 5,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        remove_unused_columns=False,
        report_to="none",
        seed=int(config.get("seed", 42))
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_rows,
        eval_dataset=val_rows,
        data_collator=collator,
        processing_class=tokenizer
    )

    print("Starting QLoRA training...")
    train_start = time.monotonic()
    train_res = trainer.train()
    train_duration = time.monotonic() - train_start
    print(f"Training completed in {train_duration / 60:.2f} minutes")

    adapter_dest = output / "best_adapter"
    model.save_pretrained(adapter_dest)
    tokenizer.save_pretrained(adapter_dest)
    write_json(output / "train_result.json", {
        "train_runtime_seconds": train_duration,
        "global_step": trainer.state.global_step,
        "best_metric": trainer.state.best_metric,
        "best_checkpoint": trainer.state.best_model_checkpoint
    })

    # --- INFERENCE ON TEST SET ---
    print("\n--- Running Evaluation on Test Set ---")
    model.eval()
    model.config.use_cache = True
    tokenizer.padding_side = "left"

    gen_cfg = config.get("generation", {})
    max_new_tokens = int(gen_cfg.get("max_new_tokens", 1024))
    rep_penalty = float(gen_cfg.get("repetition_penalty", 1.15))
    no_repeat_ngram = int(gen_cfg.get("no_repeat_ngram_size", 5))

    records, predictions = [], []
    eval_start = time.monotonic()
    pred_path = output / "predictions.jsonl"

    with pred_path.open("w", encoding="utf-8", newline="\n") as writer:
        for idx, sample in enumerate(test_rows, 1):
            chat = tokenizer.apply_chat_template([{"role": "user", "content": sample["prompt"]}],
                                                 tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(chat, return_tensors="pt", add_special_tokens=False)
            input_tokens = inputs["input_ids"].shape[-1]
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            t0 = time.monotonic()
            with torch.inference_mode():
                out = model.generate(
                    **inputs,
                    do_sample=False,
                    max_new_tokens=max_new_tokens,
                    repetition_penalty=rep_penalty,
                    no_repeat_ngram_size=no_repeat_ngram,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
            sec = time.monotonic() - t0
            ans = out[0, input_tokens:]
            raw = tokenizer.decode(ans, skip_special_tokens=True)
            status, parsed = parse_report(raw)

            row = {
                "case_id": sample["case_id"],
                "status": status,
                "parsed": parsed,
                "raw_output": raw,
                "new_tokens": len(ans),
                "hit_token_limit": len(ans) >= max_new_tokens,
                "seconds": sec
            }
            writer.write(json.dumps(row, ensure_ascii=False) + "\n")
            writer.flush()
            predictions.append(row)

            reference = strict_loads(sample["completion"])
            records.append({
                "case_id": sample["case_id"],
                "status": status,
                "reference": reference,
                "prediction": parsed,
                "raw_output": raw,
                "hit_token_limit": row["hit_token_limit"],
                "seconds": sec
            })
            print(f"Test [{idx}/{len(test_rows)}] - status: {status}, tokens: {len(ans)}, time: {sec:.1f}s")

    eval_duration = time.monotonic() - eval_start
    print(f"Inference completed in {eval_duration / 60:.2f} minutes")

    # Compute metrics
    metrics = {
        "model": "Qwen3-VL-4B-Instruct-Finetuned-8epochs",
        "patients": len(records),
        "format_valid": sum(r["status"] == "ok" for r in records),
        "format_valid_rate": sum(r["status"] == "ok" for r in records) / len(records),
        "format_invalid_json": sum(r["status"] == "invalid_json" for r in records),
        "hit_token_limit": sum(r["hit_token_limit"] for r in records),
        "mean_generation_seconds": sum(r["seconds"] for r in records) / len(records),
        "total_test_seconds": eval_duration,
        "train_minutes": train_duration / 60
    }
    nlp_scores = compute_metrics(records, model_name="bert-base-multilingual-cased")
    metrics.update(nlp_scores)
    write_json(output / "metrics.json", metrics)

    # Export human review CSV
    review_path = output / "human_review_qwen3_finetuned.csv"
    with review_path.open("w", encoding="utf-8-sig", newline="") as h:
        fields = ["case_id", "format_status", "reference_findings", "reference_impression",
                  "prediction_findings", "prediction_impression", "unsupported_claim",
                  "wrong_level", "omission", "contradiction", "reviewer", "notes"]
        w = csv.DictWriter(h, fieldnames=fields)
        w.writeheader()
        for r in records:
            p = r["prediction"] or {}
            w.writerow({
                "case_id": r["case_id"],
                "format_status": r["status"],
                "reference_findings": r["reference"]["findings"],
                "reference_impression": r["reference"]["impression"],
                "prediction_findings": p.get("findings", ""),
                "prediction_impression": p.get("impression", "")
            })

    print("\n" + "=" * 50)
    print("FINAL EVALUATION METRICS:")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print("=" * 50)
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/qwen3_finetune.yaml")
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    train_and_eval(args.config, args.input_root, args.output, args.smoke)
