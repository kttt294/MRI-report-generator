"""V2-2 pilot: QLoRA from grading facts to doctor-authored Vietnamese reports."""
import argparse
from pathlib import Path
import yaml

from src.data.v2_2_dataset import load_samples
from src.data.v2_dataset import TextCollator
from src.io_utils import file_hash, write_json, environment_receipt


def train(config, patients, master, output, smoke=False, max_runtime_minutes=180):
    import torch
    from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
                              TrainingArguments, EarlyStoppingCallback, set_seed)
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from src.models.runtime import choose_dtype
    from src.training.trainer import FiniteLossTrainer
    from src.training.checkpoints import CompleteCheckpointCallback
    from src.train_v1 import code_fingerprint

    fold = int(config["fold"])
    train_rows, train_counts = load_samples(patients, master, fold, "train")
    val_rows, val_counts = load_samples(patients, master, fold, "val")
    if not torch.cuda.is_available():
        raise RuntimeError("V2-2 QLoRA requires a CUDA GPU")
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use a fresh empty output directory")
    if smoke:
        train_rows, val_rows = train_rows[:2], val_rows[:2]
    set_seed(int(config.get("seed", 42)))
    model_cfg = config["model"]
    tokenizer = AutoTokenizer.from_pretrained(model_cfg["name_or_path"],
        revision=model_cfg.get("revision"), trust_remote_code=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    collator = TextCollator(tokenizer, int(config["max_sequence_length"]))
    lengths = []
    for row in train_rows + val_rows:
        lengths.append(int(collator([row])["input_ids"].shape[-1]))
    dtype = choose_dtype("auto")
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                               bnb_4bit_compute_dtype=dtype)
    model = AutoModelForCausalLM.from_pretrained(model_cfg["name_or_path"],
        revision=model_cfg.get("revision"), torch_dtype=dtype,
        quantization_config=quant, device_map={"": 0}, trust_remote_code=False)
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM"))
    model.config.use_cache = False
    identity = {"experiment": "v2-2/doctor-original/1", "code": code_fingerprint(),
        "patients_sha256": file_hash(patients), "master_sha256": file_hash(master),
        "config": config, "smoke": smoke,
        "base_revision": getattr(model.config, "_commit_hash", None)}
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "run_manifest.json", {"identity": identity,
        "environment": environment_receipt(), "train_cases": len(train_rows),
        "val_cases": len(val_rows), "train_missing": train_counts, "val_missing": val_counts,
        "max_sequence_tokens": max(lengths),
        "scope_warning": "Reports may describe findings absent from the eight grading fields; clinical audit required."})
    interval = 2 if smoke else int(config.get("save_steps", 10))
    args = TrainingArguments(output_dir=str(output),
        num_train_epochs=float(config.get("num_train_epochs", 2)), max_steps=2 if smoke else -1,
        per_device_train_batch_size=1, per_device_eval_batch_size=1,
        gradient_accumulation_steps=1 if smoke else int(config.get("gradient_accumulation_steps", 8)),
        learning_rate=float(config.get("learning_rate", 5e-5)),
        eval_strategy="steps", save_strategy="steps", eval_steps=interval, save_steps=interval,
        save_total_limit=2, logging_steps=1 if smoke else 2,
        load_best_model_at_end=True, metric_for_best_model="eval_loss", greater_is_better=False,
        prediction_loss_only=True, eval_accumulation_steps=1,
        fp16=dtype == torch.float16, bf16=dtype == torch.bfloat16,
        gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
        remove_unused_columns=False, report_to="none", seed=int(config.get("seed", 42)))
    callbacks = [CompleteCheckpointCallback(tokenizer, identity, max_runtime_minutes)]
    if not smoke:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=2))
    trainer = FiniteLossTrainer(model=model, args=args, train_dataset=train_rows,
        eval_dataset=val_rows, data_collator=collator, processing_class=tokenizer,
        callbacks=callbacks)
    trainer.train()
    # load_best_model_at_end restores the selected adapter before this save.
    destination = output / ("smoke_adapter" if smoke else "best_adapter")
    model.save_pretrained(destination)
    tokenizer.save_pretrained(destination)
    write_json(output / "train_result.json", {"smoke": smoke,
        "global_step": trainer.state.global_step, "max_steps": trainer.state.max_steps,
        "best_metric": trainer.state.best_metric,
        "best_checkpoint": trainer.state.best_model_checkpoint,
        "adapter_path": str(destination),
        "completed_schedule": trainer.state.global_step >= trainer.state.max_steps})
    return destination


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "patients", "master", "output"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--max-runtime-minutes", type=float, default=180)
    a = p.parse_args()
    print(train(yaml.safe_load(Path(a.config).read_text(encoding="utf-8")), a.patients,
                a.master, a.output, a.smoke, a.max_runtime_minutes))
