"""Optional text-only QLoRA, gated by human-reviewed scoped targets."""
import argparse
from pathlib import Path
import yaml
from src.data.v2_dataset import load_reviewed_targets, TextCollator
from src.io_utils import file_hash, write_json, environment_receipt


def train_text(config, resume_from=None, smoke=False, max_runtime_minutes=60):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments, set_seed
    from src.training.trainer import FiniteLossTrainer as Trainer
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from src.models.runtime import choose_dtype
    from src.training.checkpoints import CompleteCheckpointCallback, prepare_resume
    from src.train_v1 import code_fingerprint
    # Validate every row and split before allocating model/GPU memory.
    train = load_reviewed_targets(config["targets"], config["fold"], "train")
    val = load_reviewed_targets(config["targets"], config["fold"], "val")
    if not torch.cuda.is_available(): raise RuntimeError("V2 QLoRA training requires CUDA")
    output = Path(config["output_dir"])
    if output.exists() and any(output.iterdir()): raise ValueError("Use a new empty output directory")
    set_seed(config.get("seed", 42))
    cfg = config["model"]
    dtype = choose_dtype(cfg.get("torch_dtype", "auto"))
    tokenizer = AutoTokenizer.from_pretrained(cfg["name_or_path"], revision=cfg.get("revision"), trust_remote_code=False)
    if tokenizer.pad_token_id is None: tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    collator = TextCollator(tokenizer, config.get("max_sequence_length", 8192))
    if smoke: train, val = train[:2], val[:2]
    for row in train + val: collator([row])
    quantization = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=dtype) if cfg.get("use_4bit_quantization", True) else None
    model = AutoModelForCausalLM.from_pretrained(cfg["name_or_path"], revision=cfg.get("revision"),
        torch_dtype=dtype, quantization_config=quantization, device_map={"": 0}, trust_remote_code=False)
    if quantization: model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], task_type="CAUSAL_LM"))
    model.config.use_cache = False
    identity = {"code": code_fingerprint(), "targets": file_hash(config["targets"]),
                "config": {k: v for k, v in config.items() if k not in {"targets", "output_dir"}},
                "revision": getattr(model.config, "_commit_hash", None), "smoke": smoke}
    if resume_from: resume_from = str(prepare_resume(resume_from, output / "resume_input", identity))
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "run_manifest.json", {"identity": identity, "environment": environment_receipt(),
               "train_cases": len(train), "val_cases": len(val)})
    interval = 2 if smoke else config.get("save_steps", 25)
    args = TrainingArguments(output_dir=str(output), num_train_epochs=config.get("num_train_epochs", 3),
        max_steps=2 if smoke else -1, per_device_train_batch_size=1, per_device_eval_batch_size=1,
        gradient_accumulation_steps=1 if smoke else config.get("gradient_accumulation_steps", 8),
        learning_rate=float(config.get("learning_rate", 1e-4)), save_steps=interval, eval_steps=interval,
        eval_strategy="steps", save_strategy="steps", save_total_limit=2, logging_steps=1 if smoke else 5,
        load_best_model_at_end=True, metric_for_best_model="eval_loss", greater_is_better=False,
        fp16=dtype == torch.float16, bf16=dtype == torch.bfloat16, gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False}, remove_unused_columns=False,
        report_to="none", seed=config.get("seed", 42))
    trainer = Trainer(model=model, args=args, train_dataset=train, eval_dataset=val,
        data_collator=collator, processing_class=tokenizer,
        callbacks=[CompleteCheckpointCallback(tokenizer, identity, max_runtime_minutes)])
    trainer.train(resume_from_checkpoint=resume_from)
    complete = trainer.state.global_step >= trainer.state.max_steps
    destination = output / ("final_adapter" if complete else "latest_adapter")
    model.save_pretrained(destination)
    tokenizer.save_pretrained(destination)
    if smoke:
        model.load_adapter(str(destination), adapter_name="smoke_reload", is_trainable=False)
        model.set_adapter("smoke_reload")
        model.eval()
        with torch.inference_mode():
            loss = model(**{k: v.to(model.device) for k, v in collator([val[0]]).items()}).loss
        if not torch.isfinite(loss): raise RuntimeError("Reloaded adapter smoke loss is not finite")
        write_json(output / "smoke_reload.json", {"finite_loss": True, "loss": float(loss)})
    write_json(output / "train_result.json", {"completed": complete, "global_step": trainer.state.global_step,
               "adapter_path": str(destination), "smoke": smoke})


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--resume-from")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--max-runtime-minutes", type=float, default=60)
    a = p.parse_args()
    train_text(yaml.safe_load(Path(a.config).read_text(encoding="utf-8")), a.resume_from, a.smoke, a.max_runtime_minutes)
