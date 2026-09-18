"""Train V1 using shared CLI/config; preflight always precedes model loading."""
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import yaml
from src.io_utils import file_hash, content_hash, write_json, environment_receipt
from src.data.v1_dataset import SpineVLMDataset
from src.data.vlm_collator import VLMDataCollator


def make_dataset(config, split, require_images=True):
    d = config["data"]
    return SpineVLMDataset(d["master_csv"], d["nifti_dir"], split, d.get("fold", 1), d.get("language", "vi"),
        image_cache_dir=d.get("image_cache_dir"), target_size=d.get("image_size", [384, 384]),
        offsets=d.get("slice_offsets", [0]), slice_selection=d.get("slice_selection", "middle"),
        require_both_sections=d.get("require_both_sections", True), require_images=require_images)


def code_fingerprint():
    root = Path(__file__).resolve().parents[1]
    return content_hash({str(p.relative_to(root)).replace(chr(92), "/"): file_hash(p)
                         for folder in ("src", "scripts") for p in sorted((root / folder).rglob("*.py"))})


def train(config, resume_from=None, max_runtime_minutes=None, smoke=False):
    from src.models.v1_vlm import load_vlm_and_processor
    from src.models.runtime import choose_dtype
    from src.training.checkpoints import CompleteCheckpointCallback, prepare_resume
    from transformers import TrainingArguments, set_seed
    from src.training.trainer import FiniteLossTrainer as Trainer
    import torch
    cfg = config["training"]
    set_seed(cfg.get("seed", 42))
    output = Path(cfg["output_dir"])
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use a new empty run output directory, including when resuming")
    train_set, val_set = make_dataset(config, "train"), make_dataset(config, "val")
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "data_manifest.json", [train_set.manifest(), val_set.manifest()])
    if smoke:
        train_set.samples = train_set.samples[:2]
        val_set.samples = val_set.samples[:2]
    model, processor = load_vlm_and_processor(config)
    collator = VLMDataCollator(processor, config["data"].get("max_sequence_length", 4096))
    lengths = []
    # Verify every eligible sequence before training; this also validates all images.
    for dataset in (train_set, val_set):
        for i in range(len(dataset)):
            batch = collator([dataset[i]])
            lengths.append({"tokens": int(batch["attention_mask"].sum()), "target_tokens": int((batch["labels"] != -100).sum())})
    write_json(output / "token_preflight.json", lengths)
    stable_training = {k: v for k, v in cfg.items() if k not in {"output_dir", "logging_steps", "save_total_limit"}}
    identity = {"code": code_fingerprint(), "master": file_hash(config["data"]["master_csv"]),
                "model": config["model"], "resolved_revision": getattr(model.config, "_commit_hash", None),
                "training": stable_training, "lora": config["lora"], "smoke": smoke,
                "data": {k: v for k, v in config["data"].items() if k not in {"master_csv", "nifti_dir", "image_cache_dir"}}}
    if resume_from:
        resume_from = prepare_resume(resume_from, output / "resume_input", identity)
    write_json(output / "run_manifest.json", {"identity": identity, "config": config,
        "environment": environment_receipt(), "resumed_from": str(resume_from) if resume_from else None})
    dtype = choose_dtype(config["model"].get("torch_dtype", "auto"))
    interval = 2 if smoke else cfg.get("save_steps", 25)
    arguments = TrainingArguments(output_dir=str(output), num_train_epochs=cfg["num_train_epochs"],
        max_steps=2 if smoke else cfg.get("max_steps", -1),
        per_device_train_batch_size=cfg.get("per_device_train_batch_size", 1), per_device_eval_batch_size=1,
        gradient_accumulation_steps=1 if smoke else cfg.get("gradient_accumulation_steps", 8),
        learning_rate=float(cfg["learning_rate"]), warmup_ratio=cfg.get("warmup_ratio", .05),
        lr_scheduler_type=cfg.get("lr_scheduler_type", "cosine"),
        logging_steps=1 if smoke else cfg.get("logging_steps", 5), save_steps=interval, eval_steps=interval,
        eval_strategy="steps", save_strategy="steps", save_total_limit=cfg.get("save_total_limit", 2),
        load_best_model_at_end=True, metric_for_best_model="eval_loss", greater_is_better=False,
        fp16=dtype == torch.float16, bf16=dtype == torch.bfloat16,
        gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="adamw_torch", remove_unused_columns=False, report_to="none", seed=cfg.get("seed", 42))
    trainer = Trainer(model=model, args=arguments, train_dataset=train_set, eval_dataset=val_set,
        data_collator=collator, processing_class=processor,
        callbacks=[CompleteCheckpointCallback(processor, identity, max_runtime_minutes)])
    result = trainer.train(resume_from_checkpoint=str(resume_from) if resume_from else None)
    complete = trainer.state.global_step >= trainer.state.max_steps
    destination = output / ("final_adapter" if complete else "latest_adapter")
    trainer.model.save_pretrained(destination)
    processor.save_pretrained(destination)
    if smoke:
        # Reload the saved adapter into the same base (avoids a second GPU copy).
        trainer.model.load_adapter(str(destination), adapter_name="smoke_reload", is_trainable=False)
        trainer.model.set_adapter("smoke_reload")
        trainer.model.eval()
        with torch.inference_mode():
            batch = {k: v.to(trainer.model.device) for k, v in collator([val_set[0]]).items()}
            loss = trainer.model(**batch).loss
        if not torch.isfinite(loss): raise RuntimeError("Reloaded adapter smoke loss is not finite")
        write_json(output / "smoke_reload.json", {"finite_loss": True, "loss": float(loss)})
    write_json(output / "train_result.json", {"completed": complete, "global_step": trainer.state.global_step,
        "planned_steps": trainer.state.max_steps, "best_model_checkpoint": trainer.state.best_model_checkpoint,
        "metrics": result.metrics, "adapter_path": str(destination), "smoke": smoke})
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/v1_config.yaml")
    parser.add_argument("--master_csv")
    parser.add_argument("--nifti_dir")
    parser.add_argument("--output_dir")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--resume-from")
    parser.add_argument("--max-runtime-minutes", type=float)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    for key in ("master_csv", "nifti_dir"):
        if getattr(args, key): config["data"][key] = getattr(args, key)
    if args.output_dir: config["training"]["output_dir"] = args.output_dir
    if args.epochs: config["training"]["num_train_epochs"] = args.epochs
    train(config, args.resume_from, args.max_runtime_minutes, args.smoke)


if __name__ == "__main__":
    main()
