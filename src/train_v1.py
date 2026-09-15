"""
Script: src.train_v1
Mục đích: Huấn luyện mô hình V1 End-to-End Vision-Language Model
trên dữ liệu MRI cột sống bằng kỹ thuật LoRA.
"""

import os
import sys
import yaml
import argparse
from pathlib import Path
from typing import Dict, List, Any
import torch
from torch.utils.data import DataLoader
from transformers import (
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq
)

# Thêm thư mục gốc vào path để import
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.data.v1_dataset import SpineVLMDataset
from src.models.v1_vlm import load_vlm_and_processor


class VLMDataCollator:
    """
    Data Collator định dạng batch cho Vision-Language Model.
    Tự động gắn thẻ ảnh, tạo input_ids và che nhãn (mask) phần câu hỏi,
    chỉ tính đạo hàm loss trên phần câu trả lời của bác sĩ.
    """
    def __init__(self, processor, max_length: int = 512):
        self.processor = processor
        self.max_length = max_length

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        texts = []
        images = []
        
        for item in batch:
            conv = item["conversation"]
            # Áp dụng template hội thoại của mô hình
            text = self.processor.apply_chat_template(
                conv, tokenize=False, add_generation_prompt=False
            )
            texts.append(text)
            images.append(item["image"])

        # Mã hoá ảnh và chữ đồng thời
        batch_inputs = self.processor(
            text=texts,
            images=images,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        labels = batch_inputs["input_ids"].clone()
        # Thay thế token pad bằng -100 để không tính loss
        if self.processor.tokenizer.pad_token_id is not None:
            labels[labels == self.processor.tokenizer.pad_token_id] = -100
            
        # Che token ảnh để không tính loss trên điểm ảnh
        image_token_id = getattr(self.processor.tokenizer, "image_token_id", None)
        if image_token_id is not None:
            labels[labels == image_token_id] = -100

        batch_inputs["labels"] = labels
        return batch_inputs


def parse_args():
    parser = argparse.ArgumentParser(description="Huấn luyện V1 End-to-End VLM")
    parser.add_argument("--config", type=str, default="configs/v1_config.yaml", help="Đường dẫn file cấu hình YAML")
    parser.add_argument("--master_csv", type=str, default=None, help="Đè đường dẫn file CSV tổng hợp")
    parser.add_argument("--nifti_dir", type=str, default=None, help="Đè đường dẫn thư mục NIfTI")
    parser.add_argument("--output_dir", type=str, default=None, help="Đè thư mục lưu checkpoint")
    parser.add_argument("--epochs", type=int, default=None, help="Đè số epoch huấn luyện")
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Nạp file cấu hình
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
        
    # Ghi đè tham số dòng lệnh (nếu có)
    if args.master_csv: config["data"]["master_csv"] = args.master_csv
    if args.nifti_dir: config["data"]["nifti_dir"] = args.nifti_dir
    if args.output_dir: config["training"]["output_dir"] = args.output_dir
    if args.epochs: config["training"]["num_train_epochs"] = args.epochs

    train_cfg = config["training"]
    data_cfg = config["data"]
    
    output_dir = Path(train_cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("KHỞI ĐỘNG PIPELINE HUẤN LUYỆN V1 (END-TO-END VLM)")
    print(f"Mô hình: {config['model']['name_or_path']}")
    print(f"Ngôn ngữ: {data_cfg['language']} | Fold: {data_cfg['fold']}")
    print(f"Đầu ra: {output_dir.resolve()}")
    print("=" * 60)

    # 1. Nạp Model và Processor
    model, processor = load_vlm_and_processor(config)

    # 2. Chuẩn bị Dataset
    print("\n[Data] Đang chuẩn bị tập dữ liệu Train và Validation...")
    train_dataset = SpineVLMDataset(
        master_csv=data_cfg["master_csv"],
        nifti_dir=data_cfg["nifti_dir"],
        split="train",
        fold=data_cfg["fold"],
        language=data_cfg["language"],
        image_cache_dir=data_cfg.get("image_cache_dir"),
        target_size=tuple(data_cfg.get("image_size", [384, 384])),
        processor=processor
    )
    
    val_dataset = SpineVLMDataset(
        master_csv=data_cfg["master_csv"],
        nifti_dir=data_cfg["nifti_dir"],
        split="val",
        fold=data_cfg["fold"],
        language=data_cfg["language"],
        image_cache_dir=data_cfg.get("image_cache_dir"),
        target_size=tuple(data_cfg.get("image_size", [384, 384])),
        processor=processor
    )

    data_collator = VLMDataCollator(processor=processor)

    # 3. Cấu hình TrainingArguments
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=train_cfg["num_train_epochs"],
        per_device_train_batch_size=train_cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        learning_rate=float(train_cfg["learning_rate"]),
        warmup_ratio=train_cfg.get("warmup_ratio", 0.05),
        lr_scheduler_type=train_cfg.get("lr_scheduler_type", "cosine"),
        logging_steps=train_cfg.get("logging_steps", 5),
        save_steps=train_cfg.get("save_steps", 25),
        eval_steps=train_cfg.get("eval_steps", 25),
        eval_strategy=train_cfg.get("evaluation_strategy", "steps"),
        save_total_limit=train_cfg.get("save_total_limit", 2),
        fp16=train_cfg.get("fp16", False),
        bf16=train_cfg.get("bf16", True) if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else False,
        remove_unused_columns=False,
        report_to="none",
        seed=train_cfg.get("seed", 42)
    )

    # 4. Khởi tạo Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator
    )

    # 5. Bắt đầu huấn luyện
    print("\n[Training] Bắt đầu quá trình tối ưu hoá trọng số LoRA...")
    train_result = trainer.train()

    # 6. Lưu LoRA Adapter và Processor
    print(f"\n[Save] Đang lưu LoRA adapter cuối cùng tại {output_dir}...")
    trainer.model.save_pretrained(str(output_dir / "final_adapter"))
    processor.save_pretrained(str(output_dir / "final_adapter"))

    print("\n" + "=" * 60)
    print(f"HUẤN LUYỆN V1 THÀNH CÔNG! Checkpoint lưu tại: {output_dir / 'final_adapter'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
