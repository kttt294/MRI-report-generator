"""Inference never includes the target response or grading in model messages."""
import argparse
from pathlib import Path
import yaml
from src.io_utils import write_jsonl, content_hash
from src.train_v1 import make_dataset


def infer(config, adapter, output, split="test", limit=None):
    import torch
    from src.models.v1_vlm import load_vlm_and_processor
    dataset = make_dataset(config, split)
    model, processor = load_vlm_and_processor(config, training=False, adapter_path=adapter)
    model.eval()
    results = []
    generation = config.get("generation", {"max_new_tokens": 768})
    for i in range(min(limit or len(dataset), len(dataset))):
        sample = dataset[i]
        messages = sample["conversation"][:1]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=sample["images"], truncation=False, return_tensors="pt").to(model.device)
        if inputs.input_ids.shape[1] + generation["max_new_tokens"] > config["data"].get("max_sequence_length", 4096):
            raise ValueError("Inference context budget exceeded")
        with torch.inference_mode():
            ids = model.generate(**inputs, max_new_tokens=generation["max_new_tokens"], do_sample=False)
        answer_ids = ids[0, inputs.input_ids.shape[1]:]
        eos = model.generation_config.eos_token_id
        eos = eos if isinstance(eos, list) else [eos]
        answer = processor.decode(answer_ids, skip_special_tokens=True)
        results.append({"case_id": sample["patient_id"], "split": split, "prediction": answer,
            "truncated": not len(answer_ids) or answer_ids[-1].item() not in eos,
            "prompt_sha256": content_hash(text), "model": config["model"], "generation": generation,
            "adapter": str(adapter), "has_both_sections": "[MÔ TẢ]" in answer and "[KẾT LUẬN]" in answer})
    write_jsonl(output, results)
    return results


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--adapter", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--split", choices=["train", "val", "test"], default="test")
    p.add_argument("--limit", type=int)
    a = p.parse_args()
    infer(yaml.safe_load(Path(a.config).read_text(encoding="utf-8")), a.adapter, a.output, a.split, a.limit)
