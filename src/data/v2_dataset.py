"""Only explicitly reviewed, input-hash-bound targets are eligible for V2 SFT."""
from pathlib import Path
from src.io_utils import strict_loads, content_hash
from src.contracts.report_input import ReportRequest
from src.contracts.report_output import ReportDraft
from src.report.pipeline import make_prompt
from src.report.planner import make_plan
from src.report.validate import validate_draft


def load_reviewed_targets(path, fold, split):
    samples, seen = [], set()
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip(): continue
        row = strict_loads(line)
        pid = row["case_id"]
        if pid in seen: raise ValueError("Duplicate patient in reviewed targets")
        seen.add(pid)
        if row.get("review_status") != "accepted" or not row.get("reviewed_by") or row.get("target_kind") != "human_reviewed_scoped":
            raise ValueError("Only human-reviewed scoped targets may enter V2 training")
        request = ReportRequest.model_validate(row["request"])
        if request.case_id != pid or row["input_sha256"] != content_hash(request.model_dump()):
            raise ValueError("Reviewed target does not match the exact inference input")
        if set(row["folds"]) != {f"fold{i}" for i in range(1, 6)} or any(x not in {"train", "val", "test"} for x in row["folds"].values()):
            raise ValueError("Invalid reviewed target folds")
        target = ReportDraft.model_validate(row["target"])
        _, errors = validate_draft(target.model_dump_json(), make_plan(request))
        if errors:
            raise ValueError("Reviewed target violates the controlled R3 policy: " + "; ".join(errors))
        if row["folds"][f"fold{fold}"] == split:
            samples.append({"case_id": pid, "prompt": make_prompt(request, make_plan(request)),
                            "completion": target.model_dump_json()})
    if not samples: raise ValueError(f"No accepted {split} V2 targets")
    return samples


class TextCollator:
    def __init__(self, tokenizer, max_length=8192):
        self.tokenizer, self.max_length = tokenizer, max_length

    def __call__(self, batch):
        import torch
        conversations = [[{"role": "user", "content": r["prompt"]},
                          {"role": "assistant", "content": r["completion"]}] for r in batch]
        texts = [self.tokenizer.apply_chat_template(c, tokenize=False, add_generation_prompt=False) for c in conversations]
        prefixes = [self.tokenizer.apply_chat_template(c[:1], tokenize=False, add_generation_prompt=True) for c in conversations]
        inputs = self.tokenizer(texts, padding=True, truncation=False, add_special_tokens=False, return_tensors="pt")
        prompt_ids = self.tokenizer(prefixes, add_special_tokens=False)["input_ids"]
        labels = torch.full_like(inputs["input_ids"], -100)
        for i, prefix in enumerate(prompt_ids):
            positions = inputs["attention_mask"][i].nonzero(as_tuple=True)[0]
            sequence = inputs["input_ids"][i][positions]
            if len(sequence) > self.max_length or sequence[:len(prefix)].tolist() != prefix or len(sequence) <= len(prefix):
                raise ValueError("Invalid token boundary/context overflow in V2 target")
            labels[i, positions[len(prefix):]] = sequence[len(prefix):]
        inputs["labels"] = labels
        return inputs
