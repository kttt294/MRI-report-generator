from pathlib import Path
import pytest
import torch
from src.data.vlm_collator import VLMDataCollator
from src.data.v2_dataset import TextCollator
from src.training.checkpoints import mark_checkpoint, verify_checkpoint, prepare_resume, CompleteCheckpointCallback
from src.io_utils import write_json


class FakeProcessor:
    def __init__(self, left=False, broken=False): self.left, self.broken = left, broken
    def apply_chat_template(self, messages, **kwargs): return "full" if len(messages) == 2 else "prompt"
    def __call__(self, text, **kwargs):
        full = text[0] == "full"
        sequences = [[1, 2, 3, 9] if full else [1, 2], [1, 2, 4, 5, 9] if full else [1, 2]]
        if self.broken and not full: sequences = [[8, 8], [8, 8]]
        size = max(map(len, sequences))
        ids, mask = [], []
        for s in sequences:
            pad = [9] * (size-len(s))
            ids.append(pad+s if self.left else s+pad)
            mask.append([0]*len(pad)+[1]*len(s) if self.left else [1]*len(s)+[0]*len(pad))
        if kwargs.get("return_tensors") == "pt":
            return {"input_ids": torch.tensor(ids), "attention_mask": torch.tensor(mask)}
        return {"input_ids": sequences}


@pytest.mark.parametrize("left", [True, False])
def test_assistant_mask_preserves_eos_equal_pad(left):
    batch = [{"conversation": [{"role": "user"}, {"role": "assistant"}], "images": [None]}] * 2
    result = VLMDataCollator(FakeProcessor(left))(batch)
    assert result["labels"][result["labels"] != -100].tolist() == [3, 9, 4, 5, 9]
    assert (result["labels"][result["attention_mask"] == 0] == -100).all()
    with pytest.raises(ValueError): VLMDataCollator(FakeProcessor(left), max_length=4)(batch)
    with pytest.raises(ValueError): VLMDataCollator(FakeProcessor(left, True))(batch)


def test_text_mask():
    batch = [{"prompt": "x", "completion": "y"}] * 2
    result = TextCollator(FakeProcessor())(batch)
    assert result["labels"][result["labels"] != -100].tolist() == [3, 9, 4, 5, 9]


def test_checkpoint_rejects_partial_and_tamper(tmp_path):
    ckpt = tmp_path / "checkpoint-1"
    ckpt.mkdir()
    with pytest.raises(ValueError): mark_checkpoint(ckpt, {"test": 1})
    for name in ("trainer_state.json", "optimizer.pt", "scheduler.pt", "rng_state.pth"):
        (ckpt / name).write_text("{}")
    mark_checkpoint(ckpt, {"test": 1})
    verify_checkpoint(ckpt, {"test": 1})
    with pytest.raises(ValueError): verify_checkpoint(ckpt, {"test": 2})
    (ckpt / "optimizer.pt").write_text("changed")
    with pytest.raises(ValueError): verify_checkpoint(ckpt)


def test_time_budget_requests_complete_save(monkeypatch):
    from types import SimpleNamespace
    import src.training.checkpoints as checkpoints
    clock = [100.]
    monkeypatch.setattr(checkpoints.time, "monotonic", lambda: clock[0])
    callback = CompleteCheckpointCallback(None, {}, 1)
    control = SimpleNamespace(should_save=False, should_training_stop=False)
    callback.on_train_begin(None, None, control)
    clock[0] = 161.
    callback.on_step_end(None, None, control)
    assert control.should_save and control.should_training_stop
    with pytest.raises(ValueError): CompleteCheckpointCallback(None, {}, -1)


def test_resume_relocates_best_checkpoint_without_changing_source(tmp_path):
    from src.io_utils import file_hash, strict_loads
    for step in (1, 2):
        folder = tmp_path / "input" / f"checkpoint-{step}"
        folder.mkdir(parents=True)
        for name in ("optimizer.pt", "scheduler.pt", "rng_state.pth"): (folder / name).write_text("synthetic")
        write_json(folder / "trainer_state.json", {"best_model_checkpoint": "C:\\old\\run\\checkpoint-1"})
        mark_checkpoint(folder, {"test": 1})
    source = tmp_path / "input/checkpoint-2"
    digest = file_hash(source / "trainer_state.json")
    relocated = prepare_resume(source, tmp_path / "working/resume", {"test": 1})
    assert file_hash(source / "trainer_state.json") == digest
    state = strict_loads((relocated / "trainer_state.json").read_text())
    assert Path(state["best_model_checkpoint"]).is_dir()
    verify_checkpoint(relocated, {"test": 1})


def test_real_trainer_resume_and_adapter_reload(tmp_path):
    from transformers import GPT2Config, GPT2LMHeadModel, TrainingArguments, TrainerCallback, set_seed
    from src.training.trainer import FiniteLossTrainer as Trainer
    from peft import LoraConfig, get_peft_model, PeftModel
    torch.set_num_threads(1)
    cfg = GPT2Config(vocab_size=32, n_positions=16, n_embd=8, n_layer=1, n_head=1,
                     resid_pdrop=0, embd_pdrop=0, attn_pdrop=0)
    data = [{"input_ids": torch.tensor([1, 2, 3, 4]), "labels": torch.tensor([1, 2, 3, 4])}] * 8
    class Processor:
        def save_pretrained(self, path): write_json(Path(path) / "processor_test.json", {"synthetic": True})
    class Stop(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            if state.global_step == 2: control.should_save = True; control.should_training_stop = True
            return control
    def trainer(path, stop=False):
        set_seed(42)
        model = get_peft_model(GPT2LMHeadModel(cfg), LoraConfig(r=2, target_modules=["c_attn"], task_type="CAUSAL_LM"))
        args = TrainingArguments(output_dir=str(path), max_steps=4, per_device_train_batch_size=1,
            save_steps=1, logging_steps=1, use_cpu=True, report_to="none", disable_tqdm=True,
            remove_unused_columns=False, seed=42)
        return Trainer(model=model, args=args, train_dataset=data,
                       callbacks=[CompleteCheckpointCallback(Processor(), {"synthetic": 1})] + ([Stop()] if stop else []))
    first = trainer(tmp_path / "first", True)
    first.train()
    assert first.state.global_step == 2
    source = tmp_path / "first/checkpoint-2"
    verify_checkpoint(source, {"synthetic": 1})
    relocated = prepare_resume(source, tmp_path / "relocated", {"synthetic": 1})
    resumed = trainer(tmp_path / "resumed")
    resumed.train(resume_from_checkpoint=str(relocated))
    assert resumed.state.global_step == 4
    full = trainer(tmp_path / "full")
    full.train()
    for name, value in resumed.model.named_parameters():
        if "lora_" in name: torch.testing.assert_close(value, dict(full.model.named_parameters())[name], atol=1e-6, rtol=1e-5)
    saved = tmp_path / "adapter"
    resumed.model.save_pretrained(saved)
    set_seed(42)
    loaded = PeftModel.from_pretrained(GPT2LMHeadModel(cfg), saved).eval()
    with torch.inference_mode(): assert torch.isfinite(loaded(**data[0], return_dict=True).loss)
