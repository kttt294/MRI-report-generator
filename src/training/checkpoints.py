"""Complete checkpoint receipts and bounded runtime. No claim of remote durability."""
import time
import shutil
import math
from pathlib import Path
from transformers import TrainerCallback
from src.io_utils import file_hash, strict_loads, write_json

REQUIRED_STATE = ("trainer_state.json", "optimizer.pt", "scheduler.pt", "rng_state.pth")


def mark_checkpoint(path, identity):
    path = Path(path)
    if any(not (path / name).is_file() for name in REQUIRED_STATE):
        raise ValueError("Checkpoint lacks full trainer/optimizer/scheduler/RNG state")
    files = {str(p.relative_to(path)).replace("\\", "/"): file_hash(p)
             for p in sorted(path.rglob("*")) if p.is_file() and p.name != "COMPLETE.json"}
    write_json(path / "COMPLETE.json", {"identity": identity, "files": files})


def verify_checkpoint(path, identity=None):
    path = Path(path)
    receipt = strict_loads((path / "COMPLETE.json").read_text(encoding="utf-8"))
    if identity is not None and receipt["identity"] != identity:
        raise ValueError("Resume code/model/data/config identity differs from checkpoint")
    if not receipt["files"]:
        raise ValueError("Empty checkpoint receipt")
    if not set(REQUIRED_STATE) <= set(receipt["files"]):
        raise ValueError("Checkpoint receipt omits required training state")
    for name, expected in receipt["files"].items():
        target = (path / name).resolve()
        if not target.is_relative_to(path.resolve()) or not target.is_file() or file_hash(target) != expected:
            raise ValueError(f"Incomplete or changed checkpoint file: {name}")
    return receipt


def prepare_resume(source, destination, identity):
    """Verify read-only input, then relocate Trainer's absolute best-checkpoint path."""
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if destination.is_relative_to(source) or source.is_relative_to(destination):
        raise ValueError("Resume copy must be outside source checkpoint")
    verify_checkpoint(source, identity)
    state = strict_loads((source / "trainer_state.json").read_text(encoding="utf-8"))
    best = state.get("best_model_checkpoint")
    # A checkpoint saved on another OS can have either path separator.
    best_name = best.replace("\\", "/").rstrip("/").split("/")[-1] if best else None
    sources = [source]
    if best_name and best_name != source.name:
        best_source = source.parent / best_name
        verify_checkpoint(best_source, identity)
        sources.append(best_source)
    for item in sources:
        shutil.copytree(item, destination / item.name)
    copied = destination / source.name
    if best_name:
        state["best_model_checkpoint"] = str(destination / best_name)
        write_json(copied / "trainer_state.json", state)
        mark_checkpoint(copied, identity)
    return copied


class CompleteCheckpointCallback(TrainerCallback):
    def __init__(self, processor, identity, max_runtime_minutes=None):
        if max_runtime_minutes is not None and (not math.isfinite(max_runtime_minutes) or max_runtime_minutes <= 0):
            raise ValueError("Training time budget must be a positive finite number of minutes")
        self.processor, self.identity = processor, identity
        self.seconds = max_runtime_minutes * 60 if max_runtime_minutes else None
        self.started = None

    def on_train_begin(self, args, state, control, **kwargs):
        self.started = time.monotonic()

    def on_step_end(self, args, state, control, **kwargs):
        if self.seconds and time.monotonic() - self.started >= self.seconds:
            control.should_save = True
            control.should_training_stop = True
        return control

    def on_save(self, args, state, control, **kwargs):
        if state.is_world_process_zero:
            path = Path(args.output_dir) / f"checkpoint-{state.global_step}"
            self.processor.save_pretrained(path)
            mark_checkpoint(path, self.identity)
        return control
