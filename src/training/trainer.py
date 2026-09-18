"""Fail on non-finite loss instead of reporting a misleading filtered training log."""
import torch
from transformers import Trainer


class FiniteLossTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        result = super().compute_loss(model, inputs, return_outputs=return_outputs, **kwargs)
        loss = result[0] if return_outputs else result
        if not torch.isfinite(loss).all():
            raise FloatingPointError("Non-finite model loss; run stopped before accepting this step")
        return result
