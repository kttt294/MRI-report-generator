"""Shared dtype decisions for training and inference."""
def choose_dtype(name="auto"):
    import torch
    if name == "auto":
        return torch.float32 if not torch.cuda.is_available() else torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    choices = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    if name not in choices:
        raise ValueError("dtype must be auto/float32/float16/bfloat16")
    if name == "bfloat16" and torch.cuda.is_available() and not torch.cuda.is_bf16_supported():
        raise ValueError("Selected GPU does not support bfloat16; use auto or float16")
    if name == "float16" and not torch.cuda.is_available():
        raise ValueError("CPU execution requires float32")
    return choices[name]
