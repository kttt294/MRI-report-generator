"""Qwen2.5-VL QLoRA loader with explicit language-decoder target selection."""
import re
from src.models.runtime import choose_dtype


def load_vlm_and_processor(config, training=True, adapter_path=None):
    import torch
    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
    cfg = config["model"]
    if cfg.get("require_cuda", True) and not torch.cuda.is_available():
        raise RuntimeError("V1 model execution requires CUDA; use CPU data preflight first")
    dtype = choose_dtype(cfg.get("torch_dtype", "auto"))
    model_id, revision = cfg["name_or_path"], cfg.get("revision")
    kwargs = dict(revision=revision, trust_remote_code=False, local_files_only=cfg.get("local_files_only", False))
    processor = AutoProcessor.from_pretrained(adapter_path or model_id, **({} if adapter_path else kwargs),
        min_pixels=cfg.get("min_pixels", 256 * 28 * 28), max_pixels=cfg.get("max_pixels", 512 * 28 * 28))
    processor.tokenizer.padding_side = "right" if training else "left"
    model_kwargs = dict(kwargs, torch_dtype=dtype, attn_implementation="sdpa")
    if torch.cuda.is_available():
        model_kwargs["device_map"] = {"": torch.cuda.current_device()}
    quantized = cfg.get("use_4bit_quantization", True)
    if quantized:
        if not torch.cuda.is_available():
            raise RuntimeError("4-bit QLoRA requires CUDA")
        model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True,
            bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=dtype, bnb_4bit_use_double_quant=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_id, **model_kwargs)
    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path, is_trainable=training)
    elif training:
        if quantized:
            model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
        allowed = set(config["lora"]["target_modules"])
        targets = [name for name, _ in model.named_modules()
                   if re.search(r"(?:^|\.)layers\.\d+\.(?:self_attn|mlp)\.", name)
                   and not any(x in name for x in ("visual", "vision")) and name.split(".")[-1] in allowed]
        if not targets:
            raise ValueError("No language decoder LoRA modules matched; inspect this model revision")
        lora = config["lora"]
        model = get_peft_model(model, LoraConfig(r=lora["r"], lora_alpha=lora["lora_alpha"],
            lora_dropout=lora["lora_dropout"], target_modules=targets, bias="none", task_type="CAUSAL_LM"))
        model.print_trainable_parameters()
    model.config.use_cache = not training
    return model, processor
