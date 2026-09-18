"""Local text LLM. Prompt-only generation; catalog validation is downstream."""
from src.report.backends.base import Completion


class HFLocalBackend:
    def __init__(self, model_path, revision=None, max_new_tokens=4096,
                 max_input_tokens=8192, max_time_seconds=120, use_4bit=False,
                 local_files_only=True, adapter_path=None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from src.models.runtime import choose_dtype

        self.model_id = f"{model_path}@{revision or 'local'}"
        self.max_new_tokens, self.max_input_tokens = max_new_tokens, max_input_tokens
        self.max_time_seconds = max_time_seconds
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, revision=revision,
                                                       local_files_only=local_files_only, trust_remote_code=False)
        kwargs = {"revision": revision, "local_files_only": local_files_only, "trust_remote_code": False,
                  "torch_dtype": choose_dtype("auto")}
        if torch.cuda.is_available():
            kwargs["device_map"] = {"": torch.cuda.current_device()}
        if use_4bit:
            if not torch.cuda.is_available():
                raise RuntimeError("4-bit backend requires CUDA")
            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True,
                bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=choose_dtype("auto"))
        self.model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs).eval()
        if adapter_path:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, adapter_path).eval()
        self.model_id = f"{model_path}@{getattr(self.model.config, '_commit_hash', None) or revision or 'local'}"
        if adapter_path:
            from pathlib import Path
            from src.io_utils import file_hash, content_hash
            directory = Path(adapter_path)
            if not directory.is_dir(): raise ValueError("adapter_path must be a local saved adapter directory")
            digest = content_hash({str(p.relative_to(directory)): file_hash(p) for p in sorted(directory.rglob("*")) if p.is_file()})
            self.model_id += f"+adapter:{digest}"

    def generate(self, prompt, schema=None):
        import torch
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(text, return_tensors="pt", add_special_tokens=False, truncation=False)
        length = inputs.input_ids.shape[-1]
        capacity = getattr(self.model.config, "max_position_embeddings", 32768)
        if length > self.max_input_tokens or length + self.max_new_tokens > capacity:
            raise ValueError("Context budget exceeded; input was not truncated")
        inputs = inputs.to(self.model.device)
        constraints = {}
        if schema:
            from lmformatenforcer import JsonSchemaParser
            from lmformatenforcer.integrations.transformers import build_transformers_prefix_allowed_tokens_fn
            constraints["prefix_allowed_tokens_fn"] = build_transformers_prefix_allowed_tokens_fn(self.tokenizer, JsonSchemaParser(schema))
        with torch.inference_mode():
            ids = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False,
                max_time=self.max_time_seconds, pad_token_id=self.tokenizer.eos_token_id, **constraints)
        answer = ids[0, length:]
        eos = self.model.generation_config.eos_token_id
        eos_ids = eos if isinstance(eos, list) else [eos]
        ended = len(answer) > 0 and answer[-1].item() in eos_ids
        return Completion(self.tokenizer.decode(answer, skip_special_tokens=True), not ended, length, len(answer))
