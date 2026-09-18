"""Assistant-only loss with checked token-prefix boundaries, never truncation."""
import torch


class VLMDataCollator:
    def __init__(self, processor, max_length=4096):
        self.processor, self.max_length = processor, max_length

    def __call__(self, batch):
        conversations = [x["conversation"] for x in batch]
        if any(len(c) != 2 or c[-1]["role"] != "assistant" for c in conversations):
            raise ValueError("Expected one user turn followed by one assistant target")
        images = [image for item in batch for image in (item["images"] if "images" in item else [item["image"]])]
        full = [self.processor.apply_chat_template(c, tokenize=False, add_generation_prompt=False) for c in conversations]
        prompt = [self.processor.apply_chat_template(c[:-1], tokenize=False, add_generation_prompt=True) for c in conversations]
        inputs = self.processor(text=full, images=images, padding=True, truncation=False, return_tensors="pt")
        prefixes = self.processor(text=prompt, images=images, padding=True, truncation=False, return_tensors="pt")
        labels = torch.full_like(inputs["input_ids"], -100)
        for i in range(len(batch)):
            positions = inputs["attention_mask"][i].nonzero(as_tuple=True)[0]
            prefix = prefixes["input_ids"][i][prefixes["attention_mask"][i].bool()]
            sequence = inputs["input_ids"][i][positions]
            if self.max_length and len(sequence) > self.max_length:
                raise ValueError(f"Sequence of {len(sequence)} tokens exceeds {self.max_length}; no truncation performed")
            if len(sequence) <= len(prefix) or not torch.equal(sequence[:len(prefix)], prefix):
                raise ValueError("Chat template does not provide an exact assistant token boundary")
            labels[i, positions[len(prefix):]] = sequence[len(prefix):]
        inputs["labels"] = labels
        return inputs
