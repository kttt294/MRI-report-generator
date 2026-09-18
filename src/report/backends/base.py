from dataclasses import dataclass
from typing import Protocol


@dataclass
class Completion:
    text: str
    truncated: bool = False
    input_tokens: int | None = None
    output_tokens: int | None = None


class Backend(Protocol):
    model_id: str

    def generate(self, prompt: str) -> Completion: ...
