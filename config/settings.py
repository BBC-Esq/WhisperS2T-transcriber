from dataclasses import dataclass
from typing import List

@dataclass
class TranscriptionSettings:
    model_key: str
    device: str
    beam_size: int
    batch_size: int
    output_format: str
    task_mode: str
    language: str
    recursive: bool
    selected_extensions: List[str]

    def validate(self) -> List[str]:
        warnings = []

        if self.device.lower() == "cpu" and self.batch_size > 8:
            warnings.append(
                "CPU batch size > 8 may reduce performance. "
                "Consider reducing batch size for better results."
            )

        return warnings
