from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class TranscriptionSettings:
    """Settings passed to the server worker on each /transcribe call."""
    model_key: str
    device: str
    beam_size: int
    batch_size: int
    language: str
    task_mode: str
    output_format: str
    include_timestamps: bool = False
    recursive: bool = False
    selected_extensions: List[str] = field(default_factory=list)
