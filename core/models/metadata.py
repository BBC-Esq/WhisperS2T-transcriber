from __future__ import annotations

from typing import Dict, List, Set

from config.constants import (
    DISTIL_MODELS,
    MODEL_NAMES,
    MODEL_PRECISIONS,
    WHISPER_MODELS,
)


class ModelMetadata:
    """Wraps the WHISPER_MODELS catalog with helpers used by the GUI and server."""

    @classmethod
    def get_all_model_names(cls) -> List[str]:
        return list(MODEL_NAMES)

    @classmethod
    def get_precisions(cls, model_name: str) -> List[str]:
        return list(MODEL_PRECISIONS.get(model_name, []))

    @classmethod
    def supports_translation(cls, model_name: str) -> bool:
        # English-only Whisper checkpoints end in ".en"; all Distil variants
        # are distilled from English models and also don't translate reliably.
        if model_name.endswith(".en"):
            return False
        if model_name in DISTIL_MODELS:
            return False
        return True

    @classmethod
    def is_english_only(cls, model_name: str) -> bool:
        return model_name.endswith(".en")

    @classmethod
    def get_quantization_options(
        cls,
        model_name: str,
        device: str,
        supported_quantizations: Dict[str, List[str]],
    ) -> List[str]:
        available_for_model: Set[str] = set(MODEL_PRECISIONS.get(model_name, []))
        hw_supported: Set[str] = set(supported_quantizations.get(device, []))

        options = [p for p in MODEL_PRECISIONS.get(model_name, []) if p in hw_supported and p in available_for_model]

        if device == "cpu":
            options = [p for p in options if p not in ("float16", "bfloat16")]

        if not options and device == "cpu":
            options = ["float32"]

        return options

    @classmethod
    def resolve_model_key(cls, model_name: str, precision: str) -> str:
        return f"{model_name} - {precision}"

    @classmethod
    def get_model_info(cls, model_name: str, precision: str) -> dict | None:
        key = cls.resolve_model_key(model_name, precision)
        return WHISPER_MODELS.get(key)

    @classmethod
    def get_all_models_with_precisions(cls) -> Dict[str, dict]:
        return dict(WHISPER_MODELS)

    @classmethod
    def get_description(cls, model_name: str) -> str:
        if model_name in DISTIL_MODELS:
            return "Distilled Whisper variant: faster, English-only."
        if model_name.endswith(".en"):
            return "English-only Whisper checkpoint."
        if "turbo" in model_name:
            return "Whisper large-v3-turbo: fast multilingual with translation."
        return "Multilingual Whisper checkpoint."
