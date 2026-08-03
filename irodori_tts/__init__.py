"""Irodori-TTS package: text-conditioned RF diffusion over DACVAE latents."""

from .config import ModelConfig, TrainConfig
from .dataset import ManifestIndex, build_manifest_index
from .inference_runtime import ReferenceCondition, load_checkpoint_for_inference
from .lora import LORA_TARGET_PRESETS
from .model import TextToLatentRFDiT
from .tokenizer import PretrainedTextTokenizer


__all__ = [
    "LORA_TARGET_PRESETS",
    "ManifestIndex",
    "ModelConfig",
    "PretrainedTextTokenizer",
    "ReferenceCondition",
    "TextToLatentRFDiT",
    "TrainConfig",
    "build_manifest_index",
    "load_checkpoint_for_inference",
]
