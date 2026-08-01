from __future__ import annotations

import gc
import hashlib
import json
import math
import secrets
import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
from safetensors import safe_open
from safetensors.torch import load_file as load_safetensors_file

from .codec import DACVAECodec, patchify_latent, unpatchify_latent
from .config import ModelConfig
from .duration import build_duration_features
from .lora import checkpoint_state_uses_lora, is_lora_adapter_dir, load_lora_adapter
from .model import EncodedConditions, TextToLatentRFDiT
from .rf import (
    TrajectoryIntervention,
    TrajectoryObserver,
    VelocityFieldGuidance,
    sample_euler_rf_cfg,
)
from .speaker_inversion import (
    load_speaker_inversion_payload,
    speaker_inversion_batch_tensors,
)
from .text_normalization import normalize_text
from .tokenizer import PretrainedTextTokenizer
from .watermark import SilentCipherWatermarker
from .waveex import WaveExConfig


def _is_mps_available() -> bool:
    backends = getattr(torch, "backends", None)
    if backends is None or not hasattr(backends, "mps"):
        return False
    return bool(torch.backends.mps.is_available())


def _is_xpu_available() -> bool:
    try:
        return bool(torch.xpu.is_available())
    except AttributeError:
        return False


def resolve_runtime_device(device: str | torch.device) -> torch.device:
    resolved = torch.device(device)
    if resolved.type == "cpu":
        return resolved
    if resolved.type == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA device requested but torch.cuda.is_available() is False.")
        return resolved
    if resolved.type == "mps":
        if resolved.index is not None:
            raise ValueError("MPS device index is not supported. Use 'mps'.")
        if not _is_mps_available():
            raise ValueError("MPS device requested but torch.backends.mps.is_available() is False.")
        return torch.device("mps")
    if resolved.type == "xpu":
        if resolved.index is not None:
            raise ValueError("XPU device index is not supported. Use 'xpu'.")
        if not _is_xpu_available():
            raise ValueError("XPU device requested but torch.xpu.is_available() is False.")
        return torch.device("xpu")
    raise ValueError(
        f"Unsupported inference device={resolved!s}. Expected one of: cpu, cuda, mps, xpu."
    )


def list_available_runtime_devices() -> list[str]:
    devices: list[str] = []
    if torch.cuda.is_available():
        devices.append("cuda")
    if _is_mps_available():
        devices.append("mps")
    if _is_xpu_available():
        devices.append("xpu")
    devices.append("cpu")
    return devices


def default_runtime_device() -> str:
    return list_available_runtime_devices()[0]


class ContextCapacityExceededError(ValueError):
    """
    エンコード済み条件の有効トークン数が推論実装の固定容量を超えたことを表す。
    """

    def __init__(
        self,
        *,
        actual: int,
        capacity: int,
        text_tokens: int,
        speaker_tokens: int,
        caption_tokens: int,
    ) -> None:
        super().__init__(
            "Encoded context capacity exceeded: "
            f"actual={actual}, capacity={capacity}, text={text_tokens}, "
            f"speaker={speaker_tokens}, caption={caption_tokens}."
        )
        self.actual = actual
        self.capacity = capacity
        self.text_tokens = text_tokens
        self.speaker_tokens = speaker_tokens
        self.caption_tokens = caption_tokens


def list_available_runtime_precisions(device: str | torch.device) -> list[str]:
    resolved = resolve_runtime_device(device)
    if resolved.type == "cuda":
        return ["fp32", "bf16", "fp16"]
    if resolved.type == "xpu":
        return ["fp32", "bf16"]
    return ["fp32"]


def _sync_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        mps = getattr(torch, "mps", None)
        if mps is not None and hasattr(mps, "synchronize"):
            mps.synchronize()
    elif device.type == "xpu":
        xpu = getattr(torch, "xpu", None)
        if xpu is not None and hasattr(xpu, "synchronize"):
            xpu.synchronize()


def _sync_devices(*devices: torch.device) -> None:
    seen: set[tuple[str, int | None]] = set()
    for device in devices:
        key = (device.type, device.index)
        if key in seen:
            continue
        _sync_device(device)
        seen.add(key)


def _measure_start(device: torch.device, *extra_devices: torch.device) -> float:
    _sync_devices(device, *extra_devices)
    return time.perf_counter()


def _measure_end(device: torch.device, t0: float, *extra_devices: torch.device) -> float:
    _sync_devices(device, *extra_devices)
    return time.perf_counter() - t0


def _coerce_latent_shape(latent: torch.Tensor, latent_dim: int) -> torch.Tensor:
    if latent.ndim == 3 and latent.shape[0] == 1:
        latent = latent[0]
    if latent.ndim != 2:
        raise ValueError(f"Unsupported latent shape: {tuple(latent.shape)}")
    if latent.shape[1] == latent_dim:
        return latent
    if latent.shape[0] == latent_dim:
        return latent.transpose(0, 1).contiguous()
    raise ValueError(
        f"Could not infer latent layout for shape={tuple(latent.shape)} and latent_dim={latent_dim}"
    )


def find_flattening_point(
    latent: torch.Tensor,
    target_value: float = 0.0,
    window_size: int = 20,
    std_threshold: float = 0.05,
    mean_threshold: float = 0.1,
) -> int:
    """
    Echo-style heuristic: find first index where a trailing window becomes near-flat and near-zero.

    Args:
      latent: (T, D) latent sequence.
    Returns:
      Flattening index in [0, T].
    """
    if latent.ndim != 2:
        raise ValueError(f"Expected latent shape (T, D), got {tuple(latent.shape)}")
    return int(
        find_flattening_points(
            latent.unsqueeze(0),
            target_value=target_value,
            window_size=window_size,
            std_threshold=std_threshold,
            mean_threshold=mean_threshold,
        )[0].item()
    )


def find_flattening_points(
    latents: torch.Tensor,
    target_value: float = 0.0,
    window_size: int = 20,
    std_threshold: float = 0.05,
    mean_threshold: float = 0.1,
) -> torch.Tensor:
    """
    Echo-style heuristic: find first index where each trailing window becomes near-flat and near-zero.

    Args:
      latents: (B, T, D) latent sequence batch.
    Returns:
      Flattening indices in [0, T], shape (B,).
    """
    if latents.ndim != 3:
        raise ValueError(f"Expected latents shape (B, T, D), got {tuple(latents.shape)}")
    batch_size = int(latents.shape[0])
    total_steps = int(latents.shape[1])
    if total_steps <= 0 or window_size <= 0:
        return torch.full(
            (batch_size,),
            total_steps,
            dtype=torch.long,
            device=latents.device,
        )

    pad = torch.zeros(
        (batch_size, window_size, latents.shape[2]),
        device=latents.device,
        dtype=latents.dtype,
    )
    padded = torch.cat([latents, pad], dim=1)
    windows = padded.unfold(1, window_size, 1)[:, :total_steps]
    flattened_windows = windows.reshape(batch_size, total_steps, -1)
    window_std = flattened_windows.std(dim=2, unbiased=False)
    window_mean = flattened_windows.mean(dim=2)
    matches = (window_std < std_threshold) & (
        torch.abs(window_mean - target_value) < mean_threshold
    )
    first_matches = torch.argmax(matches.to(torch.long), dim=1)
    fallback = torch.full_like(first_matches, total_steps)
    return torch.where(matches.any(dim=1), first_matches, fallback)


@dataclass(frozen=True)
class RuntimeKey:
    checkpoint: str
    model_device: str
    codec_repo: str = "Aratako/Semantic-DACVAE-Japanese-32dim"
    model_precision: str = "fp32"
    codec_device: str = "cpu"
    codec_precision: str = "fp32"
    codec_deterministic_encode: bool = True
    codec_deterministic_decode: bool = True
    compile_model: bool = False
    compile_dynamic: bool = False
    enable_watermark: bool = False
    attention_backend: str = "auto"


@dataclass
class SpeakerCondition:
    """
    話者条件の状態と有効トークン範囲を保持する。

    Attributes:
        state (torch.Tensor): Speaker Encoder が返した `(batch, tokens, dim)` の話者状態
        mask (torch.Tensor): 話者状態の有効トークンを示す `(batch, tokens)` のマスク
    """

    state: torch.Tensor
    mask: torch.Tensor


@dataclass
class SamplingRequest:
    text: str
    caption: str | None = None
    ref_wav: str | None = None
    ref_latent: str | None = None
    ref_embed: str | None = None
    no_ref: bool = False
    ref_normalize_db: float | None = -16.0
    ref_ensure_max: bool = True
    num_candidates: int = 1
    decode_mode: str = "sequential"
    seconds: float | None = None
    duration_scale: float = 1.0
    min_seconds: float = 0.5
    max_seconds: float = 30.0
    max_ref_seconds: float | None = 30.0
    max_text_len: int | None = None
    max_caption_len: int | None = None
    latent_patch_bucket_multiple: int | None = None
    num_steps: int = 40
    cfg_scale_text: float = 3.0
    cfg_scale_caption: float = 3.0
    cfg_scale_speaker: float = 5.0
    cfg_scale_condition: float = 0.0
    cfg_guidance_mode: str = "independent"
    cfg_scale: float | None = None
    cfg_min_t: float = 0.5
    cfg_max_t: float = 1.0
    truncation_factor: float | None = None
    rescale_k: float | None = None
    rescale_sigma: float | None = None
    context_kv_cache: bool = True
    speaker_kv_scale: float | None = None
    speaker_kv_min_t: float | None = None
    speaker_kv_max_layers: int | None = None
    speaker_uncond_mode: str = "mask"
    seed: int | None = None
    noise_precision: str | None = None
    chunk_seed_mode: str = "same"
    t_schedule_mode: str = "linear"
    sway_coeff: float = -1.0
    trim_tail: bool = True
    tail_window_size: int = 20
    tail_std_threshold: float = 0.05
    tail_mean_threshold: float = 0.1
    lora_adapter: str | None = None
    speaker_condition_tokens: list[str] | None = None
    condition_token_scales: dict[str, float] | None = None
    waveex: WaveExConfig | None = None
    initial_noise: torch.Tensor | None = None
    initial_noise_offset: int = 0
    retry_chunk_index: int = 0
    # speaker_state のパディング設定
    ## 参照音声長による SDPA 形状変動を抑え、cuDNN カーネルキャッシュヒット率を上げる
    speaker_ref_fixed_length: int | None = None
    speaker_ref_bucket_sizes: list[int] | None = None
    # Speaker Inversion の話者条件とは分離し、生成後の話者豹変判定だけに参照音声を使う
    speaker_failure_ref_wav: str | None = None
    # 後方互換のため、新規フィールドは既存の位置引数列の末尾へ追加
    velocity_field_guidance: VelocityFieldGuidance | None = None
    trajectory_intervention: TrajectoryIntervention | None = None
    trajectory_observer: TrajectoryObserver | None = None
    # 事前計算したキャプション条件は必ず状態とマスクを組にして渡す
    caption_state_override: torch.Tensor | None = None
    caption_mask_override: torch.Tensor | None = None
    # チャンク間で話者条件を引き継ぐ場合は参照音声の再エンコードを省く
    speaker_condition_override: SpeakerCondition | None = None
    capture_generated_speaker_condition: bool = False


@dataclass
class SamplingResult:
    audio: torch.Tensor
    audios: list[torch.Tensor]
    sample_rate: int
    stage_timings: list[tuple[str, float]]
    total_to_decode: float
    used_seed: int
    messages: list[str]
    latent_steps: int = 0
    patched_steps: int = 0
    sampling_patched_steps: int = 0
    predicted_duration_frames: float | None = None
    scaled_duration_frames: float | None = None
    rounded_duration_frames: int | None = None
    min_duration_frames: int | None = None
    max_duration_frames: int | None = None
    duration_was_clamped: bool = False
    base_seed: int | None = None
    retry_seed: int | None = None
    retry_attempts: int = 1
    seed_retry_base_score: float | None = None
    seed_retry_candidate_score: float | None = None
    is_retry_adopted: bool = False
    speaker_condition: SpeakerCondition | None = None


@dataclass(frozen=True)
class _ReferenceCacheKey:
    source_type: str
    path: str
    mtime_ns: int
    size: int
    max_ref_seconds: float | None
    ref_normalize_db: float | None
    ref_ensure_max: bool
    model_device: str
    model_dtype: str
    latent_patch_size: int
    speaker_patch_size: int
    lora_adapter: str | None
    condition_token_ids: tuple[int, ...] = ()
    condition_token_scales: tuple[float, ...] = ()
    condition_vocabulary_hash: str | None = None
    # パディング設定もキャッシュキーに含め、異なるパディング条件でキャッシュ混在を防ぐ
    speaker_ref_fixed_length: int | None = None
    speaker_ref_bucket_sizes: tuple[int, ...] | None = None


@dataclass(frozen=True)
class _ReferenceCondition:
    latent: torch.Tensor
    mask: torch.Tensor


@dataclass(frozen=True)
class _SpeakerCondition:
    state: torch.Tensor
    mask: torch.Tensor


@dataclass(frozen=True)
class _CaptionCacheKey:
    text: str
    max_caption_len: int
    model_device: str
    model_dtype: str
    lora_adapter: str | None
    condition_token_ids: tuple[int, ...] = ()
    condition_token_scales: tuple[float, ...] = ()
    condition_vocabulary_hash: str | None = None


@dataclass(frozen=True)
class _CaptionCondition:
    state: torch.Tensor
    mask: torch.Tensor


@dataclass(frozen=True)
class _ConditionVocabulary:
    token_to_id: dict[str, int]
    vocabulary_hash: str


@dataclass(frozen=True)
class _ResolvedConditionTokens:
    token_names: tuple[str, ...]
    token_ids: tuple[int, ...]
    token_scales: tuple[float, ...]
    vocabulary_hash: str


def _maybe_compile_inference_model(
    model: TextToLatentRFDiT,
    *,
    enabled: bool,
    dynamic: bool,
) -> TextToLatentRFDiT:
    if not enabled:
        return model
    if not hasattr(torch, "compile"):
        raise RuntimeError("compile_model=True requires torch.compile (PyTorch 2+).")
    compile_kwargs = {"dynamic": bool(dynamic)}
    model.encode_conditions = torch.compile(model.encode_conditions, **compile_kwargs)
    model.build_context_kv_cache = torch.compile(model.build_context_kv_cache, **compile_kwargs)
    model.forward_with_encoded_conditions = torch.compile(
        model.forward_with_encoded_conditions,
        **compile_kwargs,
    )
    return model


def _configure_attention_backend(attention_backend: str) -> None:
    """
    Configure the global SDPA backend preference for inference.

    Args:
        attention_backend (str): Backend name (`auto`, `mem_efficient`, `cudnn`, `math`, `flash`)

    Raises:
        ValueError: Unsupported backend name was specified
    """

    backend = str(attention_backend).strip().lower().replace("-", "_")
    if backend in {"", "auto", "default"}:
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_cudnn_sdp(True)
        torch.backends.cuda.enable_math_sdp(True)
        return
    if backend in {"mem", "memory_efficient", "mem_efficient", "efficient"}:
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_cudnn_sdp(False)
        torch.backends.cuda.enable_math_sdp(False)
        return
    if backend == "cudnn":
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_cudnn_sdp(True)
        torch.backends.cuda.enable_math_sdp(False)
        return
    if backend == "math":
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_cudnn_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
        return
    if backend == "flash":
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_cudnn_sdp(False)
        torch.backends.cuda.enable_math_sdp(False)
        return
    raise ValueError(
        "attention_backend must be one of: auto, mem_efficient, cudnn, math, flash. "
        f"Got {attention_backend!r}."
    )


def _move_inference_module(
    module: torch.nn.Module,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.nn.Module:
    # GPU 上で FP32 の中間テンソルを作らないよう、device と dtype を同時に確定する
    module.to(device=device, dtype=dtype)
    return module


def _empty_cuda_cache_if_needed(*devices: torch.device) -> None:
    if torch.cuda.is_available() is False:
        return
    for device in devices:
        if device.type == "cuda":
            torch.cuda.empty_cache()
            return


def resolve_runtime_dtype(*, precision: str, device: torch.device) -> torch.dtype:
    mode = str(precision).strip().lower()
    if mode == "fp32":
        return torch.float32
    if mode == "bf16":
        if device.type not in ("cuda", "xpu"):
            raise ValueError("precision='bf16' currently requires CUDA or XPU device.")
        return torch.bfloat16
    if mode == "fp16":
        if device.type != "cuda":
            raise ValueError("precision='fp16' currently requires CUDA device.")
        return torch.float16
    raise ValueError(f"Unsupported precision={precision!r}. Expected one of: fp32, bf16, fp16.")


def resolve_cfg_scales(
    *,
    cfg_guidance_mode: str,
    cfg_scale_text: float,
    cfg_scale_caption: float,
    cfg_scale_speaker: float,
    cfg_scale: float | None,
    cfg_scale_condition: float = 0.0,
    use_caption_condition: bool = True,
    use_speaker_condition: bool = True,
    use_token_condition: bool = False,
) -> tuple[float, float, float, float, list[str]]:
    """
    Normalize/validate CFG scales for guidance mode.
    """
    messages: list[str] = []
    text_val = float(cfg_scale_text)
    caption_val = float(cfg_scale_caption)
    speaker_val = float(cfg_scale_speaker)
    condition_val = float(cfg_scale_condition)

    if cfg_scale is not None:
        text_val = float(cfg_scale)
        caption_val = float(cfg_scale)
        speaker_val = float(cfg_scale)
        condition_val = float(cfg_scale)
    if not use_speaker_condition:
        if speaker_val > 0.0:
            messages.append(
                "info: speaker conditioning is disabled for this checkpoint or request; "
                "ignoring cfg_scale_speaker."
            )
        speaker_val = 0.0
    if not use_token_condition:
        if condition_val > 0.0:
            messages.append(
                "info: condition token guidance is disabled for this request; "
                "ignoring cfg_scale_condition."
            )
        condition_val = 0.0

    mode = str(cfg_guidance_mode).strip().lower()
    enabled_vals = [value for value in (text_val, speaker_val, condition_val) if value > 0.0]
    if use_caption_condition and caption_val > 0.0:
        enabled_vals.append(caption_val)
    if mode == "joint" and enabled_vals and (max(enabled_vals) - min(enabled_vals) > 1e-6):
        raise ValueError(
            "cfg_guidance_mode='joint' requires equal enabled cfg_scale_text/cfg_scale_caption/cfg_scale_speaker/cfg_scale_condition, "
            "or set cfg_scale."
        )

    return text_val, caption_val, speaker_val, condition_val, messages


def _load_torch_checkpoint_payload(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError(f"Unsupported checkpoint payload type: {type(payload)!r}")
    return payload


_CONFIG_META_KEY = "config_json"
_INFERENCE_CONFIG_KEYS = {
    "max_text_len",
    "max_caption_len",
    "fixed_target_latent_steps",
}


def _load_checkpoint_from_pt(
    path: Path,
) -> tuple[dict[str, torch.Tensor], dict[str, Any], dict[str, Any] | None]:
    ckpt = _load_torch_checkpoint_payload(path)
    model_state = ckpt.get("model")
    model_cfg = ckpt.get("model_config")
    train_cfg = ckpt.get("train_config")

    if not isinstance(model_state, dict):
        raise ValueError(f"Checkpoint missing model weights dictionary: {path}")
    if not isinstance(model_cfg, dict):
        raise ValueError(f"Checkpoint missing model_config dictionary: {path}")
    if train_cfg is not None and not isinstance(train_cfg, dict):
        raise ValueError(f"Checkpoint train_config must be a dictionary when present: {path}")

    if checkpoint_state_uses_lora(model_state):
        raise ValueError(
            f"LoRA checkpoints must be loaded from adapter directories or merged safetensors: {path}"
        )
    return model_state, model_cfg, _extract_inference_train_config(train_cfg)


def _parse_json_mapping(
    raw: str | None,
    *,
    field: str,
    path: Path,
    required: bool = False,
) -> dict[str, Any] | None:
    if raw is None:
        if required:
            raise ValueError(f"Missing required metadata field '{field}' in checkpoint: {path}")
        return None
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in '{field}' metadata for checkpoint: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Metadata field '{field}' must decode to an object: {path}")
    return payload


def _extract_inference_train_config(raw: dict[str, Any] | None) -> dict[str, int] | None:
    if raw is None:
        return None

    inference_cfg: dict[str, int] = {}
    for key in _INFERENCE_CONFIG_KEYS:
        value = raw.get(key)
        if value is None:
            continue
        if not isinstance(value, int):
            raise ValueError(f"Inference config key '{key}' must be int, got {type(value)!r}.")
        inference_cfg[key] = int(value)

    return inference_cfg or None


def _split_flat_checkpoint_config(
    path: Path, flat_config: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, int] | None]:
    model_cfg: dict[str, object] = {}
    inference_cfg: dict[str, int] = {}
    for key, value in flat_config.items():
        if key in _INFERENCE_CONFIG_KEYS:
            if not isinstance(value, int):
                raise ValueError(
                    f"Inference config key '{key}' must be int in checkpoint metadata: {path}"
                )
            inference_cfg[key] = int(value)
            continue
        model_cfg[key] = value
    return model_cfg, (inference_cfg or None)


def _load_checkpoint_from_safetensors(
    path: Path,
) -> tuple[dict[str, torch.Tensor], dict[str, Any], dict[str, Any] | None]:
    model_state = load_safetensors_file(str(path), device="cpu")
    if not model_state:
        raise ValueError(f"Safetensors checkpoint has no model weights: {path}")

    with safe_open(str(path), framework="pt", device="cpu") as handle:
        metadata = handle.metadata() or {}

    flat_config = _parse_json_mapping(
        metadata.get(_CONFIG_META_KEY),
        field=_CONFIG_META_KEY,
        path=path,
        required=True,
    )
    model_cfg, inference_cfg = _split_flat_checkpoint_config(path=path, flat_config=flat_config)
    return model_state, model_cfg, inference_cfg


def _load_checkpoint_for_inference(
    path: Path,
) -> tuple[dict[str, torch.Tensor], dict[str, Any], dict[str, Any] | None]:
    if path.suffix.lower() == ".safetensors":
        return _load_checkpoint_from_safetensors(path)
    return _load_checkpoint_from_pt(path)


class InferenceRuntime:
    def __init__(
        self,
        *,
        key: RuntimeKey,
        model_cfg: ModelConfig,
        train_cfg: dict[str, Any] | None,
        model: TextToLatentRFDiT,
        tokenizer: PretrainedTextTokenizer,
        caption_tokenizer: PretrainedTextTokenizer | None,
        codec: DACVAECodec,
        default_text_max_len: int,
        default_caption_max_len: int,
        max_encoded_context_tokens: int | None = None,
    ) -> None:
        self.key = key
        self.model_device = resolve_runtime_device(key.model_device)
        self.codec_device = resolve_runtime_device(key.codec_device)
        self.model_cfg = model_cfg
        self.train_cfg = train_cfg
        self.model = model
        self.tokenizer = tokenizer
        self.caption_tokenizer = caption_tokenizer
        self.codec = codec
        self.default_text_max_len = default_text_max_len
        self.default_caption_max_len = default_caption_max_len
        if max_encoded_context_tokens is not None and max_encoded_context_tokens <= 0:
            raise ValueError(
                "max_encoded_context_tokens must be greater than zero when provided: "
                f"{max_encoded_context_tokens}"
            )
        # 固定長エンジンを接続した製品ランタイムだけが上限を設定し、通常の eager 推論は従来どおり無制限に扱う
        self.max_encoded_context_tokens = max_encoded_context_tokens
        self.watermarker = (
            SilentCipherWatermarker(device=str(self.codec_device))
            if bool(self.key.enable_watermark)
            else None
        )
        self._infer_lock = threading.Lock()
        self._model_dtype = next(self.model.parameters()).dtype
        self._lora_adapter_names: dict[str, str] = {}
        self._reference_condition_cache: OrderedDict[_ReferenceCacheKey, _ReferenceCondition] = (
            OrderedDict()
        )
        self._reference_condition_cache_max_entries = 32
        self._speaker_condition_cache: OrderedDict[_ReferenceCacheKey, _SpeakerCondition] = (
            OrderedDict()
        )
        self._speaker_condition_cache_max_entries = 32
        self._caption_condition_cache: OrderedDict[_CaptionCacheKey, _CaptionCondition] = (
            OrderedDict()
        )
        self._caption_condition_cache_max_entries = 64
        self._condition_vocabulary_cache: dict[str, _ConditionVocabulary] = {}

    @classmethod
    def from_key(cls, key: RuntimeKey) -> InferenceRuntime:
        model_device = resolve_runtime_device(key.model_device)
        codec_device = resolve_runtime_device(key.codec_device)
        _configure_attention_backend(key.attention_backend)
        model_dtype = resolve_runtime_dtype(
            precision=key.model_precision,
            device=model_device,
        )
        codec_dtype = resolve_runtime_dtype(
            precision=key.codec_precision,
            device=codec_device,
        )

        model_state, model_cfg_dict, train_cfg = _load_checkpoint_for_inference(
            Path(key.checkpoint)
        )
        model_cfg = ModelConfig(**model_cfg_dict)

        # checkpoint は FP32 のため、GPU へ載せる前に CPU 上で state_dict を反映する
        model = TextToLatentRFDiT(model_cfg)
        model.load_state_dict(model_state)
        del model_state
        model = _move_inference_module(model, device=model_device, dtype=model_dtype)
        model.eval()
        model = _maybe_compile_inference_model(
            model,
            enabled=bool(key.compile_model),
            dynamic=bool(key.compile_dynamic),
        )

        tokenizer = PretrainedTextTokenizer.from_pretrained(
            repo_id=model_cfg.text_tokenizer_repo,
            add_bos=bool(model_cfg.text_add_bos),
            local_files_only=False,
        )
        if tokenizer.vocab_size != model_cfg.text_vocab_size:
            raise ValueError(
                f"text_vocab_size mismatch: checkpoint text_vocab_size={model_cfg.text_vocab_size} but tokenizer "
                f"({model_cfg.text_tokenizer_repo}) vocab_size={tokenizer.vocab_size}."
            )
        caption_tokenizer = None
        if model_cfg.use_caption_condition:
            caption_tokenizer = PretrainedTextTokenizer.from_pretrained(
                repo_id=model_cfg.caption_tokenizer_repo_resolved,
                add_bos=model_cfg.caption_add_bos_resolved,
                local_files_only=False,
            )
            if caption_tokenizer.vocab_size != model_cfg.caption_vocab_size_resolved:
                raise ValueError(
                    f"caption_vocab_size mismatch: checkpoint caption_vocab_size={model_cfg.caption_vocab_size_resolved} but tokenizer ({model_cfg.caption_tokenizer_repo_resolved}) "
                    f"vocab_size={caption_tokenizer.vocab_size}."
                )

        default_text_max_len = 256
        default_caption_max_len = default_text_max_len
        if isinstance(train_cfg, dict):
            ckpt_text_max_len = train_cfg.get("max_text_len")
            if isinstance(ckpt_text_max_len, int) and ckpt_text_max_len > 0:
                default_text_max_len = int(ckpt_text_max_len)
            ckpt_caption_max_len = train_cfg.get("max_caption_len")
            if isinstance(ckpt_caption_max_len, int) and ckpt_caption_max_len > 0:
                default_caption_max_len = int(ckpt_caption_max_len)
            else:
                default_caption_max_len = default_text_max_len

        codec = DACVAECodec.load(
            repo_id=key.codec_repo,
            device=str(codec_device),
            dtype=codec_dtype,
            deterministic_encode=bool(key.codec_deterministic_encode),
            deterministic_decode=bool(key.codec_deterministic_decode),
        )
        _empty_cuda_cache_if_needed(model_device, codec_device)
        if model_cfg.latent_dim != codec.latent_dim:
            raise ValueError(
                f"Latent dimension mismatch: checkpoint latent_dim={model_cfg.latent_dim} but codec latent_dim={codec.latent_dim}. "
                "Use a compatible codec/checkpoint pair."
            )

        return cls(
            key=key,
            model_cfg=model_cfg,
            train_cfg=train_cfg if isinstance(train_cfg, dict) else None,
            model=model,
            tokenizer=tokenizer,
            caption_tokenizer=caption_tokenizer,
            codec=codec,
            default_text_max_len=default_text_max_len,
            default_caption_max_len=default_caption_max_len,
        )

    def _validate_encoded_context_capacity(
        self,
        encoded_conditions: EncodedConditions,
    ) -> None:
        """
        RF の K/V 構築前に、固定長エンジンへ収まる有効条件トークン数か検査する。

        Args:
            encoded_conditions (EncodedConditions): 検査するエンコード済み条件
        """

        capacity = self.max_encoded_context_tokens
        if capacity is None:
            return

        _, text_mask, _, speaker_mask, _, caption_mask = encoded_conditions
        text_lengths = text_mask.to(dtype=torch.int64).sum(dim=1)
        speaker_lengths = (
            torch.zeros_like(text_lengths)
            if speaker_mask is None
            else speaker_mask.to(dtype=torch.int64).sum(dim=1)
        )
        caption_lengths = (
            torch.zeros_like(text_lengths)
            if caption_mask is None
            else caption_mask.to(dtype=torch.int64).sum(dim=1)
        )
        context_lengths = text_lengths + speaker_lengths + caption_lengths
        maximum_context_length = int(context_lengths.max().item())
        if maximum_context_length <= capacity:
            return

        # 超過した候補と同じ行の内訳を返し、話者参照・本文・キャプションのどこを短縮すべきか判別できるようにする
        maximum_index = int(context_lengths.argmax().item())
        raise ContextCapacityExceededError(
            actual=maximum_context_length,
            capacity=capacity,
            text_tokens=int(text_lengths[maximum_index].item()),
            speaker_tokens=int(speaker_lengths[maximum_index].item()),
            caption_tokens=int(caption_lengths[maximum_index].item()),
        )

    def _resolve_lora_adapter_path(self, adapter_path: str | None) -> str | None:
        if adapter_path is None:
            return None
        raw = str(adapter_path).strip()
        if raw.lower() in {"", "none", "null", "off", "disable", "disabled", "base"}:
            return None

        path = Path(raw).expanduser()
        if not path.is_dir():
            raise FileNotFoundError(f"LoRA adapter directory not found: {path}")
        if not is_lora_adapter_dir(path):
            raise ValueError(
                f"LoRA adapter directory must contain adapter_config.json and adapter weights: {path}"
            )
        return str(path.resolve())

    @staticmethod
    def _adapter_name_for_path(path: str) -> str:
        digest = hashlib.sha1(path.encode("utf-8")).hexdigest()[:16]
        return f"runtime_{digest}"

    def _load_condition_vocabulary(self, adapter_path: str) -> _ConditionVocabulary:
        cached = self._condition_vocabulary_cache.get(adapter_path)
        if cached is not None:
            return cached

        metadata_path = Path(adapter_path) / "irodori_lora_metadata.json"
        if not metadata_path.is_file():
            raise ValueError(
                "Condition tokens require irodori_lora_metadata.json in the LoRA adapter directory."
            )
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        condition_vocabulary = payload.get("condition_vocabulary")
        if not isinstance(condition_vocabulary, dict):
            raise ValueError(f"LoRA metadata is missing condition_vocabulary: {metadata_path}")
        token_to_id_raw = condition_vocabulary.get("token_to_id")
        if not isinstance(token_to_id_raw, dict):
            raise ValueError(
                f"LoRA condition_vocabulary.token_to_id must be an object: {metadata_path}"
            )
        token_to_id = {str(token): int(token_id) for token, token_id in token_to_id_raw.items()}
        raw_hash = payload.get("condition_vocabulary_hash") or condition_vocabulary.get("sha256")
        if raw_hash is None:
            # metadata に hash がない古い adapter でも、キャッシュ分離に使える安定 hash を作る
            encoded = json.dumps(token_to_id, ensure_ascii=False, sort_keys=True).encode("utf-8")
            vocabulary_hash = hashlib.sha256(encoded).hexdigest()
        else:
            vocabulary_hash = str(raw_hash)
        vocabulary = _ConditionVocabulary(
            token_to_id=token_to_id,
            vocabulary_hash=vocabulary_hash,
        )
        self._condition_vocabulary_cache[adapter_path] = vocabulary
        return vocabulary

    def _ensure_condition_embedding_for_adapter(self, adapter_path: str) -> None:
        if not (Path(adapter_path) / "irodori_lora_metadata.json").is_file():
            return
        vocabulary = self._load_condition_vocabulary(adapter_path)
        vocab_size = len(vocabulary.token_to_id)
        if vocab_size <= 0:
            return
        current_embedding = getattr(self.model, "speaker_condition_embedding", None)
        if current_embedding is not None:
            if int(current_embedding.num_embeddings) != int(vocab_size):
                raise ValueError(
                    "LoRA condition vocabulary size mismatch: "
                    f"model={int(current_embedding.num_embeddings)} adapter={vocab_size}"
                )
            return

        # base checkpoint は条件 embedding を持たないため、PEFT が modules_to_save を復元する前に器を作る
        self.model.speaker_condition_embedding = torch.nn.Embedding(
            int(vocab_size),
            int(self.model_cfg.speaker_dim),
        )
        self.model.cfg.speaker_condition_vocab_size = int(vocab_size)
        self.model.cfg.speaker_condition_vocab_hash = vocabulary.vocabulary_hash
        self.model_cfg.speaker_condition_vocab_size = int(vocab_size)
        self.model_cfg.speaker_condition_vocab_hash = vocabulary.vocabulary_hash
        self.model = _move_inference_module(
            self.model,
            device=self.model_device,
            dtype=self._model_dtype,
        )

    def _resolve_condition_tokens(
        self,
        req: SamplingRequest,
        *,
        lora_adapter: str | None,
    ) -> _ResolvedConditionTokens | None:
        raw_tokens = req.speaker_condition_tokens
        if raw_tokens is None or len(raw_tokens) == 0:
            if req.condition_token_scales:
                raise ValueError(
                    "condition_token_scales cannot be specified without speaker_condition_tokens."
                )
            return None
        if lora_adapter is None:
            raise ValueError("speaker_condition_tokens require a LoRA adapter.")
        if self.model_cfg.use_speaker_condition_resolved is False:
            raise ValueError("speaker_condition_tokens require speaker conditioning support.")

        vocabulary = self._load_condition_vocabulary(lora_adapter)
        token_names = tuple(str(token).strip() for token in raw_tokens)
        if any(token == "" for token in token_names):
            raise ValueError("speaker_condition_tokens must not contain empty token names.")

        token_ids: list[int] = []
        unknown_tokens: list[str] = []
        for token_name in token_names:
            token_id = vocabulary.token_to_id.get(token_name)
            if token_id is None:
                unknown_tokens.append(token_name)
                continue
            token_ids.append(int(token_id))
        if unknown_tokens:
            allowed = ", ".join(sorted(vocabulary.token_to_id))
            raise ValueError(
                "Unknown speaker condition token(s): "
                f"{', '.join(unknown_tokens)}. Allowed tokens: {allowed}"
            )

        scale_map = req.condition_token_scales or {}
        unknown_scale_tokens = sorted(
            str(token) for token in scale_map if str(token) not in token_names
        )
        if unknown_scale_tokens:
            raise ValueError(
                "condition_token_scales contains token(s) not present in speaker_condition_tokens: "
                f"{', '.join(unknown_scale_tokens)}"
            )
        token_scales: list[float] = []
        for token_name in token_names:
            scale = float(scale_map.get(token_name, 1.0))
            if not math.isfinite(scale):
                raise ValueError(f"condition token scale must be finite: {token_name}")
            token_scales.append(scale)

        return _ResolvedConditionTokens(
            token_names=token_names,
            token_ids=tuple(token_ids),
            token_scales=tuple(token_scales),
            vocabulary_hash=vocabulary.vocabulary_hash,
        )

    def _prepare_lora_for_request(
        self,
        adapter_path: str | None,
        *,
        messages: list[str],
        stage_timings: list[tuple[str, float]],
        log_fn: Callable[[str], None],
    ) -> Any:
        should_time = adapter_path is not None and str(adapter_path).strip() != ""
        t0 = _measure_start(self.model_device) if should_time else None
        try:
            return self._prepare_lora_for_request_inner(
                adapter_path,
                messages=messages,
                log_fn=log_fn,
            )
        finally:
            if t0 is not None:
                stage_sec = _measure_end(self.model_device, t0)
                stage_timings.append(("prepare_lora", stage_sec))
                log_fn(f"[runtime] prepare_lora: {stage_sec * 1000.0:.1f} ms")

    def _prepare_lora_for_request_inner(
        self,
        adapter_path: str | None,
        *,
        messages: list[str],
        log_fn: Callable[[str], None],
    ) -> Any:
        resolved_path = self._resolve_lora_adapter_path(adapter_path)
        if resolved_path is None:
            disable_adapter = getattr(self.model, "disable_adapter", None)
            if callable(disable_adapter):
                msg = "info: dynamic LoRA disabled for this request; using base model."
                messages.append(msg)
                log_fn(msg)
                return disable_adapter()
            return nullcontext()

        if self.key.compile_model:
            raise RuntimeError("Dynamic LoRA loading is not compatible with compile_model=True.")

        self._ensure_condition_embedding_for_adapter(resolved_path)

        adapter_name = self._lora_adapter_names.get(resolved_path)
        if adapter_name is None:
            adapter_name = self._adapter_name_for_path(resolved_path)
            msg = f"info: loading LoRA adapter: {resolved_path}"
            messages.append(msg)
            log_fn(msg)
        else:
            msg = f"info: using cached LoRA adapter: {resolved_path}"
            messages.append(msg)
            log_fn(msg)

        self.model = load_lora_adapter(
            self.model,
            resolved_path,
            is_trainable=False,
            adapter_name=adapter_name,
            torch_device=str(self.model_device),
        )
        self._lora_adapter_names[resolved_path] = adapter_name
        self.model = _move_inference_module(
            self.model,
            device=self.model_device,
            dtype=self._model_dtype,
        )
        self.model.eval()
        return nullcontext()

    def _reference_cache_key(
        self,
        req: SamplingRequest,
        *,
        resolved_condition_tokens: _ResolvedConditionTokens | None,
        lora_adapter: str | None,
    ) -> _ReferenceCacheKey | None:
        source_type: str
        source_path: str | None
        if req.ref_latent is not None:
            source_type = "latent"
            source_path = req.ref_latent
        elif req.ref_wav is not None:
            source_type = "wav"
            source_path = req.ref_wav
        else:
            return None

        path = Path(str(source_path)).expanduser()
        stat = path.stat()
        return _ReferenceCacheKey(
            source_type=source_type,
            path=str(path.resolve()),
            mtime_ns=int(stat.st_mtime_ns),
            size=int(stat.st_size),
            max_ref_seconds=None if req.max_ref_seconds is None else float(req.max_ref_seconds),
            ref_normalize_db=None if req.ref_normalize_db is None else float(req.ref_normalize_db),
            ref_ensure_max=bool(req.ref_ensure_max),
            model_device=str(self.model_device),
            model_dtype=str(self._model_dtype),
            latent_patch_size=int(self.model_cfg.latent_patch_size),
            speaker_patch_size=int(self.model_cfg.speaker_patch_size),
            lora_adapter=lora_adapter,
            condition_token_ids=()
            if resolved_condition_tokens is None
            else resolved_condition_tokens.token_ids,
            condition_token_scales=()
            if resolved_condition_tokens is None
            else resolved_condition_tokens.token_scales,
            condition_vocabulary_hash=None
            if resolved_condition_tokens is None
            else resolved_condition_tokens.vocabulary_hash,
            speaker_ref_fixed_length=req.speaker_ref_fixed_length,
            speaker_ref_bucket_sizes=(
                tuple(req.speaker_ref_bucket_sizes)
                if req.speaker_ref_bucket_sizes is not None
                else None
            ),
        )

    def _get_cached_reference_condition(
        self,
        key: _ReferenceCacheKey | None,
        *,
        batch_size: int,
        messages: list[str],
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if key is None:
            return None, None
        cached = self._reference_condition_cache.get(key)
        if cached is None:
            return None, None
        self._reference_condition_cache.move_to_end(key)
        messages.append("info: using cached reference conditioning.")
        return self._expand_reference_condition(cached, batch_size=batch_size)

    def _put_cached_reference_condition(
        self,
        key: _ReferenceCacheKey | None,
        condition: _ReferenceCondition,
    ) -> None:
        if key is None:
            return
        self._reference_condition_cache[key] = condition
        self._reference_condition_cache.move_to_end(key)
        while len(self._reference_condition_cache) > self._reference_condition_cache_max_entries:
            self._reference_condition_cache.popitem(last=False)

    @staticmethod
    def _expand_reference_condition(
        condition: _ReferenceCondition,
        *,
        batch_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if batch_size <= 1:
            return condition.latent, condition.mask
        return condition.latent.repeat(batch_size, 1, 1), condition.mask.repeat(batch_size, 1)

    def _get_cached_speaker_condition(
        self,
        key: _ReferenceCacheKey | None,
        *,
        batch_size: int,
        messages: list[str],
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if key is None:
            return None, None
        cached = self._speaker_condition_cache.get(key)
        if cached is None:
            return None, None
        self._speaker_condition_cache.move_to_end(key)
        messages.append("info: using cached speaker conditioning.")
        return self._expand_speaker_condition(cached, batch_size=batch_size)

    def _put_cached_speaker_condition(
        self,
        key: _ReferenceCacheKey | None,
        condition: _SpeakerCondition,
    ) -> None:
        if key is None:
            return
        self._speaker_condition_cache[key] = condition
        self._speaker_condition_cache.move_to_end(key)
        while len(self._speaker_condition_cache) > self._speaker_condition_cache_max_entries:
            self._speaker_condition_cache.popitem(last=False)

    def encode_speaker_condition(
        self,
        request: SamplingRequest,
        *,
        log_fn: Callable[[str], None] | None = None,
    ) -> SpeakerCondition:
        """
        1件の参照音声、latent、Speaker Inversion 埋め込みを話者状態へ変換する。

        呼び出し側は返された状態を話者 ID と対応付け、通常合成の `speaker_condition_override` へ渡すことで参照音声の再処理を省ける。

        Args:
            request (SamplingRequest): 変換対象の話者条件を持つ要求
            log_fn (Callable[[str], None] | None): 処理メッセージの出力先

        Returns:
            SpeakerCondition: バッチ数1の話者状態と有効トークンマスク

        Raises:
            ValueError: 話者条件が指定されていない場合
            RuntimeError: チェックポイントが話者条件に対応していない場合
        """

        if request.no_ref is True:
            raise ValueError("Speaker encoding requires a reference or inversion embedding.")
        if request.ref_wav is None and request.ref_latent is None and request.ref_embed is None:
            raise ValueError("Speaker encoding requires ref_wav, ref_latent, or ref_embed.")
        if self.model_cfg.use_speaker_condition_resolved is False:
            raise RuntimeError("Speaker conditioning is disabled for this checkpoint.")

        lora_adapter = self._resolve_lora_adapter_path(request.lora_adapter)
        resolved_condition_tokens = self._resolve_condition_tokens(
            request,
            lora_adapter=lora_adapter,
        )
        messages: list[str] = []
        # 通常合成と同じロック内で計算し、CUDA のモデル実行と参照読込を直列化する
        with self._infer_lock, torch.inference_mode():
            # Speaker Inversion は保存済み状態をモデルの dtype とデバイスへ展開するだけでよい
            if request.ref_embed is not None:
                state, mask = self._load_speaker_embedding_condition(
                    req=request,
                    batch_size=1,
                    messages=messages,
                )
                if state is None or mask is None:
                    raise RuntimeError("Failed to load speaker inversion embedding.")
            else:
                ref_latent, ref_mask = self._load_reference_latent(
                    req=request,
                    resolved_condition_tokens=resolved_condition_tokens,
                    lora_adapter=lora_adapter,
                    batch_size=1,
                    messages=messages,
                )
                if ref_latent is None or ref_mask is None:
                    raise RuntimeError("Failed to load speaker reference.")
                state, mask = self.model.encode_speaker_condition(
                    batch_size=1,
                    dtype=self._model_dtype,
                    device=self.model_device,
                    ref_latent=ref_latent,
                    ref_mask=ref_mask,
                    speaker_uncond_mode=request.speaker_uncond_mode,
                )
                # 呼び出し側が speaker_state を保持するため、大きい参照 latent はランタイムに残さない
                cache_key = self._reference_cache_key(
                    request,
                    resolved_condition_tokens=resolved_condition_tokens,
                    lora_adapter=lora_adapter,
                )
                if cache_key is not None:
                    self._reference_condition_cache.pop(cache_key, None)

        if log_fn is not None:
            for message in messages:
                log_fn(message)
        return SpeakerCondition(state=state.detach(), mask=mask.detach())

    @staticmethod
    def _expand_speaker_condition(
        condition: _SpeakerCondition,
        *,
        batch_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if batch_size <= 1:
            return condition.state, condition.mask
        return condition.state.repeat(batch_size, 1, 1), condition.mask.repeat(batch_size, 1)

    def _load_cached_speaker_condition(
        self,
        *,
        req: SamplingRequest,
        resolved_condition_tokens: _ResolvedConditionTokens | None,
        lora_adapter: str | None,
        ref_latent: torch.Tensor | None,
        ref_mask: torch.Tensor | None,
        batch_size: int,
        messages: list[str],
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if not self.model_cfg.use_speaker_condition_resolved:
            return None, None
        if req.no_ref or req.ref_embed is not None:
            return None, None

        cache_key = self._reference_cache_key(
            req,
            resolved_condition_tokens=resolved_condition_tokens,
            lora_adapter=lora_adapter,
        )
        cached_state, cached_mask = self._get_cached_speaker_condition(
            cache_key,
            batch_size=batch_size,
            messages=messages,
        )
        if cached_state is not None and cached_mask is not None:
            return cached_state, cached_mask
        if ref_latent is None or ref_mask is None:
            return None, None

        state, mask = self.model.encode_speaker_condition(
            batch_size=1,
            dtype=self._model_dtype,
            device=self.model_device,
            ref_latent=ref_latent[:1],
            ref_mask=ref_mask[:1],
            speaker_uncond_mode=req.speaker_uncond_mode,
        )
        condition = _SpeakerCondition(state=state.detach(), mask=mask.detach())
        self._put_cached_speaker_condition(cache_key, condition)
        # speaker_state の生成後は大きい参照 latent を保持する必要がない
        ## 次回以降は話者状態を直接返し、GPU メモリと参照音声の再エンコードを両方省く
        if cache_key is not None:
            self._reference_condition_cache.pop(cache_key, None)
        messages.append("info: cached speaker conditioning.")
        return self._expand_speaker_condition(condition, batch_size=batch_size)

    def _caption_cache_key(
        self,
        req: SamplingRequest,
        caption_text: str,
        caption_max_len: int,
        *,
        resolved_condition_tokens: _ResolvedConditionTokens | None,
        lora_adapter: str | None,
    ) -> _CaptionCacheKey:
        """
        キャプション条件キャッシュのキーを作る。

        Args:
            req (SamplingRequest): 合成リクエスト
            caption_text (str): VoiceDesign キャプション本文
            caption_max_len (int): キャプションの最大トークン長

        Returns:
            _CaptionCacheKey: 同じ条件だけを再利用するためのキャッシュキー
        """

        # LoRA と dtype が変わると同じ文章でも潜在表現が変わるため、テキスト以外の推論条件もキーに含める
        return _CaptionCacheKey(
            text=caption_text,
            max_caption_len=int(caption_max_len),
            model_device=str(self.model_device),
            model_dtype=str(self._model_dtype),
            lora_adapter=lora_adapter,
            condition_token_ids=()
            if resolved_condition_tokens is None
            else resolved_condition_tokens.token_ids,
            condition_token_scales=()
            if resolved_condition_tokens is None
            else resolved_condition_tokens.token_scales,
            condition_vocabulary_hash=None
            if resolved_condition_tokens is None
            else resolved_condition_tokens.vocabulary_hash,
        )

    def _get_cached_caption_condition(
        self,
        key: _CaptionCacheKey,
        *,
        batch_size: int,
        messages: list[str],
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        cached = self._caption_condition_cache.get(key)
        if cached is None:
            return None, None
        self._caption_condition_cache.move_to_end(key)
        messages.append("info: using cached caption conditioning.")
        return self._expand_caption_condition(cached, batch_size=batch_size)

    def _put_cached_caption_condition(
        self,
        key: _CaptionCacheKey,
        condition: _CaptionCondition,
    ) -> None:
        """
        キャプション条件を LRU キャッシュへ保存する。
        """

        # キャプションは voice ほど種類が固定されないため、上限を超えた古い条件から破棄する
        self._caption_condition_cache[key] = condition
        self._caption_condition_cache.move_to_end(key)
        while len(self._caption_condition_cache) > self._caption_condition_cache_max_entries:
            self._caption_condition_cache.popitem(last=False)

    @staticmethod
    def _expand_caption_condition(
        condition: _CaptionCondition,
        *,
        batch_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        1件分だけ保持したキャプション条件を候補数ぶんに複製する。
        """

        # 候補生成では全候補が同じキャプションを使うため、エンコード済み表現をそのまま複製できる
        if batch_size <= 1:
            return condition.state, condition.mask
        return condition.state.repeat(batch_size, 1, 1), condition.mask.repeat(batch_size, 1)

    def _load_cached_caption_condition(
        self,
        *,
        req: SamplingRequest,
        resolved_condition_tokens: _ResolvedConditionTokens | None,
        lora_adapter: str | None,
        caption_text: str,
        caption_ids: torch.Tensor,
        caption_mask: torch.Tensor,
        caption_max_len: int,
        batch_size: int,
        messages: list[str],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        VoiceDesign キャプションをエンコードし、同一キャプションなら再利用する。

        Args:
            req (SamplingRequest): 合成リクエスト
            caption_text (str): 空白除去済みのキャプション本文
            caption_ids (torch.Tensor): キャプションのトークン ID
            caption_mask (torch.Tensor): キャプションの有効トークンマスク
            caption_max_len (int): キャプションの最大トークン長
            batch_size (int): 候補生成数
            messages (list[str]): ランタイムログへ返すメッセージ

        Returns:
            tuple[torch.Tensor, torch.Tensor]: キャプション状態とマスク
        """

        # 無印 v3 などのキャプション非対応モデルでは、この経路に入ること自体が呼び出し側の不整合
        if self.model_cfg.use_caption_condition is False:
            raise RuntimeError(
                "Caption conditioning cache was requested for a model without caption support."
            )
        if self.model.caption_encoder is None or self.model.caption_norm is None:
            raise RuntimeError("Caption conditioning is enabled but caption modules are missing.")

        # 外部で事前計算した条件はキャッシュへ混ぜず、リクエスト内だけで使用する
        caption_state_override = req.caption_state_override
        caption_mask_override = req.caption_mask_override
        if (caption_state_override is None) != (caption_mask_override is None):
            raise ValueError(
                "caption_state_override and caption_mask_override must be specified together."
            )
        if caption_state_override is not None and caption_mask_override is not None:
            if caption_state_override.ndim != 3 or int(caption_state_override.shape[0]) != 1:
                raise ValueError(
                    "caption_state_override must have shape (1, tokens, dim), "
                    f"got {tuple(caption_state_override.shape)}."
                )
            if caption_mask_override.ndim != 2 or int(caption_mask_override.shape[0]) != 1:
                raise ValueError(
                    "caption_mask_override must have shape (1, tokens), "
                    f"got {tuple(caption_mask_override.shape)}."
                )
            if int(caption_state_override.shape[1]) != int(caption_mask_override.shape[1]):
                raise ValueError(
                    "caption override token length mismatch: "
                    f"state: {int(caption_state_override.shape[1])}, "
                    f"mask: {int(caption_mask_override.shape[1])}."
                )
            messages.append("info: using request caption conditioning override.")
            condition = _CaptionCondition(
                state=caption_state_override.detach().to(self.model_device),
                mask=caption_mask_override.detach().to(self.model_device),
            )
            return self._expand_caption_condition(condition, batch_size=batch_size)

        cache_key = self._caption_cache_key(
            req,
            caption_text,
            caption_max_len,
            resolved_condition_tokens=resolved_condition_tokens,
            lora_adapter=lora_adapter,
        )
        cached_state, cached_mask = self._get_cached_caption_condition(
            cache_key,
            batch_size=batch_size,
            messages=messages,
        )
        if cached_state is not None and cached_mask is not None:
            return cached_state, cached_mask

        # キャプションはリクエスト内の全チャンクで同一なので、1件分だけエンコードして保存する
        ## 候補数ぶんまとめてエンコードすると、キャッシュにも同じテンソルが重複して残ってしまう
        with torch.inference_mode():
            caption_state = self.model.caption_encoder(caption_ids[:1], caption_mask[:1])
            caption_state = self.model.caption_norm(caption_state)
        condition = _CaptionCondition(
            state=caption_state.detach(),
            mask=caption_mask[:1].detach(),
        )
        self._put_cached_caption_condition(cache_key, condition)
        messages.append("info: cached caption conditioning.")
        return self._expand_caption_condition(condition, batch_size=batch_size)

    def _load_reference_latent(
        self,
        *,
        req: SamplingRequest,
        resolved_condition_tokens: _ResolvedConditionTokens | None,
        lora_adapter: str | None,
        batch_size: int,
        messages: list[str],
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        runtime_dtype = next(self.model.parameters()).dtype
        if not self.model_cfg.use_speaker_condition_resolved:
            if req.ref_wav is not None or req.ref_latent is not None:
                messages.append(
                    "info: speaker conditioning is disabled for this checkpoint; ignoring reference input."
                )
            return None, None
        if req.no_ref:
            # cuDNN SDPA は系列長1の自己注意を扱えない環境がある
            ## no_ref は全マスク False のダミー参照を speaker_encoder に通すため、
            ## 話者条件ありの固定長パディングとは別に、cuDNN が選べる最小長を確保する
            ref_len = max(4, int(self.model_cfg.speaker_patch_size))
            ref_latent_patched = torch.zeros(
                (
                    batch_size,
                    ref_len,
                    self.model_cfg.latent_dim * self.model_cfg.latent_patch_size,
                ),
                device=self.model_device,
                dtype=runtime_dtype,
            )
            ref_mask = torch.zeros(
                (batch_size, ref_len), dtype=torch.bool, device=self.model_device
            )
            return ref_latent_patched, ref_mask

        if req.ref_wav is None and req.ref_latent is None:
            raise ValueError("Specify either ref_wav/ref_latent, or set no_ref=True.")

        cache_key = self._reference_cache_key(
            req,
            resolved_condition_tokens=resolved_condition_tokens,
            lora_adapter=lora_adapter,
        )
        cached_latent, cached_mask = self._get_cached_reference_condition(
            cache_key,
            batch_size=batch_size,
            messages=messages,
        )
        if cached_latent is not None and cached_mask is not None:
            return cached_latent, cached_mask

        max_ref_latent_steps = None
        if req.max_ref_seconds is not None and req.max_ref_seconds > 0:
            max_ref_latent_steps = max(
                1,
                math.ceil(
                    float(req.max_ref_seconds)
                    * float(self.codec.sample_rate)
                    / float(int(self.codec.model.hop_length))
                ),
            )

        if req.ref_latent is not None:
            latent_raw = torch.load(req.ref_latent, map_location="cpu", weights_only=True)
            ref_latent = _coerce_latent_shape(
                latent_raw, latent_dim=self.model_cfg.latent_dim
            ).unsqueeze(0)
            ref_latent = ref_latent.to(dtype=runtime_dtype)
        else:
            wav, sr = _load_audio(req.ref_wav)
            if req.max_ref_seconds is not None and req.max_ref_seconds > 0:
                max_ref_samples = max(1, int(float(req.max_ref_seconds) * float(sr)))
                if wav.shape[1] > max_ref_samples:
                    messages.append(
                        f"warning: reference audio exceeds max_ref_seconds ({req.max_ref_seconds}s). "
                        f"Trimming from {float(wav.shape[1]) / float(sr):.2f}s to {float(max_ref_samples) / float(sr):.2f}s."
                    )
                    wav = wav[:, :max_ref_samples]
            if req.ref_normalize_db is not None:
                messages.append(
                    f"info: reference loudness normalize enabled (target_db={float(req.ref_normalize_db):.2f}, includes peak safety scaling)."
                )
            elif req.ref_ensure_max:
                messages.append("info: reference peak safety scaling enabled (ensure_max=True).")
            ref_latent = self.codec.encode_waveform(
                wav.unsqueeze(0),
                sample_rate=int(sr),
                normalize_db=req.ref_normalize_db,
                ensure_max=bool(req.ref_ensure_max),
            ).cpu()

        if max_ref_latent_steps is not None and ref_latent.shape[1] > max_ref_latent_steps:
            messages.append(
                f"warning: reference latent steps ({ref_latent.shape[1]}) exceed max_ref_seconds bound ({max_ref_latent_steps} steps). "
                "Trimming reference latent."
            )
            ref_latent = ref_latent[:, :max_ref_latent_steps]

        ref_latent_patched = patchify_latent(ref_latent, self.model_cfg.latent_patch_size).to(
            device=self.model_device,
            dtype=runtime_dtype,
        )
        if ref_latent_patched.shape[1] == 0:
            raise ValueError(
                "Reference latent length became zero after patchify. Use longer reference audio."
            )

        # speaker_state の系列長を固定長またはバケットサイズにパディングする
        ## 参照音声ごとに speaker_state 長が異なると SDPA の K/V 形状が変わり、
        ## cuDNN が毎回カーネルを再探索して話者切替時に 600-900ms のペナルティが発生する
        actual_patches = int(ref_latent_patched.shape[1])
        target_patches = actual_patches
        if req.speaker_ref_fixed_length is not None:
            # パターンA: 全参照音声を固定長に統一
            target_patches = max(actual_patches, int(req.speaker_ref_fixed_length))
        elif req.speaker_ref_bucket_sizes is not None:
            # パターンB: 最寄りのバケットサイズにパディング
            for bucket_size in sorted(req.speaker_ref_bucket_sizes):
                if bucket_size >= actual_patches:
                    target_patches = bucket_size
                    break
            else:
                # 最大バケットより長い場合はそのまま (切詰めない)
                target_patches = actual_patches

        if target_patches > actual_patches:
            # ゼロパディングで拡張し、mask で有効範囲を記録
            padded = torch.zeros(
                (1, target_patches, ref_latent_patched.shape[2]),
                dtype=ref_latent_patched.dtype,
                device=ref_latent_patched.device,
            )
            padded[:, :actual_patches] = ref_latent_patched
            ref_latent_patched = padded
            ref_mask = torch.zeros(
                (1, target_patches),
                dtype=torch.bool,
                device=self.model_device,
            )
            ref_mask[:, :actual_patches] = True
            messages.append(
                f"[runtime] speaker ref padded: actual={actual_patches}, "
                f"target={target_patches}, mode="
                f"{'fixed' if req.speaker_ref_fixed_length is not None else 'bucket'}"
            )
        else:
            ref_mask = torch.ones(
                (1, ref_latent_patched.shape[1]),
                dtype=torch.bool,
                device=self.model_device,
            )

        condition = _ReferenceCondition(latent=ref_latent_patched, mask=ref_mask)
        self._put_cached_reference_condition(cache_key, condition)
        return self._expand_reference_condition(condition, batch_size=batch_size)

    def _load_speaker_embedding_condition(
        self,
        *,
        req: SamplingRequest,
        batch_size: int,
        messages: list[str],
    ) -> tuple[
        torch.Tensor | None,
        torch.Tensor | None,
    ]:
        if req.ref_embed is None:
            return None, None
        if not self.model_cfg.use_speaker_condition_resolved:
            messages.append(
                "info: speaker conditioning is disabled for this checkpoint; ignoring speaker embedding."
            )
            return None, None
        if req.ref_wav is not None or req.ref_latent is not None or req.no_ref:
            raise ValueError(
                "ref_embed/--ref-embed cannot be combined with ref_wav/ref_latent/no_ref. "
                "Use exactly one speaker conditioning source."
            )

        runtime_dtype = next(self.model.parameters()).dtype
        speaker_embedding = load_speaker_inversion_payload(req.ref_embed)["speaker_embedding"]
        state, mask = speaker_inversion_batch_tensors(
            speaker_embedding,
            batch_size=batch_size,
            device=self.model_device,
            dtype=runtime_dtype,
        )
        messages.append(
            "info: using speaker inversion embedding "
            f"tokens={state.shape[1]} uncond_mode={req.speaker_uncond_mode}."
        )
        return state, mask

    def synthesize(
        self,
        req: SamplingRequest,
        *,
        log_fn: Callable[[str], None] | None = None,
    ) -> SamplingResult:
        def _log(msg: str) -> None:
            if log_fn is not None:
                log_fn(msg)

        messages: list[str] = []
        _log(
            (
                "[runtime] start synthesize "
                "model_device={} model_precision={} codec_device={} codec_precision={} "
                "silentcipher_watermark={} mode={} seconds={} steps={} seed={} candidates={} decode_mode={}"
            ).format(
                self.key.model_device,
                self.key.model_precision,
                self.key.codec_device,
                self.key.codec_precision,
                self.watermarker is not None and self.watermarker.ready,
                req.cfg_guidance_mode,
                req.seconds,
                req.num_steps,
                "random" if req.seed is None else int(req.seed),
                req.num_candidates,
                req.decode_mode,
            )
        )

        manual_seconds = None if req.seconds is None else float(req.seconds)
        if manual_seconds is not None and manual_seconds <= 0:
            raise ValueError(f"seconds must be > 0 when provided, got {req.seconds}")
        duration_scale = float(req.duration_scale)
        if duration_scale <= 0:
            raise ValueError(f"duration_scale must be > 0, got {duration_scale}")
        min_seconds = float(req.min_seconds)
        max_seconds = float(req.max_seconds)
        if min_seconds <= 0:
            raise ValueError(f"min_seconds must be > 0, got {min_seconds}")
        if max_seconds < min_seconds:
            raise ValueError(
                f"max_seconds must be >= min_seconds, got min={min_seconds} max={max_seconds}"
            )
        num_candidates = int(req.num_candidates)
        if num_candidates <= 0:
            raise ValueError(f"num_candidates must be > 0, got {num_candidates}")
        decode_mode = str(req.decode_mode).strip().lower()
        if decode_mode not in {"sequential", "batch"}:
            raise ValueError(
                f"Unsupported decode_mode={req.decode_mode!r}. Expected one of: sequential, batch."
            )

        raw_text = str(req.text)
        normalized_text = normalize_text(raw_text).strip()
        if normalized_text == "":
            raise ValueError("text became empty after normalization.")

        text_max_len = (
            self.default_text_max_len if req.max_text_len is None else int(req.max_text_len)
        )
        if text_max_len <= 0:
            raise ValueError(f"max_text_len must be > 0, got {text_max_len}")
        caption_max_len = (
            self.default_caption_max_len
            if req.max_caption_len is None
            else int(req.max_caption_len)
        )
        if self.model_cfg.use_caption_condition and caption_max_len <= 0:
            raise ValueError(f"max_caption_len must be > 0, got {caption_max_len}")
        has_caption_text = bool(
            self.model_cfg.use_caption_condition
            and req.caption is not None
            and str(req.caption).strip() != ""
        )

        truncation_factor = None if req.truncation_factor is None else float(req.truncation_factor)
        rescale_k = None if req.rescale_k is None else float(req.rescale_k)
        rescale_sigma = None if req.rescale_sigma is None else float(req.rescale_sigma)
        if truncation_factor is not None and truncation_factor <= 0:
            raise ValueError(f"truncation_factor must be > 0, got {truncation_factor}")
        if (rescale_k is None) != (rescale_sigma is None):
            raise ValueError("rescale_k and rescale_sigma must be set together.")
        if rescale_k is not None and rescale_k <= 0:
            raise ValueError(f"rescale_k must be > 0, got {rescale_k}")
        if rescale_sigma is not None and rescale_sigma <= 0:
            raise ValueError(f"rescale_sigma must be > 0, got {rescale_sigma}")

        speaker_kv_scale = None if req.speaker_kv_scale is None else float(req.speaker_kv_scale)
        speaker_kv_min_t = None
        speaker_kv_max_layers = (
            None if req.speaker_kv_max_layers is None else int(req.speaker_kv_max_layers)
        )
        use_speaker_for_request = bool(
            self.model_cfg.use_speaker_condition_resolved and not req.no_ref
        )
        if speaker_kv_scale is not None:
            if not use_speaker_for_request:
                messages.append(
                    "info: speaker conditioning is disabled for this request; ignoring speaker_kv_scale."
                )
                speaker_kv_scale = None
            else:
                if speaker_kv_scale <= 0:
                    raise ValueError(f"speaker_kv_scale must be > 0, got {speaker_kv_scale}")
                speaker_kv_min_t = (
                    0.9 if req.speaker_kv_min_t is None else float(req.speaker_kv_min_t)
                )
                if not (0.0 <= speaker_kv_min_t <= 1.0):
                    raise ValueError(f"speaker_kv_min_t must be in [0, 1], got {speaker_kv_min_t}")
                if speaker_kv_max_layers is not None and speaker_kv_max_layers < 0:
                    raise ValueError(
                        f"speaker_kv_max_layers must be >= 0 when specified, got {speaker_kv_max_layers}"
                    )

        cfg_mode = str(req.cfg_guidance_mode).strip().lower()
        if cfg_mode not in {"independent", "joint", "alternating"}:
            raise ValueError(
                f"Unsupported cfg_guidance_mode={req.cfg_guidance_mode!r}. "
                "Expected one of: independent, joint, alternating."
            )

        lora_adapter = self._resolve_lora_adapter_path(req.lora_adapter)
        resolved_condition_tokens = self._resolve_condition_tokens(
            req,
            lora_adapter=lora_adapter,
        )
        (
            cfg_scale_text,
            cfg_scale_caption,
            cfg_scale_speaker,
            cfg_scale_condition,
            scale_messages,
        ) = resolve_cfg_scales(
            cfg_guidance_mode=cfg_mode,
            cfg_scale_text=req.cfg_scale_text,
            cfg_scale_caption=req.cfg_scale_caption,
            cfg_scale_speaker=req.cfg_scale_speaker,
            cfg_scale_condition=req.cfg_scale_condition,
            cfg_scale=req.cfg_scale,
            use_caption_condition=has_caption_text,
            use_speaker_condition=use_speaker_for_request,
            use_token_condition=resolved_condition_tokens is not None,
        )
        messages.extend(scale_messages)
        for msg in scale_messages:
            _log(msg)

        stage_timings: list[tuple[str, float]] = []
        if req.seed is None:
            used_seed = int(secrets.randbits(63))
            msg = f"info: seed not specified; using random seed {used_seed}."
            messages.append(msg)
            _log(msg)
        else:
            used_seed = int(req.seed)
            _log(f"[runtime] using seed: {used_seed}")
        post_load_t0 = _measure_start(self.model_device, self.codec_device)

        with (
            self._infer_lock,
            self._prepare_lora_for_request(
                req.lora_adapter,
                messages=messages,
                stage_timings=stage_timings,
                log_fn=_log,
            ),
            torch.inference_mode(),
        ):
            t0 = _measure_start(self.model_device)
            text_ids, text_mask = self.tokenizer.batch_encode(
                [normalized_text] * num_candidates,
                max_length=text_max_len,
            )
            stage_sec = _measure_end(self.model_device, t0)
            stage_timings.append(("tokenize_text", stage_sec))
            _log(f"[runtime] tokenize_text: {stage_sec * 1000.0:.1f} ms")
            text_ids = text_ids.to(self.model_device)
            text_mask = text_mask.to(self.model_device)
            condition_token_ids = None
            condition_token_mask = None
            condition_token_scales = None
            if resolved_condition_tokens is not None:
                condition_token_ids = torch.tensor(
                    [list(resolved_condition_tokens.token_ids)] * num_candidates,
                    dtype=torch.long,
                    device=self.model_device,
                )
                condition_token_mask = torch.ones_like(condition_token_ids, dtype=torch.bool)
                condition_token_scales = torch.tensor(
                    [list(resolved_condition_tokens.token_scales)] * num_candidates,
                    dtype=self._model_dtype,
                    device=self.model_device,
                )
            caption_ids = None
            caption_mask = None
            caption_state_override = None
            caption_mask_override = None
            if self.model_cfg.use_caption_condition:
                if self.caption_tokenizer is None:
                    raise RuntimeError(
                        "Caption conditioning is enabled but caption tokenizer is not loaded."
                    )
                caption_text = "" if req.caption is None else str(req.caption).strip()
                msg_count_before_caption = len(messages)
                caption_ids, caption_mask = self.caption_tokenizer.batch_encode(
                    [caption_text] * num_candidates,
                    max_length=caption_max_len,
                )
                if caption_text == "":
                    caption_mask.zero_()
                caption_ids = caption_ids.to(self.model_device)
                caption_mask = caption_mask.to(self.model_device)
                if caption_text != "":
                    caption_state_override, caption_mask_override = (
                        self._load_cached_caption_condition(
                            req=req,
                            resolved_condition_tokens=resolved_condition_tokens,
                            lora_adapter=lora_adapter,
                            caption_text=caption_text,
                            caption_ids=caption_ids,
                            caption_mask=caption_mask,
                            caption_max_len=caption_max_len,
                            batch_size=num_candidates,
                            messages=messages,
                        )
                    )
                for msg in messages[msg_count_before_caption:]:
                    _log(msg)

            t0 = _measure_start(self.model_device, self.codec_device)
            msg_count_before_ref = len(messages)
            # チャンク間で渡された話者状態は参照ファイルの読み込みやダミー参照生成より優先する
            if req.speaker_condition_override is not None:
                speaker_condition_override = req.speaker_condition_override
                if speaker_condition_override.state.ndim != 3:
                    raise ValueError(
                        "speaker condition state must have shape (batch, tokens, dim), "
                        f"got {tuple(speaker_condition_override.state.shape)}."
                    )
                if speaker_condition_override.mask.ndim != 2:
                    raise ValueError(
                        "speaker condition mask must have shape (batch, tokens), "
                        f"got {tuple(speaker_condition_override.mask.shape)}."
                    )
                if (
                    int(speaker_condition_override.state.shape[0]) != 1
                    or int(speaker_condition_override.mask.shape[0]) != 1
                ):
                    raise ValueError("speaker condition override must contain one source item.")
                if int(speaker_condition_override.state.shape[1]) != int(
                    speaker_condition_override.mask.shape[1]
                ):
                    raise ValueError(
                        "speaker condition override token length mismatch: "
                        f"state: {int(speaker_condition_override.state.shape[1])}, "
                        f"mask: {int(speaker_condition_override.mask.shape[1])}."
                    )
                speaker_state_override, speaker_mask_override = self._expand_speaker_condition(
                    _SpeakerCondition(
                        state=speaker_condition_override.state.detach().to(
                            device=self.model_device,
                            dtype=self._model_dtype,
                        ),
                        mask=speaker_condition_override.mask.detach().to(
                            device=self.model_device,
                            dtype=torch.bool,
                        ),
                    ),
                    batch_size=num_candidates,
                )
                messages.append("info: using request speaker conditioning override.")
            else:
                (
                    speaker_state_override,
                    speaker_mask_override,
                ) = self._load_speaker_embedding_condition(
                    req=req,
                    batch_size=num_candidates,
                    messages=messages,
                )
            if speaker_state_override is None:
                ref_latent = None
                ref_mask = None
                # 起動時または過去の要求で作成済みなら speaker_state を直接再利用する
                (
                    cached_speaker_state,
                    cached_speaker_mask,
                ) = self._load_cached_speaker_condition(
                    req=req,
                    resolved_condition_tokens=resolved_condition_tokens,
                    lora_adapter=lora_adapter,
                    ref_latent=None,
                    ref_mask=None,
                    batch_size=num_candidates,
                    messages=messages,
                )
                if cached_speaker_state is None or cached_speaker_mask is None:
                    ref_latent, ref_mask = self._load_reference_latent(
                        req=req,
                        resolved_condition_tokens=resolved_condition_tokens,
                        lora_adapter=lora_adapter,
                        batch_size=num_candidates,
                        messages=messages,
                    )
                    (
                        cached_speaker_state,
                        cached_speaker_mask,
                    ) = self._load_cached_speaker_condition(
                        req=req,
                        resolved_condition_tokens=resolved_condition_tokens,
                        lora_adapter=lora_adapter,
                        ref_latent=ref_latent,
                        ref_mask=ref_mask,
                        batch_size=num_candidates,
                        messages=messages,
                    )
                if cached_speaker_state is not None and cached_speaker_mask is not None:
                    speaker_state_override = cached_speaker_state
                    speaker_mask_override = cached_speaker_mask
                    ref_latent, ref_mask = None, None
                else:
                    # 直前のキャッシュ生成に使った参照 latent をそのまま推論へ渡す
                    if ref_latent is None or ref_mask is None:
                        raise RuntimeError("Reference latent is missing after cache construction.")
            else:
                ref_latent, ref_mask = None, None
            stage_sec = _measure_end(self.model_device, t0, self.codec_device)
            stage_timings.append(("prepare_reference", stage_sec))
            for msg in messages[msg_count_before_ref:]:
                _log(msg)
            _log(f"[runtime] prepare_reference: {stage_sec * 1000.0:.1f} ms")

            hop_length = int(self.codec.model.hop_length)
            encoded_conditions = None
            encoded_conditions_without_condition_tokens = None
            if manual_seconds is not None:
                clamped_seconds = min(max_seconds, max(min_seconds, manual_seconds))
                if clamped_seconds != manual_seconds:
                    duration_msg = (
                        f"warning: manual duration {manual_seconds:.3f}s was clamped to "
                        f"{clamped_seconds:.3f}s."
                    )
                    messages.append(duration_msg)
                    _log(duration_msg)
                target_samples = max(1, int(clamped_seconds * self.codec.sample_rate))
                latent_steps = math.ceil(target_samples / hop_length)
                pred_frames = None
                scaled_frames = None
                rounded_frames = int(latent_steps)
                min_frames = max(1, math.ceil(min_seconds * self.codec.sample_rate / hop_length))
                max_frames = max(1, math.floor(max_seconds * self.codec.sample_rate / hop_length))
                duration_was_clamped = clamped_seconds != manual_seconds
                duration_msg = f"info: using manual duration {clamped_seconds:.3f}s."
                messages.append(duration_msg)
                _log(duration_msg)
                t0 = _measure_start(self.model_device)
                encoded_conditions = self.model.encode_conditions(
                    text_input_ids=text_ids,
                    text_mask=text_mask,
                    ref_latent=ref_latent,
                    ref_mask=ref_mask,
                    caption_input_ids=caption_ids,
                    caption_mask=caption_mask,
                    speaker_state_override=speaker_state_override,
                    speaker_mask_override=speaker_mask_override,
                    caption_state_override=caption_state_override,
                    caption_mask_override=caption_mask_override,
                    speaker_uncond_mode=req.speaker_uncond_mode,
                    condition_token_ids=condition_token_ids,
                    condition_token_mask=condition_token_mask,
                    condition_token_scales=condition_token_scales,
                )
                if condition_token_ids is not None and cfg_scale_condition > 0.0:
                    encoded_conditions_without_condition_tokens = self.model.encode_conditions(
                        text_input_ids=text_ids,
                        text_mask=text_mask,
                        ref_latent=ref_latent,
                        ref_mask=ref_mask,
                        caption_input_ids=caption_ids,
                        caption_mask=caption_mask,
                        speaker_state_override=speaker_state_override,
                        speaker_mask_override=speaker_mask_override,
                        caption_state_override=caption_state_override,
                        caption_mask_override=caption_mask_override,
                        speaker_uncond_mode=req.speaker_uncond_mode,
                    )
                stage_sec = _measure_end(self.model_device, t0)
                stage_timings.append(("encode_conditions", stage_sec))
                _log(f"[runtime] encode_conditions: {stage_sec * 1000.0:.1f} ms")
            elif self.model_cfg.use_duration_predictor:
                t0 = _measure_start(self.model_device)
                has_speaker_duration = torch.zeros(
                    (num_candidates,), dtype=torch.bool, device=self.model_device
                )
                if speaker_mask_override is not None:
                    has_speaker_duration = speaker_mask_override.any(dim=1)
                elif self.model_cfg.use_speaker_condition_resolved and ref_mask is not None:
                    has_speaker_duration = ref_mask.any(dim=1)
                duration_features = build_duration_features(
                    [normalized_text] * num_candidates,
                    token_counts=text_mask.sum(dim=1),
                    max_text_len=text_max_len,
                    has_speaker=has_speaker_duration,
                ).to(self.model_device)
                encoded_conditions = self.model.encode_conditions(
                    text_input_ids=text_ids,
                    text_mask=text_mask,
                    ref_latent=ref_latent,
                    ref_mask=ref_mask,
                    caption_input_ids=caption_ids,
                    caption_mask=caption_mask,
                    condition_token_ids=condition_token_ids,
                    condition_token_mask=condition_token_mask,
                    condition_token_scales=condition_token_scales,
                    speaker_state_override=speaker_state_override,
                    speaker_mask_override=speaker_mask_override,
                    caption_state_override=caption_state_override,
                    caption_mask_override=caption_mask_override,
                    speaker_uncond_mode=req.speaker_uncond_mode,
                )
                (
                    duration_text_state,
                    duration_text_mask,
                    duration_speaker_state,
                    _duration_speaker_mask,
                    _duration_caption_state,
                    _duration_caption_mask,
                ) = encoded_conditions
                if condition_token_ids is not None and cfg_scale_condition > 0.0:
                    encoded_conditions_without_condition_tokens = self.model.encode_conditions(
                        text_input_ids=text_ids,
                        text_mask=text_mask,
                        ref_latent=ref_latent,
                        ref_mask=ref_mask,
                        caption_input_ids=caption_ids,
                        caption_mask=caption_mask,
                        speaker_state_override=speaker_state_override,
                        speaker_mask_override=speaker_mask_override,
                        caption_state_override=caption_state_override,
                        caption_mask_override=caption_mask_override,
                        speaker_uncond_mode=req.speaker_uncond_mode,
                    )
                pred_log_frames = self.model.predict_duration_log_frames(
                    text_state=duration_text_state,
                    text_mask=duration_text_mask,
                    speaker_state=duration_speaker_state,
                    speaker_mask=_duration_speaker_mask,
                    caption_state=_duration_caption_state,
                    caption_mask=_duration_caption_mask,
                    duration_features=duration_features,
                    has_speaker=has_speaker_duration,
                    has_caption=torch.full(
                        (num_candidates,),
                        has_caption_text,
                        dtype=torch.bool,
                        device=self.model_device,
                    )
                    if self.model_cfg.use_caption_condition
                    else None,
                )
                pred_frames = torch.expm1(pred_log_frames).float().mean().item()
                scaled_frames = pred_frames * duration_scale
                min_frames = max(1, math.ceil(min_seconds * self.codec.sample_rate / hop_length))
                max_frames = max(1, math.floor(max_seconds * self.codec.sample_rate / hop_length))
                rounded_frames = round(scaled_frames)
                latent_steps = max(min_frames, min(max_frames, rounded_frames))
                duration_was_clamped = latent_steps != rounded_frames
                target_samples = int(latent_steps * hop_length)
                stage_sec = _measure_end(self.model_device, t0)
                stage_timings.append(("predict_duration", stage_sec))
                msg = (
                    f"info: predicted duration frames={pred_frames:.1f}, "
                    f"scale={duration_scale:.3f}, using_frames={latent_steps} "
                    f"({target_samples / float(self.codec.sample_rate):.3f}s)."
                )
                messages.append(msg)
                _log(msg)
                _log(f"[runtime] predict_duration: {stage_sec * 1000.0:.1f} ms")
            else:
                fallback_seconds = 30.0
                target_samples = int(fallback_seconds * self.codec.sample_rate)
                latent_steps = math.ceil(target_samples / hop_length)
                pred_frames = None
                scaled_frames = None
                rounded_frames = None
                min_frames = None
                max_frames = None
                duration_was_clamped = False
                msg = "info: checkpoint has no duration predictor; falling back to 30.000s."
                messages.append(msg)
                _log(msg)
                # 固定 context 容量を持つ実装では、DurationPredictor の有無にかかわらず RF 前の条件検査が必要
                if self.max_encoded_context_tokens is not None:
                    encoded_conditions = self.model.encode_conditions(
                        text_input_ids=text_ids,
                        text_mask=text_mask,
                        ref_latent=ref_latent,
                        ref_mask=ref_mask,
                        caption_input_ids=caption_ids,
                        caption_mask=caption_mask,
                        speaker_state_override=speaker_state_override,
                        speaker_mask_override=speaker_mask_override,
                        caption_state_override=caption_state_override,
                        caption_mask_override=caption_mask_override,
                        speaker_uncond_mode=req.speaker_uncond_mode,
                        condition_token_ids=condition_token_ids,
                        condition_token_mask=condition_token_mask,
                        condition_token_scales=condition_token_scales,
                    )
                    if condition_token_ids is not None and cfg_scale_condition > 0.0:
                        encoded_conditions_without_condition_tokens = self.model.encode_conditions(
                            text_input_ids=text_ids,
                            text_mask=text_mask,
                            ref_latent=ref_latent,
                            ref_mask=ref_mask,
                            caption_input_ids=caption_ids,
                            caption_mask=caption_mask,
                            speaker_state_override=speaker_state_override,
                            speaker_mask_override=speaker_mask_override,
                            caption_state_override=caption_state_override,
                            caption_mask_override=caption_mask_override,
                            speaker_uncond_mode=req.speaker_uncond_mode,
                        )

            # TensorRT など固定コンテキスト容量を持つ実装では、K/V キャッシュや RF 本体へ入る前に超過を確定させる
            ## パディング後の shape ではなく mask の有効数を数えるため、通常参照・Speaker Inversion・no_ref を同じ契約で扱える
            if encoded_conditions is not None:
                self._validate_encoded_context_capacity(encoded_conditions)

            patched_steps = math.ceil(latent_steps / self.model_cfg.latent_patch_size)
            sampling_patched_steps = int(patched_steps)
            latent_mask = None
            bucket_multiple = (
                None
                if req.latent_patch_bucket_multiple is None
                else int(req.latent_patch_bucket_multiple)
            )
            if bucket_multiple is not None:
                if bucket_multiple <= 0:
                    raise ValueError(
                        "latent_patch_bucket_multiple must be > 0 when provided, "
                        f"got {req.latent_patch_bucket_multiple}"
                    )
                sampling_patched_steps = (
                    math.ceil(sampling_patched_steps / bucket_multiple) * bucket_multiple
                )
                if sampling_patched_steps != patched_steps:
                    latent_mask = torch.zeros(
                        (num_candidates, sampling_patched_steps),
                        dtype=torch.bool,
                        device=self.model_device,
                    )
                    latent_mask[:, :patched_steps] = True
                    msg = (
                        "info: latent patch bucket enabled "
                        f"(actual_patches={patched_steps}, sampling_patches={sampling_patched_steps}, "
                        f"multiple={bucket_multiple})."
                    )
                    messages.append(msg)
                    _log(msg)

            if isinstance(self.train_cfg, dict):
                fixed_steps = self.train_cfg.get("fixed_target_latent_steps")
                if isinstance(fixed_steps, int) and fixed_steps > 0 and latent_steps > fixed_steps:
                    msg = (
                        f"warning: requested latent length ({latent_steps}) exceeds fixed_target_latent_steps ({fixed_steps}) "
                        "used in training. Long-tail stability may degrade."
                    )
                    messages.append(msg)
                    _log(msg)

            t0 = _measure_start(self.model_device)
            noise_dtype = None
            if req.noise_precision is not None and str(req.noise_precision).strip().lower() not in {
                "",
                "model",
                "runtime",
            }:
                noise_dtype = resolve_runtime_dtype(
                    precision=str(req.noise_precision),
                    device=self.model_device,
                )

            def sample_latent(
                *,
                seed: int,
                initial_noise: torch.Tensor | None,
                initial_noise_offset: int,
            ) -> torch.Tensor:
                """
                事前計算済み条件を再利用して1試行分の RF 潜在を生成する。

                Args:
                    seed (int): 試行に使う乱数種
                    initial_noise (torch.Tensor | None): 呼び出し側で共有する初期ノイズ
                    initial_noise_offset (int): 共有初期ノイズ内の開始位置

                Returns:
                    torch.Tensor: パッチ化された生成潜在
                """

                return sample_euler_rf_cfg(
                    model=self.model,
                    text_input_ids=text_ids,
                    text_mask=text_mask,
                    ref_latent=ref_latent,
                    ref_mask=ref_mask,
                    sequence_length=sampling_patched_steps,
                    caption_input_ids=caption_ids,
                    caption_mask=caption_mask,
                    speaker_state_override=speaker_state_override,
                    speaker_mask_override=speaker_mask_override,
                    caption_state_override=caption_state_override,
                    caption_mask_override=caption_mask_override,
                    speaker_uncond_mode=req.speaker_uncond_mode,
                    num_steps=int(req.num_steps),
                    cfg_scale_text=cfg_scale_text,
                    cfg_scale_caption=cfg_scale_caption,
                    cfg_scale_speaker=cfg_scale_speaker,
                    cfg_scale_condition=cfg_scale_condition,
                    cfg_guidance_mode=cfg_mode,
                    cfg_min_t=float(req.cfg_min_t),
                    cfg_max_t=float(req.cfg_max_t),
                    seed=seed,
                    truncation_factor=truncation_factor,
                    rescale_k=rescale_k,
                    rescale_sigma=rescale_sigma,
                    use_context_kv_cache=bool(req.context_kv_cache),
                    speaker_kv_scale=speaker_kv_scale,
                    speaker_kv_max_layers=speaker_kv_max_layers,
                    speaker_kv_min_t=speaker_kv_min_t,
                    t_schedule_mode=str(req.t_schedule_mode),
                    sway_coeff=float(req.sway_coeff),
                    waveex=req.waveex,
                    encoded_conditions=encoded_conditions,
                    encoded_conditions_without_condition_tokens=(
                        encoded_conditions_without_condition_tokens
                    ),
                    noise_dtype=noise_dtype,
                    initial_noise=initial_noise,
                    initial_noise_offset=initial_noise_offset,
                    latent_mask=latent_mask,
                    condition_token_ids=condition_token_ids,
                    condition_token_mask=condition_token_mask,
                    condition_token_scales=condition_token_scales,
                    velocity_field_guidance=req.velocity_field_guidance,
                    trajectory_intervention=req.trajectory_intervention,
                    trajectory_observer=req.trajectory_observer,
                )

            z_patched = sample_latent(
                seed=used_seed,
                initial_noise=req.initial_noise,
                initial_noise_offset=int(req.initial_noise_offset),
            )
            stage_sec = _measure_end(self.model_device, t0)
            stage_timings.append(("sample_rf", stage_sec))
            if req.waveex is not None and req.waveex.enabled:
                ode_indices = sorted(req.waveex.resolve_ode_step_indices(int(req.num_steps)))
                msg = (
                    f"info: waveex enabled (wavelet={req.waveex.wavelet}, "
                    f"taylor_order={req.waveex.taylor_order}, "
                    f"history_size={req.waveex.history_size}, "
                    f"high_freq_mode={req.waveex.high_freq_mode}, "
                    f"ode_steps={ode_indices})."
                )
                messages.append(msg)
                _log(msg)
            _log(f"[runtime] sample_rf: {stage_sec * 1000.0:.1f} ms")

            generated_speaker_condition = None
            if req.capture_generated_speaker_condition is True:
                t0 = _measure_start(self.model_device)
                # RF の実長だけを Speaker Encoder へ渡し、バケットの末尾パディングを話者条件へ混ぜない
                generated_reference_latent = z_patched[:1, :patched_steps]
                generated_reference_mask = torch.ones(
                    generated_reference_latent.shape[:2],
                    dtype=torch.bool,
                    device=self.model_device,
                )
                generated_speaker_state, generated_speaker_mask = (
                    self.model.encode_speaker_condition(
                        batch_size=1,
                        dtype=generated_reference_latent.dtype,
                        device=self.model_device,
                        ref_latent=generated_reference_latent,
                        ref_mask=generated_reference_mask,
                    )
                )
                generated_speaker_condition = SpeakerCondition(
                    state=generated_speaker_state.detach(),
                    mask=generated_speaker_mask.detach(),
                )
                stage_sec = _measure_end(self.model_device, t0)
                stage_timings.append(("capture_speaker_condition", stage_sec))
                _log(f"[runtime] capture_speaker_condition: {stage_sec * 1000.0:.1f} ms")

            t0 = _measure_start(self.model_device)
            z = unpatchify_latent(
                z_patched,
                patch_size=self.model_cfg.latent_patch_size,
                latent_dim=self.model_cfg.latent_dim,
            )
            stage_sec = _measure_end(self.model_device, t0)
            stage_timings.append(("unpatchify_latent", stage_sec))
            _log(f"[runtime] unpatchify_latent: {stage_sec * 1000.0:.1f} ms")
            z = z[:, :latent_steps]

            base_seed = used_seed

            t0 = _measure_start(self.model_device, self.codec_device)
            hop_length = int(self.codec.model.hop_length)
            max_samples_per_candidate = torch.full(
                (num_candidates,),
                int(target_samples),
                dtype=torch.long,
                device=self.model_device,
            )
            if bool(req.trim_tail):
                flattening_points = find_flattening_points(
                    z,
                    window_size=max(1, int(req.tail_window_size)),
                    std_threshold=float(req.tail_std_threshold),
                    mean_threshold=float(req.tail_mean_threshold),
                )
                flattening_samples = flattening_points * hop_length
                max_samples_per_candidate = torch.where(
                    flattening_samples > 0,
                    torch.minimum(max_samples_per_candidate, flattening_samples),
                    max_samples_per_candidate,
                )
            max_samples_list = [
                int(max_samples)
                for max_samples in max_samples_per_candidate.detach().cpu().tolist()
            ]
            trimmed_audios: list[torch.Tensor] = []
            if decode_mode == "batch":
                audio_batch = self.codec.decode_latent(z).cpu()
                for i in range(num_candidates):
                    audio_i = audio_batch[i]
                    trimmed_audios.append(audio_i[:, : max_samples_list[i]])
            else:
                for i in range(num_candidates):
                    audio_i = self.codec.decode_latent(z[i : i + 1]).cpu()[0]
                    trimmed_audios.append(audio_i[:, : max_samples_list[i]])
            stage_sec = _measure_end(self.model_device, t0, self.codec_device)
            stage_timings.append(("decode_latent", stage_sec))
            _log(f"[runtime] decode_latent ({decode_mode}): {stage_sec * 1000.0:.1f} ms")

            if self.watermarker is not None and self.watermarker.ready:
                t0 = _measure_start(self.codec_device)
                trimmed_audios = self.watermarker.encode_batch(
                    trimmed_audios,
                    sample_rate=int(self.codec.sample_rate),
                )
                stage_sec = _measure_end(self.codec_device, t0)
                stage_timings.append(("silentcipher_watermark", stage_sec))
                _log(f"[runtime] silentcipher_watermark: {stage_sec * 1000.0:.1f} ms")
            elif self.watermarker is not None:
                msg = (
                    "warning: SilentCipher watermark is unavailable; generated audio was not "
                    "watermarked."
                )
                messages.append(msg)
                _log(msg)

            total_to_decode = _measure_end(self.model_device, post_load_t0, self.codec_device)
            _log(f"[runtime] total_to_decode: {total_to_decode:.3f} s")

        _log("[runtime] done synthesize")
        return SamplingResult(
            audio=trimmed_audios[0],
            audios=trimmed_audios,
            sample_rate=int(self.codec.sample_rate),
            stage_timings=stage_timings,
            total_to_decode=total_to_decode,
            used_seed=used_seed,
            messages=messages,
            latent_steps=int(latent_steps),
            patched_steps=int(patched_steps),
            sampling_patched_steps=int(sampling_patched_steps),
            predicted_duration_frames=None if pred_frames is None else float(pred_frames),
            scaled_duration_frames=None if scaled_frames is None else float(scaled_frames),
            rounded_duration_frames=None if rounded_frames is None else int(rounded_frames),
            min_duration_frames=None if min_frames is None else int(min_frames),
            max_duration_frames=None if max_frames is None else int(max_frames),
            duration_was_clamped=bool(duration_was_clamped),
            base_seed=base_seed,
            retry_seed=None,
            retry_attempts=1,
            seed_retry_base_score=None,
            seed_retry_candidate_score=None,
            is_retry_adopted=False,
            speaker_condition=generated_speaker_condition,
        )

    def unload(self) -> None:
        del self.model
        del self.tokenizer
        del self.codec
        gc.collect()
        for device in (self.model_device, self.codec_device):
            if device.type == "cuda":
                torch.cuda.empty_cache()
            elif device.type == "mps":
                mps = getattr(torch, "mps", None)
                if mps is not None and hasattr(mps, "empty_cache"):
                    mps.empty_cache()
            elif device.type == "xpu":
                xpu = getattr(torch, "xpu", None)
                if xpu is not None and hasattr(xpu, "empty_cache"):
                    xpu.empty_cache()


_RUNTIME_CACHE_LOCK = threading.Lock()
_RUNTIME_CACHE_KEY: RuntimeKey | None = None
_RUNTIME_CACHE_VALUE: InferenceRuntime | None = None


def get_cached_runtime(key: RuntimeKey) -> tuple[InferenceRuntime, bool]:
    global _RUNTIME_CACHE_KEY, _RUNTIME_CACHE_VALUE
    with _RUNTIME_CACHE_LOCK:
        if _RUNTIME_CACHE_VALUE is not None and _RUNTIME_CACHE_KEY == key:
            return _RUNTIME_CACHE_VALUE, False

        old_runtime = _RUNTIME_CACHE_VALUE
        runtime = InferenceRuntime.from_key(key)
        _RUNTIME_CACHE_KEY = key
        _RUNTIME_CACHE_VALUE = runtime

    if old_runtime is not None:
        old_runtime.unload()

    return runtime, True


def clear_cached_runtime() -> None:
    global _RUNTIME_CACHE_KEY, _RUNTIME_CACHE_VALUE
    with _RUNTIME_CACHE_LOCK:
        runtime = _RUNTIME_CACHE_VALUE
        _RUNTIME_CACHE_KEY = None
        _RUNTIME_CACHE_VALUE = None

    if runtime is not None:
        runtime.unload()


def _load_audio(path: str | Path) -> tuple[torch.Tensor, int]:
    # SoundFile の (frame, channel) を推論内部の (channel, frame) へ変換する
    data, sample_rate = sf.read(str(path), dtype="float32", always_2d=True)
    waveform = torch.from_numpy(np.ascontiguousarray(data.T))
    return waveform, int(sample_rate)


def save_wav(path: str | Path, audio: torch.Tensor, sample_rate: int) -> Path:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    audio_cpu = audio.detach().to(device="cpu", dtype=torch.float32)
    audio_np = audio_cpu.squeeze(0).numpy() if audio_cpu.shape[0] == 1 else audio_cpu.T.numpy()
    # Torchaudio の既定と揃え、FLAC は PCM24、WAV は libsndfile 既定の PCM16 で保存する
    output_subtype = "PCM_24" if out_path.suffix.lower() == ".flac" else "PCM_16"
    sf.write(str(out_path), audio_np, sample_rate, subtype=output_subtype)
    return out_path
