#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import shutil
import sys
import warnings
from contextlib import nullcontext
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader as TorchDataLoader
from torch.utils.data import DistributedSampler, Sampler
from torchdata.stateful_dataloader import StatefulDataLoader
from torchdata.stateful_dataloader.sampler import StatefulDistributedSampler

from irodori_tts.codec import DACVAE_LATENT_FRAMES_PER_SECOND
from irodori_tts.config import (
    ModelConfig,
    TrainConfig,
    dump_configs,
    load_config_yaml,
    merge_dataclass_overrides,
)
from irodori_tts.dataset import LatentTextDataset, TTSCollator, build_manifest_index
from irodori_tts.duration import set_duration_has_speaker_feature
from irodori_tts.lora import (
    LORA_METADATA_NAME,
    LORA_TARGET_PRESETS,
    LORA_TRAIN_CONFIG_FIELDS,
    LORA_TRAINER_STATE_NAME,
    apply_lora,
    count_parameters,
    is_lora_adapter_dir,
    load_lora_adapter,
    train_config_uses_lora,
)
from irodori_tts.model import (
    DURATION_ARCHITECTURES,
    DURATION_CAPTION_FUSIONS,
    DURATION_CAPTION_POOLINGS,
    DURATION_SPEAKER_FUSIONS,
    TextToLatentRFDiT,
)
from irodori_tts.optim import (
    build_optimizer,
    build_scheduler,
    current_lr,
    current_pretrained_text_encoder_lr,
)
from irodori_tts.progress import TrainProgress
from irodori_tts.rf import (
    rf_interpolate,
    rf_velocity_target,
    sample_logit_normal_t,
    sample_stratified_logit_normal_t,
)
from irodori_tts.speaker_inversion import (
    SPEAKER_EMBEDDING_KEY,
    SPEAKER_INVERSION_SAFETENSORS_SUFFIX,
    SPEAKER_PRE_NORM_EMBEDDING_KEY,
    SpeakerInversionEmbedding,
    load_speaker_inversion_base_payload,
    load_speaker_inversion_payload,
    save_speaker_inversion_checkpoint,
    speaker_inversion_state_dict,
)
from irodori_tts.tokenizer import PretrainedTextTokenizer


WANDB_MODES = {"online", "offline", "disabled"}
TRAIN_MODES = {"rf", "duration_only"}
CHECKPOINT_STEP_RE = re.compile(
    rf"^checkpoint_(\d+)(?:\.pt|{re.escape(SPEAKER_INVERSION_SAFETENSORS_SUFFIX)})?$"
)
CHECKPOINT_BEST_VAL_LOSS_RE = re.compile(
    rf"^checkpoint_best_val_loss_(\d+)_(-?\d+(?:\.\d+)?)"
    rf"(?:\.pt|{re.escape(SPEAKER_INVERSION_SAFETENSORS_SUFFIX)})?$"
)
DATALOADER_STATE_KEY = "dataloader_state"
RUNTIME_STATE_KEY = "runtime_state"
RNG_STATE_KEY = "rng_state"
SPEAKER_INVERSION_MODULE_STATE_KEY = "speaker_inversion_module_state"
SPEAKER_INVERSION_RESUME_CONTRACT_KEY = "speaker_inversion_resume_contract"
SPEAKER_INVERSION_TRAINER_STATE_SUFFIX = ".speaker.trainer.pt"
SAFETENSORS_CONFIG_META_KEY = "config_json"
SAFETENSORS_TEXT_ENCODER_CONFIG_META_KEY = "text_encoder_config_json"
SAFETENSORS_INFERENCE_CONFIG_KEYS = {
    "max_text_len",
    "max_caption_len",
    "fixed_target_latent_steps",
    "ref_max_seconds",
}
DURATION_CONDITION_GROUPS = (
    "speaker",
    "no_speaker",
    "caption",
    "no_caption",
    "speaker_caption",
    "speaker_no_caption",
    "no_speaker_caption",
    "no_speaker_no_caption",
)
DURATION_CONDITION_GROUP_TOTAL_SIZE = len(DURATION_CONDITION_GROUPS) * 3
SPEAKER_INVERSION_RESUME_MUTABLE_TRAIN_FIELDS = {
    "checkpoint_best_n",
    "log_every",
    "max_steps",
    "output_dir",
    "progress",
    "progress_all_ranks",
    "save_every",
    "speaker_inversion_base_embedding",
    "speaker_inversion_init_embedding",
    "valid_every",
    "wandb_enabled",
    "wandb_entity",
    "wandb_mode",
    "wandb_project",
    "wandb_run_name",
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _capture_rng_state() -> dict[str, Any]:
    """Capture random-number generator states needed for exact training resume."""

    numpy_state = np.random.get_state()
    return {
        "python": random.getstate(),
        # weights_only=True で安全に読める基本型と Tensor だけへ変換する
        "numpy": {
            "bit_generator": numpy_state[0],
            "state": torch.from_numpy(numpy_state[1].copy()),
            "position": int(numpy_state[2]),
            "has_gauss": int(numpy_state[3]),
            "cached_gaussian": float(numpy_state[4]),
        },
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _restore_rng_state(payload: dict[str, Any]) -> None:
    """Restore random-number generator states from a training checkpoint."""

    random.setstate(payload["python"])
    numpy_state = payload["numpy"]
    if not isinstance(numpy_state, dict):
        raise ValueError("Checkpoint NumPy rng_state must be a dictionary.")
    np.random.set_state(
        (
            str(numpy_state["bit_generator"]),
            numpy_state["state"].cpu().numpy(),
            int(numpy_state["position"]),
            int(numpy_state["has_gauss"]),
            float(numpy_state["cached_gaussian"]),
        )
    )
    torch.set_rng_state(payload["torch"])
    cuda_state = payload["cuda"]
    if cuda_state is not None and torch.cuda.is_available():
        # 保存時と現在の CUDA デバイス数が違うと set_rng_state_all が失敗するためスキップする
        saved_device_count = len(cuda_state)
        current_device_count = int(torch.cuda.device_count())
        if saved_device_count != current_device_count:
            warnings.warn(
                "Skipping CUDA RNG restore because device count changed: "
                f"checkpoint={saved_device_count} current={current_device_count}",
                stacklevel=2,
            )
        else:
            torch.cuda.set_rng_state_all(cuda_state)


def _collect_rng_state(
    *,
    distributed: bool,
    rank: int,
    world_size: int,
) -> dict[str, Any]:
    """Gather per-rank RNG states so distributed resume keeps rank-local randomness."""

    local_state = _capture_rng_state()
    if not distributed:
        return {
            "version": 1,
            "world_size": 1,
            "rank_states": [local_state],
        }

    rank_states: list[dict[str, Any] | None] = [None for _ in range(world_size)]
    dist.all_gather_object(rank_states, local_state)
    return {
        "version": 1,
        "world_size": int(world_size),
        "rank_states": rank_states,
        "saved_by_rank": int(rank),
    }


def _select_rng_state_for_rank(
    payload: dict[str, Any],
    *,
    distributed: bool,
    rank: int,
    world_size: int,
) -> dict[str, Any] | None:
    """
    Select the RNG payload for the current rank.

    Legacy checkpoints store a flat RNG dict captured on rank 0. Newer checkpoints
    store per-rank states under ``rank_states``, matching dataloader resume.
    """

    state = payload.get(RNG_STATE_KEY)
    if state is None:
        return None
    if not isinstance(state, dict):
        raise ValueError("Checkpoint rng_state must be a dictionary when present.")

    # 旧形式は rank 0 の単一状態だけを持つため、全 rank が同じ状態を復元する
    if "rank_states" not in state and "torch" in state:
        return state

    rank_states = state.get("rank_states")
    if not isinstance(rank_states, list):
        raise ValueError("Checkpoint rng_state.rank_states must be a list.")
    saved_world_size = int(state.get("world_size", len(rank_states)))
    expected_world_size = int(world_size) if distributed else 1
    if saved_world_size != expected_world_size or len(rank_states) != expected_world_size:
        warnings.warn(
            "Discarding rng_state because world_size changed: "
            f"checkpoint={saved_world_size} current={expected_world_size}"
        )
        return None
    state_rank = int(rank) if distributed else 0
    rank_state = rank_states[state_rank]
    if rank_state is not None and not isinstance(rank_state, dict):
        raise ValueError(f"Checkpoint rng_state for rank {state_rank} must be a dictionary.")
    return rank_state


def _speaker_inversion_trainer_state_path(path: str | Path) -> Path:
    """Return the trainer-state sidecar path for a Speaker Inversion embedding."""

    embedding_path = Path(path)
    suffix = SPEAKER_INVERSION_SAFETENSORS_SUFFIX
    if not embedding_path.name.endswith(suffix):
        raise ValueError(f"Speaker Inversion checkpoint must end with {suffix}: {embedding_path}")
    stem = embedding_path.name[: -len(suffix)]
    return embedding_path.with_name(f"{stem}{SPEAKER_INVERSION_TRAINER_STATE_SUFFIX}")


def restore_speaker_inversion_training_state(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    payload: dict[str, Any],
    *,
    distributed: bool,
    rank: int,
    world_size: int,
) -> int:
    """
    Speaker Inversion の学習再開に必要な更新状態を復元する。

    Args:
        model (torch.nn.Module): Speaker Inversion モジュールを持つ学習モデル
        optimizer (torch.optim.Optimizer): 状態を復元する optimizer
        scheduler (torch.optim.lr_scheduler.LRScheduler | None): 状態を復元する scheduler
        payload (dict[str, Any]): `.speaker.trainer.pt` から読み込んだ保存状態
        distributed (bool): 分散学習として再開するか
        rank (int): 現在の分散 rank
        world_size (int): 現在の分散プロセス数

    Returns:
        int: 再開する学習 step

    Raises:
        ValueError: Speaker Inversion モジュールまたは保存状態が不正な場合
    """

    speaker_inversion = getattr(model, "speaker_inversion", None)
    if not isinstance(speaker_inversion, SpeakerInversionEmbedding):
        raise ValueError("Speaker Inversion resume requires an initialized embedding module.")
    module_state = payload.get(SPEAKER_INVERSION_MODULE_STATE_KEY)
    if not isinstance(module_state, dict):
        raise ValueError("Speaker Inversion resume requires its saved module state.")

    # 埋め込み本体と optimizer を同じ保存時点へ戻し、更新量の食い違いを防ぐ
    speaker_inversion.load_state_dict(module_state)
    optimizer.load_state_dict(payload["optimizer"])
    step = int(payload["step"])

    # scheduler が保存されている場合は内部カウンタも同じ時点へ戻す
    if scheduler is not None:
        scheduler_state = payload.get("scheduler")
        if scheduler_state is not None:
            scheduler.load_state_dict(scheduler_state)
        elif step > 0:
            scheduler.last_step = step

    # 乱数状態は次のミニバッチ更新より前に復元する
    rng_state = _select_rng_state_for_rank(
        payload,
        distributed=distributed,
        rank=rank,
        world_size=world_size,
    )
    if rng_state is not None:
        _restore_rng_state(rng_state)
    return step


def echo_style_masked_mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    loss_mask: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Echo/JAX-style diffusion loss:
    - take mean squared error over loss_masked tokens
    - divide by mean valid-token ratio (short samples get up-weighted)

    If loss_mask == valid_mask, this reduces to standard masked MSE.
    """
    diff = (pred - target) ** 2
    diff = diff.mean(dim=-1)  # (B, S)
    loss_weight = loss_mask.float()
    valid_weight = valid_mask.float()

    # Keep normalization stable for degenerate samples with no valid target tokens.
    has_valid = (valid_weight.sum(dim=-1) > 0).float()[:, None]
    denom = (loss_weight * valid_weight * has_valid).mean().clamp_min(1e-6)
    return (diff * loss_weight).mean() / denom


def utterance_mean_masked_mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    diff = (pred - target) ** 2
    diff = diff.mean(dim=-1)
    weight = valid_mask.float()
    per_sample = (diff * weight).sum(dim=-1) / weight.sum(dim=-1).clamp_min(1.0)
    return per_sample.mean()


def compute_rf_loss(
    *,
    pred: torch.Tensor,
    target: torch.Tensor,
    loss_mask: torch.Tensor,
    valid_mask: torch.Tensor,
    mode: str,
) -> torch.Tensor:
    mode = str(mode).strip().lower()
    if mode == "echo":
        return echo_style_masked_mse(
            pred,
            target,
            loss_mask=loss_mask,
            valid_mask=valid_mask,
        )
    if mode == "utterance_mean":
        return utterance_mean_masked_mse(pred, target, valid_mask=valid_mask)
    raise ValueError(f"Unsupported rf_loss_mode={mode!r}. Expected 'echo' or 'utterance_mean'.")


def sha256_file(path: str | Path) -> str:
    """
    ファイル内容の SHA-256 を計算する。

    Args:
        path (str | Path): ハッシュを計算するファイル

    Returns:
        str: 16進表記の SHA-256
    """

    hasher = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if chunk == b"":
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def build_speaker_inversion_resume_contract(
    train_cfg: TrainConfig,
    *,
    world_size: int,
) -> dict[str, Any]:
    """
    Speaker Inversion の学習を同じ条件で再開するための契約を作る。

    Args:
        train_cfg (TrainConfig): 保存対象の学習設定
        world_size (int): 保存時の分散プロセス数

    Returns:
        dict[str, Any]: sidecar へ保存する再開条件
    """

    # 保存間隔や出力先は更新結果を変えないため、再開後の変更を許可する
    train_contract = {
        field: value
        for field, value in asdict(train_cfg).items()
        if field not in SPEAKER_INVERSION_RESUME_MUTABLE_TRAIN_FIELDS and field != "manifest_path"
    }
    manifest_path = None
    manifest_sha256 = None
    if train_cfg.manifest_path is not None and str(train_cfg.manifest_path).strip() != "":
        manifest_path = str(Path(train_cfg.manifest_path).expanduser().resolve())
        manifest_sha256 = sha256_file(manifest_path)
    return {
        "manifest_path": manifest_path,
        "manifest_sha256": manifest_sha256,
        "train_config": train_contract,
        "world_size": int(world_size),
    }


def validate_speaker_inversion_resume_contract(
    payload: dict[str, Any],
    *,
    train_cfg: TrainConfig,
    world_size: int,
) -> None:
    """
    Speaker Inversion の sidecar と現在の学習条件が一致することを検査する。

    Args:
        payload (dict[str, Any]): 読み込んだ sidecar の内容
        train_cfg (TrainConfig): 現在の学習設定
        world_size (int): 現在の分散プロセス数

    Raises:
        ValueError: sidecar に契約がない場合、または再現性に関わる条件が異なる場合
    """

    saved_contract = payload.get(SPEAKER_INVERSION_RESUME_CONTRACT_KEY)
    if not isinstance(saved_contract, dict):
        raise ValueError(
            "Speaker Inversion resume checkpoint is missing its exact-resume contract."
        )
    current_contract = build_speaker_inversion_resume_contract(
        train_cfg,
        world_size=world_size,
    )
    if saved_contract.get("world_size") != current_contract["world_size"]:
        raise ValueError(
            "Speaker Inversion resume world_size mismatch: "
            f"expected {saved_contract.get('world_size')!r}, "
            f"got {current_contract['world_size']!r}."
        )
    if saved_contract.get("manifest_sha256") != current_contract["manifest_sha256"]:
        raise ValueError(
            "Speaker Inversion resume manifest mismatch: "
            f"saved path {saved_contract.get('manifest_path')!r}, "
            f"current path {current_contract['manifest_path']!r}."
        )
    saved_train_config = saved_contract.get("train_config")
    if not isinstance(saved_train_config, dict):
        raise ValueError("Speaker Inversion resume contract has no train_config dictionary.")
    current_train_config = current_contract["train_config"]
    mismatches = [
        field
        for field in sorted(set(saved_train_config) | set(current_train_config))
        if saved_train_config.get(field) != current_train_config.get(field)
    ]
    if mismatches:
        details = ", ".join(
            f"{field}: saved={saved_train_config.get(field)!r} "
            f"current={current_train_config.get(field)!r}"
            for field in mismatches
        )
        raise ValueError(f"Speaker Inversion resume train config mismatch: {details}.")


def load_optional_json(path: str | Path | None) -> dict[str, Any]:
    """
    LoRA metadata に埋め込む任意の JSON ファイルを読み込む。

    Args:
        path (str | Path | None): JSON ファイルのパス

    Returns:
        dict[str, Any]: ファイルが未指定なら空 dict
    """

    if path is None or str(path).strip() == "":
        return {}
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Metadata file must contain a JSON object: {path}")
    return payload


def build_lora_metadata(
    *,
    model_cfg: ModelConfig,
    train_cfg: TrainConfig,
    base_init: dict[str, Any] | None,
) -> dict[str, Any]:
    """
    adapter と一緒に保存する実験条件の metadata を作る。

    Args:
        model_cfg (ModelConfig): 保存対象のモデル設定
        train_cfg (TrainConfig): 保存対象の学習設定
        base_init (dict[str, Any] | None): base checkpoint の初期化情報

    Returns:
        dict[str, Any]: `irodori_lora_metadata.json` に書き込む内容
    """

    manifest_metadata = load_optional_json(train_cfg.manifest_metadata_path)
    manifest_hash = None
    if train_cfg.manifest_path:
        manifest_hash = sha256_file(train_cfg.manifest_path)
    return {
        "base_init": base_init,
        "manifest": {
            "path": train_cfg.manifest_path,
            "sha256": manifest_hash,
            "metadata": manifest_metadata,
        },
        "label_filter": train_cfg.manifest_label_filter,
    }


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    step: int,
    model_cfg: ModelConfig,
    train_cfg: TrainConfig,
    *,
    base_init: dict[str, Any] | None = None,
    lora_metadata: dict[str, Any] | None = None,
    dataloader_state: dict[str, Any] | None = None,
    runtime_state: dict[str, Any] | None = None,
    rng_state: dict[str, Any] | None = None,
    speaker_inversion_resume_contract: dict[str, Any] | None = None,
) -> None:
    path = Path(path)
    if train_cfg.speaker_inversion_enabled:
        if speaker_inversion_resume_contract is None:
            raise ValueError("Speaker Inversion checkpoints require an exact-resume contract.")
        save_speaker_inversion_checkpoint(path, model=model)
        speaker_inversion_module = getattr(model, "speaker_inversion", None)
        if not isinstance(speaker_inversion_module, SpeakerInversionEmbedding):
            raise RuntimeError("Speaker Inversion checkpoint requires an enabled embedding module.")
        text_encoder_config = None
        pretrained_backbone = getattr(model, "pretrained_text_backbone", None)
        if pretrained_backbone is not None:
            raw_config = getattr(pretrained_backbone, "config_dict", None)
            if isinstance(raw_config, dict):
                text_encoder_config = dict(raw_config)
        # 推論用埋め込みと分離し、再開に必要な学習状態だけを軽量 sidecar へ保存する
        trainer_state_path = _speaker_inversion_trainer_state_path(path)
        temporary_trainer_state_path = trainer_state_path.with_name(
            f".{trainer_state_path.name}.tmp"
        )
        # 呼び出し側が集約済み状態を渡さない単体テストでは、現在 rank の状態だけを保存する
        saved_rng_state = rng_state if rng_state is not None else _capture_rng_state()
        torch.save(
            {
                "step": step,
                SPEAKER_EMBEDDING_KEY: speaker_inversion_state_dict(model)[SPEAKER_EMBEDDING_KEY],
                SPEAKER_INVERSION_MODULE_STATE_KEY: speaker_inversion_module.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": None if scheduler is None else scheduler.state_dict(),
                "model_config": asdict(model_cfg),
                "train_config": asdict(train_cfg),
                "text_encoder_config": text_encoder_config,
                "base_init": base_init,
                DATALOADER_STATE_KEY: dataloader_state,
                RUNTIME_STATE_KEY: runtime_state,
                RNG_STATE_KEY: saved_rng_state,
                SPEAKER_INVERSION_RESUME_CONTRACT_KEY: speaker_inversion_resume_contract,
            },
            temporary_trainer_state_path,
        )
        # 書き込み途中の sidecar を再開候補として拾わないよう、完成後に公開する
        temporary_trainer_state_path.replace(trainer_state_path)
        return

    if train_config_uses_lora(train_cfg):
        if lora_metadata is None:
            raise ValueError("lora_metadata is required when saving a LoRA checkpoint.")
        if path.exists():
            _safe_unlink(path)
        path.mkdir(parents=True, exist_ok=True)
        if not hasattr(model, "save_pretrained"):
            raise RuntimeError(
                "LoRA checkpoint saving requires a PEFT model with save_pretrained()."
            )
        model.save_pretrained(path)
        dump_configs(path / "config.json", model_cfg, train_cfg)
        (path / LORA_METADATA_NAME).write_text(
            json.dumps(
                lora_metadata,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        torch.save(
            {
                "step": step,
                "optimizer": optimizer.state_dict(),
                "scheduler": None if scheduler is None else scheduler.state_dict(),
                "model_config": asdict(model_cfg),
                "train_config": asdict(train_cfg),
                "base_init": base_init,
                DATALOADER_STATE_KEY: dataloader_state,
                RUNTIME_STATE_KEY: runtime_state,
            },
            path / LORA_TRAINER_STATE_NAME,
        )
        return

    text_encoder_config = None
    pretrained_backbone = getattr(model, "pretrained_text_backbone", None)
    if pretrained_backbone is not None:
        raw_config = getattr(pretrained_backbone, "config_dict", None)
        if isinstance(raw_config, dict):
            text_encoder_config = dict(raw_config)

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "step": step,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": None if scheduler is None else scheduler.state_dict(),
            "model_config": asdict(model_cfg),
            "train_config": asdict(train_cfg),
            "text_encoder_config": text_encoder_config,
            DATALOADER_STATE_KEY: dataloader_state,
            RUNTIME_STATE_KEY: runtime_state,
        },
        path,
    )


def _runtime_state_for_checkpoint(*, epoch: int, epoch_step: int) -> dict[str, int]:
    return {
        "epoch": int(epoch),
        "sampler_epoch": max(0, int(epoch) - 1),
        "epoch_step": int(epoch_step),
    }


def _collect_dataloader_state(
    loader: StatefulDataLoader,
    *,
    distributed: bool,
    rank: int,
    world_size: int,
) -> dict:
    local_state = loader.state_dict()
    if not distributed:
        return {
            "version": 1,
            "world_size": 1,
            "rank_states": [local_state],
        }

    rank_states: list[dict | None] = [None for _ in range(world_size)]
    dist.all_gather_object(rank_states, local_state)
    return {
        "version": 1,
        "world_size": int(world_size),
        "rank_states": rank_states,
        "saved_by_rank": int(rank),
    }


def _select_dataloader_state_for_rank(
    payload: dict,
    *,
    distributed: bool,
    rank: int,
    world_size: int,
) -> dict | None:
    state = payload.get(DATALOADER_STATE_KEY)
    if state is None:
        return None
    if not isinstance(state, dict):
        raise ValueError("Checkpoint dataloader_state must be a dictionary when present.")
    rank_states = state.get("rank_states")
    if not isinstance(rank_states, list):
        raise ValueError("Checkpoint dataloader_state.rank_states must be a list.")
    saved_world_size = int(state.get("world_size", len(rank_states)))
    expected_world_size = int(world_size) if distributed else 1
    if saved_world_size != expected_world_size or len(rank_states) != expected_world_size:
        warnings.warn(
            "Discarding dataloader state because world_size changed: "
            f"checkpoint={saved_world_size} current={expected_world_size}"
        )
        return None
    state_rank = int(rank) if distributed else 0
    rank_state = rank_states[state_rank]
    if rank_state is not None and not isinstance(rank_state, dict):
        raise ValueError(f"Checkpoint dataloader state for rank {state_rank} must be a dictionary.")
    return _move_state_tensors_to_cpu(rank_state)


def _move_state_tensors_to_cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _move_state_tensors_to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_move_state_tensors_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_move_state_tensors_to_cpu(item) for item in value)
    return value


def _safe_unlink(path: Path) -> None:
    try:
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
    except FileNotFoundError:
        return


def list_periodic_checkpoints(output_dir: Path) -> list[tuple[int, Path]]:
    checkpoints: list[tuple[int, Path]] = []
    for path in output_dir.glob("checkpoint_*"):
        match = CHECKPOINT_STEP_RE.match(path.name)
        if match is None:
            continue
        checkpoints.append((int(match.group(1)), path))
    checkpoints.sort(key=lambda item: item[0], reverse=True)
    return checkpoints


def enforce_periodic_checkpoint_limit(output_dir: Path, keep_count: int) -> None:
    if keep_count <= 0:
        return
    checkpoints = list_periodic_checkpoints(output_dir)
    for _, stale_path in checkpoints[keep_count:]:
        _safe_unlink(stale_path)
        if stale_path.name.endswith(SPEAKER_INVERSION_SAFETENSORS_SUFFIX):
            _safe_unlink(_speaker_inversion_trainer_state_path(stale_path))


def list_best_val_loss_checkpoints(output_dir: Path) -> list[tuple[float, int, Path]]:
    checkpoints: list[tuple[float, int, Path]] = []
    for path in output_dir.glob("checkpoint_best_val_loss_*"):
        match = CHECKPOINT_BEST_VAL_LOSS_RE.match(path.name)
        if match is None:
            continue
        step = int(match.group(1))
        score = float(match.group(2))
        checkpoints.append((score, step, path))
    checkpoints.sort(key=lambda item: (item[0], item[1]))
    return checkpoints


def prune_best_val_loss_checkpoints(
    checkpoints: list[tuple[float, int, Path]],
    keep_best_n: int,
) -> list[tuple[float, int, Path]]:
    if keep_best_n <= 0:
        return checkpoints
    checkpoints = sorted(checkpoints, key=lambda item: (item[0], item[1]))
    while len(checkpoints) > keep_best_n:
        _, _, stale_path = checkpoints.pop()
        _safe_unlink(stale_path)
        if stale_path.name.endswith(SPEAKER_INVERSION_SAFETENSORS_SUFFIX):
            _safe_unlink(_speaker_inversion_trainer_state_path(stale_path))
    return checkpoints


def maybe_save_best_val_loss_checkpoint(
    *,
    output_dir: Path,
    checkpoints: list[tuple[float, int, Path]],
    keep_best_n: int,
    val_loss: float,
    step: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    model_cfg: ModelConfig,
    train_cfg: TrainConfig,
    base_init: dict | None,
    lora_metadata: dict[str, Any] | None,
    dataloader_state: dict | None,
    runtime_state: dict | None,
    rng_state: dict | None = None,
    speaker_inversion_resume_contract: dict[str, Any] | None = None,
) -> tuple[list[tuple[float, int, Path]], Path | None]:
    if keep_best_n <= 0:
        return checkpoints, None

    checkpoints = sorted(checkpoints, key=lambda item: (item[0], item[1]))
    if len(checkpoints) >= keep_best_n:
        worst_score = checkpoints[-1][0]
        if val_loss >= worst_score:
            return checkpoints, None

    kept: list[tuple[float, int, Path]] = []
    for score, saved_step, path in checkpoints:
        if saved_step == step:
            _safe_unlink(path)
            if path.name.endswith(SPEAKER_INVERSION_SAFETENSORS_SUFFIX):
                _safe_unlink(_speaker_inversion_trainer_state_path(path))
            continue
        kept.append((score, saved_step, path))
    checkpoints = kept

    path = _best_checkpoint_path(output_dir, step=step, val_loss=val_loss, train_cfg=train_cfg)
    save_checkpoint(
        path=path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        step=step,
        model_cfg=model_cfg,
        train_cfg=train_cfg,
        base_init=base_init,
        lora_metadata=lora_metadata,
        dataloader_state=dataloader_state,
        runtime_state=runtime_state,
        rng_state=rng_state,
        speaker_inversion_resume_contract=speaker_inversion_resume_contract,
    )
    checkpoints.append((float(val_loss), int(step), path))
    checkpoints = prune_best_val_loss_checkpoints(checkpoints, keep_best_n)
    return checkpoints, path


def cli_provided(argv: list[str], flag: str) -> bool:
    return any(x == flag or x.startswith(flag + "=") for x in argv)


def _periodic_checkpoint_path(output_dir: Path, step: int, train_cfg: TrainConfig) -> Path:
    if train_cfg.speaker_inversion_enabled:
        return output_dir / f"checkpoint_{step:07d}{SPEAKER_INVERSION_SAFETENSORS_SUFFIX}"
    if train_config_uses_lora(train_cfg):
        return output_dir / f"checkpoint_{step:07d}"
    return output_dir / f"checkpoint_{step:07d}.pt"


def _best_checkpoint_path(
    output_dir: Path, *, step: int, val_loss: float, train_cfg: TrainConfig
) -> Path:
    if train_cfg.speaker_inversion_enabled:
        return (
            output_dir / f"checkpoint_best_val_loss_{step:07d}_{val_loss:.6f}"
            f"{SPEAKER_INVERSION_SAFETENSORS_SUFFIX}"
        )
    if train_config_uses_lora(train_cfg):
        return output_dir / f"checkpoint_best_val_loss_{step:07d}_{val_loss:.6f}"
    return output_dir / f"checkpoint_best_val_loss_{step:07d}_{val_loss:.6f}.pt"


def _final_checkpoint_path(output_dir: Path, train_cfg: TrainConfig) -> Path:
    if train_cfg.speaker_inversion_enabled:
        return output_dir / f"checkpoint_final{SPEAKER_INVERSION_SAFETENSORS_SUFFIX}"
    if train_config_uses_lora(train_cfg):
        return output_dir / "checkpoint_final"
    return output_dir / "checkpoint_final.pt"


def build_pretrained_tokenizer(
    *,
    repo_id: str,
    add_bos: bool,
    vocab_size: int | None,
    local_files_only: bool = False,
    revision: str | None = None,
) -> PretrainedTextTokenizer:
    tokenizer = PretrainedTextTokenizer.from_pretrained(
        repo_id=repo_id,
        add_bos=bool(add_bos),
        local_files_only=local_files_only,
        revision=revision,
    )
    if vocab_size is not None and tokenizer.vocab_size != vocab_size:
        raise ValueError(
            f"Tokenizer vocab_size mismatch: expected {vocab_size} but tokenizer "
            f"({repo_id}) vocab_size={tokenizer.vocab_size}."
        )
    return tokenizer


def build_text_tokenizer(
    model_cfg: ModelConfig,
    *,
    local_files_only: bool = False,
) -> PretrainedTextTokenizer:
    return build_pretrained_tokenizer(
        repo_id=model_cfg.text_tokenizer_repo,
        add_bos=bool(model_cfg.text_add_bos),
        vocab_size=(
            None if model_cfg.use_pretrained_text_encoder else int(model_cfg.text_vocab_size)
        ),
        local_files_only=local_files_only,
        revision=model_cfg.text_encoder_revision,
    )


def build_caption_tokenizer(
    model_cfg: ModelConfig,
    *,
    local_files_only: bool = False,
) -> PretrainedTextTokenizer:
    return build_pretrained_tokenizer(
        repo_id=model_cfg.caption_tokenizer_repo_resolved,
        add_bos=model_cfg.caption_add_bos_resolved,
        vocab_size=(
            None if model_cfg.use_pretrained_text_encoder else model_cfg.caption_vocab_size_resolved
        ),
        local_files_only=local_files_only,
        revision=model_cfg.text_encoder_revision,
    )


def validate_pretrained_backbone_dim(
    *,
    repo_id: str,
    expected_dim: int | None,
    local_files_only: bool = False,
    revision: str | None = None,
) -> int:
    try:
        from transformers import AutoConfig
    except ImportError as exc:
        raise RuntimeError(
            "transformers is required for pretrained text embedding initialization. "
            "Install with `pip install transformers sentencepiece`."
        ) from exc

    text_cfg = AutoConfig.from_pretrained(
        repo_id,
        trust_remote_code=False,
        local_files_only=local_files_only,
        revision=revision,
    )
    hidden_size = getattr(text_cfg, "hidden_size", None)
    if hidden_size is None:
        encoder_cfg = getattr(text_cfg, "encoder", None)
        hidden_size = getattr(encoder_cfg, "hidden_size", None)
        if hidden_size is None:
            hidden_size = getattr(getattr(encoder_cfg, "text_config", None), "hidden_size", None)
    if hidden_size is None:
        raise ValueError(f"Could not read hidden_size from pretrained config: {repo_id}")
    hidden_size = int(hidden_size)
    if expected_dim is not None and hidden_size != expected_dim:
        raise ValueError(
            f"Condition encoder dim mismatch: expected {expected_dim} but pretrained hidden_size={hidden_size} "
            f"for repo {repo_id}."
        )
    return hidden_size


def validate_text_backbone_dim(
    model_cfg: ModelConfig,
    *,
    local_files_only: bool = False,
) -> int:
    return validate_pretrained_backbone_dim(
        repo_id=model_cfg.text_tokenizer_repo,
        expected_dim=(None if model_cfg.use_pretrained_text_encoder else int(model_cfg.text_dim)),
        local_files_only=local_files_only,
        revision=model_cfg.text_encoder_revision,
    )


def validate_caption_backbone_dim(
    model_cfg: ModelConfig,
    *,
    local_files_only: bool = False,
) -> int:
    return validate_pretrained_backbone_dim(
        repo_id=model_cfg.caption_tokenizer_repo_resolved,
        expected_dim=(
            None if model_cfg.use_pretrained_text_encoder else model_cfg.caption_dim_resolved
        ),
        local_files_only=local_files_only,
        revision=model_cfg.text_encoder_revision,
    )


def initialize_embedding_from_pretrained(
    embedding: torch.nn.Embedding,
    *,
    repo_id: str,
    local_files_only: bool = False,
    revision: str | None = None,
) -> None:
    try:
        from transformers import AutoModel
    except ImportError as exc:
        raise RuntimeError(
            "transformers is required for pretrained text embedding initialization. "
            "Install with `pip install transformers sentencepiece`."
        ) from exc

    text_backbone = AutoModel.from_pretrained(
        repo_id,
        trust_remote_code=False,
        dtype=torch.float32,
        low_cpu_mem_usage=True,
        local_files_only=local_files_only,
        revision=revision,
    )
    pretrained_embedding = text_backbone.get_input_embeddings()
    if pretrained_embedding is None:
        raise ValueError(f"Pretrained model has no input embeddings: {repo_id}")
    src_weight = pretrained_embedding.weight.detach().to(device="cpu", dtype=torch.float32)
    tgt_weight = embedding.weight
    src_vocab, src_dim = tuple(src_weight.shape)
    tgt_vocab, tgt_dim = tuple(tgt_weight.shape)
    if src_dim != tgt_dim:
        raise ValueError(
            f"Embedding hidden size mismatch: pretrained={src_dim} model={tgt_dim} for repo={repo_id}."
        )

    copy_rows = min(src_vocab, tgt_vocab)
    with torch.no_grad():
        tgt_weight[:copy_rows].copy_(
            src_weight[:copy_rows].to(device=tgt_weight.device, dtype=tgt_weight.dtype)
        )

    del text_backbone


def initialize_text_embedding_from_pretrained(
    model: TextToLatentRFDiT,
    model_cfg: ModelConfig,
    *,
    local_files_only: bool = False,
) -> None:
    initialize_embedding_from_pretrained(
        model.text_encoder.text_embedding,
        repo_id=model_cfg.text_tokenizer_repo,
        local_files_only=local_files_only,
        revision=model_cfg.text_encoder_revision,
    )


def initialize_caption_embedding_from_pretrained(
    model: TextToLatentRFDiT,
    model_cfg: ModelConfig,
    *,
    local_files_only: bool = False,
) -> None:
    if model.caption_encoder is None:
        raise RuntimeError(
            "Caption embedding initialization requested but caption encoder is absent."
        )
    initialize_embedding_from_pretrained(
        model.caption_encoder.text_embedding,
        repo_id=model_cfg.caption_tokenizer_repo_resolved,
        local_files_only=local_files_only,
        revision=model_cfg.text_encoder_revision,
    )


def _load_model_state_from_checkpoint(
    path: Path,
) -> tuple[dict[str, torch.Tensor], dict | None, dict | None, dict | None]:
    if path.suffix.lower() == ".safetensors":
        from safetensors import safe_open
        from safetensors.torch import load_file as load_safetensors_file

        checkpoint_model_cfg = None
        text_encoder_config = None
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            metadata = dict(handle.metadata() or {})
        from irodori_tts.quantization import parse_quantization_metadata

        if parse_quantization_metadata(metadata) is not None:
            raise ValueError(
                "Quantized checkpoints are inference-only and cannot be used with "
                "--init-checkpoint. Train LoRA against the matching full-precision base model, "
                "then merge and quantize it for inference."
            )
        config_json = metadata.get(SAFETENSORS_CONFIG_META_KEY)
        if config_json:
            parsed = json.loads(config_json)
            if isinstance(parsed, dict):
                checkpoint_model_cfg = {
                    key: value
                    for key, value in parsed.items()
                    if key not in SAFETENSORS_INFERENCE_CONFIG_KEYS
                }
        text_encoder_config_json = metadata.get(SAFETENSORS_TEXT_ENCODER_CONFIG_META_KEY)
        if text_encoder_config_json:
            parsed = json.loads(text_encoder_config_json)
            if isinstance(parsed, dict):
                text_encoder_config = parsed
        return (
            load_safetensors_file(str(path), device="cpu"),
            checkpoint_model_cfg,
            None,
            text_encoder_config,
        )

    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError(f"Checkpoint payload must be a dictionary, got {type(payload)!r}.")

    raw_model = payload.get("model")
    if raw_model is None and all(isinstance(v, torch.Tensor) for v in payload.values()):
        raw_model = payload
    if not isinstance(raw_model, dict):
        raise ValueError(f"Checkpoint does not contain a model state dictionary: {path}")

    checkpoint_model_cfg = payload.get("model_config")
    if checkpoint_model_cfg is not None and not isinstance(checkpoint_model_cfg, dict):
        raise ValueError(f"Checkpoint model_config must be a dictionary when present: {path}")
    checkpoint_train_cfg = payload.get("train_config")
    if checkpoint_train_cfg is not None and not isinstance(checkpoint_train_cfg, dict):
        raise ValueError(f"Checkpoint train_config must be a dictionary when present: {path}")
    text_encoder_config = payload.get("text_encoder_config")
    if text_encoder_config is not None and not isinstance(text_encoder_config, dict):
        raise ValueError(f"Checkpoint text_encoder_config must be a dictionary: {path}")
    return raw_model, checkpoint_model_cfg, checkpoint_train_cfg, text_encoder_config


def _check_model_config_compatibility(
    checkpoint_path: Path,
    checkpoint_model_cfg: dict | None,
    current_model_cfg: ModelConfig,
    *,
    require_caption_match: bool,
    upgrade_speaker_patch: bool = False,
    upgrade_text_encoder: bool = False,
) -> None:
    if checkpoint_model_cfg is None:
        return

    checkpoint_cfg = merge_dataclass_overrides(
        ModelConfig(),
        checkpoint_model_cfg,
        section="checkpoint model_config",
    )

    comparisons: list[tuple[str, object, object]] = [
        ("latent_dim", checkpoint_cfg.latent_dim, current_model_cfg.latent_dim),
        (
            "latent_patch_size",
            checkpoint_cfg.latent_patch_size,
            current_model_cfg.latent_patch_size,
        ),
        ("model_dim", checkpoint_cfg.model_dim, current_model_cfg.model_dim),
        ("num_layers", checkpoint_cfg.num_layers, current_model_cfg.num_layers),
        ("num_heads", checkpoint_cfg.num_heads, current_model_cfg.num_heads),
        ("mlp_ratio", checkpoint_cfg.mlp_ratio, current_model_cfg.mlp_ratio),
        ("text_dim", checkpoint_cfg.text_dim, current_model_cfg.text_dim),
        ("adaln_rank", checkpoint_cfg.adaln_rank, current_model_cfg.adaln_rank),
    ]
    if not upgrade_text_encoder:
        comparisons.append(
            (
                "text_encoder_type",
                checkpoint_cfg.text_encoder_type,
                current_model_cfg.text_encoder_type,
            )
        )
    if not upgrade_text_encoder and (
        checkpoint_cfg.use_pretrained_text_encoder or current_model_cfg.use_pretrained_text_encoder
    ):
        comparisons.append(
            (
                "text_tokenizer_repo",
                checkpoint_cfg.text_tokenizer_repo,
                current_model_cfg.text_tokenizer_repo,
            )
        )
        comparisons.extend(
            [
                (
                    "text_encoder_revision",
                    checkpoint_cfg.text_encoder_revision,
                    current_model_cfg.text_encoder_revision,
                ),
                (
                    "pretrained_projector_type",
                    checkpoint_cfg.pretrained_projector_type,
                    current_model_cfg.pretrained_projector_type,
                ),
                (
                    "pretrained_projector_hidden_ratio",
                    checkpoint_cfg.pretrained_projector_hidden_ratio,
                    current_model_cfg.pretrained_projector_hidden_ratio,
                ),
                (
                    "pretrained_projector_dropout",
                    checkpoint_cfg.pretrained_projector_dropout,
                    current_model_cfg.pretrained_projector_dropout,
                ),
            ]
        )
    elif not upgrade_text_encoder:
        comparisons.extend(
            [
                (
                    "text_vocab_size",
                    checkpoint_cfg.text_vocab_size,
                    current_model_cfg.text_vocab_size,
                ),
                ("text_layers", checkpoint_cfg.text_layers, current_model_cfg.text_layers),
                ("text_heads", checkpoint_cfg.text_heads, current_model_cfg.text_heads),
                (
                    "text_mlp_ratio",
                    checkpoint_cfg.text_mlp_ratio_resolved,
                    current_model_cfg.text_mlp_ratio_resolved,
                ),
            ]
        )
    if (
        checkpoint_cfg.use_speaker_condition_resolved
        and current_model_cfg.use_speaker_condition_resolved
    ):
        comparisons.extend(
            [
                ("speaker_dim", checkpoint_cfg.speaker_dim, current_model_cfg.speaker_dim),
                ("speaker_layers", checkpoint_cfg.speaker_layers, current_model_cfg.speaker_layers),
                ("speaker_heads", checkpoint_cfg.speaker_heads, current_model_cfg.speaker_heads),
                (
                    "speaker_mlp_ratio",
                    checkpoint_cfg.speaker_mlp_ratio_resolved,
                    current_model_cfg.speaker_mlp_ratio_resolved,
                ),
            ]
        )
        if not upgrade_speaker_patch:
            comparisons.append(
                (
                    "speaker_patch_size",
                    checkpoint_cfg.speaker_patch_size,
                    current_model_cfg.speaker_patch_size,
                )
            )
        else:
            old_patch = int(checkpoint_cfg.speaker_patch_size)
            new_patch = int(current_model_cfg.speaker_patch_size)
            if (
                old_patch <= 0
                or new_patch <= 0
                or new_patch <= old_patch
                or new_patch % old_patch != 0
            ):
                raise ValueError(
                    "speaker_patch_size upgrade requires new patch to be a positive "
                    f"integer multiple of the checkpoint's: got old={old_patch} "
                    f"new={new_patch} ({checkpoint_path})"
                )
    if require_caption_match:
        comparisons.extend(
            [
                (
                    "use_caption_condition",
                    checkpoint_cfg.use_caption_condition,
                    current_model_cfg.use_caption_condition,
                ),
                (
                    "use_speaker_condition",
                    checkpoint_cfg.use_speaker_condition_resolved,
                    current_model_cfg.use_speaker_condition_resolved,
                ),
                (
                    "caption_dim",
                    checkpoint_cfg.caption_dim_resolved,
                    current_model_cfg.caption_dim_resolved,
                ),
            ]
        )
        if not upgrade_text_encoder:
            comparisons.extend(
                [
                    (
                        "caption_vocab_size",
                        checkpoint_cfg.caption_vocab_size_resolved,
                        current_model_cfg.caption_vocab_size_resolved,
                    ),
                    (
                        "caption_tokenizer_repo",
                        checkpoint_cfg.caption_tokenizer_repo_resolved,
                        current_model_cfg.caption_tokenizer_repo_resolved,
                    ),
                    (
                        "caption_add_bos",
                        checkpoint_cfg.caption_add_bos_resolved,
                        current_model_cfg.caption_add_bos_resolved,
                    ),
                    (
                        "caption_layers",
                        checkpoint_cfg.caption_layers_resolved,
                        current_model_cfg.caption_layers_resolved,
                    ),
                    (
                        "caption_heads",
                        checkpoint_cfg.caption_heads_resolved,
                        current_model_cfg.caption_heads_resolved,
                    ),
                    (
                        "caption_mlp_ratio",
                        checkpoint_cfg.caption_mlp_ratio_resolved,
                        current_model_cfg.caption_mlp_ratio_resolved,
                    ),
                ]
            )

    for key, checkpoint_value, current_value in comparisons:
        if checkpoint_value != current_value:
            raise ValueError(
                f"Checkpoint/config mismatch for '{key}': checkpoint={checkpoint_value} "
                f"current={current_value} ({checkpoint_path})"
            )


def checkpoint_uses_caption_condition(
    checkpoint_model_cfg: dict | None,
    state_dict: dict[str, torch.Tensor],
) -> bool:
    if checkpoint_model_cfg is not None:
        checkpoint_cfg = merge_dataclass_overrides(
            ModelConfig(),
            checkpoint_model_cfg,
            section="checkpoint model_config",
        )
        if checkpoint_cfg.use_caption_condition:
            return True
    return any(
        key.startswith("caption_encoder.")
        or key.startswith("caption_norm.")
        or ".wk_caption." in key
        or ".wv_caption." in key
        for key in state_dict
    )


def checkpoint_uses_duration_predictor(
    checkpoint_model_cfg: dict | None,
    state_dict: dict[str, torch.Tensor],
) -> bool:
    if checkpoint_model_cfg is not None:
        checkpoint_cfg = merge_dataclass_overrides(
            ModelConfig(),
            checkpoint_model_cfg,
            section="checkpoint model_config",
        )
        if checkpoint_cfg.use_duration_predictor:
            return True
    return any(key.startswith("duration_predictor.") for key in state_dict)


def load_model_state_partially(
    model: TextToLatentRFDiT,
    state_dict: dict[str, torch.Tensor],
    *,
    reinit_keys: set[str] | None = None,
) -> tuple[list[str], list[str], list[str]]:
    """
    Load state_dict into ``model`` non-strictly, tolerating both extra and
    missing/renamed keys.

    Keys in ``reinit_keys`` are treated as an explicit "will be reinitialized
    by the caller" contract: they are NOT reported in ``skipped_shape`` even
    if the checkpoint tensor's shape differs from the current model's, and
    they are NOT loaded (the caller is expected to overwrite the parameter
    in-place afterwards, e.g. via a tile+scale upgrade).
    """
    reinit_keys = reinit_keys or set()
    model_state = model.state_dict()
    filtered_state: dict[str, torch.Tensor] = {}
    skipped_shape: list[str] = []
    skipped_extra: list[str] = []

    for key, value in state_dict.items():
        target = model_state.get(key)
        if target is None:
            skipped_extra.append(key)
            continue
        if tuple(target.shape) != tuple(value.shape):
            if key not in reinit_keys:
                skipped_shape.append(key)
            continue
        filtered_state[key] = value

    missing_keys, unexpected_keys = model.load_state_dict(filtered_state, strict=False)
    if unexpected_keys:
        skipped_extra.extend(unexpected_keys)
    # ``reinit_keys`` will be initialized by the caller; suppress them from the
    # missing report so the validator does not treat them as an error.
    missing_keys = [key for key in missing_keys if key not in reinit_keys]
    return missing_keys, skipped_shape, skipped_extra


SPEAKER_IN_PROJ_WEIGHT_KEY = "speaker_encoder.in_proj.weight"


def _canonical_parameter_key(key: str) -> str:
    prefix = "base_model.model."
    if key.startswith(prefix):
        return key[len(prefix) :]
    return key


def is_caption_only_parameter(key: str) -> bool:
    key = _canonical_parameter_key(key)
    return (
        key.startswith("caption_encoder.")
        or key.startswith("caption_norm.")
        or ".wk_caption." in key
        or ".wv_caption." in key
    )


def is_speaker_only_parameter(key: str) -> bool:
    key = _canonical_parameter_key(key)
    return (
        key.startswith("speaker_encoder.")
        or key.startswith("speaker_norm.")
        or ".wk_speaker." in key
        or ".wv_speaker." in key
    )


def is_duration_only_parameter(key: str) -> bool:
    key = _canonical_parameter_key(key)
    return key.startswith("duration_predictor.")


def is_replaced_text_encoder_parameter(key: str) -> bool:
    key = _canonical_parameter_key(key)
    return (
        key.startswith("pretrained_text_backbone.")
        or key.startswith("text_encoder.")
        or key.startswith("caption_encoder.")
    )


def is_pretrained_projector_parameter(key: str) -> bool:
    key = _canonical_parameter_key(key)
    return key.startswith("text_encoder.") or key.startswith("caption_encoder.")


def clear_non_caption_grads(model: TextToLatentRFDiT) -> tuple[int, int]:
    caption_grad_params = 0
    cleared_grad_params = 0
    for key, param in model.named_parameters():
        if is_caption_only_parameter(key):
            if param.grad is not None:
                caption_grad_params += 1
            continue
        if param.grad is not None:
            cleared_grad_params += 1
        param.grad = None
    return caption_grad_params, cleared_grad_params


def clear_non_pretrained_projector_grads(
    model: TextToLatentRFDiT,
) -> None:
    for key, param in model.named_parameters():
        if is_pretrained_projector_parameter(key):
            continue
        param.grad = None


def freeze_for_duration_only(model: torch.nn.Module) -> tuple[int, int]:
    trainable_params = 0
    frozen_params = 0
    for key, param in model.named_parameters():
        if is_duration_only_parameter(key):
            param.requires_grad_(True)
            trainable_params += param.numel()
        else:
            param.requires_grad_(False)
            frozen_params += param.numel()
    return trainable_params, frozen_params


def freeze_for_speaker_inversion(model: torch.nn.Module) -> tuple[int, int]:
    trainable_params = 0
    frozen_params = 0
    for key, param in model.named_parameters():
        if _canonical_parameter_key(key).startswith("speaker_inversion."):
            param.requires_grad_(True)
            trainable_params += param.numel()
        else:
            param.requires_grad_(False)
            frozen_params += param.numel()
    return trainable_params, frozen_params


def validate_checkpoint_upgrade_partial_load(
    checkpoint_path: Path,
    missing_keys: list[str],
    skipped_shape: list[str],
    skipped_extra: list[str],
    *,
    allow_caption_missing: bool,
    allow_duration_missing: bool,
    allow_duration_extra: bool,
    allow_speaker_extra: bool,
    allow_text_encoder_replacement: bool = False,
) -> None:
    if skipped_shape:
        raise ValueError(
            "Checkpoint/config shape mismatch while upgrading checkpoint config: "
            f"{checkpoint_path} skipped_shape={skipped_shape[:8]}"
        )

    unexpected_extra = skipped_extra
    if allow_speaker_extra:
        unexpected_extra = [key for key in unexpected_extra if not is_speaker_only_parameter(key)]
    if allow_duration_extra:
        unexpected_extra = [key for key in unexpected_extra if not is_duration_only_parameter(key)]
    if allow_text_encoder_replacement:
        unexpected_extra = [
            key for key in unexpected_extra if not is_replaced_text_encoder_parameter(key)
        ]
    if unexpected_extra:
        raise ValueError(
            "Unexpected checkpoint keys while upgrading checkpoint config: "
            f"{checkpoint_path} skipped_extra={unexpected_extra[:8]}"
        )

    def _allowed_missing(key: str) -> bool:
        return (
            (allow_caption_missing and is_caption_only_parameter(key))
            or (allow_duration_missing and is_duration_only_parameter(key))
            or (allow_text_encoder_replacement and is_replaced_text_encoder_parameter(key))
        )

    unexpected_missing = [key for key in missing_keys if not _allowed_missing(key)]
    if unexpected_missing:
        raise ValueError(
            "Partial init from checkpoint left unexpected parameters missing: "
            f"{checkpoint_path} missing={unexpected_missing[:8]}"
        )


def _load_checkpoint_payload(path: str | Path, *, map_location) -> dict:
    checkpoint_path = Path(path)
    if checkpoint_path.is_dir():
        state_path = checkpoint_path / LORA_TRAINER_STATE_NAME
        payload = torch.load(state_path, map_location=map_location, weights_only=True)
    else:
        payload = torch.load(checkpoint_path, map_location=map_location, weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError(f"Checkpoint payload must be a dictionary, got {type(payload)!r}.")
    return payload


def _normalize_checkpoint_path(path: str | Path) -> Path:
    return Path(os.path.abspath(str(Path(path).expanduser())))


def _lora_field_cli_explicit(field: str, args: argparse.Namespace, raw_argv: list[str]) -> bool:
    if field == "lora_enabled":
        return args.lora_enabled is not None
    flag = "--" + field.replace("_", "-")
    return cli_provided(raw_argv, flag)


def _restore_resume_lora_config(
    train_cfg: TrainConfig,
    *,
    resume_train_cfg: dict | None,
    args: argparse.Namespace,
    raw_argv: list[str],
    exp_cfg: dict,
) -> TrainConfig:
    if not isinstance(resume_train_cfg, dict):
        return train_cfg

    train_overrides = exp_cfg.get("train", {})
    if not isinstance(train_overrides, dict):
        train_overrides = {}

    updates: dict[str, object] = {}
    for field in LORA_TRAIN_CONFIG_FIELDS:
        if field not in resume_train_cfg:
            continue
        explicit = _lora_field_cli_explicit(field, args, raw_argv) or field in train_overrides
        current_value = getattr(train_cfg, field)
        resume_value = resume_train_cfg[field]
        if explicit:
            if current_value != resume_value:
                raise ValueError(
                    f"Resume checkpoint expects train.{field}={resume_value!r}, "
                    f"but current config requests {current_value!r}."
                )
            continue
        updates[field] = resume_value

    if updates:
        train_cfg = replace(train_cfg, **updates)
    return train_cfg


def _restore_resume_speaker_inversion_config(
    train_cfg: TrainConfig,
    *,
    resume_train_cfg: dict | None,
    raw_argv: list[str],
) -> TrainConfig:
    if not isinstance(resume_train_cfg, dict):
        return train_cfg

    fields = (
        "speaker_inversion_enabled",
        "speaker_inversion_tokens",
        "speaker_inversion_init_std",
        "speaker_inversion_residual_regularization_weight",
        "speaker_inversion_max_relative_residual_norm",
    )
    updates: dict[str, object] = {}
    for field in fields:
        if field not in resume_train_cfg:
            continue
        resume_value = resume_train_cfg[field]
        flags = (
            ("--speaker-inversion", "--no-speaker-inversion")
            if field == "speaker_inversion_enabled"
            else ("--" + field.replace("_", "-"),)
        )
        if any(cli_provided(raw_argv, flag) for flag in flags):
            current_value = getattr(train_cfg, field)
            if current_value != resume_value:
                raise ValueError(
                    f"Resume checkpoint expects train.{field}={resume_value!r}, "
                    f"but the command line requests {current_value!r}."
                )
            continue
        updates[field] = resume_value

    if updates:
        train_cfg = replace(train_cfg, **updates)
    return train_cfg


def _initialize_base_model_from_pretrained_embeddings(
    raw_model: torch.nn.Module,
    *,
    model_cfg: ModelConfig,
    distributed: bool,
    is_main_process: bool,
) -> None:
    if model_cfg.use_pretrained_text_encoder:
        if is_main_process:
            print(
                "Using trainable pretrained text encoder with "
                f"condition projector(s): {model_cfg.text_tokenizer_repo}"
            )
        return
    if distributed:
        if is_main_process:
            print(
                f"Initializing text embedding from pretrained model: {model_cfg.text_tokenizer_repo}"
            )
            initialize_text_embedding_from_pretrained(
                raw_model,
                model_cfg,
                local_files_only=False,
            )
            if model_cfg.use_caption_condition:
                print(
                    "Initializing caption embedding from pretrained model: "
                    f"{model_cfg.caption_tokenizer_repo_resolved}"
                )
                initialize_caption_embedding_from_pretrained(
                    raw_model,
                    model_cfg,
                    local_files_only=False,
                )
        dist.barrier()
        if not is_main_process:
            initialize_text_embedding_from_pretrained(
                raw_model,
                model_cfg,
                local_files_only=True,
            )
            if model_cfg.use_caption_condition:
                initialize_caption_embedding_from_pretrained(
                    raw_model,
                    model_cfg,
                    local_files_only=True,
                )
        dist.barrier()
        return

    if is_main_process:
        print(f"Initializing text embedding from pretrained model: {model_cfg.text_tokenizer_repo}")
    initialize_text_embedding_from_pretrained(
        raw_model,
        model_cfg,
        local_files_only=False,
    )
    if model_cfg.use_caption_condition:
        if is_main_process:
            print(
                "Initializing caption embedding from pretrained model: "
                f"{model_cfg.caption_tokenizer_repo_resolved}"
            )
        initialize_caption_embedding_from_pretrained(
            raw_model,
            model_cfg,
            local_files_only=False,
        )


def _apply_base_initialization(
    raw_model: torch.nn.Module,
    *,
    model_cfg: ModelConfig,
    base_init: dict | None,
    distributed: bool,
    is_main_process: bool,
    preloaded_checkpoint: tuple[dict[str, torch.Tensor], dict | None, dict | None, dict | None]
    | None = None,
) -> None:
    mode = None if base_init is None else base_init.get("mode")
    if mode is None:
        _initialize_base_model_from_pretrained_embeddings(
            raw_model,
            model_cfg=model_cfg,
            distributed=distributed,
            is_main_process=is_main_process,
        )
        return

    if mode == "checkpoint":
        checkpoint_path = base_init.get("checkpoint_path")
        if not isinstance(checkpoint_path, str) or not checkpoint_path:
            raise ValueError("LoRA checkpoint metadata is missing base_init.checkpoint_path.")
        init_path = _normalize_checkpoint_path(checkpoint_path)
        if preloaded_checkpoint is None:
            init_state, init_model_cfg, _, _ = _load_model_state_from_checkpoint(init_path)
        else:
            init_state, init_model_cfg, _, _ = preloaded_checkpoint
        checkpoint_has_caption = checkpoint_uses_caption_condition(init_model_cfg, init_state)
        current_has_caption = bool(model_cfg.use_caption_condition)
        checkpoint_has_duration = checkpoint_uses_duration_predictor(init_model_cfg, init_state)
        current_has_duration = bool(model_cfg.use_duration_predictor)
        checkpoint_uses_pretrained_text_encoder = False
        if isinstance(init_model_cfg, dict):
            checkpoint_cfg = merge_dataclass_overrides(
                ModelConfig(),
                init_model_cfg,
                section="checkpoint model_config",
            )
            checkpoint_uses_pretrained_text_encoder = checkpoint_cfg.use_pretrained_text_encoder
        elif any(key.startswith("pretrained_text_backbone.") for key in init_state):
            checkpoint_uses_pretrained_text_encoder = True
        upgrade_text_encoder = bool(
            model_cfg.use_pretrained_text_encoder
            and not checkpoint_uses_pretrained_text_encoder
            and any(
                key.startswith("text_encoder.text_embedding.")
                or key.startswith("text_encoder.blocks.")
                for key in init_state
            )
        )
        drop_duration = checkpoint_has_duration and not current_has_duration
        if checkpoint_has_caption and not current_has_caption:
            raise ValueError(
                "Caption-conditioned checkpoint cannot initialize a caption-free config. "
                "Use a caption-enabled config for this checkpoint."
            )
        if drop_duration and not (current_has_caption and not checkpoint_has_caption):
            raise ValueError(
                "Duration-predictor checkpoint cannot initialize a duration-free config. "
                "Use a duration-enabled config for this checkpoint, or initialize a "
                "caption-enabled phase-1 VoiceDesign model from a caption-free base checkpoint."
            )

        require_caption_match = checkpoint_has_caption and current_has_caption
        checkpoint_speaker_patch = None
        if isinstance(init_model_cfg, dict):
            checkpoint_speaker_patch = checkpoint_cfg.speaker_patch_size
        upgrade_speaker_patch = bool(
            model_cfg.use_speaker_condition_resolved
            and checkpoint_speaker_patch is not None
            and int(checkpoint_speaker_patch) > 0
            and int(model_cfg.speaker_patch_size) > int(checkpoint_speaker_patch)
            and int(model_cfg.speaker_patch_size) % int(checkpoint_speaker_patch) == 0
        )
        _check_model_config_compatibility(
            init_path,
            init_model_cfg,
            model_cfg,
            require_caption_match=require_caption_match,
            upgrade_speaker_patch=upgrade_speaker_patch,
            upgrade_text_encoder=upgrade_text_encoder,
        )

        missing_keys: list[str] = []
        initialized_caption_embedding = False
        upgrade_caption = current_has_caption and not checkpoint_has_caption
        upgrade_duration = current_has_duration and not checkpoint_has_duration
        if (
            upgrade_caption
            or upgrade_duration
            or drop_duration
            or upgrade_speaker_patch
            or upgrade_text_encoder
        ):
            reinit_keys: set[str] = set()
            if upgrade_speaker_patch:
                reinit_keys.add(SPEAKER_IN_PROJ_WEIGHT_KEY)
            missing_keys, skipped_shape, skipped_extra = load_model_state_partially(
                raw_model,
                init_state,
                reinit_keys=reinit_keys,
            )
            validate_checkpoint_upgrade_partial_load(
                init_path,
                missing_keys,
                skipped_shape,
                skipped_extra,
                allow_caption_missing=upgrade_caption,
                allow_duration_missing=upgrade_duration,
                allow_duration_extra=drop_duration,
                allow_speaker_extra=(
                    upgrade_caption and not model_cfg.use_speaker_condition_resolved
                ),
                allow_text_encoder_replacement=upgrade_text_encoder,
            )
        else:
            raw_model.load_state_dict(init_state, strict=True)

        if upgrade_speaker_patch:
            old_speaker_projection = init_state.get(SPEAKER_IN_PROJ_WEIGHT_KEY)
            if old_speaker_projection is None:
                raise ValueError(
                    f"Checkpoint is missing {SPEAKER_IN_PROJ_WEIGHT_KEY!r}; cannot upgrade "
                    "speaker_patch_size."
                )
            raw_model.speaker_encoder.upgrade_patch_projection(
                old_speaker_projection,
                old_patch_size=int(checkpoint_speaker_patch),
                new_patch_size=int(model_cfg.speaker_patch_size),
            )
            if is_main_process:
                print(
                    f"Upgraded speaker_patch_size {checkpoint_speaker_patch}->"
                    f"{model_cfg.speaker_patch_size}: reinitialized "
                    f"{SPEAKER_IN_PROJ_WEIGHT_KEY} from the previous projection."
                )

        if upgrade_caption and not model_cfg.use_pretrained_text_encoder:
            if distributed:
                if is_main_process:
                    print(
                        "Initializing caption embedding from pretrained model after caption-free checkpoint load: "
                        f"{model_cfg.caption_tokenizer_repo_resolved}"
                    )
                    initialize_caption_embedding_from_pretrained(
                        raw_model,
                        model_cfg,
                        local_files_only=False,
                    )
                dist.barrier()
                if not is_main_process:
                    initialize_caption_embedding_from_pretrained(
                        raw_model,
                        model_cfg,
                        local_files_only=True,
                    )
                dist.barrier()
            else:
                if is_main_process:
                    print(
                        "Initializing caption embedding from pretrained model after caption-free checkpoint load: "
                        f"{model_cfg.caption_tokenizer_repo_resolved}"
                    )
                initialize_caption_embedding_from_pretrained(
                    raw_model,
                    model_cfg,
                    local_files_only=False,
                )
            initialized_caption_embedding = True

        if is_main_process:
            print(f"Initialized model weights from: {init_path}")
            if missing_keys:
                print(f"Partial load missing keys: {len(missing_keys)}")
            if current_has_duration and not checkpoint_has_duration:
                print("Duration predictor was randomly initialized.")
            if upgrade_text_encoder:
                print(
                    "Replaced checkpoint scratch text/caption encoders with "
                    "a trainable pretrained backbone and new projector(s): "
                    f"{model_cfg.text_tokenizer_repo}"
                )
            if initialized_caption_embedding:
                print("Caption embedding was initialized from its pretrained tokenizer backbone.")
        return

    raise ValueError(f"Unsupported base_init mode: {mode!r}")


def resolve_dist_env() -> tuple[int, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    return rank, world_size, local_rank


def setup_distributed(device_arg: str) -> tuple[int, int, int, bool, torch.device]:
    rank, world_size, local_rank = resolve_dist_env()
    distributed = world_size > 1
    if distributed:
        if not str(device_arg).startswith("cuda"):
            raise ValueError(
                f"WORLD_SIZE={world_size} detected, but --device={device_arg!r}. "
                "DDP multi-GPU training requires --device cuda."
            )
        if not torch.cuda.is_available():
            raise RuntimeError("WORLD_SIZE>1 detected, but CUDA is not available.")
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device(device_arg)
    return rank, world_size, local_rank, distributed, device


def reduce_mean(value: torch.Tensor, world_size: int, distributed: bool) -> torch.Tensor:
    reduced = value.detach().clone()
    if not distributed:
        return reduced
    dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
    reduced /= float(world_size)
    return reduced


def reduce_sum(value: torch.Tensor, distributed: bool) -> torch.Tensor:
    reduced = value.detach().clone()
    if distributed:
        dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
    return reduced


def _move_batch_to_device(batch: dict, device: torch.device) -> dict:
    return {
        key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def _record_batch_stream(batch: dict, stream: torch.cuda.Stream) -> None:
    for value in batch.values():
        if isinstance(value, torch.Tensor):
            value.record_stream(stream)


def cuda_prefetch_batches(loader, *, device: torch.device, enabled: bool):
    """
    現在バッチを専用 CUDA stream で転送し、再開用 iterator 位置を先へ進めず返す。

    次バッチの取得は現在バッチを yield した後に行うため、計算との先読み重複よりも
    チェックポイントへ保存する DataLoader 位置の一致を優先する。

    Args:
        loader (Iterable[dict]): 転送対象のバッチを返す DataLoader
        device (torch.device): バッチを転送する CUDA デバイス
        enabled (bool): 非同期転送を有効にするか

    Yields:
        dict: 指定デバイスへ転送済みのバッチ
    """

    if not enabled or device.type != "cuda":
        yield from loader
        return

    stream = torch.cuda.Stream(device=device)
    iterator = iter(loader)
    next_batch = None

    def preload() -> None:
        nonlocal next_batch
        try:
            cpu_batch = next(iterator)
        except StopIteration:
            next_batch = None
            return
        with torch.cuda.stream(stream):
            next_batch = _move_batch_to_device(cpu_batch, device)

    preload()
    while next_batch is not None:
        current_stream = torch.cuda.current_stream(device)
        current_stream.wait_stream(stream)
        batch = next_batch
        _record_batch_stream(batch, current_stream)
        yield batch
        # 再開位置を保存するまでは DataLoader iterator を次へ進めず、保存位置と学習済み位置を一致させる
        preload()


class LengthGroupedSampler(Sampler[int]):
    """
    Randomly samples the full dataset while grouping nearby-length examples into batches.

    Each epoch starts from a global random permutation. The permutation is split into
    random windows; only examples inside each window are sorted by length, then the
    resulting batches are shuffled again. This avoids a short-to-long curriculum while
    reducing per-batch padding.
    """

    def __init__(
        self,
        lengths: torch.Tensor,
        *,
        batch_size: int,
        window_batches: int,
        num_replicas: int = 1,
        rank: int = 0,
        seed: int = 0,
        drop_last: bool = True,
    ) -> None:
        if lengths.ndim != 1:
            raise ValueError(f"lengths must be 1D, got shape={tuple(lengths.shape)}")
        if batch_size <= 0:
            raise ValueError(f"batch_size must be > 0, got {batch_size}")
        if window_batches <= 0:
            raise ValueError(f"window_batches must be > 0, got {window_batches}")
        if num_replicas <= 0:
            raise ValueError(f"num_replicas must be > 0, got {num_replicas}")
        if not (0 <= rank < num_replicas):
            raise ValueError(f"rank must be in [0, {num_replicas}), got {rank}")
        self.lengths = lengths.detach().to(device="cpu", dtype=torch.int64).contiguous()
        self.batch_size = int(batch_size)
        self.window_batches = int(window_batches)
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.seed = int(seed)
        self.drop_last = bool(drop_last)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _global_batch_size(self) -> int:
        return self.batch_size * self.num_replicas

    def _target_size(self) -> int:
        dataset_size = int(self.lengths.numel())
        global_batch_size = self._global_batch_size()
        if self.drop_last:
            return (dataset_size // global_batch_size) * global_batch_size
        if self.num_replicas == 1:
            return dataset_size
        return ((dataset_size + global_batch_size - 1) // global_batch_size) * global_batch_size

    def __len__(self) -> int:
        target_size = self._target_size()
        if self.num_replicas == 1:
            return target_size
        return target_size // self.num_replicas

    @staticmethod
    def _take_permutation(size: int, *, generator: torch.Generator) -> list[int]:
        if size <= 0:
            return []
        return torch.randperm(size, generator=generator, dtype=torch.int64).tolist()

    def __iter__(self):
        dataset_size = int(self.lengths.numel())
        if dataset_size <= 0:
            return iter(())
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)

        indices = self._take_permutation(dataset_size, generator=generator)
        target_size = self._target_size()
        if target_size <= 0:
            return iter(())
        if target_size < len(indices):
            indices = indices[:target_size]
        elif target_size > len(indices):
            repeats = target_size - len(indices)
            base_indices = list(indices)
            while repeats > 0:
                take = min(repeats, len(base_indices))
                indices.extend(base_indices[:take])
                repeats -= take

        global_batch_size = self._global_batch_size()
        window_size = max(global_batch_size, self.window_batches * global_batch_size)
        rank_start = self.rank * self.batch_size
        rank_end = rank_start + self.batch_size
        local_indices: list[int] = []

        for window_start in range(0, len(indices), window_size):
            window = indices[window_start : window_start + window_size]
            window_indices = torch.tensor(window, dtype=torch.int64)
            window_lengths = self.lengths.index_select(0, window_indices)
            sorted_order = torch.argsort(window_lengths, descending=True, stable=True)
            window = window_indices.index_select(0, sorted_order).tolist()
            batches = [
                window[i : i + global_batch_size]
                for i in range(0, len(window), global_batch_size)
                if len(window[i : i + global_batch_size]) > 0
            ]
            batch_order = self._take_permutation(len(batches), generator=generator)
            for batch_index in batch_order:
                batch = batches[batch_index]
                if self.num_replicas > 1 and len(batch) < global_batch_size:
                    continue
                if len(batch) > 1:
                    order = torch.randperm(
                        len(batch),
                        generator=generator,
                        dtype=torch.int64,
                    ).tolist()
                    batch = [batch[i] for i in order]
                local_indices.extend(batch[rank_start:rank_end])

        return iter(local_indices)


def duration_condition_group_totals(
    *,
    duration_loss_per_sample: torch.Tensor,
    pred_frames: torch.Tensor,
    target_frames: torch.Tensor,
    has_speaker: torch.Tensor | None,
    has_caption: torch.Tensor | None = None,
) -> torch.Tensor:
    totals = torch.zeros(
        DURATION_CONDITION_GROUP_TOTAL_SIZE,
        device=duration_loss_per_sample.device,
        dtype=torch.float64,
    )
    device = duration_loss_per_sample.device
    speaker_mask = (
        has_speaker.to(device=device, dtype=torch.bool) if has_speaker is not None else None
    )
    caption_mask = (
        has_caption.to(device=device, dtype=torch.bool) if has_caption is not None else None
    )
    mae_per_sample = (pred_frames.float() - target_frames.float()).abs()

    group_masks: dict[str, torch.Tensor | None] = {
        "speaker": speaker_mask,
        "no_speaker": None if speaker_mask is None else ~speaker_mask,
        "caption": caption_mask,
        "no_caption": None if caption_mask is None else ~caption_mask,
        "speaker_caption": None
        if speaker_mask is None or caption_mask is None
        else speaker_mask & caption_mask,
        "speaker_no_caption": None
        if speaker_mask is None or caption_mask is None
        else speaker_mask & (~caption_mask),
        "no_speaker_caption": None
        if speaker_mask is None or caption_mask is None
        else (~speaker_mask) & caption_mask,
        "no_speaker_no_caption": None
        if speaker_mask is None or caption_mask is None
        else (~speaker_mask) & (~caption_mask),
    }

    for group_index, group_name in enumerate(DURATION_CONDITION_GROUPS):
        mask = group_masks[group_name]
        if mask is None or not mask.any():
            continue
        offset = group_index * 3
        totals[offset] = duration_loss_per_sample[mask].detach().double().sum()
        totals[offset + 1] = mae_per_sample[mask].detach().double().sum()
        totals[offset + 2] = mask.sum().double()
    return totals


def duration_condition_group_metrics(totals: torch.Tensor) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for group_index, group_name in enumerate(DURATION_CONDITION_GROUPS):
        offset = group_index * 3
        count = max(float(totals[offset + 2].item()), 0.0)
        metrics[f"duration_loss_{group_name}"] = (
            float(totals[offset].item() / count) if count > 0.0 else 0.0
        )
        metrics[f"duration_mae_frames_{group_name}"] = (
            float(totals[offset + 1].item() / count) if count > 0.0 else 0.0
        )
        metrics[f"duration_samples_{group_name}"] = count
    return metrics


def duration_condition_group_log_suffix(metrics: dict[str, float]) -> str:
    groups = [
        ("sp", "speaker"),
        ("no_sp", "no_speaker"),
        ("cap", "caption"),
        ("no_cap", "no_caption"),
        ("sp_cap", "speaker_caption"),
        ("sp_no_cap", "speaker_no_caption"),
        ("no_sp_cap", "no_speaker_caption"),
        ("no_sp_no_cap", "no_speaker_no_caption"),
    ]
    chunks: list[str] = []
    for label, group in groups:
        count = metrics.get(f"duration_samples_{group}", 0.0)
        if count <= 0.0:
            continue
        chunks.append(
            "{}={:.6f} mae_{}={:.2f} n_{}={:.0f}".format(
                f"dur_{label}",
                metrics[f"duration_loss_{group}"],
                label,
                metrics[f"duration_mae_frames_{group}"],
                label,
                count,
            )
        )
    return " ".join(chunks)


def duration_condition_group_wandb_metrics(
    prefix: str,
    metrics: dict[str, float],
) -> dict[str, float]:
    out: dict[str, float] = {}
    for group_name in DURATION_CONDITION_GROUPS:
        for metric_name in (
            f"duration_loss_{group_name}",
            f"duration_mae_frames_{group_name}",
            f"duration_samples_{group_name}",
        ):
            out[f"{prefix}/{metric_name}"] = metrics[metric_name]
    return out


def split_train_valid_indices(
    *,
    num_samples: int,
    valid_ratio: float,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if valid_ratio <= 0.0:
        return torch.arange(num_samples, dtype=torch.int64), torch.empty(0, dtype=torch.int64)
    if num_samples < 2:
        raise ValueError(
            f"Validation split requires at least 2 samples in manifest, got {num_samples}."
        )

    valid_count = int(num_samples * valid_ratio)
    valid_count = max(1, valid_count)
    if valid_count >= num_samples:
        valid_count = num_samples - 1

    generator = torch.Generator()
    generator.manual_seed(int(seed))
    perm = torch.randperm(num_samples, generator=generator)
    valid_indices = torch.sort(perm[:valid_count]).values
    train_indices = torch.sort(perm[valid_count:]).values
    if train_indices.numel() == 0 or valid_indices.numel() == 0:
        raise ValueError(
            "Failed to create non-empty train/valid split. "
            f"num_samples={num_samples} valid_ratio={valid_ratio}"
        )
    return train_indices, valid_indices


def run_validation(
    *,
    model,
    loader: TorchDataLoader,
    train_cfg: TrainConfig,
    device: torch.device,
    use_bf16: bool,
    distributed: bool,
) -> dict[str, float]:
    was_training = model.training
    model_cfg = model.module.cfg if isinstance(model, DDP) else model.cfg
    duration_only = train_cfg.train_mode == "duration_only"
    model.eval()
    totals = torch.zeros(
        6 + DURATION_CONDITION_GROUP_TOTAL_SIZE,
        device=device,
        dtype=torch.float64,
    )

    with torch.no_grad():
        for batch in loader:
            text_ids = batch["text_ids"].to(device, non_blocking=True)
            text_mask = batch["text_mask"].to(device, non_blocking=True)
            caption_ids = None
            caption_mask = None
            has_caption = None
            if model_cfg.use_caption_condition:
                caption_ids = batch["caption_ids"].to(device, non_blocking=True)
                caption_mask = batch["caption_mask"].to(device, non_blocking=True)
                has_caption = batch["has_caption"].to(device, non_blocking=True)
            num_frames = batch["num_frames"].to(device, non_blocking=True)
            duration_features = batch["duration_features"].to(device, non_blocking=True)
            ref_latent = None
            ref_mask = None
            if model_cfg.use_speaker_condition_resolved:
                ref_latent = batch["ref_latent_patched"].to(device, non_blocking=True)
                ref_mask = batch["ref_latent_mask_patched"].to(device, non_blocking=True)
                has_speaker = batch["has_speaker"].to(device, non_blocking=True)
            else:
                has_speaker = None

            bsz = text_ids.shape[0]
            x0 = None
            x_mask = None
            x_mask_valid = None
            x_t = None
            t = None
            v_target = None
            if not duration_only:
                x0 = batch["latent_patched"].to(device, non_blocking=True)
                x_mask = batch["latent_mask_patched"].to(device, non_blocking=True)
                x_mask_valid = batch["latent_mask_valid_patched"].to(device, non_blocking=True)
                if train_cfg.timestep_stratified:
                    t = sample_stratified_logit_normal_t(
                        batch_size=bsz,
                        device=device,
                        mean=train_cfg.timestep_logit_mean,
                        std=train_cfg.timestep_logit_std,
                        t_min=train_cfg.timestep_min,
                        t_max=train_cfg.timestep_max,
                    )
                else:
                    t = sample_logit_normal_t(
                        batch_size=bsz,
                        device=device,
                        mean=train_cfg.timestep_logit_mean,
                        std=train_cfg.timestep_logit_std,
                        t_min=train_cfg.timestep_min,
                        t_max=train_cfg.timestep_max,
                    )
                noise = torch.randn_like(x0)
                x_t = rf_interpolate(x0, noise, t)
                v_target = rf_velocity_target(x0, noise)

            if model_cfg.use_speaker_condition_resolved:
                if train_cfg.speaker_inversion_enabled:
                    # Speaker Inversion learns one embedding for this run, so validation
                    # should match training and treat every sample as speaker-conditioned.
                    use_speaker = torch.ones((bsz,), device=device, dtype=torch.bool)
                else:
                    use_speaker = has_speaker
                speaker_condition_dropout = ~use_speaker
                duration_has_speaker = use_speaker
                duration_features = set_duration_has_speaker_feature(
                    duration_features,
                    duration_has_speaker,
                )
            else:
                speaker_condition_dropout = None
                duration_has_speaker = None
            duration_has_caption = has_caption if model_cfg.use_caption_condition else None

            with (
                torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                if use_bf16
                else nullcontext()
            ):
                if duration_only:
                    duration_pred = model(
                        x_t=None,
                        t=None,
                        text_input_ids=text_ids,
                        text_mask=text_mask,
                        ref_latent=ref_latent,
                        ref_mask=ref_mask,
                        caption_input_ids=caption_ids,
                        caption_mask=caption_mask,
                        latent_mask=None,
                        duration_features=duration_features,
                        duration_has_speaker=duration_has_speaker,
                        duration_has_caption=duration_has_caption,
                        duration_only=True,
                    )
                    v_pred = None
                elif model_cfg.use_duration_predictor:
                    v_pred, duration_pred = model(
                        x_t=x_t,
                        t=t,
                        text_input_ids=text_ids,
                        text_mask=text_mask,
                        ref_latent=ref_latent,
                        ref_mask=ref_mask,
                        caption_input_ids=caption_ids,
                        caption_mask=caption_mask,
                        latent_mask=x_mask,
                        speaker_condition_dropout=speaker_condition_dropout,
                        duration_features=duration_features,
                        duration_has_speaker=duration_has_speaker,
                        duration_has_caption=duration_has_caption,
                    )
                else:
                    if model_cfg.use_speaker_condition_resolved:
                        ref_mask = ref_mask & use_speaker[:, None]
                        ref_latent = ref_latent * use_speaker[:, None, None].to(ref_latent.dtype)
                    v_pred = model(
                        x_t=x_t,
                        t=t,
                        text_input_ids=text_ids,
                        text_mask=text_mask,
                        ref_latent=ref_latent,
                        ref_mask=ref_mask,
                        caption_input_ids=caption_ids,
                        caption_mask=caption_mask,
                        latent_mask=x_mask,
                    )
                    duration_pred = None

            rf_loss = torch.zeros((), device=device, dtype=torch.float32)
            if not duration_only:
                if v_pred is None or v_target is None or x_mask is None or x_mask_valid is None:
                    raise RuntimeError("RF validation tensors are missing.")
                v_pred = v_pred.float()
                rf_loss = compute_rf_loss(
                    pred=v_pred,
                    target=v_target.float(),
                    loss_mask=x_mask,
                    valid_mask=x_mask_valid,
                    mode=train_cfg.rf_loss_mode,
                )
            duration_loss = torch.zeros((), device=device, dtype=torch.float32)
            duration_mae_frames = torch.zeros((), device=device, dtype=torch.float32)
            if model_cfg.use_duration_predictor:
                if duration_pred is None:
                    raise RuntimeError(
                        "Duration predictor is enabled but duration_pred is missing."
                    )
                duration_target = torch.log1p(num_frames.float())
                duration_loss_per_sample = F.huber_loss(
                    duration_pred.float(),
                    duration_target,
                    delta=float(train_cfg.duration_huber_delta),
                    reduction="none",
                )
                duration_loss = duration_loss_per_sample.mean()
                pred_frames = torch.expm1(duration_pred.float()).clamp_min(0.0)
                duration_mae_frames = (pred_frames - num_frames.float()).abs().mean()
                if duration_only:
                    totals[6:] += duration_condition_group_totals(
                        duration_loss_per_sample=duration_loss_per_sample,
                        pred_frames=pred_frames,
                        target_frames=num_frames.float(),
                        has_speaker=has_speaker,
                        has_caption=has_caption,
                    )
            if duration_only:
                loss = duration_loss
            else:
                loss = rf_loss + (float(train_cfg.duration_loss_weight) * duration_loss)

            weight = float(bsz)
            totals[0] += loss.detach().double() * weight
            totals[1] += rf_loss.detach().double() * weight
            totals[2] += duration_loss.detach().double() * weight
            totals[3] += duration_mae_frames.detach().double() * weight
            totals[4] += float(num_frames.detach().float().mean().item()) * weight
            totals[5] += weight

    if distributed:
        dist.all_reduce(totals, op=dist.ReduceOp.SUM)
    denom = max(float(totals[5].item()), 1.0)
    metrics = {
        "loss": float(totals[0].item() / denom),
        "rf_loss": float(totals[1].item() / denom),
        "duration_loss": float(totals[2].item() / denom),
        "duration_mae_frames": float(totals[3].item() / denom),
        "target_frames_mean": float(totals[4].item() / denom),
        "num_samples": float(totals[5].item()),
    }
    if duration_only:
        metrics.update(duration_condition_group_metrics(totals[6:]))
    if was_training:
        model.train()
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Irodori-TTS.")
    parser.add_argument(
        "--config",
        required=True,
        help="YAML config path containing model/train settings.",
    )
    parser.add_argument(
        "--manifest",
        required=True,
        help="JSONL manifest with text+latent_path (optional speaker_id for reference sampling).",
    )
    parser.add_argument("--output-dir", default="outputs/irodori_tts")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--precision",
        choices=["fp32", "bf16"],
        default="bf16",
        help=(
            "Compute precision for model forward pass. "
            "Model weights and optimizer states remain FP32."
        ),
    )
    parser.add_argument(
        "--tf32",
        dest="allow_tf32",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable TF32 matmul/cuDNN kernels on CUDA for speed.",
    )
    parser.add_argument(
        "--compile-model",
        dest="compile_model",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable torch.compile for the training model.",
    )
    parser.add_argument(
        "--gradient-checkpointing",
        dest="gradient_checkpointing",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable activation checkpointing on diffusion blocks to reduce memory.",
    )
    parser.add_argument(
        "--train-mode",
        choices=sorted(TRAIN_MODES),
        default=None,
        help="Training objective: rf runs DiT/RF training; duration_only trains only the duration predictor.",
    )
    parser.add_argument(
        "--resume",
        default=None,
        help="Resume full training state from a training checkpoint (.pt or LoRA checkpoint dir).",
    )
    parser.add_argument(
        "--init-checkpoint",
        default=None,
        help=(
            "Initialize model weights from a checkpoint (.pt or .safetensors) and start a new run "
            "with fresh optimizer / scheduler state."
        ),
    )
    parser.add_argument("--max-steps", type=int, default=200000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=1,
        help=(
            "Number of micro-batches to accumulate before optimizer.step(). "
            "1 disables accumulation."
        ),
    )
    parser.add_argument(
        "--max-text-len",
        type=int,
        default=256,
        help="Maximum token length for text conditioning (right-truncated).",
    )
    parser.add_argument(
        "--max-caption-len",
        type=int,
        default=None,
        help="Maximum token length for caption conditioning (defaults to max_text_len).",
    )
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument(
        "--pretrained-text-encoder-learning-rate",
        type=float,
        default=1e-5,
        help=(
            "AdamW learning rate for a trainable pretrained text/caption backbone. "
            "The main scheduler multiplier is applied to this LR as well."
        ),
    )
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--optimizer", choices=["adamw", "muon"], default="muon")
    parser.add_argument("--adam-beta1", type=float, default=0.9)
    parser.add_argument("--adam-beta2", type=float, default=0.999)
    parser.add_argument("--adam-eps", type=float, default=1e-8)
    parser.add_argument("--muon-momentum", type=float, default=0.95)
    parser.add_argument("--lr-scheduler", choices=["none", "cosine", "wsd"], default="none")
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument(
        "--caption-warmup",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "During the first caption_warmup_steps optimizer steps, update only caption-only parameters "
            "(caption encoder/norm and caption attention projections)."
        ),
    )
    parser.add_argument(
        "--caption-warmup-steps",
        type=int,
        default=0,
        help="Number of optimizer steps to run caption-only warmup for when caption_warmup is enabled.",
    )
    parser.add_argument(
        "--pretrained-projector-warmup-steps",
        type=int,
        default=0,
        help=(
            "Update only the text/caption projectors for this many initial optimizer "
            "steps, then update the rest of the trainable TTS model and backbone."
        ),
    )
    parser.add_argument("--stable-steps", type=int, default=0)
    parser.add_argument("--min-lr-scale", type=float, default=0.1)
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--latent-patch-size", type=int, default=1)
    parser.add_argument("--max-latent-steps", type=int, default=750)
    parser.add_argument(
        "--ref-min-seconds",
        type=float,
        default=1.0,
        help=(
            "Minimum reference-audio length (seconds) sampled per training step "
            "when concatenating same-speaker clips to build a long reference."
        ),
    )
    parser.add_argument(
        "--ref-max-seconds",
        type=float,
        default=120.0,
        help=(
            "Maximum reference-audio length (seconds). Concat is capped here, "
            "and the sampled target length is drawn from [min, max]."
        ),
    )
    parser.add_argument(
        "--fixed-target-latent-steps",
        type=int,
        default=None,
        help=(
            "If set, always train on this fixed target latent length "
            "(short samples are right-padded with zeros, long samples are truncated)."
        ),
    )
    parser.add_argument(
        "--fixed-target-full-mask",
        action="store_true",
        help="Use full target mask for fixed-length training (Echo-style includes padded tail in loss).",
    )
    parser.add_argument(
        "--rf-loss-mode",
        choices=["echo", "utterance_mean"],
        default=None,
        help="RF loss normalization mode.",
    )
    parser.add_argument("--duration-loss-weight", type=float, default=None)
    parser.add_argument(
        "--duration-backprop-to-condition",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Allow joint RF+duration loss to update text/caption projectors and speaker "
            "conditioning through the duration predictor."
        ),
    )
    parser.add_argument("--duration-speaker-dropout", type=float, default=None)
    parser.add_argument("--duration-caption-dropout", type=float, default=None)
    parser.add_argument("--duration-huber-delta", type=float, default=None)
    parser.add_argument(
        "--text-condition-dropout",
        type=float,
        default=0.1,
        help="Probability of dropping text conditioning during training.",
    )
    parser.add_argument(
        "--caption-condition-dropout",
        type=float,
        default=0.1,
        help="Probability of dropping caption conditioning during training.",
    )
    parser.add_argument(
        "--speaker-condition-dropout",
        type=float,
        default=0.1,
        help="Probability of dropping speaker/reference conditioning during training.",
    )
    speaker_inversion_group = parser.add_mutually_exclusive_group()
    speaker_inversion_group.add_argument(
        "--speaker-inversion",
        dest="speaker_inversion_enabled",
        action="store_true",
        help="Train only learned speaker inversion embedding tokens.",
    )
    speaker_inversion_group.add_argument(
        "--no-speaker-inversion",
        dest="speaker_inversion_enabled",
        action="store_false",
        help="Disable Speaker Inversion training.",
    )
    parser.set_defaults(speaker_inversion_enabled=None)
    parser.add_argument(
        "--speaker-inversion-tokens",
        type=int,
        default=None,
        help="Number of learned Speaker Inversion tokens.",
    )
    parser.add_argument(
        "--speaker-inversion-init-std",
        type=float,
        default=None,
        help="Stddev for random Speaker Inversion token initialization.",
    )
    parser.add_argument(
        "--speaker-inversion-init-embedding",
        default=None,
        help=(
            "Optional existing Speaker Inversion .speaker.safetensors file used to "
            "warm-start a new optimization. Use --resume with .speaker.trainer.pt "
            "for exact continuation."
        ),
    )
    parser.add_argument(
        "--speaker-inversion-base-embedding",
        default=None,
        help="Reference-derived .speaker-base.safetensors used for residual inversion.",
    )
    parser.add_argument(
        "--speaker-inversion-residual-regularization-weight",
        type=float,
        default=None,
        help="Weight for residual energy relative to the fixed reference-derived base.",
    )
    parser.add_argument(
        "--speaker-inversion-max-relative-residual-norm",
        type=float,
        default=None,
        help="Optional hard limit on residual norm relative to the fixed base norm.",
    )
    parser.add_argument(
        "--timestep-stratified",
        action="store_true",
        help="Use stratified logit-normal timestep sampling (Echo-style).",
    )
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument(
        "--checkpoint-best-n",
        type=int,
        default=0,
        help=(
            "Keep up to N best validation-loss checkpoints in addition to latest. "
            "When validation is disabled, keeps latest N+1 periodic checkpoints. "
            "Set 0 to disable checkpoint-count limiting."
        ),
    )
    parser.add_argument(
        "--valid-ratio",
        type=float,
        default=0.0,
        help=("Split ratio for validation set from the single manifest. 0 disables validation."),
    )
    parser.add_argument(
        "--valid-every",
        type=int,
        default=0,
        help=("Run validation every N training steps. Set <=0 to disable validation."),
    )
    parser.add_argument(
        "--progress",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable tqdm progress bar.",
    )
    parser.add_argument(
        "--progress-all",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Show tqdm progress bars for all ranks in DDP mode (default: rank0 only).",
    )
    wandb_group = parser.add_mutually_exclusive_group()
    wandb_group.add_argument(
        "--wandb",
        dest="wandb_enabled",
        action="store_true",
        help="Enable Weights & Biases logging.",
    )
    wandb_group.add_argument(
        "--no-wandb",
        dest="wandb_enabled",
        action="store_false",
        help="Disable Weights & Biases logging.",
    )
    parser.set_defaults(wandb_enabled=None)
    parser.add_argument(
        "--wandb-project",
        default=None,
        help="Weights & Biases project name.",
    )
    parser.add_argument(
        "--wandb-entity",
        default=None,
        help="Weights & Biases entity/team name.",
    )
    parser.add_argument(
        "--wandb-run-name",
        default=None,
        help="Weights & Biases run name.",
    )
    parser.add_argument(
        "--wandb-mode",
        choices=sorted(WANDB_MODES),
        default=None,
        help="Weights & Biases mode.",
    )
    lora_group = parser.add_mutually_exclusive_group()
    lora_group.add_argument(
        "--lora",
        dest="lora_enabled",
        action="store_true",
        help="Enable PEFT LoRA fine-tuning.",
    )
    lora_group.add_argument(
        "--no-lora",
        dest="lora_enabled",
        action="store_false",
        help="Disable PEFT LoRA fine-tuning.",
    )
    parser.set_defaults(lora_enabled=None)
    parser.add_argument("--lora-r", type=int, default=None, help="LoRA rank.")
    parser.add_argument("--lora-alpha", type=int, default=None, help="LoRA alpha scaling.")
    parser.add_argument(
        "--lora-dropout",
        type=float,
        default=None,
        help="LoRA dropout probability.",
    )
    parser.add_argument(
        "--lora-bias",
        choices=["none", "all", "lora_only"],
        default=None,
        help="Bias handling passed to PEFT LoRA.",
    )
    parser.add_argument(
        "--lora-target-modules",
        default=None,
        help=(
            "LoRA target preset, regex, or comma-separated module suffix list. "
            f"Presets: {', '.join(sorted(LORA_TARGET_PRESETS))}."
        ),
    )
    parser.add_argument(
        "--lora-modules-to-save",
        default=None,
        help=(
            "Comma-separated full modules to keep trainable and save inside the LoRA adapter. "
            "Use 'auto' to save duration_predictor for duration-enabled models, or 'none' to disable."
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    ddp_group = parser.add_mutually_exclusive_group()
    ddp_group.add_argument(
        "--ddp-find-unused-parameters",
        dest="ddp_find_unused_parameters",
        action="store_true",
        help=(
            "Enable DDP find_unused_parameters. Useful when conditional branches "
            "(e.g., speaker/text conditioning) may be fully masked in some steps."
        ),
    )
    ddp_group.add_argument(
        "--no-ddp-find-unused-parameters",
        dest="ddp_find_unused_parameters",
        action="store_false",
        help="Disable DDP find_unused_parameters.",
    )
    parser.set_defaults(ddp_find_unused_parameters=None)
    args = parser.parse_args()
    if args.resume is not None and Path(args.resume).suffix.lower() == ".safetensors":
        raise ValueError(
            "--resume expects a training checkpoint (.pt, LoRA checkpoint dir, or "
            ".speaker.trainer.pt). "
            "Use --init-checkpoint for inference-only .safetensors weights."
        )

    rank, world_size, local_rank, distributed, device = setup_distributed(args.device)
    is_main_process = rank == 0

    raw_argv = sys.argv[1:]
    exp_cfg = load_config_yaml(args.config)
    unknown_root = sorted(set(exp_cfg) - {"model", "train"})
    if unknown_root:
        raise ValueError(f"Unknown top-level config keys: {unknown_root}")
    if is_main_process:
        print(f"Loaded config: {args.config}")
    model_cfg = merge_dataclass_overrides(ModelConfig(), exp_cfg.get("model"), section="model")
    train_cfg = merge_dataclass_overrides(TrainConfig(), exp_cfg.get("train"), section="train")
    default_train_cfg = TrainConfig()

    train_cfg = replace(train_cfg, manifest_path=args.manifest)
    if train_cfg.output_dir == default_train_cfg.output_dir and not cli_provided(
        raw_argv, "--output-dir"
    ):
        train_cfg = replace(train_cfg, output_dir=args.output_dir)

    if cli_provided(raw_argv, "--output-dir"):
        train_cfg = replace(train_cfg, output_dir=args.output_dir)
    if cli_provided(raw_argv, "--precision"):
        train_cfg = replace(train_cfg, precision=args.precision)
    if args.allow_tf32 is not None:
        train_cfg = replace(train_cfg, allow_tf32=args.allow_tf32)
    if args.compile_model is not None:
        train_cfg = replace(train_cfg, compile_model=args.compile_model)
    if args.gradient_checkpointing is not None:
        train_cfg = replace(train_cfg, gradient_checkpointing=args.gradient_checkpointing)
    if cli_provided(raw_argv, "--train-mode"):
        train_cfg = replace(train_cfg, train_mode=args.train_mode)
    if cli_provided(raw_argv, "--batch-size"):
        train_cfg = replace(train_cfg, batch_size=args.batch_size)
    if cli_provided(raw_argv, "--gradient-accumulation-steps"):
        train_cfg = replace(
            train_cfg,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
        )
    if cli_provided(raw_argv, "--max-text-len"):
        train_cfg = replace(train_cfg, max_text_len=args.max_text_len)
    if cli_provided(raw_argv, "--max-caption-len"):
        train_cfg = replace(train_cfg, max_caption_len=args.max_caption_len)
    if cli_provided(raw_argv, "--num-workers"):
        train_cfg = replace(train_cfg, num_workers=args.num_workers)
    if cli_provided(raw_argv, "--lr"):
        train_cfg = replace(train_cfg, learning_rate=args.lr)
    if cli_provided(raw_argv, "--pretrained-text-encoder-learning-rate"):
        train_cfg = replace(
            train_cfg,
            pretrained_text_encoder_learning_rate=(args.pretrained_text_encoder_learning_rate),
        )
    if cli_provided(raw_argv, "--weight-decay"):
        train_cfg = replace(train_cfg, weight_decay=args.weight_decay)
    if cli_provided(raw_argv, "--optimizer"):
        train_cfg = replace(train_cfg, optimizer=args.optimizer)
    if cli_provided(raw_argv, "--adam-beta1"):
        train_cfg = replace(train_cfg, adam_beta1=args.adam_beta1)
    if cli_provided(raw_argv, "--adam-beta2"):
        train_cfg = replace(train_cfg, adam_beta2=args.adam_beta2)
    if cli_provided(raw_argv, "--adam-eps"):
        train_cfg = replace(train_cfg, adam_eps=args.adam_eps)
    if cli_provided(raw_argv, "--muon-momentum"):
        train_cfg = replace(train_cfg, muon_momentum=args.muon_momentum)
    if cli_provided(raw_argv, "--lr-scheduler"):
        train_cfg = replace(train_cfg, lr_scheduler=args.lr_scheduler)
    if cli_provided(raw_argv, "--warmup-steps"):
        train_cfg = replace(train_cfg, warmup_steps=args.warmup_steps)
    if args.caption_warmup is not None:
        train_cfg = replace(train_cfg, caption_warmup=bool(args.caption_warmup))
    if cli_provided(raw_argv, "--caption-warmup-steps"):
        train_cfg = replace(train_cfg, caption_warmup_steps=args.caption_warmup_steps)
    if cli_provided(raw_argv, "--pretrained-projector-warmup-steps"):
        train_cfg = replace(
            train_cfg,
            pretrained_projector_warmup_steps=args.pretrained_projector_warmup_steps,
        )
    if cli_provided(raw_argv, "--stable-steps"):
        train_cfg = replace(train_cfg, stable_steps=args.stable_steps)
    if cli_provided(raw_argv, "--min-lr-scale"):
        train_cfg = replace(train_cfg, min_lr_scale=args.min_lr_scale)
    if cli_provided(raw_argv, "--max-steps"):
        train_cfg = replace(train_cfg, max_steps=args.max_steps)
    if cli_provided(raw_argv, "--text-condition-dropout"):
        train_cfg = replace(train_cfg, text_condition_dropout=args.text_condition_dropout)
    if cli_provided(raw_argv, "--caption-condition-dropout"):
        train_cfg = replace(train_cfg, caption_condition_dropout=args.caption_condition_dropout)
    if cli_provided(raw_argv, "--speaker-condition-dropout"):
        train_cfg = replace(train_cfg, speaker_condition_dropout=args.speaker_condition_dropout)
    if args.speaker_inversion_enabled is not None:
        train_cfg = replace(
            train_cfg,
            speaker_inversion_enabled=bool(args.speaker_inversion_enabled),
        )
    if cli_provided(raw_argv, "--speaker-inversion-tokens"):
        train_cfg = replace(train_cfg, speaker_inversion_tokens=args.speaker_inversion_tokens)
    if cli_provided(raw_argv, "--speaker-inversion-init-std"):
        train_cfg = replace(train_cfg, speaker_inversion_init_std=args.speaker_inversion_init_std)
    if cli_provided(raw_argv, "--speaker-inversion-init-embedding"):
        train_cfg = replace(
            train_cfg,
            speaker_inversion_init_embedding=args.speaker_inversion_init_embedding,
        )
    if cli_provided(raw_argv, "--speaker-inversion-base-embedding"):
        train_cfg = replace(
            train_cfg,
            speaker_inversion_base_embedding=args.speaker_inversion_base_embedding,
        )
    if cli_provided(raw_argv, "--speaker-inversion-residual-regularization-weight"):
        train_cfg = replace(
            train_cfg,
            speaker_inversion_residual_regularization_weight=(
                args.speaker_inversion_residual_regularization_weight
            ),
        )
    if cli_provided(raw_argv, "--speaker-inversion-max-relative-residual-norm"):
        train_cfg = replace(
            train_cfg,
            speaker_inversion_max_relative_residual_norm=(
                args.speaker_inversion_max_relative_residual_norm
            ),
        )
    if cli_provided(raw_argv, "--timestep-stratified"):
        train_cfg = replace(train_cfg, timestep_stratified=True)
    if cli_provided(raw_argv, "--max-latent-steps"):
        train_cfg = replace(train_cfg, max_latent_steps=args.max_latent_steps)
    if cli_provided(raw_argv, "--ref-min-seconds"):
        train_cfg = replace(train_cfg, ref_min_seconds=args.ref_min_seconds)
    if cli_provided(raw_argv, "--ref-max-seconds"):
        train_cfg = replace(train_cfg, ref_max_seconds=args.ref_max_seconds)
    if cli_provided(raw_argv, "--fixed-target-latent-steps"):
        train_cfg = replace(train_cfg, fixed_target_latent_steps=args.fixed_target_latent_steps)
    if cli_provided(raw_argv, "--fixed-target-full-mask"):
        train_cfg = replace(train_cfg, fixed_target_full_mask=True)
    if cli_provided(raw_argv, "--rf-loss-mode"):
        train_cfg = replace(train_cfg, rf_loss_mode=args.rf_loss_mode)
    if cli_provided(raw_argv, "--duration-loss-weight"):
        train_cfg = replace(train_cfg, duration_loss_weight=args.duration_loss_weight)
    if args.duration_backprop_to_condition is not None:
        train_cfg = replace(
            train_cfg,
            duration_backprop_to_condition=bool(args.duration_backprop_to_condition),
        )
    if cli_provided(raw_argv, "--duration-speaker-dropout"):
        train_cfg = replace(train_cfg, duration_speaker_dropout=args.duration_speaker_dropout)
    if cli_provided(raw_argv, "--duration-caption-dropout"):
        train_cfg = replace(train_cfg, duration_caption_dropout=args.duration_caption_dropout)
    if cli_provided(raw_argv, "--duration-huber-delta"):
        train_cfg = replace(train_cfg, duration_huber_delta=args.duration_huber_delta)
    if cli_provided(raw_argv, "--log-every"):
        train_cfg = replace(train_cfg, log_every=args.log_every)
    if cli_provided(raw_argv, "--save-every"):
        train_cfg = replace(train_cfg, save_every=args.save_every)
    if cli_provided(raw_argv, "--checkpoint-best-n"):
        train_cfg = replace(train_cfg, checkpoint_best_n=args.checkpoint_best_n)
    if cli_provided(raw_argv, "--valid-ratio"):
        train_cfg = replace(train_cfg, valid_ratio=args.valid_ratio)
    if cli_provided(raw_argv, "--valid-every"):
        train_cfg = replace(train_cfg, valid_every=args.valid_every)
    if args.progress is not None:
        train_cfg = replace(train_cfg, progress=args.progress)
    if args.progress_all is not None:
        train_cfg = replace(train_cfg, progress_all_ranks=args.progress_all)
    if args.wandb_enabled is not None:
        train_cfg = replace(train_cfg, wandb_enabled=args.wandb_enabled)
    if cli_provided(raw_argv, "--wandb-project"):
        train_cfg = replace(train_cfg, wandb_project=args.wandb_project)
    if cli_provided(raw_argv, "--wandb-entity"):
        train_cfg = replace(train_cfg, wandb_entity=args.wandb_entity)
    if cli_provided(raw_argv, "--wandb-run-name"):
        train_cfg = replace(train_cfg, wandb_run_name=args.wandb_run_name)
    if cli_provided(raw_argv, "--wandb-mode"):
        train_cfg = replace(train_cfg, wandb_mode=args.wandb_mode)
    if args.lora_enabled is not None:
        train_cfg = replace(train_cfg, lora_enabled=args.lora_enabled)
    if cli_provided(raw_argv, "--lora-r"):
        train_cfg = replace(train_cfg, lora_r=args.lora_r)
    if cli_provided(raw_argv, "--lora-alpha"):
        train_cfg = replace(train_cfg, lora_alpha=args.lora_alpha)
    if cli_provided(raw_argv, "--lora-dropout"):
        train_cfg = replace(train_cfg, lora_dropout=args.lora_dropout)
    if cli_provided(raw_argv, "--lora-bias"):
        train_cfg = replace(train_cfg, lora_bias=args.lora_bias)
    if cli_provided(raw_argv, "--lora-target-modules"):
        train_cfg = replace(train_cfg, lora_target_modules=args.lora_target_modules)
    if cli_provided(raw_argv, "--lora-modules-to-save"):
        train_cfg = replace(train_cfg, lora_modules_to_save=args.lora_modules_to_save)
    if args.ddp_find_unused_parameters is not None:
        train_cfg = replace(
            train_cfg,
            ddp_find_unused_parameters=args.ddp_find_unused_parameters,
        )
    if cli_provided(raw_argv, "--seed"):
        train_cfg = replace(train_cfg, seed=args.seed)

    resume_path = Path(args.resume).expanduser() if args.resume is not None else None
    resume_model_cfg = None
    resume_train_cfg = None
    resume_base_init = None
    resume_text_encoder_config = None
    resume_payload: dict[str, Any] | None = None
    if args.resume is not None:
        resume_payload = _load_checkpoint_payload(resume_path, map_location="cpu")
        raw_resume_model_cfg = resume_payload.get("model_config")
        if raw_resume_model_cfg is not None and not isinstance(raw_resume_model_cfg, dict):
            raise ValueError("Resume checkpoint model_config must be a dictionary when present.")
        resume_model_cfg = raw_resume_model_cfg
        raw_resume_train_cfg = resume_payload.get("train_config")
        if raw_resume_train_cfg is not None and not isinstance(raw_resume_train_cfg, dict):
            raise ValueError("Resume checkpoint train_config must be a dictionary when present.")
        resume_train_cfg = raw_resume_train_cfg
        raw_resume_base_init = resume_payload.get("base_init")
        if raw_resume_base_init is not None and not isinstance(raw_resume_base_init, dict):
            raise ValueError("Resume checkpoint base_init must be a dictionary when present.")
        resume_base_init = raw_resume_base_init
        raw_resume_text_encoder_config = resume_payload.get("text_encoder_config")
        if raw_resume_text_encoder_config is not None and not isinstance(
            raw_resume_text_encoder_config, dict
        ):
            raise ValueError(
                "Resume checkpoint text_encoder_config must be a dictionary when present."
            )
        resume_text_encoder_config = raw_resume_text_encoder_config
        train_cfg = _restore_resume_lora_config(
            train_cfg,
            resume_train_cfg=resume_train_cfg,
            args=args,
            raw_argv=raw_argv,
            exp_cfg=exp_cfg,
        )
        train_cfg = _restore_resume_speaker_inversion_config(
            train_cfg,
            resume_train_cfg=resume_train_cfg,
            raw_argv=raw_argv,
        )

    if cli_provided(raw_argv, "--latent-dim"):
        model_cfg = replace(model_cfg, latent_dim=args.latent_dim)
    if cli_provided(raw_argv, "--latent-patch-size"):
        model_cfg = replace(model_cfg, latent_patch_size=args.latent_patch_size)

    set_seed(train_cfg.seed + rank)
    if (
        model_cfg.use_pretrained_text_encoder
        and train_cfg.pretrained_text_encoder_learning_rate <= 0
    ):
        raise ValueError(
            "pretrained_text_encoder_learning_rate must be > 0, got "
            f"{train_cfg.pretrained_text_encoder_learning_rate}."
        )
    if (
        model_cfg.use_pretrained_text_encoder
        and model_cfg.use_caption_condition
        and model_cfg.caption_tokenizer_repo_resolved != model_cfg.text_tokenizer_repo
    ):
        raise ValueError(
            "Pretrained text/caption encoder sharing requires caption_tokenizer_repo "
            "to be unset or equal to text_tokenizer_repo."
        )
    if args.resume is not None:
        if model_cfg.use_pretrained_text_encoder and resume_model_cfg is None:
            raise ValueError(
                "Pretrained text encoder resume requires checkpoint model_config metadata "
                "to verify the backbone architecture and configuration."
            )
        if resume_path is None:
            raise RuntimeError("Resume path is unexpectedly missing.")
        _check_model_config_compatibility(
            resume_path,
            resume_model_cfg,
            model_cfg,
            require_caption_match=True,
        )
    if not (0.0 <= train_cfg.text_condition_dropout <= 1.0):
        raise ValueError(
            f"text_condition_dropout must be in [0, 1], got {train_cfg.text_condition_dropout}"
        )
    if train_cfg.max_text_len <= 0:
        raise ValueError(f"max_text_len must be > 0, got {train_cfg.max_text_len}")
    if str(train_cfg.train_mode).strip().lower() not in TRAIN_MODES:
        raise ValueError(
            f"train_mode must be one of {sorted(TRAIN_MODES)}, got {train_cfg.train_mode!r}"
        )
    train_cfg = replace(train_cfg, train_mode=str(train_cfg.train_mode).strip().lower())
    if train_cfg.max_caption_len is not None and train_cfg.max_caption_len <= 0:
        raise ValueError(f"max_caption_len must be > 0, got {train_cfg.max_caption_len}")
    if train_cfg.gradient_accumulation_steps <= 0:
        raise ValueError(
            f"gradient_accumulation_steps must be > 0, got {train_cfg.gradient_accumulation_steps}"
        )
    if not (0.0 <= train_cfg.speaker_condition_dropout <= 1.0):
        raise ValueError(
            "speaker_condition_dropout must be in [0, 1], "
            f"got {train_cfg.speaker_condition_dropout}"
        )
    if train_cfg.speaker_inversion_enabled:
        if not model_cfg.use_speaker_condition_resolved:
            raise ValueError(
                "speaker_inversion_enabled=True requires a speaker-conditioned model config."
            )
        if args.init_checkpoint is None and args.resume is None:
            raise ValueError(
                "speaker_inversion_enabled=True requires --init-checkpoint for a new run "
                "or --resume for a saved Speaker Inversion trainer state."
            )
        if args.resume is not None:
            if resume_payload is None or SPEAKER_EMBEDDING_KEY not in resume_payload:
                raise ValueError(
                    "Speaker Inversion resume checkpoint is missing the saved embedding."
                )
            if resume_base_init is None:
                raise ValueError(
                    "Speaker Inversion resume checkpoint is missing base_init metadata."
                )
        if train_config_uses_lora(train_cfg):
            raise ValueError("speaker_inversion_enabled=True does not support LoRA training.")
        if train_cfg.train_mode != "rf":
            raise ValueError("speaker_inversion_enabled=True supports train_mode='rf' only.")
        if train_cfg.caption_warmup:
            raise ValueError("speaker_inversion_enabled=True does not support caption_warmup.")
        if train_cfg.speaker_inversion_tokens <= 0:
            raise ValueError(
                f"speaker_inversion_tokens must be > 0, got {train_cfg.speaker_inversion_tokens}"
            )
        if train_cfg.speaker_inversion_init_std < 0:
            raise ValueError(
                "speaker_inversion_init_std must be >= 0, "
                f"got {train_cfg.speaker_inversion_init_std}"
            )
        if (
            train_cfg.speaker_inversion_init_embedding is not None
            and train_cfg.speaker_inversion_base_embedding is not None
        ):
            raise ValueError(
                "speaker_inversion_init_embedding and speaker_inversion_base_embedding "
                "cannot be used together."
            )
        if train_cfg.speaker_inversion_residual_regularization_weight < 0.0:
            raise ValueError(
                "speaker_inversion_residual_regularization_weight must be >= 0, got "
                f"{train_cfg.speaker_inversion_residual_regularization_weight}"
            )
        if (
            train_cfg.speaker_inversion_max_relative_residual_norm is not None
            and train_cfg.speaker_inversion_max_relative_residual_norm <= 0.0
        ):
            raise ValueError(
                "speaker_inversion_max_relative_residual_norm must be > 0 when provided, got "
                f"{train_cfg.speaker_inversion_max_relative_residual_norm}"
            )
        if (
            train_cfg.speaker_inversion_base_embedding is None
            and (
                train_cfg.speaker_inversion_residual_regularization_weight > 0.0
                or train_cfg.speaker_inversion_max_relative_residual_norm is not None
            )
            and args.resume is None
        ):
            raise ValueError(
                "Residual regularization and projection require speaker_inversion_base_embedding."
            )
        optimizer_explicit = cli_provided(raw_argv, "--optimizer") or (
            isinstance(exp_cfg.get("train"), dict) and "optimizer" in exp_cfg.get("train", {})
        )
        if str(train_cfg.optimizer).strip().lower() == "muon":
            if optimizer_explicit:
                raise ValueError(
                    "speaker_inversion_enabled=True supports optimizer='adamw'. "
                    "Muon has no compatible matrix parameter when only speaker tokens are trainable."
                )
            train_cfg = replace(train_cfg, optimizer="adamw")
    if not (0.0 <= train_cfg.caption_condition_dropout <= 1.0):
        raise ValueError(
            "caption_condition_dropout must be in [0, 1], "
            f"got {train_cfg.caption_condition_dropout}"
        )
    if train_cfg.fixed_target_latent_steps is not None and train_cfg.fixed_target_latent_steps <= 0:
        raise ValueError(
            "fixed_target_latent_steps must be > 0 when provided, "
            f"got {train_cfg.fixed_target_latent_steps}"
        )
    if train_cfg.fixed_target_full_mask and train_cfg.fixed_target_latent_steps is None:
        raise ValueError(
            "fixed_target_full_mask=True requires fixed_target_latent_steps to be set."
        )
    if str(train_cfg.rf_loss_mode).strip().lower() not in {"echo", "utterance_mean"}:
        raise ValueError(
            "rf_loss_mode must be one of ['echo', 'utterance_mean'], "
            f"got {train_cfg.rf_loss_mode!r}"
        )
    if train_cfg.duration_loss_weight < 0:
        raise ValueError(f"duration_loss_weight must be >= 0, got {train_cfg.duration_loss_weight}")
    if train_cfg.duration_backprop_to_condition:
        if not model_cfg.use_duration_predictor:
            raise ValueError(
                "duration_backprop_to_condition=True requires model.use_duration_predictor=True."
            )
        if train_cfg.train_mode != "rf":
            raise ValueError(
                "duration_backprop_to_condition=True is only supported for joint "
                "train_mode='rf' training."
            )
    if not (0.0 <= train_cfg.duration_speaker_dropout <= 1.0):
        raise ValueError(
            f"duration_speaker_dropout must be in [0, 1], got {train_cfg.duration_speaker_dropout}"
        )
    if not (0.0 <= train_cfg.duration_caption_dropout <= 1.0):
        raise ValueError(
            f"duration_caption_dropout must be in [0, 1], got {train_cfg.duration_caption_dropout}"
        )
    if train_cfg.duration_huber_delta <= 0:
        raise ValueError(f"duration_huber_delta must be > 0, got {train_cfg.duration_huber_delta}")
    if train_cfg.train_mode == "duration_only" and not model_cfg.use_duration_predictor:
        raise ValueError("train_mode='duration_only' requires model.use_duration_predictor=True.")
    if train_cfg.train_mode == "duration_only" and train_config_uses_lora(train_cfg):
        raise ValueError("train_mode='duration_only' does not support LoRA training.")
    if train_cfg.train_mode == "duration_only" and train_cfg.caption_warmup:
        raise ValueError("train_mode='duration_only' does not support caption_warmup.")
    if (
        train_cfg.train_mode == "duration_only"
        and args.init_checkpoint is None
        and args.resume is None
    ):
        raise ValueError(
            "train_mode='duration_only' requires --init-checkpoint or --resume "
            "so the frozen text/speaker encoders are initialized from trained weights."
        )
    if model_cfg.use_duration_predictor:
        if model_cfg.duration_aux_dim <= 0:
            raise ValueError(f"duration_aux_dim must be > 0, got {model_cfg.duration_aux_dim}")
        if model_cfg.duration_hidden_dim <= 0:
            raise ValueError(
                f"duration_hidden_dim must be > 0, got {model_cfg.duration_hidden_dim}"
            )
        if model_cfg.duration_layers <= 0:
            raise ValueError(f"duration_layers must be > 0, got {model_cfg.duration_layers}")
        if not (0.0 <= model_cfg.duration_dropout <= 1.0):
            raise ValueError(
                f"duration_dropout must be in [0, 1], got {model_cfg.duration_dropout}"
            )
        if model_cfg.duration_attention_heads <= 0:
            raise ValueError(
                f"duration_attention_heads must be > 0, got {model_cfg.duration_attention_heads}"
            )
        if model_cfg.text_dim % model_cfg.duration_attention_heads != 0:
            raise ValueError(
                "text_dim must be divisible by duration_attention_heads: "
                f"text_dim={model_cfg.text_dim}, "
                f"duration_attention_heads={model_cfg.duration_attention_heads}"
            )
        duration_architecture = str(model_cfg.duration_architecture).strip().lower()
        if duration_architecture not in DURATION_ARCHITECTURES:
            raise ValueError(
                "duration_architecture must be one of "
                f"{sorted(DURATION_ARCHITECTURES)}, got {model_cfg.duration_architecture!r}"
            )
        if model_cfg.duration_token_init_frames <= 0:
            raise ValueError(
                "duration_token_init_frames must be > 0, "
                f"got {model_cfg.duration_token_init_frames}"
            )
        duration_speaker_fusion = str(model_cfg.duration_speaker_fusion).strip().lower()
        if duration_speaker_fusion not in DURATION_SPEAKER_FUSIONS:
            raise ValueError(
                "duration_speaker_fusion must be one of "
                f"{sorted(DURATION_SPEAKER_FUSIONS)}, got {model_cfg.duration_speaker_fusion!r}"
            )
        duration_caption_fusion = str(model_cfg.duration_caption_fusion).strip().lower()
        if duration_caption_fusion not in DURATION_CAPTION_FUSIONS:
            raise ValueError(
                "duration_caption_fusion must be one of "
                f"{sorted(DURATION_CAPTION_FUSIONS)}, got {model_cfg.duration_caption_fusion!r}"
            )
        duration_caption_pooling = str(model_cfg.duration_caption_pooling).strip().lower()
        if duration_caption_pooling not in DURATION_CAPTION_POOLINGS:
            raise ValueError(
                "duration_caption_pooling must be one of "
                f"{sorted(DURATION_CAPTION_POOLINGS)}, got {model_cfg.duration_caption_pooling!r}"
            )
        if (
            duration_architecture == "token_sum_adarn_zero_no_aux"
            and duration_speaker_fusion != "adarn_zero"
        ):
            raise ValueError(
                "duration_architecture='token_sum_adarn_zero_no_aux' requires "
                "duration_speaker_fusion='adarn_zero'."
            )
        if duration_architecture == "token_sum_dual_adarn_zero_no_aux":
            if duration_speaker_fusion != "adarn_zero":
                raise ValueError(
                    "duration_architecture='token_sum_dual_adarn_zero_no_aux' requires "
                    "duration_speaker_fusion='adarn_zero'."
                )
            if duration_caption_fusion != "adarn_zero":
                raise ValueError(
                    "duration_architecture='token_sum_dual_adarn_zero_no_aux' requires "
                    "duration_caption_fusion='adarn_zero'."
                )
            if not model_cfg.use_speaker_condition_resolved or not model_cfg.use_caption_condition:
                raise ValueError(
                    "duration_architecture='token_sum_dual_adarn_zero_no_aux' requires "
                    "both speaker and caption conditioning."
                )
        model_cfg = replace(
            model_cfg,
            duration_architecture=duration_architecture,
            duration_speaker_fusion=duration_speaker_fusion,
            duration_caption_fusion=duration_caption_fusion,
            duration_caption_pooling=duration_caption_pooling,
        )
    if train_cfg.caption_warmup_steps < 0:
        raise ValueError(f"caption_warmup_steps must be >= 0, got {train_cfg.caption_warmup_steps}")
    if train_cfg.pretrained_projector_warmup_steps < 0:
        raise ValueError(
            "pretrained_projector_warmup_steps must be >= 0, got "
            f"{train_cfg.pretrained_projector_warmup_steps}"
        )
    if train_cfg.pretrained_projector_warmup_steps > 0:
        if not model_cfg.use_pretrained_text_encoder:
            raise ValueError(
                "pretrained_projector_warmup_steps requires model.text_encoder_type='pretrained'."
            )
        if args.init_checkpoint is None and args.resume is None:
            raise ValueError(
                "pretrained projector warmup requires --init-checkpoint or --resume; "
                "it is intended for replacing an encoder in a trained TTS model."
            )
        if train_cfg.caption_warmup:
            raise ValueError(
                "pretrained projector warmup and caption_warmup cannot be enabled together."
            )
        if train_cfg.train_mode == "duration_only":
            raise ValueError(
                "pretrained projector warmup requires train_mode='rf' so projector and TTS "
                "parameters remain in the optimizer."
            )
        if train_config_uses_lora(train_cfg):
            raise ValueError("pretrained projector warmup does not support LoRA training.")
        if train_cfg.speaker_inversion_enabled:
            raise ValueError(
                "pretrained projector warmup does not support Speaker Inversion training."
            )
    if train_cfg.dataloader_prefetch_factor <= 0:
        raise ValueError(
            f"dataloader_prefetch_factor must be > 0, got {train_cfg.dataloader_prefetch_factor}"
        )
    if train_cfg.length_bucket_window_batches <= 0:
        raise ValueError(
            "length_bucket_window_batches must be > 0, "
            f"got {train_cfg.length_bucket_window_batches}"
        )
    if train_cfg.latent_length_bucket_size < 0:
        raise ValueError(
            f"latent_length_bucket_size must be >= 0, got {train_cfg.latent_length_bucket_size}"
        )
    if not (0.0 <= train_cfg.valid_ratio < 1.0):
        raise ValueError(f"valid_ratio must be in [0, 1), got {train_cfg.valid_ratio}")
    if train_cfg.valid_every < 0:
        raise ValueError(f"valid_every must be >= 0, got {train_cfg.valid_every}")
    if train_cfg.valid_ratio > 0.0 and train_cfg.valid_every <= 0:
        raise ValueError("valid_every must be > 0 when valid_ratio > 0.")
    if train_cfg.valid_ratio == 0.0 and train_cfg.valid_every > 0 and is_main_process:
        print("warning: valid_every is set but valid_ratio=0. Validation is disabled.")
    if train_cfg.checkpoint_best_n < 0:
        raise ValueError(f"checkpoint_best_n must be >= 0, got {train_cfg.checkpoint_best_n}")
    if train_cfg.wandb_mode not in WANDB_MODES:
        raise ValueError(
            f"wandb_mode must be one of {sorted(WANDB_MODES)}, got {train_cfg.wandb_mode!r}"
        )
    precision = str(train_cfg.precision).lower()
    if precision not in {"fp32", "bf16"}:
        raise ValueError(f"precision must be one of ['fp32', 'bf16'], got {train_cfg.precision!r}")
    if precision == "bf16":
        if device.type != "cuda":
            if is_main_process:
                print("warning: precision=bf16 requested on non-CUDA device. Falling back to fp32.")
            train_cfg = replace(train_cfg, precision="fp32")
        elif not torch.cuda.is_bf16_supported():
            if is_main_process:
                print("warning: CUDA bf16 is not supported on this GPU. Falling back to fp32.")
            train_cfg = replace(train_cfg, precision="fp32")
    use_bf16 = train_cfg.precision == "bf16"
    if device.type == "cuda":
        tf32_enabled = bool(train_cfg.allow_tf32)
        torch.backends.cuda.matmul.allow_tf32 = tf32_enabled
        torch.backends.cudnn.allow_tf32 = tf32_enabled
        torch.set_float32_matmul_precision("high" if tf32_enabled else "highest")
        if is_main_process:
            print(f"TF32 matmul/cuDNN: {'enabled' if tf32_enabled else 'disabled'}")
    elif train_cfg.allow_tf32 and is_main_process:
        print("warning: allow_tf32=True requested on non-CUDA device; ignoring.")

    output_dir = Path(train_cfg.output_dir)
    if is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)
        dump_configs(output_dir / "config.json", model_cfg, train_cfg)
        print(f"Compute precision={train_cfg.precision} (weights/optimizer states kept in fp32).")
    if distributed:
        dist.barrier()
    if is_main_process and distributed:
        print(f"DDP enabled: world_size={world_size} (local_rank={local_rank})")
    speaker_inversion_resume_contract = None
    if train_cfg.speaker_inversion_enabled:
        speaker_inversion_resume_contract = build_speaker_inversion_resume_contract(
            train_cfg,
            world_size=world_size,
        )
        if args.resume is not None:
            if resume_payload is None:
                raise RuntimeError("Speaker Inversion resume payload is unavailable.")
            validate_speaker_inversion_resume_contract(
                resume_payload,
                train_cfg=train_cfg,
                world_size=world_size,
            )
    wandb_run = None
    if train_cfg.wandb_enabled and is_main_process:
        try:
            import wandb
        except ImportError as exc:
            raise RuntimeError(
                "W&B logging is enabled, but `wandb` is not installed. "
                "Install it with `pip install wandb`."
            ) from exc
        wandb_run = wandb.init(
            project=train_cfg.wandb_project,
            entity=train_cfg.wandb_entity,
            name=train_cfg.wandb_run_name,
            mode=train_cfg.wandb_mode,
            dir=str(output_dir),
            config={
                "model": asdict(model_cfg),
                "train": asdict(train_cfg),
                "script": "train.py",
            },
        )
        print(
            f"W&B enabled: project={train_cfg.wandb_project} mode={train_cfg.wandb_mode} run={wandb_run.name if wandb_run is not None else train_cfg.wandb_run_name}"
        )

    if distributed:
        local_files_only = not is_main_process
        if is_main_process:
            tokenizer = build_text_tokenizer(model_cfg, local_files_only=False)
            text_hidden_size = validate_text_backbone_dim(model_cfg, local_files_only=False)
            caption_tokenizer = None
            caption_hidden_size = None
            if model_cfg.use_caption_condition:
                caption_tokenizer = build_caption_tokenizer(model_cfg, local_files_only=False)
                caption_hidden_size = validate_caption_backbone_dim(
                    model_cfg,
                    local_files_only=False,
                )
        dist.barrier()
        if not is_main_process:
            tokenizer = build_text_tokenizer(model_cfg, local_files_only=local_files_only)
            text_hidden_size = validate_text_backbone_dim(
                model_cfg,
                local_files_only=local_files_only,
            )
            caption_tokenizer = None
            caption_hidden_size = None
            if model_cfg.use_caption_condition:
                caption_tokenizer = build_caption_tokenizer(
                    model_cfg,
                    local_files_only=local_files_only,
                )
                caption_hidden_size = validate_caption_backbone_dim(
                    model_cfg,
                    local_files_only=local_files_only,
                )
        dist.barrier()
    else:
        tokenizer = build_text_tokenizer(model_cfg, local_files_only=False)
        text_hidden_size = validate_text_backbone_dim(model_cfg, local_files_only=False)
        caption_tokenizer = None
        caption_hidden_size = None
        if model_cfg.use_caption_condition:
            caption_tokenizer = build_caption_tokenizer(model_cfg, local_files_only=False)
            caption_hidden_size = validate_caption_backbone_dim(
                model_cfg,
                local_files_only=False,
            )
    if is_main_process:
        print(
            f"Text tokenizer={model_cfg.text_tokenizer_repo} vocab={tokenizer.vocab_size} add_bos={model_cfg.text_add_bos} padding_side=right "
            f"(pretrained hidden_size={text_hidden_size})."
        )
        if model_cfg.use_caption_condition and caption_tokenizer is not None:
            print(
                f"Caption tokenizer={model_cfg.caption_tokenizer_repo_resolved} vocab={caption_tokenizer.vocab_size} add_bos={model_cfg.caption_add_bos_resolved} padding_side=right "
                f"(pretrained hidden_size={caption_hidden_size})."
            )
    # 共有キャッシュへの書き込みは rank 0 に限定し、他 rank は完成後のキャッシュを読む
    manifest_index = None
    if is_main_process:
        manifest_index = build_manifest_index(
            manifest_path=Path(train_cfg.manifest_path),
            show_progress=bool(train_cfg.progress),
            progress_desc="Index Manifest",
        )
    if distributed:
        dist.barrier()
    if manifest_index is None:
        manifest_index = build_manifest_index(
            manifest_path=Path(train_cfg.manifest_path),
            show_progress=False,
            progress_desc="Index Manifest",
        )
    ref_min_frames_cfg: int | None = None
    ref_max_frames_cfg: int | None = None
    if model_cfg.use_speaker_condition_resolved and train_cfg.ref_max_seconds > 0.0:
        ref_min_frames_cfg = max(
            1,
            round(float(train_cfg.ref_min_seconds) * DACVAE_LATENT_FRAMES_PER_SECOND),
        )
        ref_max_frames_cfg = max(
            ref_min_frames_cfg,
            round(float(train_cfg.ref_max_seconds) * DACVAE_LATENT_FRAMES_PER_SECOND),
        )
        if is_main_process:
            print(
                "Reference concat enabled: "
                f"ref_min_seconds={train_cfg.ref_min_seconds} "
                f"ref_max_seconds={train_cfg.ref_max_seconds} "
                f"(frames {ref_min_frames_cfg}..{ref_max_frames_cfg} at "
                f"{DACVAE_LATENT_FRAMES_PER_SECOND} Hz)."
            )
    valid_dataset = None
    if train_cfg.valid_ratio > 0.0:
        train_indices, valid_indices = split_train_valid_indices(
            num_samples=len(manifest_index.offsets),
            valid_ratio=train_cfg.valid_ratio,
            seed=train_cfg.seed,
        )
        train_dataset = LatentTextDataset(
            manifest_path=train_cfg.manifest_path,
            latent_dim=model_cfg.latent_dim,
            max_latent_steps=train_cfg.max_latent_steps,
            subset_indices=train_indices,
            enable_caption_condition=model_cfg.use_caption_condition,
            enable_speaker_condition=model_cfg.use_speaker_condition_resolved,
            manifest_index=manifest_index,
            ref_min_frames=ref_min_frames_cfg,
            ref_max_frames=ref_max_frames_cfg,
        )
        valid_dataset = LatentTextDataset(
            manifest_path=train_cfg.manifest_path,
            latent_dim=model_cfg.latent_dim,
            max_latent_steps=train_cfg.max_latent_steps,
            subset_indices=valid_indices,
            enable_caption_condition=model_cfg.use_caption_condition,
            enable_speaker_condition=model_cfg.use_speaker_condition_resolved,
            manifest_index=manifest_index,
            ref_min_frames=ref_min_frames_cfg,
            ref_max_frames=ref_max_frames_cfg,
        )
        if is_main_process:
            print(
                f"Validation split enabled: train={len(train_dataset)} valid={len(valid_dataset)} (ratio={train_cfg.valid_ratio:.4f}, valid_every={train_cfg.valid_every} steps)."
            )
    else:
        train_dataset = LatentTextDataset(
            manifest_path=train_cfg.manifest_path,
            latent_dim=model_cfg.latent_dim,
            max_latent_steps=train_cfg.max_latent_steps,
            enable_caption_condition=model_cfg.use_caption_condition,
            enable_speaker_condition=model_cfg.use_speaker_condition_resolved,
            manifest_index=manifest_index,
            ref_min_frames=ref_min_frames_cfg,
            ref_max_frames=ref_max_frames_cfg,
        )
    effective_batch_size = train_cfg.batch_size * (world_size if distributed else 1)
    drop_last = len(train_dataset) >= effective_batch_size
    if not drop_last and is_main_process:
        print(
            f"warning: dataset size ({len(train_dataset)}) is smaller than effective batch_size "
            f"({effective_batch_size}). "
            "Using drop_last=False to avoid empty dataloader."
        )
    collator = TTSCollator(
        tokenizer=tokenizer,
        caption_tokenizer=caption_tokenizer,
        latent_dim=model_cfg.latent_dim,
        latent_patch_size=model_cfg.latent_patch_size,
        fixed_target_latent_steps=train_cfg.fixed_target_latent_steps,
        fixed_target_full_mask=train_cfg.fixed_target_full_mask,
        latent_length_bucket_size=train_cfg.latent_length_bucket_size,
        max_text_len=train_cfg.max_text_len,
        max_caption_len=(
            train_cfg.max_text_len
            if train_cfg.max_caption_len is None
            else train_cfg.max_caption_len
        ),
    )
    if train_cfg.fixed_target_latent_steps is not None and is_main_process:
        print(
            f"Fixed target latent length enabled: steps={train_cfg.fixed_target_latent_steps} full_mask={train_cfg.fixed_target_full_mask}"
        )
    elif train_cfg.latent_length_bucket_size > 0 and is_main_process:
        print(
            "Fixed latent length buckets enabled: "
            f"bucket_size={train_cfg.latent_length_bucket_size}."
        )
    if not model_cfg.use_speaker_condition_resolved and is_main_process:
        print("Speaker conditioning disabled for this model config.")
    if train_cfg.caption_warmup and is_main_process:
        if not model_cfg.use_caption_condition:
            print(
                "warning: caption_warmup=True requested, but caption conditioning is disabled. Ignoring."
            )
        elif train_cfg.caption_warmup_steps <= 0:
            print(
                "warning: caption_warmup=True requested, but caption_warmup_steps <= 0. Ignoring."
            )
        else:
            print(
                "Caption warmup enabled: only caption-only parameters will update for the first "
                f"{train_cfg.caption_warmup_steps} optimizer steps."
            )
    if train_cfg.pretrained_projector_warmup_steps > 0 and is_main_process:
        print(
            "Pretrained projector warmup enabled: only text/caption projectors will update "
            f"for the first {train_cfg.pretrained_projector_warmup_steps} optimizer steps; "
            "all model parameters update after warmup."
        )
    if train_cfg.timestep_stratified and is_main_process:
        print("Using stratified logit-normal timestep sampling.")
    train_sampler = None
    train_loader_generator = None
    if train_cfg.length_bucket_enabled:
        length_bucket_values = train_dataset.length_bucket_values()
        positive_mask = length_bucket_values > 0
        if not bool(positive_mask.any()):
            if is_main_process:
                print(
                    "warning: length_bucket_enabled=True but manifest has no positive num_frames; "
                    "falling back to normal random sampling."
                )
        else:
            if not bool(positive_mask.all()):
                fallback_length = int(length_bucket_values[positive_mask].median().item())
                length_bucket_values[~positive_mask] = fallback_length
                if is_main_process:
                    missing_count = int((~positive_mask).sum().item())
                    print(
                        "warning: length bucket found samples without num_frames; "
                        f"using median length={fallback_length} for {missing_count} samples."
                    )
            train_sampler = LengthGroupedSampler(
                length_bucket_values,
                batch_size=train_cfg.batch_size,
                window_batches=train_cfg.length_bucket_window_batches,
                num_replicas=world_size if distributed else 1,
                rank=rank if distributed else 0,
                seed=train_cfg.seed,
                drop_last=drop_last,
            )
            if is_main_process:
                window_samples = (
                    train_cfg.batch_size
                    * max(1, world_size if distributed else 1)
                    * train_cfg.length_bucket_window_batches
                )
                print(
                    "Length bucket sampling enabled: "
                    f"window_batches={train_cfg.length_bucket_window_batches} "
                    f"window_samples={window_samples} "
                    "with shuffled batch order."
                )
    if train_sampler is None and distributed:
        train_sampler = StatefulDistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=train_cfg.seed,
            drop_last=drop_last,
        )
    elif train_sampler is None:
        train_loader_generator = torch.Generator()
        train_loader_generator.manual_seed(int(train_cfg.seed))
    dataloader_common_kwargs = {
        "batch_size": train_cfg.batch_size,
        "num_workers": train_cfg.num_workers,
        "pin_memory": (device.type == "cuda"),
        "collate_fn": collator,
    }
    if train_cfg.num_workers > 0:
        dataloader_common_kwargs["persistent_workers"] = bool(
            train_cfg.dataloader_persistent_workers
        )
        dataloader_common_kwargs["prefetch_factor"] = int(train_cfg.dataloader_prefetch_factor)
    elif train_cfg.dataloader_persistent_workers and is_main_process:
        print("warning: dataloader_persistent_workers=True is ignored because num_workers=0.")
    if train_cfg.dataloader_cuda_prefetch and device.type != "cuda" and is_main_process:
        print("warning: dataloader_cuda_prefetch=True is ignored because device is not CUDA.")
    loader = StatefulDataLoader(
        dataset=train_dataset,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        drop_last=drop_last,
        generator=train_loader_generator,
        snapshot_every_n_steps=1,
        **dataloader_common_kwargs,
    )
    if len(loader) == 0:
        raise ValueError("Dataloader yielded zero batches. Check manifest and batch_size settings.")
    valid_loader = None
    valid_sampler = None
    if valid_dataset is not None:
        if distributed:
            valid_sampler = DistributedSampler(
                valid_dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=False,
                drop_last=False,
            )
        valid_loader = TorchDataLoader(
            dataset=valid_dataset,
            shuffle=False,
            sampler=valid_sampler,
            drop_last=False,
            **dataloader_common_kwargs,
        )
        if len(valid_loader) == 0:
            raise ValueError(
                "Validation dataloader yielded zero batches. Decrease batch_size or valid_ratio."
            )

    has_validation = valid_loader is not None and train_cfg.valid_every > 0
    checkpoint_retention_enabled = train_cfg.checkpoint_best_n > 0
    periodic_checkpoint_keep = 0
    if checkpoint_retention_enabled:
        periodic_checkpoint_keep = 1 if has_validation else int(train_cfg.checkpoint_best_n) + 1
    best_val_checkpoints: list[tuple[float, int, Path]] = []
    if is_main_process:
        if checkpoint_retention_enabled and has_validation:
            best_val_checkpoints = list_best_val_loss_checkpoints(output_dir)
            best_val_checkpoints = prune_best_val_loss_checkpoints(
                best_val_checkpoints,
                train_cfg.checkpoint_best_n,
            )
        if checkpoint_retention_enabled and has_validation:
            print(f"Checkpoint retention: latest=1 + best_val_loss={train_cfg.checkpoint_best_n}.")
        elif checkpoint_retention_enabled:
            print(
                f"Checkpoint retention: validation disabled, keep latest {periodic_checkpoint_keep} periodic checkpoints."
            )

    if not (0.0 <= train_cfg.lora_dropout <= 1.0):
        raise ValueError(f"lora_dropout must be in [0, 1], got {train_cfg.lora_dropout}")
    if train_cfg.lora_r <= 0:
        raise ValueError(f"lora_r must be > 0, got {train_cfg.lora_r}")
    if train_cfg.lora_alpha <= 0:
        raise ValueError(f"lora_alpha must be > 0, got {train_cfg.lora_alpha}")

    if args.resume is not None:
        if train_config_uses_lora(train_cfg):
            if resume_path is None or not is_lora_adapter_dir(resume_path):
                raise ValueError("LoRA resume expects an adapter checkpoint directory.")
        elif resume_path is not None and resume_path.is_dir():
            raise ValueError(
                "Non-LoRA resume expects a .pt training checkpoint, not a checkpoint directory."
            )
        if args.init_checkpoint is not None and not train_config_uses_lora(train_cfg):
            raise ValueError(
                "--resume and --init-checkpoint can only be combined for LoRA adapter resumes."
            )

    if train_config_uses_lora(train_cfg) and args.resume is None and args.init_checkpoint is None:
        raise ValueError(
            "LoRA fine-tuning requires --init-checkpoint for the base model, "
            "or --resume from a LoRA adapter checkpoint directory."
        )

    preloaded_init_checkpoint = None
    pretrained_backbone_config = None
    load_pretrained_backbone_weights = True
    if model_cfg.use_pretrained_text_encoder:
        if args.resume is not None and not train_config_uses_lora(train_cfg):
            if isinstance(resume_model_cfg, dict):
                checkpoint_cfg = merge_dataclass_overrides(
                    ModelConfig(),
                    resume_model_cfg,
                    section="resume checkpoint model_config",
                )
                if checkpoint_cfg.use_pretrained_text_encoder:
                    # 保存済み構成だけで backbone を再構築し、外部設定への暗黙依存を避ける
                    if resume_text_encoder_config is None:
                        raise ValueError(
                            "Non-LoRA resume with a pretrained text encoder requires "
                            "text_encoder_config in the checkpoint."
                        )
                    load_pretrained_backbone_weights = False
                    pretrained_backbone_config = resume_text_encoder_config
        else:
            pretrained_base_path = None
            if args.init_checkpoint is not None:
                pretrained_base_path = _normalize_checkpoint_path(args.init_checkpoint)
            elif train_config_uses_lora(train_cfg) and isinstance(resume_base_init, dict):
                checkpoint_path = resume_base_init.get("checkpoint_path")
                if resume_base_init.get("mode") == "checkpoint" and isinstance(
                    checkpoint_path, str
                ):
                    pretrained_base_path = _normalize_checkpoint_path(checkpoint_path)
            if pretrained_base_path is not None:
                init_checkpoint_path = pretrained_base_path
                preloaded_init_checkpoint = _load_model_state_from_checkpoint(init_checkpoint_path)
                init_state, init_model_cfg, _, init_text_encoder_config = preloaded_init_checkpoint
                checkpoint_uses_pretrained = any(
                    key.startswith("pretrained_text_backbone.") for key in init_state
                )
                if isinstance(init_model_cfg, dict):
                    checkpoint_cfg = merge_dataclass_overrides(
                        ModelConfig(),
                        init_model_cfg,
                        section="init checkpoint model_config",
                    )
                    checkpoint_uses_pretrained = checkpoint_cfg.use_pretrained_text_encoder
                if checkpoint_uses_pretrained:
                    # resume 経路と同じく、保存済み構成が無い pretrained checkpoint は拒否する
                    if init_text_encoder_config is None:
                        raise ValueError(
                            "Init checkpoint with a pretrained text encoder requires "
                            "text_encoder_config in the checkpoint."
                        )
                    load_pretrained_backbone_weights = False
                    pretrained_backbone_config = init_text_encoder_config

    raw_model: torch.nn.Module = TextToLatentRFDiT(
        model_cfg,
        pretrained_backbone_config=pretrained_backbone_config,
        load_pretrained_backbone_weights=load_pretrained_backbone_weights,
    ).to(device)
    lora_wrapped = False
    base_init: dict | None = None
    if args.resume is not None and train_config_uses_lora(train_cfg):
        base_init = resume_base_init
        if args.init_checkpoint is not None:
            override_init_path = _normalize_checkpoint_path(args.init_checkpoint)
            base_init = {"mode": "checkpoint", "checkpoint_path": str(override_init_path)}
        _apply_base_initialization(
            raw_model,
            model_cfg=model_cfg,
            base_init=base_init,
            distributed=distributed,
            is_main_process=is_main_process,
            preloaded_checkpoint=preloaded_init_checkpoint,
        )
        if resume_path is None or not is_lora_adapter_dir(resume_path):
            raise ValueError("LoRA resume expects an adapter checkpoint directory.")
        raw_model = load_lora_adapter(raw_model, resume_path, is_trainable=True)
        lora_wrapped = True
    elif args.resume is not None and train_cfg.speaker_inversion_enabled:
        base_init = resume_base_init
        _apply_base_initialization(
            raw_model,
            model_cfg=model_cfg,
            base_init=base_init,
            distributed=distributed,
            is_main_process=is_main_process,
        )
    elif args.resume is None and args.init_checkpoint is None:
        _apply_base_initialization(
            raw_model,
            model_cfg=model_cfg,
            base_init=None,
            distributed=distributed,
            is_main_process=is_main_process,
        )
        if train_config_uses_lora(train_cfg):
            raw_model = apply_lora(raw_model, train_cfg)
            lora_wrapped = True
    elif args.init_checkpoint is not None:
        init_checkpoint_path = _normalize_checkpoint_path(args.init_checkpoint)
        base_init = {"mode": "checkpoint", "checkpoint_path": str(init_checkpoint_path)}
        _apply_base_initialization(
            raw_model,
            model_cfg=model_cfg,
            base_init=base_init,
            distributed=distributed,
            is_main_process=is_main_process,
            preloaded_checkpoint=preloaded_init_checkpoint,
        )
        if train_config_uses_lora(train_cfg) and not lora_wrapped:
            raw_model = apply_lora(raw_model, train_cfg)
            lora_wrapped = True

    if train_config_uses_lora(train_cfg) and is_main_process:
        trainable_params, total_params = count_parameters(raw_model)
        print(
            "LoRA enabled: "
            f"r={train_cfg.lora_r} alpha={train_cfg.lora_alpha} "
            f"dropout={train_cfg.lora_dropout:.3f} "
            f"target_modules={train_cfg.lora_target_modules!r} "
            f"modules_to_save={train_cfg.lora_modules_to_save!r} "
            f"trainable={trainable_params:,}/{total_params:,}"
        )
    # manifest のハッシュと付随 JSON は不変なので、学習開始時に1回だけ読み込む
    lora_metadata = (
        build_lora_metadata(model_cfg=model_cfg, train_cfg=train_cfg, base_init=base_init)
        if train_config_uses_lora(train_cfg) and is_main_process
        else None
    )
    if train_cfg.speaker_inversion_enabled:
        init_embedding = None
        base_pre_norm_embedding = None
        resume_module_state: dict[str, torch.Tensor] | None = None
        if args.resume is not None:
            if resume_payload is None:
                raise RuntimeError("Speaker Inversion resume payload is unavailable.")
            raw_module_state = resume_payload.get(SPEAKER_INVERSION_MODULE_STATE_KEY)
            if raw_module_state is not None:
                if not isinstance(raw_module_state, dict):
                    raise ValueError("Speaker Inversion module state must be a dictionary.")
                resume_module_state = raw_module_state
                saved_base = resume_module_state.get("base_pre_norm_embedding")
                if isinstance(saved_base, torch.Tensor):
                    base_pre_norm_embedding = saved_base
                else:
                    init_embedding = resume_payload[SPEAKER_EMBEDDING_KEY]
            else:
                init_embedding = resume_payload[SPEAKER_EMBEDDING_KEY]
            if is_main_process:
                print(f"Loaded Speaker Inversion trainer state: {args.resume}")
        elif train_cfg.speaker_inversion_init_embedding is not None:
            init_payload = load_speaker_inversion_payload(
                train_cfg.speaker_inversion_init_embedding,
            )
            init_embedding = init_payload[SPEAKER_EMBEDDING_KEY]
            if is_main_process:
                print(
                    "Loaded Speaker Inversion init embedding: "
                    f"{train_cfg.speaker_inversion_init_embedding}"
                )
        elif train_cfg.speaker_inversion_base_embedding is not None:
            base_payload = load_speaker_inversion_base_payload(
                train_cfg.speaker_inversion_base_embedding,
                expected_checkpoint=args.init_checkpoint,
                expected_speaker_dim=model_cfg.speaker_dim,
                expected_speaker_patch_size=model_cfg.speaker_patch_size,
            )
            base_pre_norm_embedding = base_payload[SPEAKER_PRE_NORM_EMBEDDING_KEY]
            if is_main_process:
                print(
                    "Loaded reference-derived Speaker Inversion base: "
                    f"{train_cfg.speaker_inversion_base_embedding}"
                )
        speaker_inversion = raw_model.enable_speaker_inversion(
            num_tokens=train_cfg.speaker_inversion_tokens,
            init_std=train_cfg.speaker_inversion_init_std,
            init_embedding=init_embedding,
            base_pre_norm_embedding=base_pre_norm_embedding,
            max_relative_residual_norm=(train_cfg.speaker_inversion_max_relative_residual_norm),
        )
        speaker_inversion.to(device)
        if is_main_process:
            print(
                "Speaker Inversion parameters initialized: "
                f"embedding={tuple(speaker_inversion.embedding.shape)}."
            )
    if train_cfg.train_mode == "duration_only":
        trainable_duration_params, frozen_params = freeze_for_duration_only(raw_model)
        if trainable_duration_params == 0:
            raise RuntimeError(
                "No duration predictor parameters were found for duration_only mode."
            )
        if is_main_process:
            print(
                "Duration-only training enabled: "
                f"trainable={trainable_duration_params:,} frozen={frozen_params:,}."
            )
    if train_cfg.speaker_inversion_enabled:
        trainable_speaker_params, frozen_params = freeze_for_speaker_inversion(raw_model)
        if trainable_speaker_params == 0:
            raise RuntimeError("No Speaker Inversion parameters were found.")
        if is_main_process:
            print(
                "Speaker Inversion freeze applied: "
                f"trainable={trainable_speaker_params:,} frozen={frozen_params:,}."
            )
    if train_cfg.gradient_checkpointing:
        raw_model.set_gradient_checkpointing(True)
        if is_main_process:
            scope = "diffusion blocks"
            if model_cfg.use_pretrained_text_encoder:
                scope += " and the pretrained text encoder (when supported)"
            print(f"Gradient checkpointing enabled on {scope}.")
    train_model = raw_model
    if train_cfg.compile_model:
        if not hasattr(torch, "compile"):
            raise RuntimeError("compile_model=True requires torch.compile (PyTorch 2+).")
        if is_main_process:
            print("torch.compile enabled (dynamic=True).")
        train_model = torch.compile(raw_model, dynamic=True)
    ddp_find_unused_parameters = bool(train_cfg.ddp_find_unused_parameters)
    ddp_find_unused_parameters_explicit = args.ddp_find_unused_parameters is not None or (
        isinstance(exp_cfg.get("train"), dict)
        and "ddp_find_unused_parameters" in exp_cfg.get("train", {})
    )
    if distributed:
        # Auto-enable for common configs where conditional branches can be fully
        # masked in a step. Without this, DDP can hang after step 1 due to
        # unreduced gradients in ranks where a branch is entirely unused.
        if not ddp_find_unused_parameters and not ddp_find_unused_parameters_explicit:
            speaker_labeled_count = train_dataset.speaker_labeled_count
            has_partial_or_no_speaker_labels = speaker_labeled_count < len(train_dataset)
            caption_labeled_count = train_dataset.caption_labeled_count
            has_partial_or_no_caption_labels = (
                model_cfg.use_caption_condition and caption_labeled_count < len(train_dataset)
            )
            has_stochastic_cond_drop = (
                train_cfg.text_condition_dropout > 0.0
                or train_cfg.speaker_condition_dropout > 0.0
                or (model_cfg.use_caption_condition and train_cfg.caption_condition_dropout > 0.0)
            )
            if (
                has_partial_or_no_speaker_labels
                or has_partial_or_no_caption_labels
                or has_stochastic_cond_drop
            ):
                ddp_find_unused_parameters = True
                if is_main_process:
                    print(
                        "DDP find_unused_parameters auto-enabled "
                        "(conditional branches may be fully masked in some steps)."
                    )
        model = DDP(
            train_model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=ddp_find_unused_parameters,
            broadcast_buffers=False,
        )
    else:
        model = train_model
    optimizer = build_optimizer(raw_model, train_cfg)
    scheduler = build_scheduler(optimizer, train_cfg)
    if is_main_process:
        print(
            f"Optimizer={train_cfg.optimizer} Scheduler={train_cfg.lr_scheduler} lr={current_lr(optimizer):.3e}"
        )
        pretrained_lr = current_pretrained_text_encoder_lr(optimizer)
        if pretrained_lr is not None:
            print(f"Pretrained text encoder optimizer=adamw lr={pretrained_lr:.3e}.")
        if train_cfg.gradient_accumulation_steps > 1:
            print(
                f"Gradient accumulation enabled: steps={train_cfg.gradient_accumulation_steps} (effective global batch={train_cfg.batch_size * world_size * train_cfg.gradient_accumulation_steps})."
            )

    step = 0
    resume_epoch = 0
    resume_loader_state_loaded = False
    epoch_step_offset = 0
    progress: TrainProgress | None = None
    rng_state: dict[str, Any] | None = None
    if args.resume is not None:
        if resume_payload is None:
            raise RuntimeError("Resume checkpoint payload is unavailable.")
        ckpt = resume_payload
        if train_cfg.speaker_inversion_enabled:
            step = restore_speaker_inversion_training_state(
                raw_model,
                optimizer,
                scheduler,
                ckpt,
                distributed=distributed,
                rank=rank,
                world_size=world_size,
            )
            # Speaker Inversion の復元処理が使用した保存状態を、共通の再開ログにも反映する
            rng_state = _select_rng_state_for_rank(
                ckpt,
                distributed=distributed,
                rank=rank,
                world_size=world_size,
            )
        else:
            if not train_config_uses_lora(train_cfg):
                raw_model.load_state_dict(ckpt["model"])
            optimizer.load_state_dict(ckpt["optimizer"])
            step = int(ckpt["step"])
            if scheduler is not None:
                scheduler_state = ckpt.get("scheduler")
                if scheduler_state is not None:
                    scheduler.load_state_dict(scheduler_state)
                elif step > 0:
                    scheduler.last_step = step
        runtime_state = ckpt.get(RUNTIME_STATE_KEY)
        if isinstance(runtime_state, dict):
            resume_epoch = int(runtime_state.get("sampler_epoch", 0))
            epoch_step_offset = int(runtime_state.get("epoch_step", 0))
        dataloader_state = _select_dataloader_state_for_rank(
            ckpt,
            distributed=distributed,
            rank=rank,
            world_size=world_size,
        )
        if dataloader_state is not None:
            if train_sampler is not None:
                train_sampler.set_epoch(resume_epoch)
            loader.load_state_dict(dataloader_state)
            resume_loader_state_loaded = True
        if train_cfg.speaker_inversion_enabled is False:
            rng_state = _select_rng_state_for_rank(
                ckpt,
                distributed=distributed,
                rank=rank,
                world_size=world_size,
            )
            if rng_state is not None:
                _restore_rng_state(rng_state)
        if is_main_process:
            print(f"Resumed from step={step}")
            if dataloader_state is None:
                print(
                    "warning: resume checkpoint has no dataloader_state; "
                    "data iteration will restart at the beginning of an epoch."
                )
            else:
                print("Restored dataloader state for mid-epoch resume.")
            if rng_state is None:
                print("warning: resume checkpoint has no rng_state; random state was reseeded.")
            else:
                print("Restored random-number generator states.")

    progress = TrainProgress(
        max_steps=train_cfg.max_steps,
        start_step=step,
        rank=rank,
        world_size=world_size,
        enabled=train_cfg.progress,
        show_all_ranks=train_cfg.progress_all_ranks,
        description="Train Duration" if train_cfg.train_mode == "duration_only" else "Train RF",
    )
    accum_steps = int(train_cfg.gradient_accumulation_steps)
    global_batch_size = train_cfg.batch_size * world_size * accum_steps
    duration_only = train_cfg.train_mode == "duration_only"
    caption_warmup_active = bool(
        train_cfg.caption_warmup
        and model_cfg.use_caption_condition
        and train_cfg.caption_warmup_steps > 0
        and step < train_cfg.caption_warmup_steps
    )
    pretrained_projector_warmup_active = bool(
        model_cfg.use_pretrained_text_encoder
        and train_cfg.pretrained_projector_warmup_steps > 0
        and step < train_cfg.pretrained_projector_warmup_steps
    )
    if caption_warmup_active and is_main_process:
        print(
            "Caption warmup active: non-caption gradients will be cleared for the first "
            f"{train_cfg.caption_warmup_steps} optimizer steps."
        )
    if pretrained_projector_warmup_active and is_main_process:
        print(
            "Pretrained projector warmup active: non-projector gradients will be cleared "
            f"through optimizer step {train_cfg.pretrained_projector_warmup_steps}."
        )

    try:
        model.train()
        if scheduler is not None and step == 0:
            # Ensure the very first optimizer step uses warmup-scaled LR.
            scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        accum_micro_steps = 0
        accum_loss = torch.zeros((), device=device, dtype=torch.float32)
        accum_rf_loss = torch.zeros((), device=device, dtype=torch.float32)
        accum_duration_loss = torch.zeros((), device=device, dtype=torch.float32)
        accum_duration_mae_frames = torch.zeros((), device=device, dtype=torch.float32)
        accum_duration_group_totals = torch.zeros(
            DURATION_CONDITION_GROUP_TOTAL_SIZE,
            device=device,
            dtype=torch.float64,
        )
        epoch = resume_epoch
        if resume_loader_state_loaded is False:
            epoch_step_offset = 0
        last_epoch_step = epoch_step_offset
        while step < train_cfg.max_steps:
            if train_sampler is not None and not resume_loader_state_loaded:
                train_sampler.set_epoch(epoch)
            epoch += 1
            current_epoch_step_offset = epoch_step_offset if resume_loader_state_loaded else 0
            epoch_step_offset = 0
            # 復元済みローダーの状態は、この epoch のイテレーター生成時だけ有効にする
            resume_loader_state_loaded = False
            train_batches = cuda_prefetch_batches(
                loader,
                device=device,
                enabled=bool(train_cfg.dataloader_cuda_prefetch),
            )
            for raw_epoch_step, batch in enumerate(train_batches, start=1):
                epoch_step = raw_epoch_step + current_epoch_step_offset
                last_epoch_step = epoch_step
                accum_micro_steps += 1
                text_ids = batch["text_ids"].to(device, non_blocking=True)
                text_mask = batch["text_mask"].to(device, non_blocking=True)
                caption_ids = None
                caption_mask = None
                has_caption = None
                if raw_model.cfg.use_caption_condition:
                    caption_ids = batch["caption_ids"].to(device, non_blocking=True)
                    caption_mask = batch["caption_mask"].to(device, non_blocking=True)
                    has_caption = batch["has_caption"].to(device, non_blocking=True)
                num_frames = batch["num_frames"].to(device, non_blocking=True)
                duration_features = batch["duration_features"].to(device, non_blocking=True)
                ref_latent = None
                ref_mask = None
                if raw_model.cfg.use_speaker_condition_resolved:
                    ref_latent = batch["ref_latent_patched"].to(device, non_blocking=True)
                    ref_mask = batch["ref_latent_mask_patched"].to(device, non_blocking=True)
                    has_speaker = batch["has_speaker"].to(device, non_blocking=True)
                else:
                    has_speaker = None

                bsz = text_ids.shape[0]
                x_mask = None
                x_mask_valid = None
                x_t = None
                t = None
                v_target = None
                if not duration_only:
                    x0 = batch["latent_patched"].to(device, non_blocking=True)
                    x_mask = batch["latent_mask_patched"].to(device, non_blocking=True)
                    x_mask_valid = batch["latent_mask_valid_patched"].to(device, non_blocking=True)
                    if train_cfg.timestep_stratified:
                        t = sample_stratified_logit_normal_t(
                            batch_size=bsz,
                            device=device,
                            mean=train_cfg.timestep_logit_mean,
                            std=train_cfg.timestep_logit_std,
                            t_min=train_cfg.timestep_min,
                            t_max=train_cfg.timestep_max,
                        )
                    else:
                        t = sample_logit_normal_t(
                            batch_size=bsz,
                            device=device,
                            mean=train_cfg.timestep_logit_mean,
                            std=train_cfg.timestep_logit_std,
                            t_min=train_cfg.timestep_min,
                            t_max=train_cfg.timestep_max,
                        )
                    noise = torch.randn_like(x0)
                    x_t = rf_interpolate(x0, noise, t)
                    v_target = rf_velocity_target(x0, noise)

                text_cond_drop = torch.rand(bsz, device=device) < train_cfg.text_condition_dropout
                if text_cond_drop.any() and not raw_model.cfg.use_duration_predictor:
                    text_mask = text_mask.clone()
                    text_mask[text_cond_drop] = False
                caption_cond_drop = None
                caption_drop_for_model = None
                duration_has_caption = None
                if raw_model.cfg.use_caption_condition:
                    if has_caption is None or caption_mask is None:
                        raise RuntimeError(
                            "Caption conditioning is enabled but caption batch tensors are missing."
                        )
                    caption_cond_drop = (
                        torch.rand(bsz, device=device) < train_cfg.caption_condition_dropout
                    )
                    use_caption = has_caption & (~caption_cond_drop)
                    caption_drop_for_model = ~use_caption
                    duration_caption_drop = (
                        torch.rand(bsz, device=device) < train_cfg.duration_caption_dropout
                    )
                    duration_has_caption = has_caption & (~duration_caption_drop)
                    if not raw_model.cfg.use_duration_predictor:
                        caption_mask = caption_mask & use_caption[:, None]

                speaker_drop_for_model = None
                duration_has_speaker = None
                if raw_model.cfg.use_speaker_condition_resolved:
                    speaker_cond_drop = (
                        torch.rand(bsz, device=device) < train_cfg.speaker_condition_dropout
                    )
                    if train_cfg.speaker_inversion_enabled:
                        # Speaker Inversion learns one embedding for this run, so all samples are
                        # speaker-conditioned even when the manifest has no speaker_id.
                        use_speaker = ~speaker_cond_drop
                        speaker_drop_for_model = speaker_cond_drop
                    else:
                        use_speaker = has_speaker & (~speaker_cond_drop)
                        speaker_drop_for_model = ~use_speaker
                    duration_speaker_drop = (
                        torch.rand(bsz, device=device) < train_cfg.duration_speaker_dropout
                    )
                    if train_cfg.speaker_inversion_enabled:
                        duration_has_speaker = ~duration_speaker_drop
                    else:
                        duration_has_speaker = has_speaker & (~duration_speaker_drop)
                    duration_features = set_duration_has_speaker_feature(
                        duration_features,
                        duration_has_speaker,
                    )
                    if (
                        not raw_model.cfg.use_duration_predictor
                        and not train_cfg.speaker_inversion_enabled
                    ):
                        ref_mask = ref_mask & use_speaker[:, None]
                        ref_latent = ref_latent * use_speaker[:, None, None].to(ref_latent.dtype)

                should_step = (accum_micro_steps % accum_steps) == 0
                sync_context = model.no_sync() if distributed and not should_step else nullcontext()
                with sync_context:
                    with (
                        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                        if use_bf16
                        else nullcontext()
                    ):
                        if duration_only:
                            duration_pred = model(
                                x_t=None,
                                t=None,
                                text_input_ids=text_ids,
                                text_mask=text_mask,
                                ref_latent=ref_latent,
                                ref_mask=ref_mask,
                                caption_input_ids=caption_ids,
                                caption_mask=caption_mask,
                                latent_mask=None,
                                duration_features=duration_features,
                                duration_has_speaker=duration_has_speaker,
                                duration_has_caption=duration_has_caption,
                                duration_only=True,
                            )
                            v_pred = None
                        elif raw_model.cfg.use_duration_predictor:
                            v_pred, duration_pred = model(
                                x_t=x_t,
                                t=t,
                                text_input_ids=text_ids,
                                text_mask=text_mask,
                                ref_latent=ref_latent,
                                ref_mask=ref_mask,
                                caption_input_ids=caption_ids,
                                caption_mask=caption_mask,
                                latent_mask=x_mask,
                                text_condition_dropout=text_cond_drop,
                                speaker_condition_dropout=speaker_drop_for_model,
                                caption_condition_dropout=caption_drop_for_model,
                                duration_features=duration_features,
                                duration_has_speaker=duration_has_speaker,
                                duration_has_caption=duration_has_caption,
                                duration_backprop_to_condition=(
                                    train_cfg.duration_backprop_to_condition
                                ),
                            )
                        else:
                            v_pred = model(
                                x_t=x_t,
                                t=t,
                                text_input_ids=text_ids,
                                text_mask=text_mask,
                                ref_latent=ref_latent,
                                ref_mask=ref_mask,
                                caption_input_ids=caption_ids,
                                caption_mask=caption_mask,
                                latent_mask=x_mask,
                                text_condition_dropout=None,
                                speaker_condition_dropout=speaker_drop_for_model
                                if train_cfg.speaker_inversion_enabled
                                else None,
                                caption_condition_dropout=None,
                            )
                            duration_pred = None

                    rf_loss = torch.zeros((), device=device, dtype=torch.float32)
                    if not duration_only:
                        if (
                            v_pred is None
                            or v_target is None
                            or x_mask is None
                            or x_mask_valid is None
                        ):
                            raise RuntimeError("RF training tensors are missing.")
                        v_pred = v_pred.float()
                        rf_loss = compute_rf_loss(
                            pred=v_pred,
                            target=v_target.float(),
                            loss_mask=x_mask,
                            valid_mask=x_mask_valid,
                            mode=train_cfg.rf_loss_mode,
                        )
                    duration_loss = torch.zeros((), device=device, dtype=torch.float32)
                    duration_mae_frames = torch.zeros((), device=device, dtype=torch.float32)
                    duration_group_totals = torch.zeros(
                        DURATION_CONDITION_GROUP_TOTAL_SIZE,
                        device=device,
                        dtype=torch.float64,
                    )
                    if raw_model.cfg.use_duration_predictor:
                        if duration_pred is None:
                            raise RuntimeError(
                                "Duration predictor is enabled but duration_pred is missing."
                            )
                        duration_target = torch.log1p(num_frames.float())
                        duration_loss_per_sample = F.huber_loss(
                            duration_pred.float(),
                            duration_target,
                            delta=float(train_cfg.duration_huber_delta),
                            reduction="none",
                        )
                        duration_loss = duration_loss_per_sample.mean()
                        pred_frames = torch.expm1(duration_pred.float()).clamp_min(0.0)
                        duration_mae_frames = (pred_frames - num_frames.float()).abs().mean()
                        if duration_only:
                            duration_group_totals = duration_condition_group_totals(
                                duration_loss_per_sample=duration_loss_per_sample,
                                pred_frames=pred_frames,
                                target_frames=num_frames.float(),
                                has_speaker=has_speaker,
                                has_caption=has_caption,
                            )
                    if duration_only:
                        loss = duration_loss
                    else:
                        loss = rf_loss + (float(train_cfg.duration_loss_weight) * duration_loss)
                    speaker_inversion_module = getattr(raw_model, "speaker_inversion", None)
                    residual_regularization_weight = float(
                        train_cfg.speaker_inversion_residual_regularization_weight
                    )
                    # 重み0のときは計算グラフへ余計なノードを足さない
                    if (
                        isinstance(speaker_inversion_module, SpeakerInversionEmbedding)
                        and speaker_inversion_module.uses_pre_norm_residual
                        and residual_regularization_weight > 0.0
                    ):
                        loss = loss + (
                            residual_regularization_weight
                            * speaker_inversion_module.residual_regularization_loss()
                        )
                    (loss / float(accum_steps)).backward()
                    if pretrained_projector_warmup_active:
                        clear_non_pretrained_projector_grads(raw_model)
                    elif caption_warmup_active:
                        clear_non_caption_grads(raw_model)

                accum_loss += loss.detach()
                accum_rf_loss += rf_loss.detach()
                accum_duration_loss += duration_loss.detach()
                accum_duration_mae_frames += duration_mae_frames.detach()
                accum_duration_group_totals += duration_group_totals
                if not should_step:
                    continue

                should_log_step = (step + 1) % train_cfg.log_every == 0
                step_loss = accum_loss / float(accum_steps)
                step_rf_loss = accum_rf_loss / float(accum_steps)
                step_duration_loss = accum_duration_loss / float(accum_steps)
                step_duration_mae_frames = accum_duration_mae_frames / float(accum_steps)
                step_duration_group_totals = accum_duration_group_totals.clone()
                accum_loss.zero_()
                accum_rf_loss.zero_()
                accum_duration_loss.zero_()
                accum_duration_mae_frames.zero_()
                accum_duration_group_totals.zero_()
                torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip_norm)
                optimizer.step()
                speaker_inversion_module = getattr(raw_model, "speaker_inversion", None)
                if isinstance(speaker_inversion_module, SpeakerInversionEmbedding):
                    speaker_inversion_module.project_residual_()
                optimizer.zero_grad(set_to_none=True)
                if scheduler is not None:
                    scheduler.step()
                step += 1
                progress.update(step)
                if caption_warmup_active and step >= train_cfg.caption_warmup_steps:
                    caption_warmup_active = False
                    if is_main_process:
                        progress.write("caption warmup complete; all parameters are now updating.")
                if (
                    pretrained_projector_warmup_active
                    and step >= train_cfg.pretrained_projector_warmup_steps
                ):
                    pretrained_projector_warmup_active = False
                    if is_main_process:
                        progress.write(
                            "pretrained projector warmup complete; DiT, speaker, duration, and "
                            "the pretrained backbone are now updating."
                        )

                if should_log_step:
                    loss_value = reduce_mean(step_loss, world_size, distributed).item()
                    rf_loss_value = reduce_mean(step_rf_loss, world_size, distributed).item()
                    duration_loss_value = reduce_mean(
                        step_duration_loss, world_size, distributed
                    ).item()
                    duration_mae_frames_value = reduce_mean(
                        step_duration_mae_frames, world_size, distributed
                    ).item()
                    duration_group_metrics: dict[str, float] = {}
                    if duration_only:
                        duration_group_totals = reduce_sum(
                            step_duration_group_totals,
                            distributed,
                        )
                        duration_group_metrics = duration_condition_group_metrics(
                            duration_group_totals
                        )
                    lr_value = current_lr(optimizer)
                    pretrained_lr_value = current_pretrained_text_encoder_lr(optimizer)
                    progress_metrics: dict[str, float] = {
                        "loss": loss_value,
                        "rf": rf_loss_value,
                        "lr": lr_value,
                    }
                    residual_norm_value: float | None = None
                    speaker_inversion_module = getattr(raw_model, "speaker_inversion", None)
                    if (
                        isinstance(speaker_inversion_module, SpeakerInversionEmbedding)
                        and speaker_inversion_module.uses_pre_norm_residual
                    ):
                        residual_norm_value = float(
                            speaker_inversion_module.relative_residual_norm().detach().cpu()
                        )
                        progress_metrics["residual_norm"] = residual_norm_value
                    if pretrained_lr_value is not None:
                        progress_metrics["text_lr"] = pretrained_lr_value
                    if raw_model.cfg.use_duration_predictor:
                        progress_metrics["dur"] = duration_loss_value
                        progress_metrics["dur_mae"] = duration_mae_frames_value
                        if duration_only:
                            progress_metrics["dur_sp"] = duration_group_metrics[
                                "duration_loss_speaker"
                            ]
                            progress_metrics["dur_no_sp"] = duration_group_metrics[
                                "duration_loss_no_speaker"
                            ]
                            progress_metrics["dur_cap"] = duration_group_metrics[
                                "duration_loss_caption"
                            ]
                            progress_metrics["dur_no_cap"] = duration_group_metrics[
                                "duration_loss_no_caption"
                            ]
                    progress.log(
                        step=step,
                        epoch=epoch,
                        epoch_step=epoch_step,
                        epoch_total=len(loader),
                        metrics=progress_metrics,
                        global_batch_size=global_batch_size,
                    )
                    if is_main_process:
                        if raw_model.cfg.use_duration_predictor:
                            message = (
                                f"step={step} loss={loss_value:.6f} rf={rf_loss_value:.6f} "
                                f"dur={duration_loss_value:.6f} "
                                f"dur_mae={duration_mae_frames_value:.2f}"
                            )
                            if duration_only:
                                group_suffix = duration_condition_group_log_suffix(
                                    duration_group_metrics
                                )
                                if group_suffix:
                                    message += f" {group_suffix}"
                            if residual_norm_value is not None:
                                message += f" residual_norm={residual_norm_value:.6f}"
                            progress.write(f"{message} lr={lr_value:.3e}")
                        else:
                            progress.write(
                                f"step={step} loss={loss_value:.6f} rf={rf_loss_value:.6f} "
                                f"lr={lr_value:.3e}"
                            )
                        if wandb_run is not None:
                            metrics = {
                                "train/loss": loss_value,
                                "train/rf_loss": rf_loss_value,
                                "train/lr": lr_value,
                            }
                            if pretrained_lr_value is not None:
                                metrics["train/pretrained_text_encoder_lr"] = pretrained_lr_value
                            if residual_norm_value is not None:
                                metrics["train/speaker_inversion_relative_residual_norm"] = (
                                    residual_norm_value
                                )
                            if raw_model.cfg.use_duration_predictor:
                                metrics["train/duration_loss"] = duration_loss_value
                                metrics["train/duration_mae_frames"] = duration_mae_frames_value
                                if duration_only:
                                    metrics.update(
                                        duration_condition_group_wandb_metrics(
                                            "train",
                                            duration_group_metrics,
                                        )
                                    )
                            wandb_run.log(metrics, step=step)

                if step % train_cfg.save_every == 0:
                    dataloader_state = _collect_dataloader_state(
                        loader,
                        distributed=distributed,
                        rank=rank,
                        world_size=world_size,
                    )
                    rng_state = _collect_rng_state(
                        distributed=distributed,
                        rank=rank,
                        world_size=world_size,
                    )
                    runtime_state = _runtime_state_for_checkpoint(
                        epoch=epoch,
                        epoch_step=epoch_step,
                    )
                    if is_main_process:
                        save_checkpoint(
                            _periodic_checkpoint_path(output_dir, step, train_cfg),
                            raw_model,
                            optimizer,
                            scheduler,
                            step,
                            model_cfg,
                            train_cfg,
                            base_init=base_init,
                            lora_metadata=lora_metadata,
                            dataloader_state=dataloader_state,
                            runtime_state=runtime_state,
                            rng_state=rng_state,
                            speaker_inversion_resume_contract=(speaker_inversion_resume_contract),
                        )
                        enforce_periodic_checkpoint_limit(
                            output_dir=output_dir,
                            keep_count=periodic_checkpoint_keep,
                        )

                if (
                    valid_loader is not None
                    and train_cfg.valid_every > 0
                    and step % train_cfg.valid_every == 0
                ):
                    valid_metrics = run_validation(
                        model=model,
                        loader=valid_loader,
                        train_cfg=train_cfg,
                        device=device,
                        use_bf16=use_bf16,
                        distributed=distributed,
                    )
                    best_dataloader_state = None
                    best_runtime_state = None
                    best_rng_state = None
                    if checkpoint_retention_enabled:
                        best_dataloader_state = _collect_dataloader_state(
                            loader,
                            distributed=distributed,
                            rank=rank,
                            world_size=world_size,
                        )
                        best_rng_state = _collect_rng_state(
                            distributed=distributed,
                            rank=rank,
                            world_size=world_size,
                        )
                        best_runtime_state = _runtime_state_for_checkpoint(
                            epoch=epoch,
                            epoch_step=epoch_step,
                        )
                    if is_main_process:
                        if raw_model.cfg.use_duration_predictor:
                            message = (
                                "valid step={} loss={:.6f} rf={:.6f} dur={:.6f} dur_mae={:.2f}"
                            ).format(
                                step,
                                valid_metrics["loss"],
                                valid_metrics["rf_loss"],
                                valid_metrics["duration_loss"],
                                valid_metrics["duration_mae_frames"],
                            )
                            if duration_only:
                                group_suffix = duration_condition_group_log_suffix(valid_metrics)
                                if group_suffix:
                                    message += f" {group_suffix}"
                            progress.write(
                                "{} (samples={:.0f})".format(
                                    message,
                                    valid_metrics["num_samples"],
                                )
                            )
                        else:
                            progress.write(
                                ("valid step={} loss={:.6f} rf={:.6f} (samples={:.0f})").format(
                                    step,
                                    valid_metrics["loss"],
                                    valid_metrics["rf_loss"],
                                    valid_metrics["num_samples"],
                                )
                            )
                        if wandb_run is not None:
                            metrics = {
                                "valid/loss": valid_metrics["loss"],
                                "valid/rf_loss": valid_metrics["rf_loss"],
                            }
                            if raw_model.cfg.use_duration_predictor:
                                metrics["valid/duration_loss"] = valid_metrics["duration_loss"]
                                metrics["valid/duration_mae_frames"] = valid_metrics[
                                    "duration_mae_frames"
                                ]
                                if duration_only:
                                    metrics.update(
                                        duration_condition_group_wandb_metrics(
                                            "valid",
                                            valid_metrics,
                                        )
                                    )
                            wandb_run.log(metrics, step=step)
                        best_val_checkpoints, best_path = maybe_save_best_val_loss_checkpoint(
                            output_dir=output_dir,
                            checkpoints=best_val_checkpoints,
                            keep_best_n=train_cfg.checkpoint_best_n,
                            val_loss=float(valid_metrics["loss"]),
                            step=step,
                            model=raw_model,
                            optimizer=optimizer,
                            scheduler=scheduler,
                            model_cfg=model_cfg,
                            train_cfg=train_cfg,
                            base_init=base_init,
                            lora_metadata=lora_metadata,
                            dataloader_state=best_dataloader_state,
                            runtime_state=best_runtime_state,
                            rng_state=best_rng_state,
                            speaker_inversion_resume_contract=(speaker_inversion_resume_contract),
                        )
                        if best_path is not None:
                            progress.write(
                                "saved best val checkpoint: {} (loss={:.6f})".format(
                                    best_path.name,
                                    float(valid_metrics["loss"]),
                                )
                            )

                if step >= train_cfg.max_steps:
                    break

        if (
            valid_loader is not None
            and train_cfg.valid_every > 0
            and step % train_cfg.valid_every != 0
        ):
            valid_metrics = run_validation(
                model=model,
                loader=valid_loader,
                train_cfg=train_cfg,
                device=device,
                use_bf16=use_bf16,
                distributed=distributed,
            )
            final_best_dataloader_state = None
            final_best_runtime_state = None
            final_best_rng_state = None
            if checkpoint_retention_enabled:
                final_best_dataloader_state = _collect_dataloader_state(
                    loader,
                    distributed=distributed,
                    rank=rank,
                    world_size=world_size,
                )
                final_best_rng_state = _collect_rng_state(
                    distributed=distributed,
                    rank=rank,
                    world_size=world_size,
                )
                final_best_runtime_state = _runtime_state_for_checkpoint(
                    epoch=epoch,
                    epoch_step=last_epoch_step,
                )
            if is_main_process:
                if raw_model.cfg.use_duration_predictor:
                    message = (
                        "valid final step={} loss={:.6f} rf={:.6f} dur={:.6f} dur_mae={:.2f}"
                    ).format(
                        step,
                        valid_metrics["loss"],
                        valid_metrics["rf_loss"],
                        valid_metrics["duration_loss"],
                        valid_metrics["duration_mae_frames"],
                    )
                    if duration_only:
                        group_suffix = duration_condition_group_log_suffix(valid_metrics)
                        if group_suffix:
                            message += f" {group_suffix}"
                    progress.write(
                        "{} (samples={:.0f})".format(
                            message,
                            valid_metrics["num_samples"],
                        )
                    )
                else:
                    progress.write(
                        ("valid final step={} loss={:.6f} rf={:.6f} (samples={:.0f})").format(
                            step,
                            valid_metrics["loss"],
                            valid_metrics["rf_loss"],
                            valid_metrics["num_samples"],
                        )
                    )
                if wandb_run is not None:
                    metrics = {
                        "valid/loss": valid_metrics["loss"],
                        "valid/rf_loss": valid_metrics["rf_loss"],
                    }
                    if raw_model.cfg.use_duration_predictor:
                        metrics["valid/duration_loss"] = valid_metrics["duration_loss"]
                        metrics["valid/duration_mae_frames"] = valid_metrics["duration_mae_frames"]
                        if duration_only:
                            metrics.update(
                                duration_condition_group_wandb_metrics(
                                    "valid",
                                    valid_metrics,
                                )
                            )
                    wandb_run.log(metrics, step=step)
                best_val_checkpoints, best_path = maybe_save_best_val_loss_checkpoint(
                    output_dir=output_dir,
                    checkpoints=best_val_checkpoints,
                    keep_best_n=train_cfg.checkpoint_best_n,
                    val_loss=float(valid_metrics["loss"]),
                    step=step,
                    model=raw_model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    model_cfg=model_cfg,
                    train_cfg=train_cfg,
                    base_init=base_init,
                    lora_metadata=lora_metadata,
                    dataloader_state=final_best_dataloader_state,
                    runtime_state=final_best_runtime_state,
                    rng_state=final_best_rng_state,
                    speaker_inversion_resume_contract=speaker_inversion_resume_contract,
                )
                if best_path is not None:
                    progress.write(
                        "saved best val checkpoint: {} (loss={:.6f})".format(
                            best_path.name,
                            float(valid_metrics["loss"]),
                        )
                    )

        final_dataloader_state = _collect_dataloader_state(
            loader,
            distributed=distributed,
            rank=rank,
            world_size=world_size,
        )
        final_rng_state = _collect_rng_state(
            distributed=distributed,
            rank=rank,
            world_size=world_size,
        )
        final_runtime_state = _runtime_state_for_checkpoint(
            epoch=epoch,
            epoch_step=last_epoch_step,
        )
        if is_main_process:
            save_checkpoint(
                _final_checkpoint_path(output_dir, train_cfg),
                raw_model,
                optimizer,
                scheduler,
                step,
                model_cfg,
                train_cfg,
                base_init=base_init,
                lora_metadata=lora_metadata,
                dataloader_state=final_dataloader_state,
                runtime_state=final_runtime_state,
                rng_state=final_rng_state,
                speaker_inversion_resume_contract=speaker_inversion_resume_contract,
            )
            if wandb_run is not None:
                wandb_run.summary["train/final_step"] = step
            progress.write(f"Training finished at step={step}.")
    finally:
        if progress is not None:
            progress.close()
        if wandb_run is not None:
            wandb_run.finish()
        if distributed and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
