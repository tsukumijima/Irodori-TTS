import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from torch import nn
from torchdata.stateful_dataloader import StatefulDataLoader

from irodori_tts.config import ModelConfig, TrainConfig
from irodori_tts.dataset import LatentTextDataset
from irodori_tts.speaker_inversion import SpeakerInversionEmbedding
from train import (
    DATALOADER_STATE_KEY,
    RNG_STATE_KEY,
    RUNTIME_STATE_KEY,
    SPEAKER_EMBEDDING_KEY,
    SPEAKER_INVERSION_MODULE_STATE_KEY,
    _capture_rng_state,
    _restore_resume_speaker_inversion_config,
    _restore_rng_state,
    _speaker_inversion_trainer_state_path,
    enforce_periodic_checkpoint_limit,
    save_checkpoint,
    set_seed,
)


class SpeakerInversionCheckpointModel(nn.Module):
    """Provide the minimal model contract required by Speaker Inversion saving."""

    def __init__(self) -> None:
        super().__init__()
        self.speaker_inversion = SpeakerInversionEmbedding(
            num_tokens=2,
            speaker_dim=4,
            init_std=0.02,
        )


def test_set_seed_initializes_all_training_random_generators(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Initialize Python, NumPy, and PyTorch streams from the configured training seed."""

    monkeypatch.setattr(torch.cuda, "manual_seed_all", lambda _: None)
    set_seed(42)
    expected = (random.random(), float(np.random.random()), float(torch.rand(())))

    set_seed(42)
    actual = (random.random(), float(np.random.random()), float(torch.rand(())))

    assert actual == expected


def test_resume_restores_speaker_inversion_training_contract() -> None:
    """Restore token shape and residual constraints from the exact-resume sidecar."""

    restored = _restore_resume_speaker_inversion_config(
        TrainConfig(
            speaker_inversion_enabled=True,
            speaker_inversion_tokens=188,
            speaker_inversion_residual_regularization_weight=0.0,
            speaker_inversion_max_relative_residual_norm=None,
        ),
        resume_train_cfg={
            "speaker_inversion_enabled": True,
            "speaker_inversion_tokens": 32,
            "speaker_inversion_init_std": 0.02,
            "speaker_inversion_residual_regularization_weight": 0.01,
            "speaker_inversion_max_relative_residual_norm": 0.15,
        },
        raw_argv=["train.py", "--resume", "checkpoint.speaker.trainer.pt"],
    )

    assert restored.speaker_inversion_tokens == 32
    assert restored.speaker_inversion_residual_regularization_weight == 0.01
    assert restored.speaker_inversion_max_relative_residual_norm == 0.15


def test_resume_rejects_changed_speaker_inversion_training_contract() -> None:
    """Reject an explicit token-count change that would invalidate exact continuation."""

    with pytest.raises(ValueError, match="speaker_inversion_tokens"):
        _restore_resume_speaker_inversion_config(
            TrainConfig(speaker_inversion_enabled=True, speaker_inversion_tokens=64),
            resume_train_cfg={"speaker_inversion_tokens": 32},
            raw_argv=["train.py", "--speaker-inversion-tokens", "64"],
        )


def test_speaker_inversion_checkpoint_saves_inference_and_trainer_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep the inference embedding separate from the complete resume state."""

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    model = SpeakerInversionCheckpointModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    model.speaker_inversion.embedding.square().sum().backward()
    optimizer.step()
    embedding_path = tmp_path / "checkpoint_0000250.speaker.safetensors"
    dataloader_state = {"index": 7}
    runtime_state = {"epoch": 3, "sampler_epoch": 2, "epoch_step": 4}

    save_checkpoint(
        embedding_path,
        model,
        optimizer,
        scheduler=None,
        step=250,
        model_cfg=ModelConfig(speaker_dim=4),
        train_cfg=TrainConfig(speaker_inversion_enabled=True),
        base_init={"mode": "checkpoint", "checkpoint_path": "/base/model.safetensors"},
        dataloader_state=dataloader_state,
        runtime_state=runtime_state,
    )

    trainer_path = _speaker_inversion_trainer_state_path(embedding_path)
    payload = torch.load(trainer_path, map_location="cpu", weights_only=True)
    assert embedding_path.is_file()
    assert not (tmp_path / f".{trainer_path.name}.tmp").exists()
    assert payload["step"] == 250
    assert payload[DATALOADER_STATE_KEY] == dataloader_state
    assert payload[RUNTIME_STATE_KEY] == runtime_state
    assert payload["optimizer"]["state"]
    assert RNG_STATE_KEY in payload
    assert SPEAKER_INVERSION_MODULE_STATE_KEY in payload
    torch.testing.assert_close(
        payload[SPEAKER_EMBEDDING_KEY],
        model.speaker_inversion.embedding.detach(),
    )


def test_rng_state_round_trip_restores_all_cpu_generators(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Resume Python, NumPy, and PyTorch random streams at the saved position."""

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    state = _capture_rng_state()
    expected = (random.random(), float(np.random.random()), float(torch.rand(())))
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    _restore_rng_state(state)

    actual = (random.random(), float(np.random.random()), float(torch.rand(())))
    assert actual == expected


def test_stateful_dataloader_restores_worker_reference_sampling(tmp_path: Path) -> None:
    """
    保存した DataLoader 位置からワーカー内の参照抽選を再開できることを確認する。
    """

    manifest_path = tmp_path / "manifest.jsonl"
    manifest_rows: list[str] = []
    for sample_index in range(12):
        latent_path = tmp_path / f"latent_{sample_index}.pt"
        torch.save(torch.full((4, 2), float(sample_index)), latent_path)
        manifest_rows.append(
            json.dumps(
                {
                    "text": f"sample {sample_index}",
                    "latent_path": latent_path.name,
                    "speaker_id": "speaker",
                    "num_frames": 4,
                    "caption": [f"caption {sample_index} a", f"caption {sample_index} b"],
                }
            )
        )
    manifest_path.write_text("\n".join(manifest_rows) + "\n", encoding="utf-8")

    def build_loader() -> Any:
        # 親プロセスの乱数状態だけでは復元できない参照抽選を実ワーカーで検証する
        return StatefulDataLoader(
            LatentTextDataset(
                manifest_path=manifest_path,
                latent_dim=2,
                enable_caption_condition=True,
                enable_speaker_condition=True,
                ref_min_frames=4,
                ref_max_frames=12,
            ),
            batch_size=None,
            num_workers=2,
            persistent_workers=True,
            snapshot_every_n_steps=1,
        )

    uninterrupted_loader = build_loader()
    uninterrupted_iterator = iter(uninterrupted_loader)
    for _ in range(3):
        next(uninterrupted_iterator)
    saved_state = uninterrupted_loader.state_dict()
    expected_items = []
    for _ in range(4):
        item = next(uninterrupted_iterator)
        expected_items.append((item["caption"], item["ref_latent"].clone()))

    resumed_loader = build_loader()
    resumed_loader.load_state_dict(saved_state)
    resumed_iterator = iter(resumed_loader)
    actual_items = [next(resumed_iterator) for _ in range(4)]

    for actual, (expected_caption, expected_reference) in zip(
        actual_items,
        expected_items,
        strict=True,
    ):
        assert actual["caption"] == expected_caption
        torch.testing.assert_close(actual["ref_latent"], expected_reference, rtol=0.0, atol=0.0)


def test_speaker_inversion_resume_matches_uninterrupted_updates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Recover the same embedding after an interrupted stochastic optimization."""

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    def run_update(
        model: SpeakerInversionCheckpointModel,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
    ) -> None:
        # 3系統の乱数を損失へ混ぜ、いずれかの復元漏れを最終埋め込みの差として検出する
        random_scale = random.random() + float(np.random.random()) + float(torch.rand(()))
        loss = model.speaker_inversion.embedding.square().sum() * random_scale
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        scheduler.step()

    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    uninterrupted_model = SpeakerInversionCheckpointModel()
    uninterrupted_optimizer = torch.optim.AdamW(uninterrupted_model.parameters(), lr=0.01)
    uninterrupted_scheduler = torch.optim.lr_scheduler.StepLR(
        uninterrupted_optimizer,
        step_size=1,
        gamma=0.9,
    )
    for _ in range(4):
        run_update(uninterrupted_model, uninterrupted_optimizer, uninterrupted_scheduler)

    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    interrupted_model = SpeakerInversionCheckpointModel()
    interrupted_optimizer = torch.optim.AdamW(interrupted_model.parameters(), lr=0.01)
    interrupted_scheduler = torch.optim.lr_scheduler.StepLR(
        interrupted_optimizer,
        step_size=1,
        gamma=0.9,
    )
    for _ in range(2):
        run_update(interrupted_model, interrupted_optimizer, interrupted_scheduler)
    embedding_path = tmp_path / "checkpoint_0000002.speaker.safetensors"
    save_checkpoint(
        embedding_path,
        interrupted_model,
        interrupted_optimizer,
        scheduler=interrupted_scheduler,
        step=2,
        model_cfg=ModelConfig(speaker_dim=4),
        train_cfg=TrainConfig(speaker_inversion_enabled=True),
        base_init={"mode": "checkpoint", "checkpoint_path": "/base/model.safetensors"},
    )

    # 新しいプロセス相当のモデルと optimizer へ sidecar の全状態を復元する
    payload = torch.load(
        _speaker_inversion_trainer_state_path(embedding_path),
        map_location="cpu",
        weights_only=True,
    )
    resumed_model = SpeakerInversionCheckpointModel()
    with torch.no_grad():
        resumed_model.speaker_inversion.embedding.copy_(payload[SPEAKER_EMBEDDING_KEY])
    resumed_optimizer = torch.optim.AdamW(resumed_model.parameters(), lr=0.01)
    resumed_optimizer.load_state_dict(payload["optimizer"])
    resumed_scheduler = torch.optim.lr_scheduler.StepLR(
        resumed_optimizer,
        step_size=1,
        gamma=0.9,
    )
    resumed_scheduler.load_state_dict(payload["scheduler"])
    _restore_rng_state(payload[RNG_STATE_KEY])
    for _ in range(2):
        run_update(resumed_model, resumed_optimizer, resumed_scheduler)

    torch.testing.assert_close(
        resumed_model.speaker_inversion.embedding,
        uninterrupted_model.speaker_inversion.embedding,
        rtol=0.0,
        atol=0.0,
    )


def test_speaker_inversion_retention_removes_matching_trainer_state(tmp_path: Path) -> None:
    """Remove inference and resume artifacts together when pruning a checkpoint."""

    for step in (1, 2):
        embedding_path = tmp_path / f"checkpoint_{step:07d}.speaker.safetensors"
        embedding_path.touch()
        _speaker_inversion_trainer_state_path(embedding_path).touch()

    enforce_periodic_checkpoint_limit(tmp_path, keep_count=1)

    kept_embedding_path = tmp_path / "checkpoint_0000002.speaker.safetensors"
    stale_embedding_path = tmp_path / "checkpoint_0000001.speaker.safetensors"
    assert kept_embedding_path.is_file()
    assert _speaker_inversion_trainer_state_path(kept_embedding_path).is_file()
    assert not stale_embedding_path.exists()
    assert not _speaker_inversion_trainer_state_path(stale_embedding_path).exists()
