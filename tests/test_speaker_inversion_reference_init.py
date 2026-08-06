from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file
from torch import nn

from irodori_tts.speaker_inversion import (
    SPEAKER_EMBEDDING_KEY,
    SPEAKER_PRE_NORM_EMBEDDING_KEY,
    SpeakerInversionEmbedding,
    load_speaker_inversion_base_payload,
    save_speaker_inversion_base_safetensors,
    speaker_inversion_checkpoint_sha256,
    speaker_inversion_state_dict,
)


class ReferenceInitializedModel(nn.Module):
    """Provide the model members required to export composed inversion tokens."""

    def __init__(self, base: torch.Tensor, max_relative_norm: float = 0.1) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.speaker_norm = nn.RMSNorm(base.shape[-1])
        self.speaker_inversion = SpeakerInversionEmbedding(
            num_tokens=base.shape[0] + 1,
            speaker_dim=base.shape[1],
            init_std=0.02,
            base_pre_norm_embedding=base,
            max_relative_residual_norm=max_relative_norm,
        )

    def compose_speaker_condition_pre_norm(
        self,
        *,
        state: torch.Tensor,
        mask: torch.Tensor,
        batch_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply the same normalization and mean-token contract as the real model."""

        state = self.speaker_norm(state.to(device=device, dtype=dtype))
        mean_state = state.mean(dim=1, keepdim=True)
        return torch.cat([mean_state, state], dim=1), torch.cat(
            [torch.ones((batch_size, 1), dtype=torch.bool, device=device), mask],
            dim=1,
        )


def test_zero_residual_reproduces_reference_condition() -> None:
    """Keep step zero identical to the ordinary reference-derived speaker state."""

    generator = torch.Generator().manual_seed(42)
    base = torch.randn(7, 8, generator=generator)
    model = ReferenceInitializedModel(base)
    normalized = model.speaker_norm(base)
    expected = torch.cat([normalized.mean(dim=0, keepdim=True), normalized], dim=0)

    payload = speaker_inversion_state_dict(model)

    torch.testing.assert_close(payload[SPEAKER_EMBEDDING_KEY], expected)


def test_export_recomputes_mean_token_after_residual_update() -> None:
    """Derive token zero from the updated local tokens instead of optimizing it independently."""

    base = torch.arange(24, dtype=torch.float32).reshape(3, 8)
    model = ReferenceInitializedModel(base)
    with torch.no_grad():
        model.speaker_inversion.embedding.add_(0.25)

    embedding = speaker_inversion_state_dict(model)[SPEAKER_EMBEDDING_KEY]

    torch.testing.assert_close(embedding[0], embedding[1:].mean(dim=0))


def test_residual_projection_enforces_relative_norm_limit() -> None:
    """Prevent optimization from discarding the clear reference-derived starting state."""

    base = torch.ones(4, 8)
    max_relative_norm = 0.05
    model = ReferenceInitializedModel(base, max_relative_norm=max_relative_norm)
    with torch.no_grad():
        model.speaker_inversion.embedding.fill_(1.0)

    model.speaker_inversion.project_residual_()

    assert float(model.speaker_inversion.relative_residual_norm().detach()) == pytest.approx(
        max_relative_norm,
        abs=1e-6,
    )


def test_reference_base_safetensors_round_trip(tmp_path: Path) -> None:
    """Preserve the pre-normalization local tokens used to initialize training."""

    base = torch.randn(5, 8)
    path = tmp_path / "voice.speaker-base.safetensors"

    save_speaker_inversion_base_safetensors(path, base)
    payload = load_speaker_inversion_base_payload(path)

    torch.testing.assert_close(payload[SPEAKER_PRE_NORM_EMBEDDING_KEY], base)


def test_reference_base_without_metadata_reports_format_error(tmp_path: Path) -> None:
    """メタデータのない safetensors を形式エラーとして拒否する。"""

    path = tmp_path / "voice.speaker-base.safetensors"
    save_file({SPEAKER_PRE_NORM_EMBEDDING_KEY: torch.randn(5, 8)}, str(path))

    with pytest.raises(ValueError, match="unsupported or missing format_version"):
        load_speaker_inversion_base_payload(path)


def test_reference_base_save_expands_home_directory(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """
    チルダで始まる保存先を利用者のホームディレクトリへ展開する。

    Args:
        monkeypatch (pytest.MonkeyPatch): HOME を一時ディレクトリへ差し替えるフィクスチャ
        tmp_path (Path): 保存先に使う一時ホームディレクトリ
    """

    monkeypatch.setenv("HOME", str(tmp_path))

    save_speaker_inversion_base_safetensors(
        "~/voice.speaker-base.safetensors",
        torch.randn(5, 8),
    )

    assert (tmp_path / "voice.speaker-base.safetensors").is_file()


def test_reference_base_rejects_different_checkpoint(tmp_path: Path) -> None:
    """Reject a base extracted from weights other than the frozen training checkpoint."""

    checkpoint = tmp_path / "model.safetensors"
    other_checkpoint = tmp_path / "other.safetensors"
    checkpoint.write_bytes(b"original checkpoint")
    other_checkpoint.write_bytes(b"different checkpoint")
    path = tmp_path / "voice.speaker-base.safetensors"
    save_speaker_inversion_base_safetensors(
        path,
        torch.randn(5, 8),
        metadata={
            "checkpoint": str(checkpoint.resolve()),
            "checkpoint_sha256": speaker_inversion_checkpoint_sha256(checkpoint),
            "speaker_patch_size": "4",
        },
    )

    with pytest.raises(ValueError, match="checkpoint mismatch"):
        load_speaker_inversion_base_payload(
            path,
            expected_checkpoint=other_checkpoint,
            expected_speaker_dim=8,
            expected_speaker_patch_size=4,
        )


def test_reference_base_accepts_relocated_checkpoint(tmp_path: Path) -> None:
    """Identify the frozen checkpoint by content instead of its filesystem location."""

    original_checkpoint = tmp_path / "original.safetensors"
    relocated_checkpoint = tmp_path / "relocated.safetensors"
    original_checkpoint.write_bytes(b"same checkpoint contents")
    relocated_checkpoint.write_bytes(original_checkpoint.read_bytes())
    path = tmp_path / "voice.speaker-base.safetensors"
    save_speaker_inversion_base_safetensors(
        path,
        torch.randn(5, 8),
        metadata={
            "checkpoint": str(original_checkpoint.resolve()),
            "checkpoint_sha256": speaker_inversion_checkpoint_sha256(original_checkpoint),
            "speaker_patch_size": "4",
        },
    )

    payload = load_speaker_inversion_base_payload(
        path,
        expected_checkpoint=relocated_checkpoint,
        expected_speaker_dim=8,
        expected_speaker_patch_size=4,
    )

    assert payload[SPEAKER_PRE_NORM_EMBEDDING_KEY].shape == (5, 8)
