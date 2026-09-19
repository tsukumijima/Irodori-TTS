from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from irodori_tts.speaker_inversion import (
    SPEAKER_INVERSION_BASE_FORMAT_VERSION,
    SPEAKER_PRE_NORM_EMBEDDING_KEY,
    load_speaker_inversion_base_payload,
    save_speaker_inversion_base_safetensors,
    speaker_inversion_checkpoint_sha256,
)


def test_reference_base_safetensors_round_trip(tmp_path: Path) -> None:
    """Preserve the pre-normalization local tokens extracted from a reference."""

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


def test_reference_base_rejects_tensor_dimension_mismatch(tmp_path: Path) -> None:
    """Reject a tensor whose dimension disagrees with validated metadata."""

    path = tmp_path / "voice.speaker-base.safetensors"
    save_file(
        {SPEAKER_PRE_NORM_EMBEDDING_KEY: torch.randn(5, 7)},
        str(path),
        metadata={
            "format_version": SPEAKER_INVERSION_BASE_FORMAT_VERSION,
            "local_tokens": "5",
            "speaker_dim": "8",
        },
    )

    with pytest.raises(ValueError, match=r"speaker_pre_norm_embedding.*dim"):
        load_speaker_inversion_base_payload(path, expected_speaker_dim=8)


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
