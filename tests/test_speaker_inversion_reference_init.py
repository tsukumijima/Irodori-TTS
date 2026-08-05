from pathlib import Path

import torch
from torch import nn

from irodori_tts.speaker_inversion import (
    SPEAKER_EMBEDDING_KEY,
    SPEAKER_PRE_NORM_EMBEDDING_KEY,
    SpeakerInversionEmbedding,
    load_speaker_inversion_base_payload,
    save_speaker_inversion_base_safetensors,
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

    torch.manual_seed(42)
    base = torch.randn(7, 8)
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
    model = ReferenceInitializedModel(base, max_relative_norm=0.05)
    with torch.no_grad():
        model.speaker_inversion.embedding.fill_(1.0)

    model.speaker_inversion.project_residual_()

    assert float(model.speaker_inversion.relative_residual_norm().detach()) <= 0.050001


def test_reference_base_safetensors_round_trip(tmp_path: Path) -> None:
    """Preserve the pre-normalization local tokens used to initialize training."""

    base = torch.randn(5, 8)
    path = tmp_path / "voice.speaker-base.safetensors"

    save_speaker_inversion_base_safetensors(path, base)
    payload = load_speaker_inversion_base_payload(path)

    torch.testing.assert_close(payload[SPEAKER_PRE_NORM_EMBEDDING_KEY], base)
