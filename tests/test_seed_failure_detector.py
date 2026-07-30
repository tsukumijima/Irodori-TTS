from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file as save_safetensors_file
from torch import nn

from irodori_tts.seed_failure_detector import SeedRetryPredecodeSelector


def _CreateSelector() -> SeedRetryPredecodeSelector:
    """
    固定した線形スコアを返す最小の判定器を作る。

    Returns:
        SeedRetryPredecodeSelector: 先頭特徴だけを評価する判定器
    """

    weight = torch.zeros(SeedRetryPredecodeSelector.FEATURE_COUNT)
    weight[0] = 1.0
    return SeedRetryPredecodeSelector(
        effective_weight=weight,
        effective_bias=torch.tensor(0.25),
        trigger_threshold=torch.tensor(0.5),
        pair_margin_threshold=torch.tensor(0.2),
    )


def test_selector_loads_checkpoint_and_returns_calibrated_decisions(
    tmp_path: Path,
) -> None:
    selector = _CreateSelector()
    checkpoint_path = tmp_path / "selector.safetensors"
    save_safetensors_file(
        {name: value.detach().contiguous() for name, value in selector.state_dict().items()},
        checkpoint_path,
        metadata={
            "contract_version": SeedRetryPredecodeSelector.CONTRACT_VERSION,
            "feature_count": str(SeedRetryPredecodeSelector.FEATURE_COUNT),
            "decoder_block_index": str(SeedRetryPredecodeSelector.DECODER_BLOCK_INDEX),
            "temporal_positions": str(SeedRetryPredecodeSelector.TEMPORAL_POSITIONS),
            "channels": "96",
        },
    )

    loaded = SeedRetryPredecodeSelector.from_safetensors(
        checkpoint_path,
        device=torch.device("cpu"),
    )
    generated_state = torch.zeros(
        1,
        96,
        SeedRetryPredecodeSelector.TEMPORAL_POSITIONS,
    )
    score = loaded.score(generated_state, None)
    retry_decisions = loaded.should_retry(score)
    adoption_decisions = loaded.should_adopt_retry(
        base_score=score,
        retry_score=score + 0.3,
    )

    assert score.tolist() == pytest.approx([0.25])
    assert retry_decisions.tolist() == [True]
    assert adoption_decisions.tolist() == [True]


def test_selector_rejects_unknown_checkpoint_contract(tmp_path: Path) -> None:
    selector = _CreateSelector()
    checkpoint_path = tmp_path / "selector.safetensors"
    save_safetensors_file(
        {name: value.detach().contiguous() for name, value in selector.state_dict().items()},
        checkpoint_path,
        metadata={
            "contract_version": "unknown",
            "feature_count": str(SeedRetryPredecodeSelector.FEATURE_COUNT),
            "decoder_block_index": str(SeedRetryPredecodeSelector.DECODER_BLOCK_INDEX),
            "temporal_positions": str(SeedRetryPredecodeSelector.TEMPORAL_POSITIONS),
            "channels": "96",
        },
    )

    with pytest.raises(ValueError, match="contract mismatch"):
        SeedRetryPredecodeSelector.from_safetensors(
            checkpoint_path,
            device=torch.device("cpu"),
        )


def test_retry_seed_is_deterministic_and_chunk_specific() -> None:
    first = SeedRetryPredecodeSelector.derive_retry_seed(
        base_seed=123,
        chunk_index=0,
        attempt_index=1,
    )
    repeated = SeedRetryPredecodeSelector.derive_retry_seed(
        base_seed=123,
        chunk_index=0,
        attempt_index=1,
    )
    next_chunk = SeedRetryPredecodeSelector.derive_retry_seed(
        base_seed=123,
        chunk_index=1,
        attempt_index=1,
    )

    assert first == repeated
    assert next_chunk != first
    assert 0 <= first < 2**63


def test_predecode_encoder_survives_codec_decoder_release() -> None:
    selector = _CreateSelector()
    codec_model = SimpleNamespace(
        quantizer=SimpleNamespace(out_proj=nn.Conv1d(32, 96, kernel_size=1)),
        decoder=SimpleNamespace(model=nn.ModuleList([nn.Identity() for _ in range(5)])),
    )
    latent = torch.randn(1, 24, 32)

    # 完全復号器を解放する前に、判定へ必要な射影と block 4までを独立した寿命へ移す
    selector.bind_codec_model(codec_model)
    del codec_model.decoder
    encoded = selector.encode_latent(latent)

    assert encoded.shape == (1, 96, SeedRetryPredecodeSelector.TEMPORAL_POSITIONS)
    assert torch.isfinite(encoded).all().item() is True


def test_predecode_encoder_requires_bound_codec_modules() -> None:
    selector = _CreateSelector()

    with pytest.raises(RuntimeError, match="predecode encoder is not bound"):
        selector.encode_latent(torch.randn(1, 24, 32))
