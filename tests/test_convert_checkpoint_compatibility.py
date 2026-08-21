import pytest

from convert_checkpoint_to_safetensors import _condition_encoder_compatibility_mismatches
from irodori_tts.config import ModelConfig


@pytest.mark.parametrize(
    ("base_cfg", "adapter_cfg", "expected_fields"),
    [
        (
            ModelConfig(text_add_bos=True),
            ModelConfig(text_add_bos=False),
            ["text_add_bos"],
        ),
        (
            ModelConfig(use_caption_condition=True),
            ModelConfig(use_caption_condition=False),
            ["caption_condition_disabled"],
        ),
    ],
)
def test_condition_encoder_compatibility_rejects_silent_merge_mismatches(
    base_cfg: ModelConfig,
    adapter_cfg: ModelConfig,
    expected_fields: list[str],
) -> None:
    """
    重み形状だけでは検出できない条件エンコーダ契約の差を結合前に拒否する。
    """

    mismatches = _condition_encoder_compatibility_mismatches(base_cfg, adapter_cfg)

    for field in expected_fields:
        assert field in mismatches


def test_condition_encoder_compatibility_allows_caption_upgrade() -> None:
    """
    caption 非対応 base から caption 対応 adapter への upgrade だけを許可する。
    """

    base_cfg = ModelConfig(use_caption_condition=False)
    adapter_cfg = ModelConfig(
        use_caption_condition=True,
        caption_tokenizer_repo="adapter-caption-tokenizer",
        caption_add_bos=True,
    )

    mismatches = _condition_encoder_compatibility_mismatches(base_cfg, adapter_cfg)

    assert not any(field.startswith("caption_") for field in mismatches)
