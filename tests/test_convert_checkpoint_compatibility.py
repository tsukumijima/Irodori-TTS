from convert_checkpoint_to_safetensors import _condition_encoder_compatibility_mismatches
from irodori_tts.config import ModelConfig


def test_text_encoder_compatibility_rejects_text_add_bos_mismatch() -> None:
    """
    BOS 契約の差は次元一致だけでは見えないため、結合前に拒否する。
    """

    base_cfg = ModelConfig(text_add_bos=True)
    adapter_cfg = ModelConfig(text_add_bos=False)

    mismatches = _condition_encoder_compatibility_mismatches(base_cfg, adapter_cfg)

    assert "text_add_bos" in mismatches


def test_caption_compatibility_rejects_tokenizer_and_bos_mismatch() -> None:
    """
    caption の入力契約が違う adapter は、重み形状が一致しても拒否する。
    """

    base_cfg = ModelConfig(
        use_caption_condition=True,
        caption_tokenizer_repo="base-caption-tokenizer",
        caption_add_bos=True,
    )
    adapter_cfg = ModelConfig(
        use_caption_condition=True,
        caption_tokenizer_repo="adapter-caption-tokenizer",
        caption_add_bos=False,
    )

    mismatches = _condition_encoder_compatibility_mismatches(base_cfg, adapter_cfg)

    assert "caption_tokenizer_repo_resolved" in mismatches
    assert "caption_add_bos_resolved" in mismatches


def test_caption_compatibility_allows_caption_upgrade() -> None:
    """
    caption を持たない base からの既存 upgrade 経路を維持する。
    """

    base_cfg = ModelConfig(use_caption_condition=False)
    adapter_cfg = ModelConfig(
        use_caption_condition=True,
        caption_tokenizer_repo="adapter-caption-tokenizer",
        caption_add_bos=True,
    )

    mismatches = _condition_encoder_compatibility_mismatches(base_cfg, adapter_cfg)

    assert not any(field.startswith("caption_") for field in mismatches)


def test_caption_compatibility_rejects_caption_downgrade() -> None:
    """
    caption 付き base の重みを caption 無効の adapter と結合しない。
    """

    base_cfg = ModelConfig(use_caption_condition=True)
    adapter_cfg = ModelConfig(use_caption_condition=False)

    mismatches = _condition_encoder_compatibility_mismatches(base_cfg, adapter_cfg)

    assert any(field.startswith("caption_") for field in mismatches)
