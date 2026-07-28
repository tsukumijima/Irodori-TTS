from __future__ import annotations

import torch

from irodori_tts.packed_attention import (
    build_packed_attention_plan,
    should_use_packed_attention,
)


def test_shape_policy_matches_measured_product_boundaries() -> None:
    # 通常参照の長い固定コンテキストは、短い潜在系列から詰め替え対象になる
    assert should_use_packed_attention(batch_size=1, query_length=32, key_length=1569) is True

    # Speaker Inversion は CFG バッチ数ごとに実測上の損益分岐点を使う
    assert should_use_packed_attention(batch_size=4, query_length=128, key_length=912) is False
    assert should_use_packed_attention(batch_size=4, query_length=160, key_length=944) is True
    assert should_use_packed_attention(batch_size=1, query_length=224, key_length=1008) is False
    assert should_use_packed_attention(batch_size=1, query_length=256, key_length=1040) is True

    # 参照なしの3分岐 CFG は Q=128 から詰め替え費用を回収できる
    assert should_use_packed_attention(batch_size=3, query_length=96, key_length=869) is False
    assert should_use_packed_attention(batch_size=3, query_length=128, key_length=901) is True


def test_plan_preserves_valid_key_order_and_lengths() -> None:
    latent_mask = torch.tensor([[True, True], [True, False]])
    text_mask = torch.tensor([[True, False], [False, True]])
    speaker_mask = torch.ones((2, 1500), dtype=torch.bool)
    plan = build_packed_attention_plan(
        latent_mask=latent_mask,
        text_mask=text_mask,
        speaker_mask=speaker_mask,
        caption_mask=None,
    )

    assert plan is not None
    key_mask = torch.cat([latent_mask, text_mask, speaker_mask], dim=1)
    for batch_index in range(key_mask.shape[0]):
        valid_length = int(plan.key_lengths[batch_index].item())
        expected_indices = torch.nonzero(key_mask[batch_index], as_tuple=False).flatten()
        assert torch.equal(
            plan.packed_indices[batch_index, :valid_length],
            expected_indices,
        )
    assert plan.query_lengths.flatten().tolist() == [2, 1]
