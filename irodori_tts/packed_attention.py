"""高精度な可変長 cuDNN Attention の事前計算と実行を扱う。"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from importlib import import_module
from typing import Any

import torch


@dataclass(frozen=True)
class PackedAttentionPlan:
    """RF 反復を通して再利用する K/V 詰め替え計画。"""

    packed_indices: torch.Tensor
    query_lengths: torch.Tensor
    key_lengths: torch.Tensor


def should_use_packed_attention(*, batch_size: int, query_length: int, key_length: int) -> bool:
    """RTX 5090 の製品形状実測に基づき、詰め替え費用を回収できる形状を判定する。"""

    context_length = key_length - query_length
    if context_length >= 1500:
        return True
    if batch_size >= 4:
        return query_length >= 160
    if batch_size == 3:
        return query_length >= 128
    return query_length >= 256


def build_packed_attention_plan(
    *,
    latent_mask: torch.Tensor,
    text_mask: torch.Tensor,
    speaker_mask: torch.Tensor | None,
    caption_mask: torch.Tensor | None,
) -> PackedAttentionPlan | None:
    """Attention マスクから RF 全層共通の詰め替え添字を1回だけ作る。"""

    masks = [latent_mask, text_mask]
    if speaker_mask is not None:
        masks.append(speaker_mask)
    if caption_mask is not None:
        masks.append(caption_mask)
    key_mask = torch.cat(masks, dim=1).to(dtype=torch.bool)
    batch_size, key_length = key_mask.shape
    query_length = latent_mask.shape[1]
    if (
        should_use_packed_attention(
            batch_size=batch_size,
            query_length=query_length,
            key_length=key_length,
        )
        is False
    ):
        return None

    # 安定ソートにより各領域の有効キーを元の順序のまま前方へ詰める
    packed_indices = torch.argsort(
        key_mask.to(torch.int8),
        dim=-1,
        descending=True,
        stable=True,
    )
    query_lengths = latent_mask.sum(dim=-1, dtype=torch.int32).reshape(batch_size, 1, 1, 1)
    key_lengths = key_mask.sum(dim=-1, dtype=torch.int32).reshape(batch_size, 1, 1, 1)
    return PackedAttentionPlan(
        packed_indices=packed_indices,
        query_lengths=query_lengths,
        key_lengths=key_lengths,
    )


def packed_cudnn_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    plan: PackedAttentionPlan,
) -> torch.Tensor:
    """事前計算済み添字で K/V を詰め、cuDNN の可変長 SDPA を実行する。"""

    # PyTorch の gather は現在の CUDA ストリーム上で動き、追加同期を発生させない
    gather_indices = plan.packed_indices[:, None, :, None].expand_as(key)
    packed_key = torch.gather(key, dim=2, index=gather_indices)
    packed_value = torch.gather(value, dim=2, index=gather_indices)
    cudnn_attention = _load_cudnn_sdpa()
    return cudnn_attention(
        query,
        packed_key,
        packed_value,
        scale=1.0 / math.sqrt(query.shape[-1]),
        seq_len_q=plan.query_lengths,
        seq_len_kv=plan.key_lengths,
    )


@lru_cache(maxsize=1)
def _load_cudnn_sdpa() -> Any:
    """任意依存の cuDNN Frontend を初回利用時だけ読み込む。"""

    try:
        return import_module("cudnn.experimental.ops").scaled_dot_product_attention
    except ImportError as ex:
        raise RuntimeError(
            "Packed cuDNN attention requires the nvidia-cudnn-frontend package."
        ) from ex
