from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest
import torch

from irodori_tts.inference_runtime import InferenceRuntime, SamplingRequest


class RecordingSpeakerModel:
    """Speaker Encoder へ渡された参照状態を記録するテスト用モデル。"""

    def __init__(self) -> None:
        self.ref_latent: torch.Tensor | None = None

    def encode_speaker_condition(
        self,
        *,
        batch_size: int,
        dtype: torch.dtype,
        device: torch.device,
        ref_latent: torch.Tensor,
        ref_mask: torch.Tensor,
        speaker_uncond_mode: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        入力を記録し、固定形状の話者状態を返す。

        Args:
            batch_size (int): 返す話者状態のバッチ数
            dtype (torch.dtype): 返す話者状態のデータ型
            device (torch.device): 返す話者状態のデバイス
            ref_latent (torch.Tensor): 記録対象の参照 latent
            ref_mask (torch.Tensor): 未使用の参照マスク
            speaker_uncond_mode (str): 未使用の無条件話者状態の作成方式

        Returns:
            tuple[torch.Tensor, torch.Tensor]: 固定形状の話者状態と有効トークンマスク
        """

        del ref_mask, speaker_uncond_mode
        self.ref_latent = ref_latent
        return (
            torch.ones(batch_size, 3, 4, dtype=dtype, device=device),
            torch.ones(batch_size, 3, dtype=torch.bool, device=device),
        )


def test_encode_speaker_condition_returns_server_cache_value() -> None:
    """単一参照の変換 API は speaker_state を返し、大きい参照 latent を残さない。"""

    runtime = InferenceRuntime.__new__(InferenceRuntime)
    runtime.model_cfg = SimpleNamespace(use_speaker_condition_resolved=True)
    runtime.model_device = torch.device("cpu")
    runtime._model_dtype = torch.float32
    runtime._infer_lock = threading.Lock()
    runtime._reference_condition_cache = {"cache-key": object()}
    runtime.model = RecordingSpeakerModel()
    runtime._resolve_lora_adapter_path = lambda _path: None
    runtime._resolve_condition_tokens = lambda _request, lora_adapter: None
    runtime._reference_cache_key = (
        lambda _request, resolved_condition_tokens, lora_adapter: "cache-key"
    )
    runtime._load_reference_latent = lambda **_kwargs: (
        torch.zeros(1, 5, 6),
        torch.ones(1, 5, dtype=torch.bool),
    )

    condition = runtime.encode_speaker_condition(
        SamplingRequest(text="", ref_wav="reference.flac"),
    )

    assert condition.state.shape == (1, 3, 4)
    assert condition.mask.shape == (1, 3)
    assert runtime.model.ref_latent is not None
    assert runtime._reference_condition_cache == {}


@pytest.mark.parametrize(
    "sampling_request",
    [
        SamplingRequest(text="", no_ref=True),
        SamplingRequest(text=""),
    ],
)
def test_encode_speaker_condition_rejects_missing_source(
    sampling_request: SamplingRequest,
) -> None:
    """話者条件を持たない要求はモデルへ入る前に拒否する。"""

    runtime = InferenceRuntime.__new__(InferenceRuntime)
    runtime.model_cfg = SimpleNamespace(use_speaker_condition_resolved=True)

    with pytest.raises(ValueError):
        runtime.encode_speaker_condition(sampling_request)
