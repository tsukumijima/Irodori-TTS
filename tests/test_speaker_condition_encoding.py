from __future__ import annotations

import threading
from collections import OrderedDict
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch

import irodori_tts.inference_runtime as inference_runtime_module
from irodori_tts.inference_runtime import InferenceRuntime, SamplingRequest, SpeakerCondition


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

    def encode_speaker_condition_pre_norm(
        self,
        *,
        ref_latent: torch.Tensor,
        ref_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return fixed pre-normalization tokens while recording the reference latent."""

        self.ref_latent = ref_latent
        return torch.full((1, 3, 4), 2.0), ref_mask[:, :3]

    def compose_speaker_condition_pre_norm(
        self,
        *,
        state: torch.Tensor,
        mask: torch.Tensor,
        batch_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the test pre-normalization tokens with a synthetic mean token."""

        state = state.to(device=device, dtype=dtype).expand(batch_size, -1, -1)
        mask = mask.to(device=device).expand(batch_size, -1)
        return (
            torch.cat([state.mean(dim=1, keepdim=True), state], dim=1),
            torch.cat(
                [torch.ones((batch_size, 1), dtype=torch.bool, device=device), mask],
                dim=1,
            ),
        )


class RecordingReferenceCodec:
    """参照音声の符号化前サンプル数を記録するテスト用 codec。"""

    def __init__(self) -> None:
        self.sample_rate = 10
        self.model = SimpleNamespace(hop_length=2)
        self.input_lengths: list[int] = []

    def encode_waveform(self, waveform: torch.Tensor, **_kwargs: object) -> torch.Tensor:
        """
        入力サンプル数に対応する固定幅 latent を返す。

        Args:
            waveform (torch.Tensor): 符号化前の波形
            **_kwargs (object): 実 codec と同じ呼び出しを受ける未使用の追加引数

        Returns:
            torch.Tensor: ホップ長2で換算したゼロ埋め latent
        """

        sample_count = int(waveform.shape[-1])
        self.input_lengths.append(sample_count)
        latent_steps = (sample_count + 1) // 2
        return torch.zeros((1, latent_steps, 1), dtype=torch.float32)


@pytest.mark.parametrize("speaker_patch_size", [1, 4])
def test_no_ref_dummy_preserves_minimum_length_after_speaker_patching(
    speaker_patch_size: int,
) -> None:
    runtime = cast(Any, InferenceRuntime.__new__(InferenceRuntime))
    runtime.model = torch.nn.Linear(1, 1)
    runtime.model_cfg = SimpleNamespace(
        use_speaker_condition_resolved=True,
        speaker_patch_size=speaker_patch_size,
        latent_dim=32,
        latent_patch_size=1,
    )
    runtime.model_device = torch.device("cpu")
    runtime.default_max_ref_seconds = 120.0

    latent, mask = runtime._load_reference_latent(
        req=SamplingRequest(text="", no_ref=True),
        lora_adapter=None,
        batch_size=2,
        messages=[],
    )

    assert latent is not None
    assert mask is not None
    assert latent.shape == (2, 4 * speaker_patch_size, 32)
    assert mask.shape == (2, 4 * speaker_patch_size)
    assert not mask.any().item()


def test_encode_speaker_condition_returns_reusable_condition() -> None:
    """単一参照の変換 API は speaker_state を返し、大きい参照 latent を残さない。"""

    runtime = cast(Any, InferenceRuntime.__new__(InferenceRuntime))
    runtime.model_cfg = SimpleNamespace(use_speaker_condition_resolved=True)
    runtime.model_device = torch.device("cpu")
    runtime._model_dtype = torch.float32
    runtime._infer_lock = threading.Lock()
    unrelated_cache_value = object()
    runtime._reference_condition_cache = OrderedDict(
        (
            ("cache-key", object()),
            ("unrelated-key", unrelated_cache_value),
        )
    )
    runtime.model = RecordingSpeakerModel()
    runtime._resolve_lora_adapter_path = lambda _path: None
    runtime._reference_cache_key = lambda _request, lora_adapter: "cache-key"
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
    assert runtime._reference_condition_cache == OrderedDict(
        (("unrelated-key", unrelated_cache_value),)
    )


def test_multiple_reference_waveforms_trim_each_clip_to_remaining_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    複数参照の後続クリップが残り潜在長へ合わせて切り詰められることを検証する。

    Args:
        monkeypatch (pytest.MonkeyPatch): 参照音声読込を固定波形へ置き換えるフィクスチャ
    """

    runtime = cast(Any, InferenceRuntime.__new__(InferenceRuntime))
    runtime.model = torch.nn.Linear(1, 1)
    runtime.model_cfg = SimpleNamespace(
        use_speaker_condition_resolved=True,
        latent_dim=1,
        latent_patch_size=1,
        speaker_patch_size=1,
    )
    runtime.model_device = torch.device("cpu")
    runtime.default_max_ref_seconds = 1.0
    runtime.codec = RecordingReferenceCodec()
    runtime._reference_cache_key = lambda _request, lora_adapter: None
    runtime._reference_condition_cache = OrderedDict()

    waveforms = {
        "first.wav": (torch.zeros((1, 6)), 10),
        "second.wav": (torch.zeros((1, 10)), 10),
        "third.wav": (torch.zeros((1, 10)), 10),
    }
    monkeypatch.setattr(
        inference_runtime_module,
        "load_audio",
        lambda path: waveforms[path],
    )

    messages: list[str] = []
    latent, mask = runtime._load_reference_latent(
        req=SamplingRequest(
            text="",
            ref_wavs=["first.wav", "second.wav", "third.wav"],
        ),
        lora_adapter=None,
        batch_size=1,
        messages=messages,
    )

    assert latent is not None
    assert mask is not None
    assert runtime.codec.input_lengths == [6, 4]
    assert latent.shape == (1, 5, 1)
    assert any("1 skipped after reaching the reference limit" in message for message in messages)


def test_no_ref_request_uses_inherited_speaker_condition_for_cfg(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    参照なし長文から引き継いだ話者状態を後続チャンクの CFG へ反映する。

    Args:
        monkeypatch (pytest.MonkeyPatch): CFG 解決時の話者条件判定を記録するフィクスチャ
    """

    runtime = cast(Any, InferenceRuntime.__new__(InferenceRuntime))
    runtime.key = SimpleNamespace(
        model_device="cpu",
        model_precision="fp32",
        codec_device="cpu",
        codec_precision="fp32",
    )
    runtime.watermarker = None
    runtime.default_text_max_len = 16
    runtime.default_caption_max_len = 16
    runtime.model_cfg = SimpleNamespace(
        use_caption_condition=False,
        use_speaker_condition_resolved=True,
    )
    runtime._resolve_lora_adapter_path = lambda _value: None
    captured: dict[str, bool] = {}

    def record_speaker_condition(**kwargs: Any) -> tuple[float, float, float, list[str]]:
        captured["use_speaker_condition"] = bool(kwargs["use_speaker_condition"])
        raise RuntimeError("speaker condition captured")

    monkeypatch.setattr(
        inference_runtime_module,
        "resolve_cfg_scales",
        record_speaker_condition,
    )
    request = SamplingRequest(
        text="テスト",
        no_ref=True,
        speaker_condition_override=SpeakerCondition(
            state=torch.ones((1, 3, 4)),
            mask=torch.ones((1, 3), dtype=torch.bool),
        ),
    )

    with pytest.raises(RuntimeError, match="speaker condition captured"):
        runtime.synthesize(request)

    assert captured["use_speaker_condition"] is True
