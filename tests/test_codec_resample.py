from unittest.mock import patch

import torch
import torchaudio

from irodori_tts.codec import DACVAECodec


class _IdentityCodec(torch.nn.Module):
    """
    Codec 前処理後の波形をそのまま返すテスト用モデル。
    """

    def encode(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        入力波形を変更せず返す。

        Args:
            waveform (torch.Tensor): Codec 前処理後の波形

        Returns:
            torch.Tensor: 入力と同じ波形
        """

        return waveform


def test_cached_resampler_matches_upstream_functional_path() -> None:
    """
    再利用する TorchAudio カーネルが upstream の関数呼び出しと同じ波形を返す。
    """

    sample_rate = 44100
    target_sample_rate = 48000
    waveform = torch.randn(
        (1, 1, sample_rate * 2),
        generator=torch.Generator().manual_seed(20260806),
    )
    codec = DACVAECodec(
        model=_IdentityCodec(),
        sample_rate=target_sample_rate,
        latent_dim=32,
        device=torch.device("cpu"),
        dtype=torch.float32,
        deterministic_encode=False,
        deterministic_decode=False,
        normalize_db=None,
    )

    # Identity を Codec として使い、リサンプル直後の波形をそのまま比較する
    with patch(
        "irodori_tts.codec.torchaudio.transforms.Resample",
        wraps=torchaudio.transforms.Resample,
    ) as resampler_factory:
        actual = codec.encode_waveform(waveform, sample_rate).transpose(1, 2)
        repeated = codec.encode_waveform(waveform, sample_rate).transpose(1, 2)

    assert resampler_factory.call_count == 1
    expected = torchaudio.functional.resample(waveform, sample_rate, target_sample_rate)

    torch.testing.assert_close(actual, expected, rtol=0.0, atol=1e-5)

    # 同じ入力条件の2回目も、数値を変えずに既存カーネルを再利用する
    torch.testing.assert_close(repeated, expected, rtol=0.0, atol=1e-5)


def test_resampler_cache_evicts_least_recently_used_entry() -> None:
    """
    リサンプラーの上限超過時に最も古い入力条件を作り直す。
    """

    codec = DACVAECodec(
        model=_IdentityCodec(),
        sample_rate=48000,
        latent_dim=32,
        device=torch.device("cpu"),
        dtype=torch.float32,
        deterministic_encode=False,
        deterministic_decode=False,
        normalize_db=None,
    )
    waveform = torch.zeros((1, 1, 16), dtype=torch.float32)

    with patch(
        "irodori_tts.codec.torchaudio.transforms.Resample",
        return_value=torch.nn.Identity(),
    ) as resampler_factory:
        for sample_rate in range(32000, 32033):
            codec.encode_waveform(waveform, sample_rate)
        codec.encode_waveform(waveform, 32032)
        codec.encode_waveform(waveform, 32000)

    assert resampler_factory.call_count == 34
