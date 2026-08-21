from unittest.mock import patch

import torch
import torchaudio

from irodori_tts.codec import DACVAECodec


class _IdentityCodec(torch.nn.Module):
    def encode(self, waveform: torch.Tensor) -> torch.Tensor:
        return waveform


def test_cached_resampler_matches_functional_resample_and_reuses_kernel() -> None:
    """
    Resample キャッシュが functional.resample と同じ波形を返し、同一条件ではカーネルを再利用する。
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

    with patch(
        "irodori_tts.codec.torchaudio.transforms.Resample",
        wraps=torchaudio.transforms.Resample,
    ) as resampler_factory:
        actual = codec.encode_waveform(waveform, sample_rate).transpose(1, 2)
        repeated = codec.encode_waveform(waveform, sample_rate).transpose(1, 2)

    assert resampler_factory.call_count == 1
    expected = torchaudio.functional.resample(waveform, sample_rate, target_sample_rate)
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=1e-5)
    torch.testing.assert_close(repeated, expected, rtol=0.0, atol=1e-5)
