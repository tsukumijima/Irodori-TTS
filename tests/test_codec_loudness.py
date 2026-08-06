import pytest
import torch

from irodori_tts.codec import DACVAECodec


@pytest.mark.parametrize(
    ("sample_rate", "expected_sine", "expected_noise"),
    (
        (16000, -29.75902557373047, -37.32304000854492),
        (32000, -29.75699234008789, -36.99609375),
        (44100, -29.756685256958008, -36.891578674316406),
        (48000, -29.756589889526367, -36.87405776977539),
    ),
)
def test_measure_loudness_matches_audiotools_reference_values(
    sample_rate: int,
    expected_sine: float,
    expected_noise: float,
) -> None:
    """AudioTools 0.7.2 で事前計測した BS.1770 値との数値差を監視する。

    各サンプルレートで1秒の 440Hz 正弦波と seed 1 の白色雑音を作り、
    `AudioSignal.loudness()` へ渡した値を参照値として固定する。
    """

    time = torch.arange(sample_rate, dtype=torch.float32) / sample_rate
    sine = 0.05 * torch.sin(2.0 * torch.pi * 440.0 * time)
    noise = (
        torch.randn(
            sample_rate,
            generator=torch.Generator().manual_seed(1),
            dtype=torch.float32,
        )
        * 0.01
    )

    torch.testing.assert_close(
        DACVAECodec.measure_loudness(sine, sample_rate),
        torch.tensor(expected_sine),
        atol=1e-3,
        rtol=0.0,
    )
    torch.testing.assert_close(
        DACVAECodec.measure_loudness(noise, sample_rate),
        torch.tensor(expected_noise),
        atol=1e-3,
        rtol=0.0,
    )


def test_measure_loudness_returns_finite_silence_for_short_and_silent_audio() -> None:
    for waveform in (
        torch.zeros(100, dtype=torch.float32),
        torch.zeros(48000, dtype=torch.float32),
    ):
        measured_db = DACVAECodec.measure_loudness(waveform, 48000)

        torch.testing.assert_close(measured_db, torch.tensor(-70.0))


def test_normalize_loudness_reaches_target_without_peak_clipping() -> None:
    sample_rate = 48000
    time = torch.arange(sample_rate, dtype=torch.float32) / sample_rate
    waveform = 0.05 * torch.sin(2.0 * torch.pi * 440.0 * time)

    normalized = DACVAECodec._normalize_loudness(
        waveform,
        sample_rate=sample_rate,
        target_db=-30.0,
    )

    # 正規化後の統合ラウドネスが指定値へ到達していることを確認する
    measured_db = DACVAECodec.measure_loudness(
        normalized,
        sample_rate,
    )
    torch.testing.assert_close(measured_db, torch.tensor(-30.0), atol=1e-4, rtol=0.0)
    assert normalized.abs().max() < 1.0


def test_normalize_loudness_limits_peak_after_gain() -> None:
    sample_rate = 48000
    time = torch.arange(sample_rate, dtype=torch.float32) / sample_rate
    waveform = 0.9 * torch.sin(2.0 * torch.pi * 440.0 * time)

    normalized = DACVAECodec._normalize_loudness(
        waveform,
        sample_rate=sample_rate,
        target_db=0.0,
    )

    # 目標ラウドネスで増幅しても出力波形の絶対ピークを1.0へ収める
    torch.testing.assert_close(normalized.abs().max(), torch.tensor(1.0))


def test_normalize_loudness_matches_audiotools_for_intermediate_filter_peaks() -> None:
    """IIR の段間ピークが1.0を超える波形でも AudioTools 0.7.2 と一致させる。"""

    sample_rate = 48000
    generator = torch.Generator().manual_seed(20260806)
    waveform = torch.randn(sample_rate, generator=generator, dtype=torch.float32) * 0.35
    waveform = waveform.clamp(-1.0, 1.0)

    normalized = DACVAECodec._normalize_loudness(
        waveform,
        sample_rate=sample_rate,
        target_db=-16.0,
    )

    # AudioTools 0.7.2 の AudioSignal.normalize() と ensure_max_of_audio() の固定値
    torch.testing.assert_close(
        DACVAECodec.measure_loudness(waveform, sample_rate),
        torch.tensor(-6.046573638916016),
        atol=1e-4,
        rtol=0.0,
    )
    torch.testing.assert_close(
        normalized.abs().max(), torch.tensor(0.31792792677879333), atol=1e-6, rtol=0.0
    )


def test_normalize_loudness_preserves_audiotools_minimum_loudness_gain() -> None:
    """測定下限に達する微小波形にも AudioTools 0.7.2 と同じゲインを適用する。"""

    waveform = torch.full((48000,), 1e-8, dtype=torch.float32)

    normalized = DACVAECodec._normalize_loudness(
        waveform,
        sample_rate=48000,
        target_db=-16.0,
    )

    expected_gain = torch.exp(torch.tensor(54.0) * (torch.log(torch.tensor(10.0)) / 20.0))
    torch.testing.assert_close(normalized, waveform * expected_gain)
