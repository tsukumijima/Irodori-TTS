import torch

from irodori_tts.codec import DACVAECodec


def test_measure_loudness_matches_torchaudio_reference_values() -> None:
    """Torchaudio 2.9 で事前計測した BS.1770 値との数値差を監視する。"""

    # 参照音声で現れる主要なサンプルレートごとに、Torchaudio 2.9 の値を固定する
    for sample_rate, expected_sine, expected_noise in (
        (16000, -29.7590274810791, -37.323036193847656),
        (32000, -29.75702667236328, -36.99609375),
        (44100, -29.75666618347168, -36.891578674316406),
        (48000, -29.75687599182129, -36.87406921386719),
    ):
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
            DACVAECodec._measure_loudness(sine, sample_rate),
            torch.tensor(expected_sine),
            atol=1e-3,
            rtol=0.0,
        )
        torch.testing.assert_close(
            DACVAECodec._measure_loudness(noise, sample_rate),
            torch.tensor(expected_noise),
            atol=1e-3,
            rtol=0.0,
        )


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
    measured_db = DACVAECodec._measure_loudness(
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
