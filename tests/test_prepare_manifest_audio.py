import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch
from datasets import Audio, Dataset

from prepare_manifest import _coerce_audio


@pytest.mark.parametrize("extension", ("wav", "flac", "m4a"))
def test_coerce_audio_preserves_datasets_source_sample_rate(
    tmp_path: Path,
    extension: str,
) -> None:
    """
    Datasets の正規デコード経路が元サンプルレートの channel-first 波形を返す。

    Args:
        tmp_path (Path): 入力音声を作成する一時ディレクトリ
        extension (str): 検証する音声コンテナの拡張子
    """

    source_sample_rate = 24000
    waveform = np.linspace(-0.5, 0.5, source_sample_rate, dtype=np.float32)
    audio_path = tmp_path / f"sample.{extension}"
    if extension == "m4a":
        if shutil.which("ffmpeg") is None:
            pytest.skip("ffmpeg is required to create the M4A fixture")
        source_path = tmp_path / "m4a_source.wav"
        sf.write(source_path, waveform, source_sample_rate, subtype="PCM_16")
        subprocess.run(
            ["ffmpeg", "-v", "error", "-i", str(source_path), "-c:a", "aac", str(audio_path)],
            check=True,
        )
    else:
        sf.write(audio_path, waveform, source_sample_rate, subtype="PCM_16")
    dataset = Dataset.from_dict({"audio": [str(audio_path)]})
    dataset = dataset.cast_column("audio", Audio())

    decoded, decoded_sample_rate = _coerce_audio(dataset[0]["audio"])

    assert decoded_sample_rate == source_sample_rate
    assert decoded.shape[0] == 1
    if extension == "m4a":
        assert decoded.shape[1] >= source_sample_rate
    else:
        assert decoded.shape[1] == source_sample_rate
    assert decoded.dtype == torch.float32


def test_coerce_audio_honors_explicit_datasets_target_sample_rate(tmp_path: Path) -> None:
    """
    明示した `--target-sample-rate` 相当の変換結果をそのまま受け取る。

    Args:
        tmp_path (Path): 入力音声を作成する一時ディレクトリ
    """

    source_sample_rate = 24000
    target_sample_rate = 48000
    waveform = np.linspace(-0.5, 0.5, source_sample_rate, dtype=np.float32)
    audio_path = tmp_path / "sample.flac"
    sf.write(audio_path, waveform, source_sample_rate, subtype="PCM_16")
    dataset = Dataset.from_dict({"audio": [str(audio_path)]})
    dataset = dataset.cast_column("audio", Audio(sampling_rate=target_sample_rate))

    decoded, decoded_sample_rate = _coerce_audio(dataset[0]["audio"])

    assert decoded_sample_rate == target_sample_rate
    assert decoded.shape == (1, target_sample_rate)
