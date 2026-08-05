from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from datasets import Audio, Dataset

from prepare_manifest import _coerce_audio


def test_coerce_audio_uses_datasets_decode_sample_rate(tmp_path: Path) -> None:
    """
    Datasets の正規デコード経路が指定サンプルレートの channel-first 波形を返す。

    Args:
        tmp_path (Path): 入力音声を作成する一時ディレクトリ
    """

    source_sample_rate = 24000
    codec_sample_rate = 48000
    waveform = np.linspace(-0.5, 0.5, source_sample_rate, dtype=np.float32)
    audio_path = tmp_path / "sample.flac"
    sf.write(audio_path, waveform, source_sample_rate, subtype="PCM_16")
    dataset = Dataset.from_dict({"audio": [str(audio_path)]})
    dataset = dataset.cast_column("audio", Audio(sampling_rate=codec_sample_rate))

    decoded, decoded_sample_rate = _coerce_audio(dataset[0]["audio"])

    assert decoded_sample_rate == codec_sample_rate
    assert decoded.shape == (1, codec_sample_rate)
    assert decoded.dtype == torch.float32
