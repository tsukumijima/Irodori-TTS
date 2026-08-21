import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch

from irodori_tts.inference_runtime import load_audio, save_wav


def test_load_audio_preserves_channel_first_float32_contract(tmp_path: Path) -> None:
    sample_rate = 44100
    waveform = np.stack(
        [
            np.linspace(-0.5, 0.5, sample_rate, dtype=np.float32),
            np.linspace(0.25, -0.25, sample_rate, dtype=np.float32),
        ],
        axis=1,
    )
    audio_path = tmp_path / "stereo.flac"
    sf.write(audio_path, waveform, sample_rate, subtype="PCM_24")

    loaded, loaded_sample_rate = load_audio(audio_path)
    expected, _ = sf.read(audio_path, dtype="float32", always_2d=True)

    assert loaded_sample_rate == sample_rate
    assert loaded.dtype == torch.float32
    assert loaded.shape == (2, sample_rate)
    torch.testing.assert_close(loaded, torch.from_numpy(expected.T.copy()), rtol=0.0, atol=0.0)


def test_load_audio_decodes_m4a_without_changing_sample_rate(tmp_path: Path) -> None:
    """
    torchaudio.load() 経由で M4A をデコードできる。

    Args:
        tmp_path (Path): 入出力音声を作成する一時ディレクトリ
    """

    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg is required to create the M4A fixture")

    sample_rate = 44100
    waveform = np.linspace(-0.25, 0.25, sample_rate, dtype=np.float32)
    wav_path = tmp_path / "source.wav"
    m4a_path = tmp_path / "source.m4a"
    sf.write(wav_path, waveform, sample_rate, subtype="PCM_16")
    subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(wav_path), "-c:a", "aac", str(m4a_path)],
        check=True,
    )

    loaded, loaded_sample_rate = load_audio(m4a_path)

    assert loaded_sample_rate == sample_rate
    assert loaded.dtype == torch.float32
    assert loaded.shape[0] == 1
    assert loaded.shape[1] >= sample_rate


def test_save_wav_preserves_previous_default_subtypes(tmp_path: Path) -> None:
    sample_rate = 48000
    waveform = torch.linspace(-0.5, 0.5, sample_rate, dtype=torch.float32).unsqueeze(0)
    wav_path = tmp_path / "output.wav"
    flac_path = tmp_path / "output.flac"

    save_wav(wav_path, waveform, sample_rate)
    save_wav(flac_path, waveform, sample_rate)

    assert sf.info(wav_path).subtype == "PCM_16"
    assert sf.info(flac_path).subtype == "PCM_24"


def test_save_wav_rejects_unsupported_output_extension(tmp_path: Path) -> None:
    waveform = torch.zeros((1, 480), dtype=torch.float32)

    with pytest.raises(ValueError, match=r"expected \.flac or \.wav"):
        save_wav(tmp_path / "output.ogg", waveform, 48000)
