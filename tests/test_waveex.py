import pytest
import torch

from irodori_tts.waveex import WaveExBuffer, WaveExConfig


def test_direct_taylor_extrapolation_continues_linear_history() -> None:
    """
    既定の2状態バッファで線形な潜在軌跡を延長できることを確認する。
    """

    buffer = WaveExBuffer(WaveExConfig())
    buffer.push(torch.tensor([1.0, 3.0]))
    buffer.push(torch.tensor([2.0, 5.0]))

    torch.testing.assert_close(buffer.predict_next(), torch.tensor([3.0, 7.0]))


@pytest.mark.parametrize("wavelet", ["haar", "db2", "db4", "sym4", "sym6"])
def test_wavelet_extrapolation_preserves_constant_history(wavelet: str) -> None:
    """
    対応する各ウェーブレット基底で一定の潜在を維持できることを確認する。

    Args:
        wavelet (str): 検証するウェーブレット基底。
    """

    history_size = 12 if wavelet == "sym6" else 8
    buffer = WaveExBuffer(
        WaveExConfig(
            wavelet=wavelet,
            history_size=history_size,
        )
    )
    for _ in range(history_size):
        buffer.push(torch.tensor([2.0, -1.0]))

    torch.testing.assert_close(
        buffer.predict_next(),
        torch.tensor([2.0, -1.0]),
        rtol=1e-5,
        atol=1e-5,
    )
