from __future__ import annotations

import pytest
import torch

import irodori_tts.inference_runtime as inference_runtime


@pytest.mark.parametrize(
    ("device_name", "availability_probe_name", "expected_precisions"),
    [
        ("mps", "_is_mps_available", ["fp32"]),
        ("xpu", "_is_xpu_available", ["fp32", "bf16"]),
    ],
)
def test_runtime_preserves_upstream_optional_device_contracts(
    monkeypatch: pytest.MonkeyPatch,
    device_name: str,
    availability_probe_name: str,
    expected_precisions: list[str],
) -> None:
    """
    実機を必要としない範囲で upstream の任意デバイス契約を確認する。

    Args:
        monkeypatch (pytest.MonkeyPatch): デバイスの利用可否判定を置き換える fixture
        device_name (str): 検証する PyTorch デバイス名
        availability_probe_name (str): 置き換える利用可否判定関数名
        expected_precisions (list[str]): デバイスで選択可能な推論精度
    """

    # 実機の有無と公開 API のデバイス契約を分けて検証する
    monkeypatch.setattr(inference_runtime, availability_probe_name, lambda: True)

    resolved = inference_runtime.resolve_runtime_device(device_name)

    assert resolved == torch.device(device_name)
    assert device_name in inference_runtime.list_available_runtime_devices()
    assert inference_runtime.list_available_runtime_precisions(device_name) == expected_precisions
