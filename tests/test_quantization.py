from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file

from irodori_tts.quantization import (
    INT4_CUDA_PACKING_FORMAT,
    QUANTIZATION_TYPE_INT4_WEIGHT_ONLY,
    QUANTIZATION_TYPE_INT8_WEIGHT_ONLY,
    flatten_quantized_state_dict,
    is_torchao_quantized_state_dict,
    quantize_model,
    unflatten_quantized_state_dict,
)


class LinearModel(torch.nn.Module):
    """
    torchao の直列化試験に使う全結合層を提供する。
    """

    def __init__(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.projection = torch.nn.Linear(128, 128, device=device, dtype=dtype)


def _round_trip_quantized_state(
    tmp_path: Path,
    *,
    model: LinearModel,
    quantization_type: str,
    packing_format: str | None = None,
) -> None:
    """
    1層の state dictionary を量子化して保存し、復元する。

    Args:
        tmp_path (Path): safetensors の保存先ディレクトリ。
        model (LinearModel): 量子化対象デバイス上のモデル。
        quantization_type (str): 検証する torchao の量子化形式。
        packing_format (str | None): デバイス固有の INT4 パッキング形式。
    """

    quantized_modules = quantize_model(
        model,
        quantization_type=quantization_type,
        profile="all-linear",
        target_device=next(model.parameters()).device,
    )
    flattened, metadata = flatten_quantized_state_dict(
        model.state_dict(),
        base_metadata={},
        quantization_type=quantization_type,
        profile="all-linear",
        compute_dtype=next(model.parameters()).dtype,
        quantized_modules=len(quantized_modules),
        int4_packing_format=packing_format,
    )
    checkpoint_path = tmp_path / "quantized.safetensors"
    save_file(flattened, checkpoint_path, metadata=metadata)

    restored, payload = unflatten_quantized_state_dict(
        load_file(checkpoint_path),
        metadata=metadata,
    )

    assert payload["quantization_type"] == quantization_type
    assert is_torchao_quantized_state_dict(restored) is True


def test_int8_quantized_state_round_trip(tmp_path: Path) -> None:
    """
    CPU の INT8 テンソルをチェックポイント経由で復元できることを確認する。
    """

    _round_trip_quantized_state(
        tmp_path,
        model=LinearModel(device=torch.device("cpu")),
        quantization_type=QUANTIZATION_TYPE_INT8_WEIGHT_ONLY,
    )


@pytest.mark.skipif(torch.cuda.is_available() is False, reason="CUDA is required for INT4 packing")
def test_cuda_int4_quantized_state_round_trip(tmp_path: Path) -> None:
    """
    CUDA の INT4 テンソルを指定のタイル形式で復元できることを確認する。
    """

    _round_trip_quantized_state(
        tmp_path,
        model=LinearModel(device=torch.device("cuda"), dtype=torch.bfloat16),
        quantization_type=QUANTIZATION_TYPE_INT4_WEIGHT_ONLY,
        packing_format=INT4_CUDA_PACKING_FORMAT,
    )
