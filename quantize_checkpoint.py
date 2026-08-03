#!/usr/bin/env python3
from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from irodori_tts import load_checkpoint_for_inference
from irodori_tts.checkpoint_export import CheckpointPublisher
from irodori_tts.config import ModelConfig, merge_dataclass_overrides
from irodori_tts.model import TextToLatentRFDiT
from irodori_tts.quantization import (
    DEFAULT_INT4_GROUP_SIZE,
    INT4_CUDA_PACKING_FORMAT,
    INT4_GROUP_SIZES,
    INT4_XPU_PACKING_FORMAT,
    QUANTIZATION_CLI_CHOICES,
    QUANTIZATION_PROFILES,
    flatten_quantized_state_dict,
    normalize_quantization_type,
    parse_quantization_metadata,
    quantization_cli_name,
    quantize_model,
    validate_quantization_device,
)


def _default_output_path(input_path: Path, quantization: str) -> Path:
    suffix = quantization_cli_name(quantization)
    return input_path.with_name(f"{input_path.stem}.{suffix}.safetensors")


def _resolve_device(raw: str) -> torch.device:
    normalized = str(raw).strip().lower()
    if normalized == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        try:
            if torch.xpu.is_available():
                return torch.device("xpu")
        except AttributeError:
            pass
        return torch.device("cpu")
    normalized_type = normalized.split(":", maxsplit=1)[0]
    if normalized_type not in {"cpu", "cuda", "xpu"}:
        raise ValueError("Quantization device must be one of: auto, cpu, cuda, xpu.")
    device = torch.device(normalized)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but torch.cuda.is_available() is False.")
    if device.type == "xpu":
        try:
            is_xpu_available = torch.xpu.is_available()
        except AttributeError as ex:
            raise ValueError("XPU was requested but torch.xpu is unavailable.") from ex
        if not is_xpu_available:
            raise ValueError("XPU was requested but torch.xpu.is_available() is False.")
    return device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert an Irodori inference safetensors checkpoint with torchao."
    )
    parser.add_argument("input_checkpoint", help="Source Irodori .safetensors checkpoint.")
    parser.add_argument(
        "--output",
        default=None,
        help="Output .safetensors path (default: <input>.<quantization>.safetensors).",
    )
    parser.add_argument(
        "--quantization",
        choices=QUANTIZATION_CLI_CHOICES,
        default="int8-weight-only",
        help="Weight/activation quantization scheme (default: int8-weight-only).",
    )
    parser.add_argument(
        "--profile",
        choices=QUANTIZATION_PROFILES,
        default="core",
        help=(
            "'core' quantizes transformer attention/MLP weights while preserving projectors, "
            "AdaLN, and duration prediction in BF16; 'all-linear' quantizes every compatible "
            "Linear."
        ),
    )
    parser.add_argument(
        "--int4-group-size",
        type=int,
        choices=INT4_GROUP_SIZES,
        default=DEFAULT_INT4_GROUP_SIZE,
        help="Group size for int4-weight-only (default: 128).",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Temporary conversion device: auto, cpu, cuda, or xpu (default: auto).",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite an existing output file.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = _resolve_device(args.device)
    quantization_type = normalize_quantization_type(args.quantization)
    validate_quantization_device(quantization_type, target_device=device)
    input_path = Path(args.input_checkpoint).expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"Input checkpoint not found: {input_path}")
    if input_path.suffix.lower() != ".safetensors":
        raise ValueError(
            "Quantization requires an inference .safetensors checkpoint. Convert training or "
            "LoRA checkpoints with convert_checkpoint_to_safetensors.py first."
        )

    output_path = (
        Path(args.output).expanduser().resolve()
        if args.output is not None
        else _default_output_path(input_path, args.quantization)
    )
    if output_path.suffix.lower() != ".safetensors":
        raise ValueError(f"Output must use .safetensors suffix: {output_path}")
    if input_path == output_path:
        raise ValueError("Input and output paths must be different.")
    if output_path.exists() and not bool(args.force):
        raise FileExistsError(f"Output already exists: {output_path} (use --force to overwrite)")

    with safe_open(str(input_path), framework="pt", device="cpu") as handle:
        source_metadata = handle.metadata() or {}
    if parse_quantization_metadata(source_metadata) is not None:
        raise ValueError(f"Input checkpoint is already quantized: {input_path}")

    model_state, model_cfg_dict, _, text_encoder_config = load_checkpoint_for_inference(input_path)
    model_cfg = merge_dataclass_overrides(
        ModelConfig(),
        model_cfg_dict,
        section="checkpoint model_config",
    )
    model = TextToLatentRFDiT(
        model_cfg,
        pretrained_backbone_config=text_encoder_config,
        load_pretrained_backbone_weights=not model_cfg.use_pretrained_text_encoder,
    )
    model.load_state_dict(model_state, assign=True)
    del model_state

    compute_dtype = torch.bfloat16
    model = model.eval().to(device=device, dtype=compute_dtype)
    quantized_modules = quantize_model(
        model,
        quantization_type=quantization_type,
        profile=args.profile,
        int4_group_size=args.int4_group_size,
        target_device=device,
    )
    model = model.to(device="cpu")

    flattened_state, metadata = flatten_quantized_state_dict(
        model.state_dict(),
        base_metadata=source_metadata,
        quantization_type=quantization_type,
        profile=args.profile,
        compute_dtype=compute_dtype,
        quantized_modules=len(quantized_modules),
        int4_group_size=args.int4_group_size,
        int4_packing_format={
            "cuda": INT4_CUDA_PACKING_FORMAT,
            "xpu": INT4_XPU_PACKING_FORMAT,
        }.get(device.type),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # モデルと tokenizer を一時領域で準備し、両方の成功後に公開する
    with tempfile.TemporaryDirectory(
        prefix=f".{output_path.stem}.quantize-",
        dir=output_path.parent,
    ) as temporary_dir_name:
        temporary_dir = Path(temporary_dir_name)
        temporary_output_path = temporary_dir / output_path.name
        temporary_tokenizer_dir = CheckpointPublisher.copy_bundled_tokenizer(
            input_path,
            temporary_dir / "tokenizer",
        )
        save_file(flattened_state, str(temporary_output_path), metadata=metadata)
        tokenizer_dir = CheckpointPublisher.publish(
            staged_checkpoint=temporary_output_path,
            output_checkpoint=output_path,
            staged_tokenizer=temporary_tokenizer_dir,
            temporary_directory=temporary_dir,
            force=bool(args.force),
        )

    tensor_bytes = sum(
        tensor.numel() * tensor.element_size() for tensor in flattened_state.values()
    )
    print(f"Input: {input_path}")
    print(f"Saved: {output_path}")
    print(f"Quantization: {args.quantization}")
    if args.quantization == "int4-weight-only":
        print(f"INT4 group size: {args.int4_group_size}")
    print(f"Profile: {args.profile}")
    print(f"Quantized Linear modules: {len(quantized_modules):,}")
    print(f"Stored tensor bytes: {tensor_bytes / (1024**3):.2f} GiB")
    if tokenizer_dir is not None:
        print(f"Tokenizer: {tokenizer_dir}")


if __name__ == "__main__":
    main()
