#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from irodori_tts.inference_runtime import (
    InferenceRuntime,
    RuntimeKey,
    SamplingRequest,
    default_runtime_device,
    download_hf_checkpoint,
)
from irodori_tts.speaker_inversion import save_speaker_inversion_base_safetensors


def main() -> None:
    """
    Extract pre-normalization speaker tokens from an ordinary reference.
    """

    # Keep checkpoint and reference selection unambiguous because both determine token identity.
    parser = argparse.ArgumentParser(
        description="Prepare a reference-derived Speaker Inversion base embedding."
    )
    checkpoint_group = parser.add_mutually_exclusive_group(required=True)
    checkpoint_group.add_argument("--checkpoint", default=None)
    checkpoint_group.add_argument("--hf-checkpoint", default=None)
    reference_group = parser.add_mutually_exclusive_group(required=True)
    reference_group.add_argument("--ref-wav", default=None)
    reference_group.add_argument("--ref-wavs", nargs="+", default=None, metavar="PATH")
    reference_group.add_argument("--ref-latent", default=None)
    reference_group.add_argument("--ref-latents", nargs="+", default=None, metavar="PATH")
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-ref-seconds", type=float, default=None)
    parser.add_argument("--ref-normalize-db", type=float, default=-16.0)
    parser.add_argument("--model-device", default=default_runtime_device())
    parser.add_argument("--model-precision", choices=["fp32", "bf16", "fp16"], default="fp32")
    parser.add_argument("--codec-device", default=default_runtime_device())
    parser.add_argument("--codec-precision", choices=["fp32", "bf16", "fp16"], default="fp32")
    parser.add_argument(
        "--codec-repo",
        default="Aratako/Semantic-DACVAE-Japanese-32dim",
    )
    args = parser.parse_args()

    # Resolve remote checkpoints before runtime construction so provenance has one local path.
    if args.checkpoint is not None:
        checkpoint_path = Path(str(args.checkpoint)).expanduser()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    else:
        checkpoint_path = Path(download_hf_checkpoint(str(args.hf_checkpoint)))

    # Use the public inference runtime so extraction matches ordinary zero-shot preprocessing.
    runtime = InferenceRuntime.from_key(
        RuntimeKey(
            checkpoint=str(checkpoint_path),
            model_device=str(args.model_device),
            codec_repo=str(args.codec_repo),
            model_precision=str(args.model_precision),
            codec_device=str(args.codec_device),
            codec_precision=str(args.codec_precision),
        )
    )
    # Export local tokens before speaker normalization; training recomputes the mean token.
    condition = runtime.encode_speaker_inversion_base(
        SamplingRequest(
            text="",
            ref_wav=args.ref_wav,
            ref_wavs=args.ref_wavs,
            ref_latent=args.ref_latent,
            ref_latents=args.ref_latents,
            max_ref_seconds=(None if args.max_ref_seconds is None else float(args.max_ref_seconds)),
            ref_normalize_db=float(args.ref_normalize_db),
        ),
        log_fn=print,
    )
    # Persist only the fixed base because the learned residual belongs to training checkpoints.
    output_path = Path(str(args.output)).expanduser()
    save_speaker_inversion_base_safetensors(
        output_path,
        condition.state,
        metadata={
            "checkpoint": str(checkpoint_path.resolve()),
            "speaker_patch_size": str(int(runtime.model_cfg.speaker_patch_size)),
            "codec_repo": str(args.codec_repo),
            "max_ref_seconds": (
                "checkpoint_default"
                if args.max_ref_seconds is None
                else str(float(args.max_ref_seconds))
            ),
        },
    )
    print(
        f"Saved Speaker Inversion base: {output_path} "
        f"local_tokens={condition.state.shape[1]} "
        f"exported_tokens={condition.condition_state.shape[1]}"
    )


if __name__ == "__main__":
    main()
