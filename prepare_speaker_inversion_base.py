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
from irodori_tts.speaker_inversion import (
    SPEAKER_INVERSION_BASE_SAFETENSORS_SUFFIX,
    save_speaker_inversion_base_safetensors,
    speaker_inversion_checkpoint_sha256,
)


def main() -> None:
    """
    Extract pre-normalization speaker tokens from an ordinary reference.
    """

    # Keep checkpoint and reference selection unambiguous because both determine token identity.
    parser = argparse.ArgumentParser(
        description="Prepare a reference-derived Speaker Inversion base embedding."
    )
    checkpoint_group = parser.add_mutually_exclusive_group(required=True)
    checkpoint_group.add_argument(
        "--checkpoint",
        default=None,
        help="Local base checkpoint used to encode the reference.",
    )
    checkpoint_group.add_argument(
        "--hf-checkpoint",
        default=None,
        help="Hugging Face checkpoint ID or URL used to encode the reference.",
    )
    reference_group = parser.add_mutually_exclusive_group(required=True)
    reference_group.add_argument(
        "--ref-wav",
        default=None,
        help="Single reference waveform path.",
    )
    reference_group.add_argument(
        "--ref-wavs",
        nargs="+",
        default=None,
        metavar="PATH",
        help="Reference waveform paths concatenated in the given order.",
    )
    reference_group.add_argument(
        "--ref-latent",
        default=None,
        help="Single pre-encoded reference latent path.",
    )
    reference_group.add_argument(
        "--ref-latents",
        nargs="+",
        default=None,
        metavar="PATH",
        help="Pre-encoded reference latent paths concatenated in the given order.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output .speaker-base.safetensors path.",
    )
    parser.add_argument(
        "--max-ref-seconds",
        type=float,
        default=None,
        help="Maximum combined reference duration; omitted uses the checkpoint setting.",
    )
    parser.add_argument(
        "--expected-local-tokens",
        type=int,
        default=None,
        help="Reject the export unless the reference produces this many local tokens.",
    )
    parser.add_argument(
        "--ref-normalize-db",
        type=float,
        default=-16.0,
        help="Reference loudness target in dB; default: -16.0.",
    )
    parser.add_argument(
        "--model-device",
        default=default_runtime_device(),
        help="Device used by the text-to-latent model.",
    )
    parser.add_argument(
        "--model-precision",
        choices=["fp32", "bf16", "fp16"],
        default="fp32",
        help="Text-to-latent model precision; default: fp32.",
    )
    parser.add_argument(
        "--codec-device",
        default=default_runtime_device(),
        help="Device used by the reference audio codec.",
    )
    parser.add_argument(
        "--codec-precision",
        choices=["fp32", "bf16", "fp16"],
        default="fp32",
        help="Reference audio codec precision; default: fp32.",
    )
    parser.add_argument(
        "--codec-repo",
        default="Aratako/Semantic-DACVAE-Japanese-32dim",
        help="DACVAE repository or local path used to encode reference audio.",
    )
    args = parser.parse_args()
    output_path = Path(str(args.output)).expanduser()
    if not output_path.name.endswith(SPEAKER_INVERSION_BASE_SAFETENSORS_SUFFIX):
        raise ValueError(
            "Speaker Inversion base output must use the "
            f"{SPEAKER_INVERSION_BASE_SAFETENSORS_SUFFIX!r} suffix: {output_path}"
        )
    if args.expected_local_tokens is not None and int(args.expected_local_tokens) <= 0:
        raise ValueError(
            "--expected-local-tokens must be > 0 when provided, "
            f"got {int(args.expected_local_tokens)}."
        )

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
    if args.expected_local_tokens is not None and int(condition.state.shape[1]) != int(
        args.expected_local_tokens
    ):
        raise ValueError(
            "Reference length produced an unexpected number of local speaker tokens: "
            f"expected {int(args.expected_local_tokens)}, got {int(condition.state.shape[1])}."
        )
    # Persist only the fixed base because the learned residual belongs to training checkpoints.
    save_speaker_inversion_base_safetensors(
        output_path,
        condition.state,
        metadata={
            "checkpoint_sha256": speaker_inversion_checkpoint_sha256(checkpoint_path),
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
        f"normalized_condition_tokens={condition.condition_state.shape[1]}"
    )


if __name__ == "__main__":
    main()
