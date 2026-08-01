from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
import soxr
import torch
from huggingface_hub import hf_hub_download
from scipy.signal import lfilter


_CODEC_DEFAULT = object()


def patchify_latent(latent: torch.Tensor, patch_size: int) -> torch.Tensor:
    """
    Convert latent from (B, T, D) -> (B, T//patch, D*patch).
    Extra tail tokens are dropped.
    """
    if patch_size <= 1:
        return latent
    bsz, seq_len, dim = latent.shape
    usable = (seq_len // patch_size) * patch_size
    latent = latent[:, :usable]
    latent = latent.reshape(bsz, usable // patch_size, dim * patch_size)
    return latent


def unpatchify_latent(patched: torch.Tensor, patch_size: int, latent_dim: int) -> torch.Tensor:
    """
    Convert latent from (B, T_p, D*patch) -> (B, T_p*patch, D).
    """
    if patch_size <= 1:
        return patched
    return patched.reshape(patched.shape[0], patched.shape[1] * patch_size, latent_dim)


@dataclass
class DACVAECodec:
    model: torch.nn.Module
    sample_rate: int
    latent_dim: int
    device: torch.device
    dtype: torch.dtype
    deterministic_encode: bool
    deterministic_decode: bool
    normalize_db: float | None

    @classmethod
    def load(
        cls,
        repo_id: str = "Aratako/Semantic-DACVAE-Japanese-32dim",
        device: str = "cuda",
        dtype: torch.dtype | None = None,
        deterministic_encode: bool = True,
        deterministic_decode: bool = True,
        normalize_db: float | None = -16.0,
    ) -> DACVAECodec:
        # Prefer installed package; fallback to local clone at ../dacvae.
        try:
            from dacvae import DACVAE
        except ImportError:
            local_repo = Path(__file__).resolve().parents[2] / "dacvae"
            if local_repo.exists():
                sys.path.insert(0, str(local_repo))
            from dacvae import DACVAE

        location = str(repo_id).strip()
        if location.startswith("hf://"):
            location = location[len("hf://") :]
        if not Path(location).exists() and "/" in location and not location.endswith(".pth"):
            try:
                location = hf_hub_download(repo_id=location, filename="weights.pth")
                print(f"[codec] dacvae: hf://{repo_id} -> {location}", flush=True)
            except Exception:
                # Let DACVAE.load surface a clearer error if this is not a valid path/repo.
                pass

        # GPU 上で FP32 の VAE を一度展開するとロード直後の VRAM が膨らむため、
        # CPU 上で構築したモデルを目的 dtype のまま GPU へ移す
        model = DACVAE.load(location).eval()
        if dtype is not None:
            model = model.to(device=device, dtype=dtype)
        else:
            model = model.to(device=device)

        decoder = getattr(model, "decoder", None)
        if decoder is not None and hasattr(decoder, "alpha"):
            decoder.alpha = 0.0
            if hasattr(decoder, "wm_model"):
                # Irodori checkpoints were trained without the DACVAE watermark branch.
                # Keep decode output mono while skipping that encode/decode path.
                def _watermark_passthrough(
                    x: torch.Tensor,
                    message: torch.Tensor | None = None,
                    _decoder=decoder,
                ) -> torch.Tensor:
                    del message
                    return _decoder.wm_model.encoder_block.forward_no_conv(x)

                decoder.watermark = _watermark_passthrough

        if deterministic_decode:
            cls._configure_deterministic_decode(model=model, device=device)

        model_dtype = next(model.parameters()).dtype
        # Infer latent dimension by encoding a tiny random signal.
        dummy = torch.zeros(1, 1, 2048, device=device, dtype=model_dtype)
        with torch.inference_mode():
            z = model.encode(dummy)  # (B, D, T)
        return cls(
            model=model,
            sample_rate=int(model.sample_rate),
            latent_dim=int(z.shape[1]),
            device=torch.device(device),
            dtype=model_dtype,
            deterministic_encode=bool(deterministic_encode),
            deterministic_decode=bool(deterministic_decode),
            normalize_db=None if normalize_db is None else float(normalize_db),
        )

    @staticmethod
    def _configure_deterministic_decode(model: torch.nn.Module, device: str | torch.device) -> None:
        decoder = getattr(model, "decoder", None)
        wm_model = getattr(decoder, "wm_model", None)
        msg_processor = getattr(wm_model, "msg_processor", None)
        if wm_model is None or msg_processor is None:
            return
        nbits = int(msg_processor.nbits)
        message_device = torch.device(device)

        def _fixed_message(batch_size: int) -> torch.Tensor:
            return torch.zeros((batch_size, nbits), dtype=torch.float32, device=message_device)

        wm_model.random_message = _fixed_message

    @staticmethod
    def _measure_loudness(wav: torch.Tensor, sample_rate: int) -> torch.Tensor:
        """
        ITU-R BS.1770-4 の統合ラウドネスを CPU で測定する。
        """

        waveform = wav.detach().to(device="cpu", dtype=torch.float32).numpy()
        if waveform.ndim != 1:
            raise ValueError(
                f"measure_loudness expects a mono waveform, got {tuple(waveform.shape)}"
            )

        def apply_biquad(
            audio: np.ndarray,
            numerator: tuple[np.float32, np.float32, np.float32],
            denominator: tuple[np.float32, np.float32, np.float32],
        ) -> np.ndarray:
            """
            Torchaudio と同じゼロ初期状態と段間 clamp で2次 IIR を適用する。
            """

            filtered = lfilter(
                np.asarray(numerator, dtype=np.float32),
                np.asarray(denominator, dtype=np.float32),
                audio,
            )
            return np.clip(filtered, -1.0, 1.0).astype(np.float32, copy=False)

        # Torchaudio の BS.1770 実装と同じ RBJ high-shelf 係数を float32 で生成する
        high_shelf_frequency = np.float32(1500.0)
        high_shelf_q = np.float32(1.0 / np.sqrt(2.0))
        high_shelf_gain = np.float32(4.0)
        angular_frequency = np.float32(2.0 * np.pi) * high_shelf_frequency / np.float32(sample_rate)
        alpha = np.sin(angular_frequency, dtype=np.float32) / np.float32(2.0) / high_shelf_q
        amplitude = np.exp(
            high_shelf_gain / np.float32(40.0) * np.float32(np.log(10.0)),
            dtype=np.float32,
        )
        temporary_1 = np.float32(2.0) * np.sqrt(amplitude, dtype=np.float32) * alpha
        temporary_2 = (amplitude - np.float32(1.0)) * np.cos(
            angular_frequency,
            dtype=np.float32,
        )
        temporary_3 = (amplitude + np.float32(1.0)) * np.cos(
            angular_frequency,
            dtype=np.float32,
        )
        waveform = apply_biquad(
            waveform,
            (
                amplitude * ((amplitude + np.float32(1.0)) + temporary_2 + temporary_1),
                np.float32(-2.0) * amplitude * ((amplitude - np.float32(1.0)) + temporary_3),
                amplitude * ((amplitude + np.float32(1.0)) + temporary_2 - temporary_1),
            ),
            (
                (amplitude + np.float32(1.0)) - temporary_2 + temporary_1,
                np.float32(2.0) * ((amplitude - np.float32(1.0)) - temporary_3),
                (amplitude + np.float32(1.0)) - temporary_2 - temporary_1,
            ),
        )

        # 続く38Hz high-pass も同じ係数と float32 演算で適用する
        high_pass_frequency = np.float32(38.0)
        high_pass_q = np.float32(0.5)
        angular_frequency = np.float32(2.0 * np.pi) * high_pass_frequency / np.float32(sample_rate)
        cosine = np.cos(angular_frequency, dtype=np.float32)
        alpha = np.sin(angular_frequency, dtype=np.float32) / np.float32(2.0) / high_pass_q
        waveform = apply_biquad(
            waveform,
            (
                (np.float32(1.0) + cosine) / np.float32(2.0),
                -np.float32(1.0) - cosine,
                (np.float32(1.0) + cosine) / np.float32(2.0),
            ),
            (
                np.float32(1.0) + alpha,
                np.float32(-2.0) * cosine,
                np.float32(1.0) - alpha,
            ),
        )

        # 400ms窓を75%重複させ、絶対ゲートと相対ゲートを順に適用する
        gate_samples = round(0.4 * sample_rate)
        step_samples = round(gate_samples * 0.25)
        block_starts = range(0, waveform.shape[-1] - gate_samples + 1, step_samples)
        energy = np.asarray(
            [
                np.mean(
                    np.square(waveform[start : start + gate_samples]),
                    dtype=np.float32,
                )
                for start in block_starts
            ],
            dtype=np.float32,
        )
        # 無音ブロックは規格どおり -inf となるため、NumPy の診断だけを抑えてゲートで除外する
        with np.errstate(divide="ignore", invalid="ignore"):
            block_loudness = np.float32(-0.691) + np.float32(10.0) * np.log10(energy)
            gated_blocks = block_loudness > np.float32(-70.0)
            gated_energy = np.sum(
                energy[gated_blocks],
                dtype=np.float32,
            ) / np.count_nonzero(gated_blocks)
            relative_gate = (
                np.float32(-0.691) + np.float32(10.0) * np.log10(gated_energy) - np.float32(10.0)
            )
            gated_blocks = np.logical_and(gated_blocks, block_loudness > relative_gate)
            gated_energy = np.sum(
                energy[gated_blocks],
                dtype=np.float32,
            ) / np.count_nonzero(gated_blocks)
            measured_db = np.float32(-0.691) + np.float32(10.0) * np.log10(gated_energy)
        return torch.tensor(measured_db, dtype=torch.float32)

    @staticmethod
    def _normalize_loudness(
        wav: torch.Tensor, sample_rate: int, target_db: float | None
    ) -> torch.Tensor:
        if target_db is None:
            return wav
        wav_device = wav.device
        wav = wav.to(dtype=torch.float32)
        if wav.ndim == 2:
            if wav.shape[0] == 1:
                wav = wav[0]
            elif wav.shape[1] == 1:
                wav = wav[:, 0]
            else:
                wav = wav.mean(dim=0)
        if wav.ndim != 1:
            raise ValueError(
                "normalize_loudness expects a mono waveform with shape (T,) "
                f"or singleton-channel (1, T)/(T, 1), got {tuple(wav.shape)}"
            )

        # BS.1770 の測定窓を満たすように短い参照音声だけ右側をゼロ埋めする
        minimum_samples = int(sample_rate * 0.5)
        loudness_input = wav
        if wav.shape[-1] < minimum_samples:
            loudness_input = torch.nn.functional.pad(
                wav,
                (0, minimum_samples - wav.shape[-1]),
            )

        # GPU 世代や CUDA ライブラリに左右されない SciPy の CPU IIR 経路で測る
        measured_db = DACVAECodec._measure_loudness(
            loudness_input,
            int(sample_rate),
        ).clamp_min(-70.0)
        gain = torch.exp(
            (torch.as_tensor(float(target_db)) - measured_db)
            * (torch.log(torch.tensor(10.0)) / 20.0)
        ).to(device=wav.device)
        normalized = wav * gain

        # ラウドネス調整後にピークが1.0を超える場合だけ全体を縮小する
        peak = normalized.abs().max()
        if torch.isfinite(peak) and peak > 1.0:
            normalized = normalized / peak
        return normalized.to(dtype=torch.float32, device=wav_device)

    @torch.inference_mode()
    def encode_waveform(
        self,
        waveform: torch.Tensor,
        sample_rate: int,
        *,
        normalize_db: float | None | object = _CODEC_DEFAULT,
        ensure_max: bool | None = None,
    ) -> torch.Tensor:
        """
        Input:
          waveform: (B, C, T) or (C, T)
          normalize_db: Optional target loudness (LUFS-like dB) applied before encode
          ensure_max: If True and normalize_db is None, scale down only when abs peak exceeds 1.0
        Output:
          latent: (B, T_latent, D_latent)
        """
        if waveform.ndim == 2:
            waveform = waveform.unsqueeze(0)
        if waveform.ndim != 3:
            raise ValueError(f"Expected waveform ndim=3, got shape={tuple(waveform.shape)}")

        if waveform.shape[1] != 1:
            waveform = waveform.mean(dim=1, keepdim=True)
        if sample_rate != self.sample_rate:
            # Reference audio normally arrives on CPU from SoundFile, and the short
            # per-utterance tensors pay more scheduling cost than useful work in torch resample.
            resampled_waveforms = []
            for wav in waveform.squeeze(1):
                wav_np = wav.detach().to(device="cpu", dtype=torch.float32).numpy()
                resampled_np = soxr.resample(
                    wav_np,
                    float(sample_rate),
                    float(self.sample_rate),
                    quality="HQ",
                )
                resampled_waveforms.append(
                    torch.from_numpy(np.asarray(resampled_np, dtype=np.float32))
                )
            waveform = torch.stack(resampled_waveforms, dim=0).unsqueeze(1)

        if normalize_db is _CODEC_DEFAULT:
            effective_normalize_db = self.normalize_db
        elif normalize_db is None:
            effective_normalize_db = None
        else:
            effective_normalize_db = float(normalize_db)
        # audiotools normalization already applies ensure_max_of_audio(), so codec-side
        # peak scaling is only needed when normalization is disabled.
        effective_ensure_max = (
            effective_normalize_db is None and bool(ensure_max) if ensure_max is not None else False
        )

        # audiotools accepts CUDA tensors, and codec encode immediately runs on the same device.
        waveform = waveform.to(self.device, dtype=torch.float32)
        if effective_normalize_db is not None or effective_ensure_max:
            # Keep behavior deterministic per utterance by normalizing each waveform independently.
            processed: list[torch.Tensor] = []
            for wav in waveform.squeeze(1):
                if effective_normalize_db is not None:
                    wav = self._normalize_loudness(
                        wav, sample_rate=self.sample_rate, target_db=effective_normalize_db
                    )
                wav = wav.squeeze()
                if wav.ndim != 1:
                    raise RuntimeError(
                        "Expected mono per-item waveform after preprocessing, "
                        f"got shape={tuple(wav.shape)}"
                    )
                if effective_ensure_max:
                    peak = wav.abs().max()
                    if torch.isfinite(peak) and peak > 1.0:
                        wav = wav * (1.0 / float(peak))
                processed.append(wav)
            waveform = torch.stack(processed, dim=0).unsqueeze(1)

        waveform = waveform.to(self.device, dtype=self.dtype)
        if self.deterministic_encode:
            required_paths_present = (
                hasattr(self.model, "encoder")
                and hasattr(self.model, "_pad")
                and hasattr(self.model, "quantizer")
                and hasattr(self.model.quantizer, "in_proj")
            )
            if not required_paths_present:
                raise RuntimeError(
                    "deterministic_encode=True requires encoder/_pad/quantizer.in_proj on DACVAE model."
                )
            z = self.model.encoder(self.model._pad(waveform))
            mean, _scale = self.model.quantizer.in_proj(z).chunk(2, dim=1)
            encoded = mean
        else:
            encoded = self.model.encode(waveform)  # (B, D, T)
        return encoded.transpose(1, 2).contiguous()  # (B, T, D)

    @torch.inference_mode()
    def decode_latent(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Input:
          latent: (B, T, D)
        Output:
          audio: (B, 1, samples)
        """
        if latent.ndim != 3:
            raise ValueError(f"Expected latent ndim=3, got shape={tuple(latent.shape)}")
        z = latent.transpose(1, 2).contiguous().to(self.device, dtype=self.dtype)  # (B, D, T)
        return self.model.decode(z)

    def encode_file(self, path: str | Path) -> torch.Tensor:
        # SoundFile で常に (frame, channel) として読み、従来の (channel, frame) 契約へ戻す
        data, sr = sf.read(str(path), dtype="float32", always_2d=True)
        wav = torch.from_numpy(np.ascontiguousarray(data.T))
        wav = wav.unsqueeze(0)  # (1, C, T)
        return self.encode_waveform(wav, sr).cpu()
