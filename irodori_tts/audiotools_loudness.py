# Portions of this file are adapted from descript-audiotools 0.7.2:
# https://github.com/descriptinc/audiotools/tree/49b8b6bb9259a99bda5f2138a1a4df055dd172ef
# and PyLoudNorm 0.1.1:
# https://github.com/csteinmetz1/pyloudnorm/tree/v0.1.1
#
# Copyright (c) 2023-Present, Descript
# Copyright (c) 2018 Christian Steinmetz
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio


class AudioToolsLoudness:
    """Provide the AudioTools 0.7.2 loudness operations used by Irodori-TTS."""

    MIN_LOUDNESS = -70.0
    GAIN_FACTOR = math.log(10.0) / 20.0

    @staticmethod
    def _filter_coefficients(
        *,
        gain_db: float,
        quality_factor: float,
        frequency: float,
        sample_rate: int,
        filter_type: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate biquad filter coefficients for the AudioTools weighting filters.

        Args:
            gain_db (float): Filter gain in dB.
            quality_factor (float): Filter quality factor.
            frequency (float): Center frequency in Hz.
            sample_rate (int): Audio sample rate in Hz.
            filter_type (str): Filter shape, either ``high_shelf`` or ``high_pass``.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Denominator and numerator coefficients.

        Raises:
            ValueError: The filter type is unsupported.
        """

        amplitude = 10.0 ** (gain_db / 40.0)
        angular_frequency = 2.0 * np.pi * (frequency / sample_rate)
        alpha = np.sin(angular_frequency) / (2.0 * quality_factor)
        cosine = np.cos(angular_frequency)

        # Generate the two filter shapes used by the default AudioTools K-weighting meter.
        if filter_type == "high_shelf":
            square_root_amplitude = np.sqrt(amplitude)
            numerator = np.array(
                [
                    amplitude
                    * (
                        (amplitude + 1.0)
                        + (amplitude - 1.0) * cosine
                        + 2.0 * square_root_amplitude * alpha
                    ),
                    -2.0 * amplitude * ((amplitude - 1.0) + (amplitude + 1.0) * cosine),
                    amplitude
                    * (
                        (amplitude + 1.0)
                        + (amplitude - 1.0) * cosine
                        - 2.0 * square_root_amplitude * alpha
                    ),
                ],
                dtype=np.float64,
            )
            denominator = np.array(
                [
                    (amplitude + 1.0)
                    - (amplitude - 1.0) * cosine
                    + 2.0 * square_root_amplitude * alpha,
                    2.0 * ((amplitude - 1.0) - (amplitude + 1.0) * cosine),
                    (amplitude + 1.0)
                    - (amplitude - 1.0) * cosine
                    - 2.0 * square_root_amplitude * alpha,
                ],
                dtype=np.float64,
            )
        elif filter_type == "high_pass":
            numerator = np.array(
                [
                    (1.0 + cosine) / 2.0,
                    -(1.0 + cosine),
                    (1.0 + cosine) / 2.0,
                ],
                dtype=np.float64,
            )
            denominator = np.array(
                [1.0 + alpha, -2.0 * cosine, 1.0 - alpha],
                dtype=np.float64,
            )
        else:
            raise ValueError(f"Unsupported AudioTools filter type: {filter_type}")

        # PyLoudNorm normalizes both coefficient arrays by a0 before AudioTools casts to float32.
        numerator /= denominator[0]
        denominator /= denominator[0]
        return (
            torch.from_numpy(denominator).to(dtype=torch.float32),
            torch.from_numpy(numerator).to(dtype=torch.float32),
        )

    @classmethod
    def measure(cls, waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
        """Compute integrated gated loudness using the AudioTools 0.7.2 CPU path.

        Args:
            waveform (torch.Tensor): One-dimensional mono audio data.
            sample_rate (int): Audio sample rate in Hz.

        Returns:
            torch.Tensor: Integrated loudness as a float32 CPU tensor.

        Raises:
            ValueError: The input is not a one-dimensional mono waveform.
        """

        if waveform.ndim != 1:
            raise ValueError(
                f"AudioTools loudness expects a mono waveform, got {tuple(waveform.shape)}"
            )
        audio = waveform.detach().to(device="cpu", dtype=torch.float32)

        # AudioTools pads signals shorter than 0.5 seconds before applying the 400 ms gate.
        minimum_samples = int(sample_rate * 0.5)
        if audio.shape[-1] < minimum_samples:
            audio = F.pad(audio, (0, minimum_samples - audio.shape[-1]))
        audio = audio[None, :, None]

        # Apply the AudioTools CPU filters in the original order with clamp disabled.
        for filter_parameters in (
            (4.0, 1.0 / np.sqrt(2.0), 1500.0, "high_shelf"),
            (0.0, 0.5, 38.0, "high_pass"),
        ):
            denominator, numerator = cls._filter_coefficients(
                gain_db=filter_parameters[0],
                quality_factor=filter_parameters[1],
                frequency=filter_parameters[2],
                sample_rate=sample_rate,
                filter_type=filter_parameters[3],
            )
            filtered = torchaudio.functional.lfilter(
                audio.permute(0, 2, 1),
                denominator,
                numerator,
                clamp=False,
            )
            audio = filtered.permute(0, 2, 1)

        # Match julius.core.unfold by including the final samples in at least one window.
        block_samples = int(0.4 * sample_rate)
        step_samples = int(block_samples * 0.25)
        frame_count = (
            math.ceil((max(audio.shape[1], block_samples) - block_samples) / step_samples) + 1
        )
        target_samples = (frame_count - 1) * step_samples + block_samples
        padded = F.pad(audio.permute(0, 2, 1), (0, target_samples - audio.shape[1])).contiguous()
        windows = padded.as_strided(
            (padded.shape[0], padded.shape[1], frame_count, block_samples),
            (padded.stride(0), padded.stride(1), step_samples, 1),
        ).transpose(-1, -2)

        # Apply the AudioTools absolute and relative gates using the original comparisons.
        energy = windows.square().sum(2) / float(block_samples)
        block_loudness = -0.691 + 10.0 * torch.log10(energy)
        absolute_mask = block_loudness > cls.MIN_LOUDNESS
        absolute_energy = energy.masked_fill(~absolute_mask, 0.0)
        absolute_average = absolute_energy.sum(2) / absolute_mask.sum(2)
        relative_threshold = -0.691 + 10.0 * torch.log10(absolute_average.sum(1)) - 10.0
        relative_mask = block_loudness > relative_threshold[:, None, None]
        combined_mask = absolute_mask & relative_mask
        gated_energy = energy.masked_fill(~combined_mask, 0.0).sum(2) / combined_mask.sum(2)
        gated_energy = torch.where(
            torch.isnan(gated_energy),
            torch.zeros_like(gated_energy),
            gated_energy,
        )
        gated_energy = torch.nan_to_num(
            gated_energy,
            nan=0.0,
            posinf=torch.finfo(torch.float32).max,
            neginf=torch.finfo(torch.float32).min,
        )
        loudness = -0.691 + 10.0 * torch.log10(gated_energy.sum(1))
        return loudness.maximum(torch.full_like(loudness, cls.MIN_LOUDNESS))[0]

    @classmethod
    def normalize(
        cls,
        waveform: torch.Tensor,
        sample_rate: int,
        target_db: float | None,
    ) -> torch.Tensor:
        """Normalize loudness and peak level in the AudioTools 0.7.2 order.

        Args:
            waveform (torch.Tensor): Mono or singleton-channel audio data.
            sample_rate (int): Audio sample rate in Hz.
            target_db (float | None): Target loudness, or ``None`` to return the input unchanged.

        Returns:
            torch.Tensor: Float32 mono audio on the input device.

        Raises:
            ValueError: The input cannot be represented as mono audio.
        """

        if target_db is None:
            return waveform
        waveform_device = waveform.device
        mono_waveform = waveform.to(dtype=torch.float32)
        if mono_waveform.ndim == 2:
            if mono_waveform.shape[0] == 1:
                mono_waveform = mono_waveform[0]
            elif mono_waveform.shape[1] == 1:
                mono_waveform = mono_waveform[:, 0]
            else:
                mono_waveform = mono_waveform.mean(dim=0)
        if mono_waveform.ndim != 1:
            raise ValueError(
                "AudioTools normalization expects a mono waveform with shape (T,) "
                f"or singleton-channel (1, T)/(T, 1), got {tuple(mono_waveform.shape)}"
            )

        # AudioTools normalize() converts the LUFS difference into linear gain.
        measured_db = cls.measure(mono_waveform, sample_rate)
        gain = torch.exp((torch.as_tensor(float(target_db)) - measured_db) * cls.GAIN_FACTOR).to(
            device=mono_waveform.device
        )
        normalized = mono_waveform * gain

        # AudioTools ensure_max_of_audio() scales only signals whose absolute peak exceeds 1.0.
        peak = normalized.abs().max()
        if torch.isfinite(peak) and peak > 1.0:
            normalized = normalized / peak
        return normalized.to(dtype=torch.float32, device=waveform_device)
