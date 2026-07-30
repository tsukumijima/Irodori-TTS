from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import torch

from .model import EncodedConditions, TextToLatentRFDiT
from .packed_attention import PackedAttentionPlan, build_packed_attention_plan
from .speaker_inversion import SPEAKER_INVERSION_UNCOND_MODES
from .waveex import WaveExBuffer, WaveExConfig


@dataclass(frozen=True)
class VelocityFieldGuidance:
    """同一潜在上の条件対から速度差を作り、通常の速度へ加算する。

    caption と speaker のどちらか一方の条件対を指定する。未指定側は通常の
    条件を共有するため、話者を保ったキャプション操作と、キャプションを
    保った話者テクスチャ移植を同じ計算で扱える。

    Args:
        alpha (float): `target - opposite` の速度差へ掛ける係数
        min_t (float): 速度差を加える拡散時刻の下限
        max_t (float): 速度差を加える拡散時刻の上限
        target_caption_state (torch.Tensor | None): 目標側の caption state
        target_caption_mask (torch.Tensor | None): 目標側の caption mask
        opposite_caption_state (torch.Tensor | None): 反対側の caption state
        opposite_caption_mask (torch.Tensor | None): 反対側の caption mask
        target_speaker_state (torch.Tensor | None): 目標側の speaker state
        target_speaker_mask (torch.Tensor | None): 目標側の speaker mask
        opposite_speaker_state (torch.Tensor | None): 反対側の speaker state
        opposite_speaker_mask (torch.Tensor | None): 反対側の speaker mask
    """

    alpha: float
    min_t: float = 0.0
    max_t: float = 1.0
    target_caption_state: torch.Tensor | None = None
    target_caption_mask: torch.Tensor | None = None
    opposite_caption_state: torch.Tensor | None = None
    opposite_caption_mask: torch.Tensor | None = None
    target_speaker_state: torch.Tensor | None = None
    target_speaker_mask: torch.Tensor | None = None
    opposite_speaker_state: torch.Tensor | None = None
    opposite_speaker_mask: torch.Tensor | None = None


@dataclass(frozen=True)
class TrajectoryCheckpoint:
    """途中時刻の完成予測を軌道介入コールバックへ渡す。

    Args:
        step_index (int): 0 始まりのサンプリングステップ番号
        t (float): 速度を評価した現在時刻
        t_next (float): 状態を進める次時刻
        x0_hat (torch.Tensor): `rf_predict_x0()` で求めた完成潜在予測
    """

    step_index: int
    t: float
    t_next: float
    x0_hat: torch.Tensor


@dataclass(frozen=True)
class TrajectoryObservation:
    """
    途中時刻の完成予測を読み取り専用の観測処理へ渡す。

    Args:
        step_index (int): 0始まりのサンプリングステップ番号。
        t (float): 速度を評価した現在時刻。
        t_next (float): 状態を進める次時刻。
        x0_hat (torch.Tensor): 実際の状態更新へ使う速度から求めた完成潜在予測。
        latent_mask (torch.Tensor | None): パディングを除く有効な潜在位置。
    """

    step_index: int
    t: float
    t_next: float
    x0_hat: torch.Tensor
    latent_mask: torch.Tensor | None


@dataclass(frozen=True)
class TrajectoryObserver:
    """
    指定した完全 ODE ステップの完成潜在予測を読み取り専用で観測する。

    Args:
        step_indices (tuple[int, ...]): コールバックを呼ぶ0始まりのステップ番号。
        callback (Callable[[TrajectoryObservation], None]): 観測値を受け取る処理。
    """

    step_indices: tuple[int, ...]
    callback: Callable[[TrajectoryObservation], None]


@dataclass(frozen=True)
class TrajectoryIntervention:
    """指定ステップで完成潜在予測を観測し、必要なら別ノイズへ分岐する。

    コールバックが `None` を返す場合は通常の Euler 更新を続ける
    ノイズを返す場合は完成予測とそのノイズから次時刻の状態を再構成する

    Args:
        step_indices (tuple[int, ...]): コールバックを呼ぶ 0 始まりのステップ番号
        callback (Callable[[TrajectoryCheckpoint], torch.Tensor | None]):
            観測のみなら `None`、分岐する場合は完成予測と同じ形の別ノイズを返す処理
    """

    step_indices: tuple[int, ...]
    callback: Callable[[TrajectoryCheckpoint], torch.Tensor | None]


def _make_rng(seed: int, device: torch.device) -> tuple[torch.Generator, torch.device]:
    # MPS generators are not available on some PyTorch builds; use CPU generator as fallback.
    try:
        return torch.Generator(device=device).manual_seed(seed), device
    except RuntimeError:
        return torch.Generator(device="cpu").manual_seed(seed), torch.device("cpu")


def sample_logit_normal_t(
    batch_size: int,
    device: torch.device,
    mean: float = 0.0,
    std: float = 1.0,
    t_min: float = 1e-3,
    t_max: float = 0.999,
) -> torch.Tensor:
    z = torch.randn(batch_size, device=device) * std + mean
    t = torch.sigmoid(z)
    return t.clamp(min=t_min, max=t_max)


def sample_stratified_logit_normal_t(
    batch_size: int,
    device: torch.device,
    mean: float = 0.0,
    std: float = 1.0,
    t_min: float = 1e-3,
    t_max: float = 0.999,
) -> torch.Tensor:
    """
    Stratified sampling for logit-normal timesteps.

    u ~ stratified U(0, 1), z = mean + std * Phi^{-1}(u), t = sigmoid(z)
    """
    if batch_size <= 0:
        return torch.empty((0,), device=device)
    u = (
        torch.arange(batch_size, device=device, dtype=torch.float32)
        + torch.rand(batch_size, device=device)
    ) / float(batch_size)
    u = u.clamp(1e-6, 1.0 - 1e-6)
    # Phi^{-1}(u) = sqrt(2) * erfinv(2u - 1)
    z = torch.erfinv(2.0 * u - 1.0) * (2.0**0.5)
    z = z * std + mean
    t = torch.sigmoid(z)
    # Randomize assignment order so dataset ordering does not correlate with t bins.
    t = t[torch.randperm(batch_size, device=device)]
    return t.clamp(min=t_min, max=t_max)


def rf_interpolate(x0: torch.Tensor, noise: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    # Straight line interpolation: x_t = (1-t) x0 + t z.
    return (1.0 - t[:, None, None]) * x0 + t[:, None, None] * noise


def rf_velocity_target(x0: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
    # For x_t = (1-t) x0 + t z, velocity is d/dt x_t = z - x0.
    return noise - x0


def rf_predict_x0(x_t: torch.Tensor, v_pred: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    # x_t = x0 + t * v  =>  x0 = x_t - t * v
    return x_t - t[:, None, None] * v_pred


def temporal_score_rescale(
    v_pred: torch.Tensor,
    x_t: torch.Tensor,
    t: float | torch.Tensor,
    rescale_k: float,
    rescale_sigma: float,
) -> torch.Tensor:
    """
    Temporal score rescaling from https://arxiv.org/pdf/2510.01184.
    """
    t_value = float(t.item()) if isinstance(t, torch.Tensor) else float(t)
    if t_value >= 1.0:
        return v_pred
    one_minus_t = 1.0 - t_value
    snr = (one_minus_t * one_minus_t) / (t_value * t_value)
    sigma_sq = float(rescale_sigma) * float(rescale_sigma)
    ratio = (snr * sigma_sq + 1.0) / (snr * sigma_sq / float(rescale_k) + 1.0)
    return (ratio * (one_minus_t * v_pred + x_t) - x_t) / one_minus_t


def scale_speaker_kv_cache(
    context_kv_cache: list[tuple[torch.Tensor, ...]],
    scale: float,
    max_layers: int | None = None,
) -> None:
    """
    In-place scaling of speaker K/V tensors in precomputed context cache.
    """
    if max_layers is None:
        n_layers = len(context_kv_cache)
    else:
        n_layers = max(0, min(int(max_layers), len(context_kv_cache)))
    for i in range(n_layers):
        layer_kv = context_kv_cache[i]
        if len(layer_kv) < 4:
            raise ValueError(
                f"Expected at least 4 tensors in context KV cache entry, got {len(layer_kv)}"
            )
        k_speaker = layer_kv[2]
        v_speaker = layer_kv[3]
        k_speaker.mul_(scale)
        v_speaker.mul_(scale)


@torch.inference_mode()
def sample_euler_rf_cfg(
    model: TextToLatentRFDiT,
    text_input_ids: torch.Tensor,
    text_mask: torch.Tensor,
    ref_latent: torch.Tensor | None,
    ref_mask: torch.Tensor | None,
    sequence_length: int,
    caption_input_ids: torch.Tensor | None = None,
    caption_mask: torch.Tensor | None = None,
    speaker_state_override: torch.Tensor | None = None,
    speaker_mask_override: torch.Tensor | None = None,
    caption_state_override: torch.Tensor | None = None,
    caption_mask_override: torch.Tensor | None = None,
    encoded_conditions: EncodedConditions | None = None,
    encoded_conditions_without_condition_tokens: EncodedConditions | None = None,
    speaker_uncond_mode: str = "mask",
    num_steps: int = 40,
    cfg_scale_text: float = 3.0,
    cfg_scale_caption: float = 3.0,
    cfg_scale_speaker: float = 5.0,
    cfg_scale_condition: float = 0.0,
    cfg_guidance_mode: str = "independent",
    cfg_min_t: float = 0.5,
    cfg_max_t: float = 1.0,
    seed: int = 0,
    cfg_scale: float | None = None,
    truncation_factor: float | None = None,
    rescale_k: float | None = None,
    rescale_sigma: float | None = None,
    use_context_kv_cache: bool = True,
    use_cudnn_packed_attention: bool = False,
    speaker_kv_scale: float | None = None,
    speaker_kv_max_layers: int | None = None,
    speaker_kv_min_t: float | None = None,
    t_schedule_mode: str = "linear",
    sway_coeff: float = -1.0,
    waveex: WaveExConfig | None = None,
    noise_dtype: torch.dtype | None = None,
    initial_noise: torch.Tensor | None = None,
    initial_noise_offset: int = 0,
    latent_mask: torch.Tensor | None = None,
    condition_token_ids: torch.Tensor | None = None,
    condition_token_mask: torch.Tensor | None = None,
    condition_token_scales: torch.Tensor | None = None,
    velocity_field_guidance: VelocityFieldGuidance | None = None,
    trajectory_intervention: TrajectoryIntervention | None = None,
    trajectory_observer: TrajectoryObserver | None = None,
) -> torch.Tensor:
    """
    Euler sampling over RF ODE with text/reference/caption conditioning CFG.

    Returns:
      latent sequence in patched space, shape (B, sequence_length, patched_latent_dim)
    """
    device = model.device
    dtype = model.dtype
    batch_size = text_input_ids.shape[0]
    latent_dim = model.cfg.patched_latent_dim

    rng, rng_device = _make_rng(seed=seed, device=device)
    if initial_noise is None:
        x_t = torch.randn(
            (batch_size, sequence_length, latent_dim),
            device=rng_device,
            dtype=dtype if noise_dtype is None else noise_dtype,
            generator=rng,
        )
        if rng_device != device:
            x_t = x_t.to(device=device)
        if x_t.dtype != dtype:
            x_t = x_t.to(dtype=dtype)
    else:
        if int(initial_noise_offset) < 0:
            raise ValueError(
                f"initial_noise_offset must be non-negative, got {initial_noise_offset!r}."
            )
        if initial_noise.ndim != 3:
            raise ValueError(
                f"initial_noise must have shape (B, T, C), got {tuple(initial_noise.shape)}."
            )
        if initial_noise.shape[0] != batch_size:
            raise ValueError(
                f"initial_noise batch size mismatch: expected {batch_size}, got {initial_noise.shape[0]}."
            )
        if initial_noise.shape[2] != latent_dim:
            raise ValueError(
                f"initial_noise latent dim mismatch: expected {latent_dim}, got {initial_noise.shape[2]}."
            )
        noise_start = int(initial_noise_offset)
        noise_end = noise_start + int(sequence_length)
        if noise_end > int(initial_noise.shape[1]):
            raise ValueError(
                f"initial_noise is too short: need end index {noise_end}, "
                f"available {initial_noise.shape[1]}."
            )

        # Request-level streaming experiments pass one long noise tensor and let each
        # chunk consume a non-overlapping span. Clone because the ODE loop updates x_t in-place.
        x_t = initial_noise[:, noise_start:noise_end, :].to(device=device, dtype=dtype).clone()
    if truncation_factor is not None:
        x_t = x_t * float(truncation_factor)
    if latent_mask is not None:
        if latent_mask.ndim != 2:
            raise ValueError(f"latent_mask must have shape (B, S), got {tuple(latent_mask.shape)}.")
        if latent_mask.shape[0] != batch_size or latent_mask.shape[1] != sequence_length:
            raise ValueError(
                "latent_mask shape mismatch: "
                f"expected ({batch_size}, {sequence_length}), got {tuple(latent_mask.shape)}."
            )
        latent_mask = latent_mask.to(device=device, dtype=torch.bool)

    if cfg_scale is not None:
        # Backward compatibility for old single-scale caller.
        cfg_scale_text = float(cfg_scale)
        cfg_scale_caption = float(cfg_scale)
        cfg_scale_speaker = float(cfg_scale)
    if not model.cfg.use_speaker_condition_resolved:
        cfg_scale_speaker = 0.0
        speaker_kv_scale = None
    speaker_uncond_mode = str(speaker_uncond_mode).strip().lower()
    if speaker_uncond_mode not in SPEAKER_INVERSION_UNCOND_MODES:
        raise ValueError(
            f"speaker_uncond_mode must be one of {sorted(SPEAKER_INVERSION_UNCOND_MODES)}, "
            f"got {speaker_uncond_mode!r}"
        )

    cfg_guidance_mode = str(cfg_guidance_mode).strip().lower()
    if cfg_guidance_mode not in {"independent", "joint", "alternating"}:
        raise ValueError(
            f"Unsupported cfg_guidance_mode={cfg_guidance_mode!r}. "
            "Expected one of: independent, joint, alternating."
        )

    init_scale = 0.999
    t_schedule_mode_norm = str(t_schedule_mode).strip().lower()
    sway_coeff_value = float(sway_coeff)
    if not math.isfinite(sway_coeff_value):
        raise ValueError(f"sway_coeff must be finite, got {sway_coeff!r}.")
    if t_schedule_mode_norm == "linear":
        u = torch.linspace(0.0, 1.0, num_steps + 1, device=device)
    elif t_schedule_mode_norm == "sway":
        # F5-TTS-style Sway Sampling. Negative sway_coeff densifies the noise
        # side of the schedule (early steps); positive densifies the data side.
        u = torch.linspace(0.0, 1.0, num_steps + 1, device=device)
        u = u + sway_coeff_value * (torch.cos(0.5 * math.pi * u) + u - 1.0)
        u = u.clamp(0.0, 1.0)
    else:
        raise ValueError(
            f"Unsupported t_schedule_mode={t_schedule_mode!r}. Expected 'linear' or 'sway'."
        )
    t_schedule = (1.0 - u) * init_scale
    # GPU 上の時刻 Tensor は ODE 演算へ残し、分岐用の値だけを要求ごとに1回まとめて CPU へ移す
    ## 反復中の CUDA スカラー読み出しは各ステップで CPU と GPU を同期させるため避ける
    t_schedule_values = tuple(float(value) for value in t_schedule.detach().cpu())
    if not all(
        current_t > next_t
        for current_t, next_t in zip(
            t_schedule_values[:-1],
            t_schedule_values[1:],
            strict=True,
        )
    ):
        raise ValueError("t_schedule must be strictly decreasing; adjust num_steps or sway_coeff.")
    trajectory_step_indices: set[int] = set()
    if trajectory_intervention is not None:
        trajectory_step_indices = set(trajectory_intervention.step_indices)
        if len(trajectory_step_indices) != len(trajectory_intervention.step_indices):
            raise ValueError("trajectory_intervention step_indices must not contain duplicates.")
        if any(step_index < 0 or step_index >= num_steps for step_index in trajectory_step_indices):
            raise ValueError(
                f"trajectory_intervention step_indices must be within [0, {num_steps - 1}]."
            )
        # WaveEx の外挿ステップには速度予測がないため、完成予測の定義を混在させない
        if waveex is not None and waveex.enabled:
            raise ValueError("trajectory_intervention cannot be combined with WaveEx.")
    observation_step_indices: set[int] = set()
    if trajectory_observer is not None:
        observation_step_indices = set(trajectory_observer.step_indices)
        if len(observation_step_indices) != len(trajectory_observer.step_indices):
            raise ValueError("trajectory_observer step_indices must not contain duplicates.")
        if any(
            step_index < 0 or step_index >= num_steps for step_index in observation_step_indices
        ):
            raise ValueError(
                f"trajectory_observer step_indices must be within [0, {num_steps - 1}]."
            )
        if waveex is not None and waveex.enabled:
            waveex_observation_indices = waveex.resolve_ode_step_indices(num_steps)
            if not observation_step_indices.issubset(waveex_observation_indices):
                raise ValueError(
                    "trajectory_observer step_indices must select full ODE steps when WaveEx is enabled."
                )
    use_independent_cfg = cfg_guidance_mode == "independent"
    use_joint_cfg = cfg_guidance_mode == "joint"
    use_alternating_cfg = cfg_guidance_mode == "alternating"

    if encoded_conditions is None:
        (
            text_state_cond,
            text_mask_cond,
            speaker_state_cond,
            speaker_mask_cond,
            caption_state_cond,
            caption_mask_cond,
        ) = model.encode_conditions(
            text_input_ids=text_input_ids,
            text_mask=text_mask,
            ref_latent=ref_latent,
            ref_mask=ref_mask,
            caption_input_ids=caption_input_ids,
            caption_mask=caption_mask,
            speaker_state_override=speaker_state_override,
            speaker_mask_override=speaker_mask_override,
            caption_state_override=caption_state_override,
            caption_mask_override=caption_mask_override,
            speaker_uncond_mode=speaker_uncond_mode,
            condition_token_ids=condition_token_ids,
            condition_token_mask=condition_token_mask,
            condition_token_scales=condition_token_scales,
        )
    else:
        (
            text_state_cond,
            text_mask_cond,
            speaker_state_cond,
            speaker_mask_cond,
            caption_state_cond,
            caption_mask_cond,
        ) = encoded_conditions
    text_state_uncond = torch.zeros_like(text_state_cond)
    text_mask_uncond = torch.zeros_like(text_mask_cond)
    speaker_state_uncond = None
    speaker_mask_uncond = None
    if model.cfg.use_speaker_condition_resolved:
        if speaker_state_cond is None or speaker_mask_cond is None:
            raise RuntimeError(
                "Speaker conditioning is enabled but encoded speaker state is missing."
            )
        if speaker_uncond_mode == "noise":
            speaker_noise = torch.randn(
                speaker_state_cond.shape,
                device=rng_device,
                dtype=speaker_state_cond.dtype,
                generator=rng,
            )
            if rng_device != device:
                speaker_noise = speaker_noise.to(device=device)
            speaker_state_uncond = speaker_noise * speaker_state_cond.std().clamp_min(1e-6)
            speaker_mask_uncond = torch.ones_like(speaker_mask_cond)
        else:
            speaker_state_uncond = torch.zeros_like(speaker_state_cond)
            speaker_mask_uncond = torch.zeros_like(speaker_mask_cond)
    condition_token_mask_cond = None
    if condition_token_ids is not None:
        if condition_token_mask is None:
            condition_token_mask_cond = torch.ones_like(condition_token_ids, dtype=torch.bool)
        else:
            condition_token_mask_cond = condition_token_mask.to(device=device, dtype=torch.bool)
    has_condition_cfg = (
        model.cfg.use_speaker_condition_resolved
        and cfg_scale_condition > 0
        and condition_token_mask_cond is not None
        and bool(condition_token_mask_cond.any().item())
    )
    condition_token_uncond_bundle = None
    if has_condition_cfg:
        if encoded_conditions_without_condition_tokens is None:
            # 条件 CFG では reference speaker は残し、LoRA で学習した条件 token だけを外す
            encoded_conditions_without_condition_tokens = model.encode_conditions(
                text_input_ids=text_input_ids,
                text_mask=text_mask,
                ref_latent=ref_latent,
                ref_mask=ref_mask,
                caption_input_ids=caption_input_ids,
                caption_mask=caption_mask,
                speaker_state_override=speaker_state_override,
                speaker_mask_override=speaker_mask_override,
                caption_state_override=caption_state_override,
                caption_mask_override=caption_mask_override,
                speaker_uncond_mode=speaker_uncond_mode,
            )
        condition_token_uncond_bundle = encoded_conditions_without_condition_tokens
        if condition_token_uncond_bundle[2] is not None and speaker_state_cond is not None:
            condition_speaker_state = condition_token_uncond_bundle[2]
            condition_speaker_mask = condition_token_uncond_bundle[3]
            if condition_speaker_mask is None:
                raise RuntimeError("Condition-token uncond speaker mask is missing.")
            cond_tokens = int(speaker_state_cond.shape[1])
            uncond_tokens = int(condition_speaker_state.shape[1])
            if uncond_tokens < cond_tokens:
                # 条件 CFG では token を無効化するだけなので、形合わせ用の末尾 token は mask=False にする
                pad_tokens = cond_tokens - uncond_tokens
                state_pad = torch.zeros(
                    (
                        condition_speaker_state.shape[0],
                        pad_tokens,
                        condition_speaker_state.shape[2],
                    ),
                    device=condition_speaker_state.device,
                    dtype=condition_speaker_state.dtype,
                )
                mask_pad = torch.zeros(
                    (condition_speaker_mask.shape[0], pad_tokens),
                    device=condition_speaker_mask.device,
                    dtype=torch.bool,
                )
                condition_speaker_state = torch.cat([condition_speaker_state, state_pad], dim=1)
                condition_speaker_mask = torch.cat([condition_speaker_mask, mask_pad], dim=1)
                condition_token_uncond_bundle = (
                    condition_token_uncond_bundle[0],
                    condition_token_uncond_bundle[1],
                    condition_speaker_state,
                    condition_speaker_mask,
                    condition_token_uncond_bundle[4],
                    condition_token_uncond_bundle[5],
                )
            elif uncond_tokens != cond_tokens:
                raise RuntimeError(
                    "Condition-token uncond speaker state is longer than the conditioned state."
                )
    caption_state_uncond = None
    caption_mask_uncond = None
    if model.cfg.use_caption_condition:
        if caption_state_cond is None or caption_mask_cond is None:
            raise RuntimeError(
                "Caption conditioning is enabled but encoded caption state is missing."
            )
        caption_state_uncond = torch.zeros_like(caption_state_cond)
        caption_mask_uncond = torch.zeros_like(caption_mask_cond)

    has_text_cfg = cfg_scale_text > 0
    has_caption_cfg = (
        model.cfg.use_caption_condition
        and cfg_scale_caption > 0
        and caption_mask_cond is not None
        and bool(caption_mask_cond.any().item())
    )
    has_speaker_cfg = cfg_scale_speaker > 0

    if velocity_field_guidance is not None:
        caption_values = (
            velocity_field_guidance.target_caption_state,
            velocity_field_guidance.target_caption_mask,
            velocity_field_guidance.opposite_caption_state,
            velocity_field_guidance.opposite_caption_mask,
        )
        speaker_values = (
            velocity_field_guidance.target_speaker_state,
            velocity_field_guidance.target_speaker_mask,
            velocity_field_guidance.opposite_speaker_state,
            velocity_field_guidance.opposite_speaker_mask,
        )
        has_caption_pair = all(value is not None for value in caption_values)
        has_speaker_pair = all(value is not None for value in speaker_values)
        has_partial_caption_pair = (
            any(value is not None for value in caption_values) and not has_caption_pair
        )
        has_partial_speaker_pair = (
            any(value is not None for value in speaker_values) and not has_speaker_pair
        )

        # 片側だけ指定された条件対は通常条件への暗黙置換を招くため、入口で明示的に拒否
        if has_partial_caption_pair or has_partial_speaker_pair:
            raise ValueError(
                "velocity_field_guidance requires all four state/mask values for each specified channel."
            )
        # 条件チャネルを同時に2つ変えると差分の意味が曖昧になるため1対だけを受け付ける
        if has_caption_pair == has_speaker_pair:
            raise ValueError(
                "velocity_field_guidance requires exactly one complete caption or speaker pair."
            )
        if not math.isfinite(float(velocity_field_guidance.alpha)):
            raise ValueError("velocity_field_guidance alpha must be finite.")
        if not 0.0 <= velocity_field_guidance.min_t <= velocity_field_guidance.max_t <= 1.0:
            raise ValueError(
                "velocity_field_guidance time range must satisfy 0 <= min_t <= max_t <= 1."
            )

        # 追加 forward が通常条件と同じバッチ契約を使えることを、サンプリング開始前に検証
        active_values = caption_values if has_caption_pair else speaker_values
        target_state, target_mask, opposite_state, opposite_mask = active_values
        assert target_state is not None and target_mask is not None
        assert opposite_state is not None and opposite_mask is not None
        channel_name = "caption" if has_caption_pair else "speaker"
        for side_name, state, mask in (
            ("target", target_state, target_mask),
            ("opposite", opposite_state, opposite_mask),
        ):
            if state.ndim != 3 or mask.ndim != 2:
                raise ValueError(
                    f"velocity_field_guidance {side_name} {channel_name} state/mask must have "
                    f"shapes (B, T, C) and (B, T), got {tuple(state.shape)} and {tuple(mask.shape)}."
                )
            if state.shape[:2] != mask.shape:
                raise ValueError(
                    f"velocity_field_guidance {side_name} {channel_name} state/mask shape mismatch."
                )
            if state.shape[0] != batch_size:
                raise ValueError(
                    f"velocity_field_guidance {side_name} {channel_name} batch size mismatch: "
                    f"expected {batch_size}, got {state.shape[0]}."
                )
            if state.device != device or mask.device != device:
                raise ValueError(
                    f"velocity_field_guidance {side_name} {channel_name} tensors must be on {device}."
                )
            if state.dtype != dtype or mask.dtype != torch.bool:
                raise ValueError(
                    f"velocity_field_guidance {side_name} {channel_name} state/mask must use "
                    f"{dtype} and torch.bool."
                )

    def _bundle(
        *,
        text_state: torch.Tensor,
        text_mask_val: torch.Tensor,
        speaker_state: torch.Tensor | None,
        speaker_mask_val: torch.Tensor | None,
        caption_state: torch.Tensor | None,
        caption_mask_val: torch.Tensor | None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
    ]:
        return (
            text_state,
            text_mask_val,
            speaker_state,
            speaker_mask_val,
            caption_state,
            caption_mask_val,
        )

    cond_bundle = _bundle(
        text_state=text_state_cond,
        text_mask_val=text_mask_cond,
        speaker_state=speaker_state_cond,
        speaker_mask_val=speaker_mask_cond,
        caption_state=caption_state_cond,
        caption_mask_val=caption_mask_cond,
    )
    enabled_cfg_names: list[str] = []
    cfg_scales: dict[str, float] = {}
    if has_text_cfg:
        enabled_cfg_names.append("text")
        cfg_scales["text"] = float(cfg_scale_text)
    if has_speaker_cfg:
        enabled_cfg_names.append("speaker")
        cfg_scales["speaker"] = float(cfg_scale_speaker)
    if has_condition_cfg:
        enabled_cfg_names.append("condition")
        cfg_scales["condition"] = float(cfg_scale_condition)
    if has_caption_cfg:
        enabled_cfg_names.append("caption")
        cfg_scales["caption"] = float(cfg_scale_caption)

    independent_bundles = [cond_bundle]
    independent_names = ["cond"]
    if use_independent_cfg:
        for name in enabled_cfg_names:
            independent_names.append(name)
            independent_bundles.append(
                _bundle(
                    text_state=text_state_uncond if name == "text" else text_state_cond,
                    text_mask_val=text_mask_uncond if name == "text" else text_mask_cond,
                    speaker_state=(
                        speaker_state_uncond if name == "speaker" else speaker_state_cond
                    ),
                    speaker_mask_val=(
                        speaker_mask_uncond if name == "speaker" else speaker_mask_cond
                    ),
                    caption_state=(
                        caption_state_uncond if name == "caption" else caption_state_cond
                    ),
                    caption_mask_val=(
                        caption_mask_uncond if name == "caption" else caption_mask_cond
                    ),
                )
            )
        if has_condition_cfg and condition_token_uncond_bundle is not None:
            condition_index = independent_names.index("condition")
            independent_bundles[condition_index] = _bundle(
                text_state=text_state_cond,
                text_mask_val=text_mask_cond,
                speaker_state=condition_token_uncond_bundle[2],
                speaker_mask_val=condition_token_uncond_bundle[3],
                caption_state=caption_state_cond,
                caption_mask_val=caption_mask_cond,
            )
    cfg_batch_mult = len(independent_bundles)

    def _cat_optional_tensors(values: list[torch.Tensor | None]) -> torch.Tensor | None:
        present = [value for value in values if value is not None]
        if not present:
            return None
        if len(present) != len(values):
            raise ValueError("Cannot concatenate optional condition tensors with mixed presence.")
        return torch.cat(present, dim=0)

    independent_text_state = torch.cat([bundle[0] for bundle in independent_bundles], dim=0)
    independent_text_mask = torch.cat([bundle[1] for bundle in independent_bundles], dim=0)
    independent_speaker_state = _cat_optional_tensors([bundle[2] for bundle in independent_bundles])
    independent_speaker_mask = _cat_optional_tensors([bundle[3] for bundle in independent_bundles])
    independent_caption_state = _cat_optional_tensors([bundle[4] for bundle in independent_bundles])
    independent_caption_mask = _cat_optional_tensors([bundle[5] for bundle in independent_bundles])
    independent_latent_mask = (
        None if latent_mask is None else torch.cat([latent_mask] * cfg_batch_mult, dim=0)
    )

    joint_uncond_bundle = _bundle(
        text_state=text_state_uncond,
        text_mask_val=text_mask_uncond,
        speaker_state=speaker_state_uncond,
        speaker_mask_val=speaker_mask_uncond,
        caption_state=caption_state_uncond,
        caption_mask_val=caption_mask_uncond,
    )
    if has_condition_cfg and not has_speaker_cfg and condition_token_uncond_bundle is not None:
        # speaker CFG を使わない joint CFG では、話者参照を残したまま条件 token だけを外す
        joint_uncond_bundle = _bundle(
            text_state=text_state_uncond if has_text_cfg else text_state_cond,
            text_mask_val=text_mask_uncond if has_text_cfg else text_mask_cond,
            speaker_state=condition_token_uncond_bundle[2],
            speaker_mask_val=condition_token_uncond_bundle[3],
            caption_state=caption_state_uncond if has_caption_cfg else caption_state_cond,
            caption_mask_val=caption_mask_uncond if has_caption_cfg else caption_mask_cond,
        )

    alternating_bundles: dict[
        str,
        tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor | None,
            torch.Tensor | None,
            torch.Tensor | None,
            torch.Tensor | None,
        ],
    ] = {
        "text": _bundle(
            text_state=text_state_uncond,
            text_mask_val=text_mask_uncond,
            speaker_state=speaker_state_cond,
            speaker_mask_val=speaker_mask_cond,
            caption_state=caption_state_cond,
            caption_mask_val=caption_mask_cond,
        ),
        "caption": _bundle(
            text_state=text_state_cond,
            text_mask_val=text_mask_cond,
            speaker_state=speaker_state_cond,
            speaker_mask_val=speaker_mask_cond,
            caption_state=caption_state_uncond,
            caption_mask_val=caption_mask_uncond,
        ),
    }
    if has_speaker_cfg:
        alternating_bundles["speaker"] = _bundle(
            text_state=text_state_cond,
            text_mask_val=text_mask_cond,
            speaker_state=speaker_state_uncond,
            speaker_mask_val=speaker_mask_uncond,
            caption_state=caption_state_cond,
            caption_mask_val=caption_mask_cond,
        )
    if has_condition_cfg and condition_token_uncond_bundle is not None:
        alternating_bundles["condition"] = _bundle(
            text_state=text_state_cond,
            text_mask_val=text_mask_cond,
            speaker_state=condition_token_uncond_bundle[2],
            speaker_mask_val=condition_token_uncond_bundle[3],
            caption_state=caption_state_cond,
            caption_mask_val=caption_mask_cond,
        )

    attention_plan_cond: PackedAttentionPlan | None = None
    attention_plan_cfg: PackedAttentionPlan | None = None
    attention_plan_joint_uncond: PackedAttentionPlan | None = None
    attention_plan_alternating: dict[str, PackedAttentionPlan | None] = {}
    if use_cudnn_packed_attention:
        # 潜在マスク未指定時も明示的な全有効マスクに直し、RF 反復前に添字を確定する
        effective_latent_mask = (
            latent_mask
            if latent_mask is not None
            else torch.ones(
                (batch_size, sequence_length),
                dtype=torch.bool,
                device=device,
            )
        )
        attention_plan_cond = build_packed_attention_plan(
            latent_mask=effective_latent_mask,
            text_mask=text_mask_cond,
            speaker_mask=speaker_mask_cond,
            caption_mask=caption_mask_cond,
        )
        if use_independent_cfg and cfg_batch_mult > 1:
            attention_plan_cfg = build_packed_attention_plan(
                latent_mask=effective_latent_mask.repeat(cfg_batch_mult, 1),
                text_mask=independent_text_mask,
                speaker_mask=independent_speaker_mask,
                caption_mask=independent_caption_mask,
            )
        elif use_joint_cfg and enabled_cfg_names:
            attention_plan_joint_uncond = build_packed_attention_plan(
                latent_mask=effective_latent_mask,
                text_mask=joint_uncond_bundle[1],
                speaker_mask=joint_uncond_bundle[3],
                caption_mask=joint_uncond_bundle[5],
            )
        elif use_alternating_cfg:
            for name in enabled_cfg_names:
                bundle = alternating_bundles[name]
                attention_plan_alternating[name] = build_packed_attention_plan(
                    latent_mask=effective_latent_mask,
                    text_mask=bundle[1],
                    speaker_mask=bundle[3],
                    caption_mask=bundle[5],
                )

    # Force-speaker scaling operates on projected speaker K/V, so it requires context KV caches.
    effective_use_context_kv_cache = bool(use_context_kv_cache or (speaker_kv_scale is not None))

    context_kv_cond = None
    context_kv_cfg = None
    context_kv_joint_uncond = None
    context_kv_alternating: dict[str, list[tuple[torch.Tensor, ...]]] = {}
    if effective_use_context_kv_cache:
        context_kv_cond = model.build_context_kv_cache(
            text_state=text_state_cond,
            speaker_state=speaker_state_cond,
            caption_state=caption_state_cond,
        )
        if use_independent_cfg and cfg_batch_mult > 1:
            context_kv_cfg = model.build_context_kv_cache(
                text_state=independent_text_state,
                speaker_state=independent_speaker_state,
                caption_state=independent_caption_state,
            )
        elif use_joint_cfg:
            if enabled_cfg_names:
                context_kv_joint_uncond = model.build_context_kv_cache(
                    text_state=joint_uncond_bundle[0],
                    speaker_state=joint_uncond_bundle[2],
                    caption_state=joint_uncond_bundle[4],
                )
        elif use_alternating_cfg:
            for name in enabled_cfg_names:
                bundle = alternating_bundles[name]
                context_kv_alternating[name] = model.build_context_kv_cache(
                    text_state=bundle[0],
                    speaker_state=bundle[2],
                    caption_state=bundle[4],
                )
    if speaker_kv_scale is not None:
        scale_speaker_kv_cache(
            context_kv_cache=context_kv_cond,
            scale=float(speaker_kv_scale),
            max_layers=speaker_kv_max_layers,
        )
        if context_kv_cfg is not None:
            scale_speaker_kv_cache(
                context_kv_cache=context_kv_cfg,
                scale=float(speaker_kv_scale),
                max_layers=speaker_kv_max_layers,
            )
        for cache in context_kv_alternating.values():
            scale_speaker_kv_cache(
                context_kv_cache=cache,
                scale=float(speaker_kv_scale),
                max_layers=speaker_kv_max_layers,
            )
    speaker_kv_active = speaker_kv_scale is not None

    waveex_cfg = waveex if (waveex is not None and waveex.enabled) else None
    waveex_buffer: WaveExBuffer | None = None
    waveex_ode_indices: set[int] = set()
    waveex_min_history = 0
    if waveex_cfg is not None:
        waveex_buffer = WaveExBuffer(waveex_cfg)
        # Note: do NOT seed the buffer with the initial pure-noise latent;
        # only real ODE outputs should populate the history so the wavelet
        # extrapolation operates on a proper trajectory.
        waveex_ode_indices = waveex_cfg.resolve_ode_step_indices(num_steps)
        waveex_min_history = max(2, int(waveex_cfg.history_size))

    for i in range(num_steps):
        t = t_schedule[i]
        t_next = t_schedule[i + 1]
        t_value = t_schedule_values[i]
        t_next_value = t_schedule_values[i + 1]
        # 時刻は既存の GPU Tensor をビューとして拡張し、スカラー値の CPU 経由コピーを発生させない
        tt = t.expand(batch_size)

        use_cfg = bool(enabled_cfg_names) and (cfg_min_t <= t_value <= cfg_max_t)
        is_full_ode_index = waveex_cfg is None or i in waveex_ode_indices
        use_taylor_step = (
            waveex_buffer is not None
            and not is_full_ode_index
            and len(waveex_buffer) >= waveex_min_history
        )
        if use_taylor_step:
            x_t = waveex_buffer.predict_next()
            waveex_buffer.push(x_t)
            if (
                speaker_kv_active
                and speaker_kv_min_t is not None
                and t_next_value < speaker_kv_min_t
                and t_value >= speaker_kv_min_t
            ):
                inv_scale = 1.0 / float(speaker_kv_scale)
                scale_speaker_kv_cache(
                    context_kv_cache=context_kv_cond,
                    scale=inv_scale,
                    max_layers=speaker_kv_max_layers,
                )
                if context_kv_cfg is not None:
                    scale_speaker_kv_cache(
                        context_kv_cache=context_kv_cfg,
                        scale=inv_scale,
                        max_layers=speaker_kv_max_layers,
                    )
                for cache in context_kv_alternating.values():
                    scale_speaker_kv_cache(
                        context_kv_cache=cache,
                        scale=inv_scale,
                        max_layers=speaker_kv_max_layers,
                    )
                speaker_kv_active = False
            continue

        if use_cfg:
            if use_independent_cfg:
                x_t_cfg = torch.cat([x_t] * cfg_batch_mult, dim=0).to(dtype)
                tt_cfg = tt.repeat(cfg_batch_mult)
                v_out = model.forward_with_encoded_conditions(
                    x_t=x_t_cfg,
                    t=tt_cfg,
                    text_state=independent_text_state,
                    text_mask=independent_text_mask,
                    speaker_state=independent_speaker_state,
                    speaker_mask=independent_speaker_mask,
                    caption_state=independent_caption_state,
                    caption_mask=independent_caption_mask,
                    context_kv_cache=context_kv_cfg,
                    latent_mask=independent_latent_mask,
                    packed_attention_plan=attention_plan_cfg,
                )
                chunks = v_out.chunk(cfg_batch_mult, dim=0)
                v = chunks[0]
                for name, chunk in zip(independent_names[1:], chunks[1:], strict=True):
                    v = v + cfg_scales[name] * (chunks[0] - chunk)
            else:
                v_cond = model.forward_with_encoded_conditions(
                    x_t=x_t.to(dtype),
                    t=tt,
                    text_state=text_state_cond,
                    text_mask=text_mask_cond,
                    speaker_state=speaker_state_cond,
                    speaker_mask=speaker_mask_cond,
                    caption_state=caption_state_cond,
                    caption_mask=caption_mask_cond,
                    context_kv_cache=context_kv_cond,
                    latent_mask=latent_mask,
                    packed_attention_plan=attention_plan_cond,
                )
                if use_joint_cfg:
                    if len(enabled_cfg_names) > 1:
                        joint_scales = [cfg_scales[name] for name in enabled_cfg_names]
                        if max(joint_scales) - min(joint_scales) > 1e-6:
                            raise ValueError(
                                "cfg_guidance_mode='joint' expects equal enabled guidance scales; "
                                "set matching text/speaker/caption scales or use --cfg-scale."
                            )
                    joint_scale = cfg_scales[enabled_cfg_names[0]]
                    v_uncond_joint = model.forward_with_encoded_conditions(
                        x_t=x_t.to(dtype),
                        t=tt,
                        text_state=joint_uncond_bundle[0],
                        text_mask=joint_uncond_bundle[1],
                        speaker_state=joint_uncond_bundle[2],
                        speaker_mask=joint_uncond_bundle[3],
                        caption_state=joint_uncond_bundle[4],
                        caption_mask=joint_uncond_bundle[5],
                        context_kv_cache=context_kv_joint_uncond,
                        latent_mask=latent_mask,
                        packed_attention_plan=attention_plan_joint_uncond,
                    )
                    v = v_cond + joint_scale * (v_cond - v_uncond_joint)
                elif use_alternating_cfg:
                    alt_name = enabled_cfg_names[i % len(enabled_cfg_names)]
                    alt_bundle = alternating_bundles[alt_name]
                    v_uncond_alt = model.forward_with_encoded_conditions(
                        x_t=x_t.to(dtype),
                        t=tt,
                        text_state=alt_bundle[0],
                        text_mask=alt_bundle[1],
                        speaker_state=alt_bundle[2],
                        speaker_mask=alt_bundle[3],
                        caption_state=alt_bundle[4],
                        caption_mask=alt_bundle[5],
                        context_kv_cache=context_kv_alternating.get(alt_name),
                        latent_mask=latent_mask,
                        packed_attention_plan=attention_plan_alternating.get(alt_name),
                    )
                    v = v_cond + cfg_scales[alt_name] * (v_cond - v_uncond_alt)
                else:
                    raise RuntimeError(f"Unexpected cfg_guidance_mode: {cfg_guidance_mode}")
        else:
            v = model.forward_with_encoded_conditions(
                x_t=x_t.to(dtype),
                t=tt,
                text_state=text_state_cond,
                text_mask=text_mask_cond,
                speaker_state=speaker_state_cond,
                speaker_mask=speaker_mask_cond,
                caption_state=caption_state_cond,
                caption_mask=caption_mask_cond,
                context_kv_cache=context_kv_cond,
                latent_mask=latent_mask,
                packed_attention_plan=attention_plan_cond,
            )

        # 通常の CFG 速度へ、同一潜在上で測った条件対の速度差を直接加算
        if (
            velocity_field_guidance is not None
            and velocity_field_guidance.alpha != 0.0
            and velocity_field_guidance.min_t <= t_value <= velocity_field_guidance.max_t
        ):
            target_speaker_state = (
                velocity_field_guidance.target_speaker_state
                if velocity_field_guidance.target_speaker_state is not None
                else speaker_state_cond
            )
            target_speaker_mask = (
                velocity_field_guidance.target_speaker_mask
                if velocity_field_guidance.target_speaker_mask is not None
                else speaker_mask_cond
            )
            opposite_speaker_state = (
                velocity_field_guidance.opposite_speaker_state
                if velocity_field_guidance.opposite_speaker_state is not None
                else speaker_state_cond
            )
            opposite_speaker_mask = (
                velocity_field_guidance.opposite_speaker_mask
                if velocity_field_guidance.opposite_speaker_mask is not None
                else speaker_mask_cond
            )
            target_caption_state = (
                velocity_field_guidance.target_caption_state
                if velocity_field_guidance.target_caption_state is not None
                else caption_state_cond
            )
            target_caption_mask = (
                velocity_field_guidance.target_caption_mask
                if velocity_field_guidance.target_caption_mask is not None
                else caption_mask_cond
            )
            opposite_caption_state = (
                velocity_field_guidance.opposite_caption_state
                if velocity_field_guidance.opposite_caption_state is not None
                else caption_state_cond
            )
            opposite_caption_mask = (
                velocity_field_guidance.opposite_caption_mask
                if velocity_field_guidance.opposite_caption_mask is not None
                else caption_mask_cond
            )
            target_velocity = model.forward_with_encoded_conditions(
                x_t=x_t.to(dtype),
                t=tt,
                text_state=text_state_cond,
                text_mask=text_mask_cond,
                speaker_state=target_speaker_state,
                speaker_mask=target_speaker_mask,
                caption_state=target_caption_state,
                caption_mask=target_caption_mask,
                context_kv_cache=None,
                latent_mask=latent_mask,
            )
            opposite_velocity = model.forward_with_encoded_conditions(
                x_t=x_t.to(dtype),
                t=tt,
                text_state=text_state_cond,
                text_mask=text_mask_cond,
                speaker_state=opposite_speaker_state,
                speaker_mask=opposite_speaker_mask,
                caption_state=opposite_caption_state,
                caption_mask=opposite_caption_mask,
                context_kv_cache=None,
                latent_mask=latent_mask,
            )
            v = v + velocity_field_guidance.alpha * (target_velocity - opposite_velocity)

        if rescale_k is not None and rescale_sigma is not None:
            v = temporal_score_rescale(
                v_pred=v,
                x_t=x_t,
                t=t_value,
                rescale_k=float(rescale_k),
                rescale_sigma=float(rescale_sigma),
            )

        if trajectory_observer is not None and i in observation_step_indices:
            # 実際の状態更新へ使う補正後の速度から完成潜在を作り、WaveEx の履歴更新前に観測する
            trajectory_observer.callback(
                TrajectoryObservation(
                    step_index=i,
                    t=t_value,
                    t_next=t_next_value,
                    x0_hat=rf_predict_x0(x_t=x_t, v_pred=v, t=tt),
                    latent_mask=latent_mask,
                )
            )

        if (
            speaker_kv_active
            and speaker_kv_min_t is not None
            and t_next_value < speaker_kv_min_t
            and t_value >= speaker_kv_min_t
        ):
            inv_scale = 1.0 / float(speaker_kv_scale)
            scale_speaker_kv_cache(
                context_kv_cache=context_kv_cond,
                scale=inv_scale,
                max_layers=speaker_kv_max_layers,
            )
            if context_kv_cfg is not None:
                scale_speaker_kv_cache(
                    context_kv_cache=context_kv_cfg,
                    scale=inv_scale,
                    max_layers=speaker_kv_max_layers,
                )
            for cache in context_kv_alternating.values():
                scale_speaker_kv_cache(
                    context_kv_cache=cache,
                    scale=inv_scale,
                    max_layers=speaker_kv_max_layers,
                )
            speaker_kv_active = False

        replacement_noise = None
        if trajectory_intervention is not None and i in trajectory_step_indices:
            # CFG と時間方向補正をすべて反映した速度から完成潜在を予測
            x0_hat = rf_predict_x0(x_t=x_t, v_pred=v, t=tt)
            replacement_noise = trajectory_intervention.callback(
                TrajectoryCheckpoint(
                    step_index=i,
                    t=t_value,
                    t_next=t_next_value,
                    x0_hat=x0_hat,
                )
            )
            if replacement_noise is not None:
                if replacement_noise.shape != x0_hat.shape:
                    raise ValueError(
                        "trajectory_intervention callback noise shape mismatch: "
                        f"expected {tuple(x0_hat.shape)}, got {tuple(replacement_noise.shape)}."
                    )
                replacement_noise = replacement_noise.to(device=device, dtype=dtype)

        # 分岐時は完成予測と別ノイズから次時刻を作り、次ステップで速度を再評価
        if replacement_noise is None:
            x_t = x_t + v * (t_next - t)
        else:
            tt_next = t_next.expand(batch_size)
            x_t = rf_interpolate(x0=x0_hat, noise=replacement_noise, t=tt_next)
        if waveex_buffer is not None:
            waveex_buffer.push(x_t)

    return x_t
