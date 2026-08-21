from __future__ import annotations

import unittest
from types import SimpleNamespace
from typing import Any, cast

import torch

from irodori_tts.model import EncodedConditions
from irodori_tts.rf import VelocityFieldGuidance, sample_euler_rf_cfg


class FakeVelocityModel:
    """速度場ガイダンスの forward 回数と KV キャッシュ構築回数を記録する。"""

    def __init__(self) -> None:
        self.device = torch.device("cpu")
        self.dtype = torch.float32
        self.cfg = SimpleNamespace(
            patched_latent_dim=1,
            use_caption_condition=True,
            use_speaker_condition_resolved=False,
        )
        self.caption_sums: list[float] = []
        self.context_cache_builds = 0

    def build_context_kv_cache(self, **_kwargs: Any) -> list[tuple[torch.Tensor, ...]]:
        self.context_cache_builds += 1
        return []

    def encode_conditions(self, **kwargs: Any) -> EncodedConditions:
        caption_state = cast(torch.Tensor, kwargs["caption_state_override"])
        caption_mask = cast(torch.Tensor, kwargs["caption_mask_override"])
        text_state = torch.zeros((1, 1, 1), dtype=self.dtype)
        text_mask = torch.ones((1, 1), dtype=torch.bool)
        return EncodedConditions(text_state, text_mask, None, None, caption_state, caption_mask)

    def forward_with_encoded_conditions(self, **kwargs: Any) -> torch.Tensor:
        x_t = cast(torch.Tensor, kwargs["x_t"])
        caption_state = cast(torch.Tensor, kwargs["caption_state"])
        caption_sum = float(caption_state.sum().item())
        self.caption_sums.append(caption_sum)
        return torch.full_like(x_t, caption_sum)


class FakeSpeakerVelocityModel(FakeVelocityModel):
    def __init__(self) -> None:
        super().__init__()
        self.cfg.use_caption_condition = False
        self.cfg.use_speaker_condition_resolved = True
        self.speaker_sums: list[float] = []

    def encode_conditions(self, **_kwargs: Any) -> EncodedConditions:
        text_state = torch.zeros((1, 1, 1), dtype=self.dtype)
        text_mask = torch.ones((1, 1), dtype=torch.bool)
        speaker_state = torch.ones((1, 2, 1), dtype=self.dtype)
        speaker_mask = torch.ones((1, 2), dtype=torch.bool)
        return EncodedConditions(
            text_state,
            text_mask,
            speaker_state,
            speaker_mask,
            None,
            None,
        )

    def forward_with_encoded_conditions(self, **kwargs: Any) -> torch.Tensor:
        x_t = cast(torch.Tensor, kwargs["x_t"])
        speaker_state = cast(torch.Tensor, kwargs["speaker_state"])
        speaker_sum = float(speaker_state.sum().item())
        self.speaker_sums.append(speaker_sum)
        return torch.full_like(x_t, speaker_sum)


class VelocityFieldGuidanceTest(unittest.TestCase):
    def _sample(
        self,
        guidance: VelocityFieldGuidance | None,
        *,
        model: FakeVelocityModel | None = None,
        num_steps: int = 3,
        use_context_kv_cache: bool = False,
    ) -> tuple[torch.Tensor, FakeVelocityModel]:
        runtime_model = FakeVelocityModel() if model is None else model
        caption_state = torch.ones((1, 2, 1), dtype=torch.float32)
        caption_mask = torch.ones((1, 2), dtype=torch.bool)
        result = sample_euler_rf_cfg(
            model=cast(Any, runtime_model),
            text_input_ids=torch.ones((1, 1), dtype=torch.long),
            text_mask=torch.ones((1, 1), dtype=torch.bool),
            ref_latent=None,
            ref_mask=None,
            sequence_length=2,
            caption_state_override=caption_state,
            caption_mask_override=caption_mask,
            num_steps=num_steps,
            cfg_scale_text=0.0,
            cfg_scale_caption=0.0,
            cfg_scale_speaker=0.0,
            use_context_kv_cache=use_context_kv_cache,
            initial_noise=torch.ones((1, 2, 1), dtype=torch.float32),
            velocity_field_guidance=guidance,
        )
        return result, runtime_model

    def _caption_guidance(
        self,
        alpha: float = 1.0,
        *,
        min_t: float = 0.0,
        max_t: float = 1.0,
    ) -> VelocityFieldGuidance:
        target_state = torch.full((1, 2, 1), 3.0, dtype=torch.float32)
        opposite_state = torch.full((1, 2, 1), -1.0, dtype=torch.float32)
        mask = torch.ones((1, 2), dtype=torch.bool)
        return VelocityFieldGuidance(
            alpha=alpha,
            min_t=min_t,
            max_t=max_t,
            target_caption_state=target_state,
            target_caption_mask=mask,
            opposite_caption_state=opposite_state,
            opposite_caption_mask=mask.clone(),
        )

    def test_zero_alpha_keeps_output_identical(self) -> None:
        baseline, _ = self._sample(None)
        guided, model = self._sample(self._caption_guidance(alpha=0.0))

        self.assertTrue(torch.equal(baseline, guided))
        self.assertEqual(len(model.caption_sums), 3)
        self.assertEqual(model.context_cache_builds, 0)

    def test_speaker_guidance_runs_target_and_opposite_forwards(self) -> None:
        model = FakeSpeakerVelocityModel()
        mask = torch.ones((1, 2), dtype=torch.bool)
        guidance = VelocityFieldGuidance(
            alpha=1.0,
            target_speaker_state=torch.full((1, 2, 1), 3.0),
            target_speaker_mask=mask,
            opposite_speaker_state=torch.full((1, 2, 1), -1.0),
            opposite_speaker_mask=mask.clone(),
        )

        self._sample(guidance, model=cast(Any, model))

        self.assertEqual(model.speaker_sums.count(6.0), 3)
        self.assertEqual(model.speaker_sums.count(-2.0), 3)

    def test_time_window_includes_both_boundaries(self) -> None:
        schedule = (1.0 - torch.linspace(0.0, 1.0, 4, dtype=torch.float32)) * 0.999
        first_t = float(schedule[0].item())
        second_t = float(schedule[1].item())
        guidance = self._caption_guidance(min_t=second_t, max_t=first_t)
        _result, model = self._sample(guidance, num_steps=3)

        self.assertEqual(model.caption_sums.count(6.0), 2)
        self.assertEqual(model.caption_sums.count(-2.0), 2)
        self.assertEqual(len(model.caption_sums), 7)
        self.assertEqual(model.context_cache_builds, 0)

    def test_guidance_reuses_context_cache_when_enabled(self) -> None:
        guidance = self._caption_guidance()
        _result, model = self._sample(guidance, use_context_kv_cache=True)

        self.assertEqual(model.context_cache_builds, 3)
        self.assertEqual(model.caption_sums.count(6.0), 3)
        self.assertEqual(model.caption_sums.count(-2.0), 3)


if __name__ == "__main__":
    unittest.main()
