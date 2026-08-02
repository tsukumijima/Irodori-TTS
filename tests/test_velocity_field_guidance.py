from __future__ import annotations

import math
import unittest
from dataclasses import fields, replace
from types import SimpleNamespace
from typing import Any, cast

import torch

from irodori_tts.inference_runtime import SamplingRequest
from irodori_tts.model import EncodedConditions
from irodori_tts.rf import VelocityFieldGuidance, sample_euler_rf_cfg


class FakeVelocityModel:
    """速度場ガイダンスの契約と時間窓を小さな決定論的モデルで検証する。"""

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
        """
        ガイダンス条件ごとの KV キャッシュ構築回数を記録する。

        Args:
            **_kwargs (Any): 実モデルと同じ呼び出しを受ける未使用の条件 Tensor

        Returns:
            list[tuple[torch.Tensor, ...]]: forward へ渡す空のテスト用キャッシュ
        """

        self.context_cache_builds += 1
        return []

    def encode_conditions(self, **kwargs: Any) -> EncodedConditions:
        """テスト入力をサンプラーが使う条件タプルへ変換する。"""

        caption_state = cast(torch.Tensor, kwargs["caption_state_override"])
        caption_mask = cast(torch.Tensor, kwargs["caption_mask_override"])
        text_state = torch.zeros((1, 1, 1), dtype=self.dtype)
        text_mask = torch.ones((1, 1), dtype=torch.bool)
        return EncodedConditions(text_state, text_mask, None, None, caption_state, caption_mask)

    def forward_with_encoded_conditions(self, **kwargs: Any) -> torch.Tensor:
        """キャプション状態の平均を速度として返し、追加 forward の発火も記録する。"""

        x_t = cast(torch.Tensor, kwargs["x_t"])
        caption_state = cast(torch.Tensor, kwargs["caption_state"])
        caption_sum = float(caption_state.sum().item())
        self.caption_sums.append(caption_sum)
        return torch.full_like(x_t, caption_sum)


class VelocityFieldGuidanceTest(unittest.TestCase):
    def test_sampling_request_keeps_guidance_before_new_positional_fields(self) -> None:
        field_names = [field.name for field in fields(SamplingRequest)]
        self.assertEqual(
            field_names[-6:],
            [
                "velocity_field_guidance",
                "trajectory_observer",
                "caption_state_override",
                "caption_mask_override",
                "speaker_condition_override",
                "capture_generated_speaker_condition",
            ],
        )

    def _sample(
        self,
        guidance: VelocityFieldGuidance | None,
        *,
        model: FakeVelocityModel | None = None,
        num_steps: int = 3,
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
            use_context_kv_cache=False,
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

    def test_none_keeps_output_identical(self) -> None:
        first, _ = self._sample(None)
        second, _ = self._sample(None)

        self.assertTrue(torch.equal(first, second))

    def test_zero_alpha_keeps_output_identical(self) -> None:
        baseline, _ = self._sample(None)
        guided, model = self._sample(self._caption_guidance(alpha=0.0))

        self.assertTrue(torch.equal(baseline, guided))
        self.assertEqual(len(model.caption_sums), 3)
        self.assertEqual(model.context_cache_builds, 0)

    def test_partial_caption_pair_is_rejected(self) -> None:
        partial = VelocityFieldGuidance(
            alpha=1.0,
            target_caption_state=torch.ones((1, 2, 1), dtype=torch.float32),
        )

        with self.assertRaisesRegex(ValueError, "all four state/mask values"):
            self._sample(partial)

    def test_both_condition_pairs_are_rejected(self) -> None:
        state = torch.ones((1, 2, 1), dtype=torch.float32)
        mask = torch.ones((1, 2), dtype=torch.bool)
        guidance = replace(
            self._caption_guidance(),
            target_speaker_state=state,
            target_speaker_mask=mask,
            opposite_speaker_state=state.clone(),
            opposite_speaker_mask=mask.clone(),
        )

        with self.assertRaisesRegex(ValueError, "exactly one complete"):
            self._sample(guidance)

    def test_missing_condition_pairs_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "exactly one complete"):
            self._sample(VelocityFieldGuidance(alpha=1.0))

    def test_invalid_time_ranges_are_rejected(self) -> None:
        for min_t, max_t in ((-0.1, 0.5), (0.5, 1.1), (0.8, 0.2)):
            with self.subTest(min_t=min_t, max_t=max_t):
                with self.assertRaisesRegex(ValueError, "0 <= min_t <= max_t <= 1"):
                    self._sample(self._caption_guidance(min_t=min_t, max_t=max_t))

    def test_non_finite_alpha_is_rejected(self) -> None:
        for alpha in (math.nan, math.inf, -math.inf):
            with self.subTest(alpha=alpha):
                with self.assertRaisesRegex(ValueError, "alpha must be finite"):
                    self._sample(self._caption_guidance(alpha=alpha))

    def test_time_window_includes_both_boundaries(self) -> None:
        # サンプラーと同じ float32 の時刻を境界値に使い、丸め誤差と包含判定を分離
        first_t = float(torch.tensor(0.999, dtype=torch.float32).item())
        guidance = self._caption_guidance(min_t=first_t, max_t=first_t)
        _result, model = self._sample(guidance, num_steps=3)

        # 通常 forward 3回に加え、t=0.999 の境界上だけ target/opposite を各1回計算
        self.assertEqual(model.caption_sums.count(6.0), 1)
        self.assertEqual(model.caption_sums.count(-2.0), 1)
        self.assertEqual(len(model.caption_sums), 5)
        self.assertEqual(model.context_cache_builds, 2)


if __name__ == "__main__":
    unittest.main()
