from __future__ import annotations

import unittest
from dataclasses import fields
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import patch

import torch

from irodori_tts.inference_runtime import SamplingRequest
from irodori_tts.model import EncodedConditions
from irodori_tts.rf import (
    TrajectoryCheckpoint,
    TrajectoryIntervention,
    sample_euler_rf_cfg,
)


class ConstantVelocityModel:
    """途中完成予測と再ノイズ分岐を決定論的な速度場で検証する。"""

    def __init__(self) -> None:
        self.device = torch.device("cpu")
        self.dtype = torch.float32
        self.cfg = SimpleNamespace(
            patched_latent_dim=1,
            use_caption_condition=False,
            use_speaker_condition_resolved=False,
        )

    def encode_conditions(self, **_kwargs: Any) -> EncodedConditions:
        """条件を使わない最小のサンプラー入力を返す。"""

        text_state = torch.zeros((1, 1, 1), dtype=self.dtype)
        text_mask = torch.ones((1, 1), dtype=torch.bool)
        return text_state, text_mask, None, None, None, None

    def forward_with_encoded_conditions(self, **kwargs: Any) -> torch.Tensor:
        """全時刻で一定の速度を返す。"""

        return torch.full_like(cast(torch.Tensor, kwargs["x_t"]), 0.25)


class TrajectoryInterventionTest(unittest.TestCase):
    def _sample(self, intervention: TrajectoryIntervention | None) -> torch.Tensor:
        return sample_euler_rf_cfg(
            model=cast(Any, ConstantVelocityModel()),
            text_input_ids=torch.ones((1, 1), dtype=torch.long),
            text_mask=torch.ones((1, 1), dtype=torch.bool),
            ref_latent=None,
            ref_mask=None,
            sequence_length=2,
            num_steps=2,
            cfg_scale_text=0.0,
            cfg_scale_caption=0.0,
            cfg_scale_speaker=0.0,
            use_context_kv_cache=False,
            initial_noise=torch.ones((1, 2, 1), dtype=torch.float32),
            trajectory_intervention=intervention,
        )

    def test_sampling_request_keeps_intervention_before_new_positional_fields(self) -> None:
        field_names = [field.name for field in fields(SamplingRequest)]
        self.assertEqual(
            field_names[-7:],
            [
                "velocity_field_guidance",
                "trajectory_intervention",
                "trajectory_observer",
                "caption_state_override",
                "caption_mask_override",
                "speaker_condition_override",
                "capture_generated_speaker_condition",
            ],
        )

    def test_default_sampling_does_not_read_cuda_scalars_per_step(self) -> None:
        # CPU テストでも Tensor.item() の呼び出し自体を禁止し、CUDA 同期の再混入を検出
        with patch.object(torch.Tensor, "item", return_value=True) as item_mock:
            self._sample(None)

        self.assertEqual(item_mock.call_count, 0)

    def test_observer_receives_predict_x0_without_changing_output(self) -> None:
        checkpoints: list[TrajectoryCheckpoint] = []

        def observe(checkpoint: TrajectoryCheckpoint) -> None:
            checkpoints.append(checkpoint)
            return None

        baseline = self._sample(None)
        observed = self._sample(TrajectoryIntervention(step_indices=(0,), callback=observe))

        self.assertTrue(torch.equal(baseline, observed))
        self.assertEqual(len(checkpoints), 1)
        self.assertEqual(checkpoints[0].step_index, 0)
        expected_x0 = torch.full((1, 2, 1), 0.75025, dtype=torch.float32)
        self.assertTrue(torch.allclose(checkpoints[0].x0_hat, expected_x0))

    def test_returned_noise_reconstructs_next_time_state(self) -> None:
        def branch(checkpoint: TrajectoryCheckpoint) -> torch.Tensor:
            return torch.zeros_like(checkpoint.x0_hat)

        baseline = self._sample(None)
        branched = self._sample(TrajectoryIntervention(step_indices=(0,), callback=branch))

        self.assertFalse(torch.equal(baseline, branched))
        self.assertTrue(torch.allclose(branched, torch.full_like(branched, 0.25062507)))

    def test_invalid_step_is_rejected_before_sampling(self) -> None:
        def observe(_checkpoint: TrajectoryCheckpoint) -> None:
            return None

        intervention = TrajectoryIntervention(step_indices=(2,), callback=observe)
        with self.assertRaisesRegex(ValueError, "within"):
            self._sample(intervention)


if __name__ == "__main__":
    unittest.main()
