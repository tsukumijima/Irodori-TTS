from __future__ import annotations

import unittest
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import patch

import torch

from irodori_tts.inference_runtime import SamplingRequest
from irodori_tts.model import EncodedConditions
from irodori_tts.rf import (
    TrajectoryObservation,
    TrajectoryObserver,
    sample_euler_rf_cfg,
)
from irodori_tts.waveex import WaveExConfig


class ConstantVelocityModel:
    """
    読み取り専用観測を決定論的な速度場で検証する。
    """

    def __init__(self) -> None:
        self.device = torch.device("cpu")
        self.dtype = torch.float32
        self.cfg = SimpleNamespace(
            patched_latent_dim=1,
            use_caption_condition=True,
            use_speaker_condition_resolved=False,
        )

    def encode_conditions(self, **_kwargs: Any) -> EncodedConditions:
        """
        条件を使わない最小のサンプラー入力を返す。

        Returns:
            EncodedConditions: テキスト状態とマスクを含む条件一式。
        """

        text_state = torch.zeros((1, 1, 1), dtype=self.dtype)
        text_mask = torch.ones((1, 1), dtype=torch.bool)
        caption_state = torch.zeros((1, 1, 1), dtype=self.dtype)
        caption_mask = torch.ones((1, 1), dtype=torch.bool)
        return EncodedConditions(text_state, text_mask, None, None, caption_state, caption_mask)

    def forward_with_encoded_conditions(self, **kwargs: Any) -> torch.Tensor:
        """
        全時刻で一定の速度を返す。

        Args:
            **kwargs (Any): `x_t` を含むサンプラー引数。

        Returns:
            torch.Tensor: 入力潜在と同じ形の一定速度。
        """

        velocity = torch.full_like(cast(torch.Tensor, kwargs["x_t"]), 0.25)
        latent_mask = cast(torch.Tensor | None, kwargs.get("latent_mask"))
        if latent_mask is not None:
            velocity = velocity * latent_mask.unsqueeze(-1)
        return velocity


class TrajectoryObserverTest(unittest.TestCase):
    def test_sampling_request_exposes_observer(self) -> None:
        observer = TrajectoryObserver(
            step_indices=(0,),
            callback=lambda _observation: None,
        )

        request = SamplingRequest(
            text="テスト",
            trajectory_observer=observer,
        )

        self.assertIs(request.trajectory_observer, observer)

    def _sample_encoded_conditions(
        self,
        encoded_conditions: EncodedConditions,
        *,
        caption_input_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        キャッシュ済み条件を指定して最小の RF サンプリングを実行する。

        Args:
            encoded_conditions (EncodedConditions): 検証対象の条件一式
            caption_input_ids (torch.Tensor | None): 同時指定を検証する caption ID

        Returns:
            torch.Tensor: サンプリング後の潜在
        """

        return sample_euler_rf_cfg(
            model=cast(Any, ConstantVelocityModel()),
            text_input_ids=torch.ones((1, 1), dtype=torch.long),
            text_mask=torch.ones((1, 1), dtype=torch.bool),
            ref_latent=None,
            ref_mask=None,
            sequence_length=2,
            encoded_conditions=encoded_conditions,
            caption_input_ids=caption_input_ids,
            num_steps=2,
            cfg_scale_text=0.0,
            cfg_scale_caption=1.0,
            cfg_scale_speaker=0.0,
            use_context_kv_cache=False,
            initial_noise=torch.ones((1, 2, 1), dtype=torch.float32),
        )

    def test_encoded_conditions_reject_raw_condition_overrides(self) -> None:
        """
        キャッシュ済み条件と生の条件を同時に指定した呼び出しを拒否する。
        """

        encoded_conditions = ConstantVelocityModel().encode_conditions()

        with self.assertRaisesRegex(ValueError, "cannot be combined"):
            self._sample_encoded_conditions(
                encoded_conditions,
                caption_input_ids=torch.ones((1, 1), dtype=torch.long),
            )

    def test_encoded_conditions_reject_state_mask_shape_mismatch(self) -> None:
        """
        キャッシュ済み状態とマスクの系列長不一致をモデル呼び出し前に拒否する。
        """

        encoded_conditions = ConstantVelocityModel().encode_conditions()
        encoded_conditions = EncodedConditions(
            text_state=encoded_conditions.text_state,
            text_mask=encoded_conditions.text_mask,
            ref_state=None,
            ref_mask=None,
            caption_state=encoded_conditions.caption_state,
            caption_mask=torch.ones((1, 2), dtype=torch.bool),
        )

        with self.assertRaisesRegex(ValueError, "caption shape mismatch"):
            self._sample_encoded_conditions(encoded_conditions)

    def _sample(
        self,
        observer: TrajectoryObserver | None,
        waveex: WaveExConfig | None = None,
        num_steps: int = 2,
    ) -> torch.Tensor:
        """
        読み取り専用観測を指定して最小の RF サンプリングを実行する。

        Args:
            observer (TrajectoryObserver | None): 観測設定。
            waveex (WaveExConfig | None): WaveEx 設定。
            num_steps (int): 積分ステップ数。

        Returns:
            torch.Tensor: サンプリング後の潜在。
        """

        return sample_euler_rf_cfg(
            model=cast(Any, ConstantVelocityModel()),
            text_input_ids=torch.ones((1, 1), dtype=torch.long),
            text_mask=torch.ones((1, 1), dtype=torch.bool),
            ref_latent=None,
            ref_mask=None,
            sequence_length=2,
            num_steps=num_steps,
            cfg_scale_text=0.0,
            cfg_scale_caption=1.0,
            cfg_scale_speaker=0.0,
            use_context_kv_cache=False,
            initial_noise=torch.ones((1, 2, 1), dtype=torch.float32),
            trajectory_observer=observer,
            waveex=waveex,
        )

    def test_default_sampling_does_not_read_cuda_scalars_per_step(self) -> None:
        # ステップ数を増やしても item() 回数が増えないことから、反復内の CUDA 同期再混入を検出する
        original_item = torch.Tensor.item
        item_calls = 0

        def record_item(tensor: torch.Tensor, *args: Any, **kwargs: Any) -> Any:
            nonlocal item_calls
            item_calls += 1
            return original_item(tensor, *args, **kwargs)

        with patch.object(torch.Tensor, "item", record_item):
            self._sample(None, num_steps=2)
        short_schedule_calls = item_calls
        self.assertGreater(short_schedule_calls, 0)

        item_calls = 0
        with patch.object(torch.Tensor, "item", record_item):
            self._sample(None, num_steps=5)

        self.assertEqual(item_calls, short_schedule_calls)

    def test_observer_receives_predict_x0_without_changing_output(self) -> None:
        observations: list[TrajectoryObservation] = []

        baseline = self._sample(None)
        observed = self._sample(
            TrajectoryObserver(
                step_indices=(0,),
                callback=observations.append,
            )
        )

        self.assertTrue(torch.equal(baseline, observed))
        self.assertEqual(len(observations), 1)
        self.assertEqual(observations[0].step_index, 0)
        self.assertIsNone(observations[0].latent_mask)
        expected_x0 = torch.full((1, 2, 1), 0.75025, dtype=torch.float32)
        self.assertTrue(torch.allclose(observations[0].x0_hat, expected_x0))

    def test_observer_receives_latent_mask(self) -> None:
        observations: list[TrajectoryObservation] = []
        latent_mask = torch.tensor([[True, False]])

        sample_euler_rf_cfg(
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
            latent_mask=latent_mask,
            trajectory_observer=TrajectoryObserver(
                step_indices=(0,),
                callback=observations.append,
            ),
        )

        torch.testing.assert_close(observations[0].latent_mask, latent_mask)

    def test_observer_cannot_mutate_sampler_latent_mask(self) -> None:
        """
        コールバックによる変更が後続のサンプリングへ波及しないことを確認する。
        """

        latent_mask = torch.tensor([[True, False]])

        def sample(observer: TrajectoryObserver | None) -> torch.Tensor:
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
                latent_mask=latent_mask,
                trajectory_observer=observer,
            )

        baseline = sample(None)

        def clear_observed_mask(observation: TrajectoryObservation) -> None:
            assert observation.latent_mask is not None
            observation.latent_mask.fill_(False)

        observed = sample(
            TrajectoryObserver(
                step_indices=(0,),
                callback=clear_observed_mask,
            )
        )

        torch.testing.assert_close(observed, baseline, rtol=0.0, atol=0.0)
        torch.testing.assert_close(latent_mask, torch.tensor([[True, False]]))

    def test_observer_accepts_full_waveex_step(self) -> None:
        observations: list[TrajectoryObservation] = []

        observed = self._sample(
            TrajectoryObserver(
                step_indices=(1,),
                callback=observations.append,
            ),
            WaveExConfig(
                enabled=True,
                ode_step_indices=(0, 1),
                history_size=2,
            ),
            num_steps=3,
        )

        self.assertEqual(len(observations), 1)
        self.assertEqual(observations[0].step_index, 1)
        self.assertEqual(tuple(observed.shape), (1, 2, 1))

    def test_observer_rejects_waveex_prediction_step(self) -> None:
        observer = TrajectoryObserver(
            step_indices=(1,),
            callback=lambda _observation: None,
        )

        with self.assertRaisesRegex(ValueError, "full ODE"):
            self._sample(
                observer,
                WaveExConfig(
                    enabled=True,
                    ode_step_indices=(0,),
                    history_size=2,
                ),
            )

    def test_observer_rejects_duplicate_step_indices(self) -> None:
        observer = TrajectoryObserver(
            step_indices=(0, 0),
            callback=lambda _observation: None,
        )

        with self.assertRaisesRegex(ValueError, "must not contain duplicates"):
            self._sample(observer)

    def test_observer_rejects_out_of_range_step_indices(self) -> None:
        observer = TrajectoryObserver(
            step_indices=(2,),
            callback=lambda _observation: None,
        )

        with self.assertRaisesRegex(ValueError, "must be within"):
            self._sample(observer)

    def test_sampling_rejects_non_decreasing_schedule(self) -> None:
        with (
            patch(
                "irodori_tts.rf.torch.linspace",
                return_value=torch.tensor([0.0, 0.0, 1.0]),
            ),
            self.assertRaisesRegex(ValueError, "strictly decreasing"),
        ):
            self._sample(None)

    def test_sampling_rejects_non_positive_step_count(self) -> None:
        with self.assertRaisesRegex(ValueError, "greater than zero"):
            self._sample(None, num_steps=0)


if __name__ == "__main__":
    unittest.main()
