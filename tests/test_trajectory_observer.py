from __future__ import annotations

import unittest
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import patch

import torch

from irodori_tts.inference_runtime import SamplingRequest
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
            use_caption_condition=False,
            use_speaker_condition_resolved=False,
        )

    def encode_conditions(self, **_kwargs: Any) -> tuple[torch.Tensor, ...]:
        """
        条件を使わない最小のサンプラー入力を返す。

        Returns:
            tuple[torch.Tensor, ...]: テキスト状態とマスクを含む条件一式。
        """

        text_state = torch.zeros((1, 1, 1), dtype=self.dtype)
        text_mask = torch.ones((1, 1), dtype=torch.bool)
        return text_state, text_mask, None, None, None, None

    def forward_with_encoded_conditions(self, **kwargs: Any) -> torch.Tensor:
        """
        全時刻で一定の速度を返す。

        Args:
            **kwargs (Any): `x_t` を含むサンプラー引数。

        Returns:
            torch.Tensor: 入力潜在と同じ形の一定速度。
        """

        return torch.full_like(cast(torch.Tensor, kwargs["x_t"]), 0.25)


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

    def _sample(
        self,
        observer: TrajectoryObserver | None,
        waveex: WaveExConfig | None = None,
    ) -> torch.Tensor:
        """
        読み取り専用観測を指定して最小の RF サンプリングを実行する。

        Args:
            observer (TrajectoryObserver | None): 観測設定。
            waveex (WaveExConfig | None): WaveEx 設定。

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
            num_steps=2,
            cfg_scale_text=0.0,
            cfg_scale_caption=0.0,
            cfg_scale_speaker=0.0,
            use_context_kv_cache=False,
            initial_noise=torch.ones((1, 2, 1), dtype=torch.float32),
            trajectory_observer=observer,
            waveex=waveex,
        )

    def test_default_sampling_does_not_read_cuda_scalars_per_step(self) -> None:
        # CPU テストでも Tensor.item() を禁止し、反復ごとの CUDA 同期が再混入した場合に検出する
        with patch.object(torch.Tensor, "item", return_value=True) as item_mock:
            self._sample(None)

        self.assertEqual(item_mock.call_count, 0)

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
        expected_x0 = torch.full((1, 2, 1), 0.75025, dtype=torch.float32)
        self.assertTrue(torch.allclose(observations[0].x0_hat, expected_x0))

    def test_observer_accepts_full_waveex_step(self) -> None:
        observations: list[TrajectoryObservation] = []

        observed = self._sample(
            TrajectoryObserver(
                step_indices=(0,),
                callback=observations.append,
            ),
            WaveExConfig(
                enabled=True,
                ode_step_indices=(0,),
                history_size=2,
            ),
        )

        self.assertEqual(len(observations), 1)
        self.assertEqual(observations[0].step_index, 0)
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


if __name__ == "__main__":
    unittest.main()
