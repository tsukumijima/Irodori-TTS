from __future__ import annotations

import unittest

import torch

from irodori_tts.inference_runtime import (
    ContextCapacityExceededError,
    InferenceRuntime,
)
from irodori_tts.model import EncodedConditions


class EncodedContextCapacityTest(unittest.TestCase):
    def _runtime(self, capacity: int | None) -> InferenceRuntime:
        # モデルロードを伴わない単体試験では、容量検査に必要な状態だけを設定する
        runtime = InferenceRuntime.__new__(InferenceRuntime)
        runtime.set_max_encoded_context_tokens(capacity)
        return runtime

    def test_capacity_configuration_rejects_non_positive_values(self) -> None:
        runtime = self._runtime(None)

        with self.assertRaisesRegex(ValueError, "greater than zero"):
            runtime.set_max_encoded_context_tokens(0)

    def _conditions(
        self,
        *,
        text_mask: torch.Tensor,
        speaker_mask: torch.Tensor | None,
        caption_mask: torch.Tensor | None,
    ) -> EncodedConditions:
        # 容量検査は state の値に依存しないため、mask と系列長が対応する最小 Tensor で条件を組み立てる
        batch_size = int(text_mask.shape[0])
        text_state = torch.zeros((batch_size, text_mask.shape[1], 1))
        speaker_state = (
            None if speaker_mask is None else torch.zeros((batch_size, speaker_mask.shape[1], 1))
        )
        caption_state = (
            None if caption_mask is None else torch.zeros((batch_size, caption_mask.shape[1], 1))
        )
        return EncodedConditions(
            text_state=text_state,
            text_mask=text_mask,
            ref_state=speaker_state,
            ref_mask=speaker_mask,
            caption_state=caption_state,
            caption_mask=caption_mask,
        )

    def test_capacity_boundary_accepts_864_and_rejects_865(self) -> None:
        runtime = self._runtime(864)

        # Speaker Inversion は全トークンが有効なので、本文とキャプションを加えた合計で境界を検証する
        runtime._validate_encoded_context_capacity(
            self._conditions(
                text_mask=torch.ones((1, 64), dtype=torch.bool),
                speaker_mask=torch.ones((1, 768), dtype=torch.bool),
                caption_mask=torch.ones((1, 32), dtype=torch.bool),
            )
        )
        with self.assertRaises(ContextCapacityExceededError) as raised:
            runtime._validate_encoded_context_capacity(
                self._conditions(
                    text_mask=torch.ones((1, 65), dtype=torch.bool),
                    speaker_mask=torch.ones((1, 768), dtype=torch.bool),
                    caption_mask=torch.ones((1, 32), dtype=torch.bool),
                )
            )

        self.assertEqual(raised.exception.actual, 865)
        self.assertEqual(raised.exception.capacity, 864)
        self.assertEqual(raised.exception.text_tokens, 65)
        self.assertEqual(raised.exception.speaker_tokens, 768)
        self.assertEqual(raised.exception.caption_tokens, 32)

    def test_no_ref_does_not_consume_speaker_capacity(self) -> None:
        runtime = self._runtime(3)

        # no_ref のダミー系列は物理長があっても全 mask False のため、話者トークンを消費しない
        runtime._validate_encoded_context_capacity(
            self._conditions(
                text_mask=torch.ones((1, 2), dtype=torch.bool),
                speaker_mask=torch.zeros((1, 4), dtype=torch.bool),
                caption_mask=torch.ones((1, 1), dtype=torch.bool),
            )
        )

    def test_normal_reference_ignores_false_padding(self) -> None:
        runtime = self._runtime(5)

        # 固定長へ埋めた通常参照では、有効な平均・時系列トークンだけを数え、後段の False パディングを除外する
        runtime._validate_encoded_context_capacity(
            self._conditions(
                text_mask=torch.tensor([[True, True]]),
                speaker_mask=torch.tensor([[True, True, False, False, False, False]]),
                caption_mask=torch.tensor([[True]]),
            )
        )

    def test_none_capacity_preserves_unbounded_eager_behavior(self) -> None:
        runtime = self._runtime(None)

        # 通常の eager runtime は製品エンジンの固定容量を持たないため、大きな条件も従来どおり受理する
        runtime._validate_encoded_context_capacity(
            self._conditions(
                text_mask=torch.ones((1, 1024), dtype=torch.bool),
                speaker_mask=torch.ones((1, 1024), dtype=torch.bool),
                caption_mask=None,
            )
        )


if __name__ == "__main__":
    unittest.main()
