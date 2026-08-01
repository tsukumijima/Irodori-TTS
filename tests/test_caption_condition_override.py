from __future__ import annotations

import unittest
from types import SimpleNamespace
from typing import Any, cast

import torch

from irodori_tts.inference_runtime import InferenceRuntime, SamplingRequest


class CaptionConditionOverrideTest(unittest.TestCase):
    def _runtime(self) -> InferenceRuntime:
        runtime = InferenceRuntime.__new__(InferenceRuntime)
        runtime.model_cfg = cast(Any, SimpleNamespace(use_caption_condition=True))
        runtime.model = cast(
            Any,
            SimpleNamespace(
                caption_encoder=object(),
                caption_norm=object(),
            ),
        )
        runtime.model_device = torch.device("cpu")
        return runtime

    def _load(
        self,
        request: SamplingRequest,
        *,
        batch_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
        runtime = self._runtime()
        messages: list[str] = []

        # 上書き経路ではトークン ID と通常キャッシュを使用しない
        state, mask = runtime._load_cached_caption_condition(
            req=request,
            resolved_condition_tokens=None,
            lora_adapter=None,
            caption_text="テスト",
            caption_ids=torch.ones((batch_size, 2), dtype=torch.long),
            caption_mask=torch.ones((batch_size, 2), dtype=torch.bool),
            caption_max_len=2,
            batch_size=batch_size,
            messages=messages,
        )
        return state, mask, messages

    def test_caption_condition_override_preserves_single_batch(self) -> None:
        source_state = torch.arange(6, dtype=torch.float32).reshape(1, 2, 3)
        source_mask = torch.tensor([[True, False]])

        state, mask, messages = self._load(
            SamplingRequest(
                text="テスト",
                caption_state_override=source_state,
                caption_mask_override=source_mask,
            ),
            batch_size=1,
        )

        torch.testing.assert_close(state, source_state)
        torch.testing.assert_close(mask, source_mask)
        self.assertIn("info: using request caption conditioning override.", messages)

    def test_caption_condition_override_expands_to_requested_batch(self) -> None:
        source_state = torch.arange(6, dtype=torch.float32).reshape(1, 2, 3)
        source_mask = torch.tensor([[True, False]])

        state, mask, _ = self._load(
            SamplingRequest(
                text="テスト",
                caption_state_override=source_state,
                caption_mask_override=source_mask,
            ),
            batch_size=3,
        )

        self.assertEqual(tuple(state.shape), (3, 2, 3))
        self.assertEqual(tuple(mask.shape), (3, 2))
        for batch_index in range(3):
            torch.testing.assert_close(state[batch_index], source_state[0])
            torch.testing.assert_close(mask[batch_index], source_mask[0])

    def test_caption_condition_override_requires_state_and_mask(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be specified together"):
            self._load(
                SamplingRequest(
                    text="テスト",
                    caption_state_override=torch.ones((1, 2, 3)),
                ),
                batch_size=1,
            )
        with self.assertRaisesRegex(ValueError, "must be specified together"):
            self._load(
                SamplingRequest(
                    text="テスト",
                    caption_mask_override=torch.ones((1, 2), dtype=torch.bool),
                ),
                batch_size=1,
            )


if __name__ == "__main__":
    unittest.main()
