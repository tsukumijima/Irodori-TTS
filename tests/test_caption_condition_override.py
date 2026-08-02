from __future__ import annotations

import threading
import unittest
from collections import OrderedDict
from types import SimpleNamespace
from typing import Any, cast

import torch

from irodori_tts.inference_runtime import InferenceRuntime, SamplingRequest
from irodori_tts.model import TextToLatentRFDiT


class RecordingCaptionEncoder:
    """事前学習済みバックボーンを受け取ったことを記録するテスト用エンコーダー。"""

    def __init__(self) -> None:
        self.backbone: object | None = None

    def __call__(
        self,
        backbone: object,
        input_ids: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        self.backbone = backbone
        return input_ids.unsqueeze(-1).to(dtype=torch.float32) * mask.unsqueeze(-1)


class RecordingCaptionModel:
    """公開キャプション API を実モデルと同じ実装で呼ぶテスト用モデル。"""

    def __init__(self) -> None:
        self.cfg = SimpleNamespace(use_caption_condition=True)
        self.caption_encoder: object = object()
        self.caption_norm: object = object()
        self.pretrained_text_backbone: object | None = None

    def encode_caption_condition(
        self,
        *,
        input_ids: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        return TextToLatentRFDiT.encode_caption_condition(
            cast(Any, self),
            input_ids=input_ids,
            mask=mask,
        )


class RecordingCaptionTokenizer:
    """入力文と最大長を記録するテスト用トークナイザー。"""

    def __init__(self) -> None:
        self.captions: list[str] = []
        self.max_length: int | None = None

    def batch_encode(
        self,
        captions: list[str],
        *,
        max_length: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.captions = captions
        self.max_length = max_length
        return torch.tensor([[3, 4]]), torch.tensor([[True, True]])


class CaptionConditionOverrideTest(unittest.TestCase):
    def _runtime(self) -> InferenceRuntime:
        runtime = InferenceRuntime.__new__(InferenceRuntime)
        runtime.model_cfg = cast(Any, SimpleNamespace(use_caption_condition=True))
        runtime.model = cast(Any, RecordingCaptionModel())
        runtime.model_device = torch.device("cpu")
        runtime._model_dtype = torch.float32
        runtime._caption_condition_cache = OrderedDict()
        runtime._caption_condition_cache_max_entries = 4
        runtime.caption_tokenizer = None
        runtime.default_caption_max_len = 16
        runtime._infer_lock = threading.Lock()
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

    def test_caption_cache_uses_pretrained_backbone(self) -> None:
        runtime = self._runtime()
        model = cast(RecordingCaptionModel, runtime.model)
        encoder = RecordingCaptionEncoder()
        backbone = object()
        model.caption_encoder = encoder
        model.caption_norm = torch.nn.Identity()
        model.pretrained_text_backbone = backbone
        state, mask = runtime._load_cached_caption_condition(
            req=SamplingRequest(text="テスト"),
            lora_adapter=None,
            caption_text="落ち着いた声",
            caption_ids=torch.tensor([[1, 2]], dtype=torch.long),
            caption_mask=torch.tensor([[True, True]]),
            caption_max_len=2,
            batch_size=1,
            messages=[],
        )

        self.assertIs(encoder.backbone, backbone)
        self.assertEqual(tuple(state.shape), (1, 2, 1))
        torch.testing.assert_close(mask, torch.tensor([[True, True]]))

    def test_public_caption_encoding_owns_tokenization_and_model_dispatch(self) -> None:
        runtime = self._runtime()
        model = cast(RecordingCaptionModel, runtime.model)
        encoder = RecordingCaptionEncoder()
        tokenizer = RecordingCaptionTokenizer()
        backbone = object()
        model.caption_encoder = encoder
        model.caption_norm = torch.nn.Identity()
        model.pretrained_text_backbone = backbone
        runtime.caption_tokenizer = cast(Any, tokenizer)

        condition = runtime.encode_caption_condition(" 落ち着いた声 ", max_length=8)

        self.assertEqual(tokenizer.captions, ["落ち着いた声"])
        self.assertEqual(tokenizer.max_length, 8)
        self.assertIs(encoder.backbone, backbone)
        self.assertEqual(tuple(condition.state.shape), (1, 2, 1))
        torch.testing.assert_close(condition.mask, torch.tensor([[True, True]]))


if __name__ == "__main__":
    unittest.main()
