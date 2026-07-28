from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from irodori_tts.config import ModelConfig
from irodori_tts.inference_runtime import InferenceRuntime, SamplingRequest, resolve_cfg_scales


class ConditionTokenRuntimeTest(unittest.TestCase):
    def _runtime(self) -> InferenceRuntime:
        runtime = InferenceRuntime.__new__(InferenceRuntime)
        runtime.model_cfg = ModelConfig(use_speaker_condition=True)
        runtime._condition_vocabulary_cache = {}
        return runtime

    def _adapter_dir(self, tmp_path: Path) -> str:
        adapter_dir = tmp_path / "adapter"
        adapter_dir.mkdir()
        metadata = {
            "condition_vocabulary": {
                "token_to_id": {
                    "sex:female": 0,
                    "style:囁き": 16,
                },
                "sha256": "vocab-hash",
            },
            "condition_vocabulary_hash": "vocab-hash",
        }
        (adapter_dir / "irodori_lora_metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False),
            encoding="utf-8",
        )
        return str(adapter_dir)

    def test_resolve_condition_tokens(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            runtime = self._runtime()
            adapter_dir = self._adapter_dir(Path(temp_dir))

            resolved = runtime._resolve_condition_tokens(
                SamplingRequest(
                    text="テスト",
                    lora_adapter=adapter_dir,
                    speaker_condition_tokens=["sex:female", "style:囁き"],
                    condition_token_scales={"style:囁き": 0.5},
                ),
                lora_adapter=adapter_dir,
            )

        self.assertIsNotNone(resolved)
        assert resolved is not None
        self.assertEqual(resolved.token_ids, (0, 16))
        self.assertEqual(resolved.token_scales, (1.0, 0.5))
        self.assertEqual(resolved.vocabulary_hash, "vocab-hash")

    def test_unknown_condition_token_is_error(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            runtime = self._runtime()
            adapter_dir = self._adapter_dir(Path(temp_dir))

            with self.assertRaisesRegex(ValueError, "Unknown speaker condition token"):
                runtime._resolve_condition_tokens(
                    SamplingRequest(
                        text="テスト",
                        lora_adapter=adapter_dir,
                        speaker_condition_tokens=["style:missing"],
                    ),
                    lora_adapter=adapter_dir,
                )

    def test_condition_scale_requires_listed_token(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            runtime = self._runtime()
            adapter_dir = self._adapter_dir(Path(temp_dir))

            with self.assertRaisesRegex(ValueError, "not present in speaker_condition_tokens"):
                runtime._resolve_condition_tokens(
                    SamplingRequest(
                        text="テスト",
                        lora_adapter=adapter_dir,
                        speaker_condition_tokens=["sex:female"],
                        condition_token_scales={"style:囁き": 0.5},
                    ),
                    lora_adapter=adapter_dir,
                )

    def test_cfg_scale_condition_is_disabled_without_tokens(self) -> None:
        scales = resolve_cfg_scales(
            cfg_guidance_mode="independent",
            cfg_scale_text=1.0,
            cfg_scale_caption=0.0,
            cfg_scale_speaker=0.0,
            cfg_scale_condition=2.0,
            cfg_scale=None,
            use_caption_condition=False,
            use_speaker_condition=True,
            use_token_condition=False,
        )

        self.assertEqual(scales[:4], (1.0, 0.0, 0.0, 0.0))
        self.assertTrue(any("condition token guidance" in message for message in scales[4]))

    def test_cfg_scale_condition_must_match_joint_scale(self) -> None:
        with self.assertRaisesRegex(ValueError, "cfg_guidance_mode='joint'"):
            resolve_cfg_scales(
                cfg_guidance_mode="joint",
                cfg_scale_text=1.0,
                cfg_scale_caption=0.0,
                cfg_scale_speaker=0.0,
                cfg_scale_condition=2.0,
                cfg_scale=None,
                use_caption_condition=False,
                use_speaker_condition=True,
                use_token_condition=True,
            )


if __name__ == "__main__":
    unittest.main()
