from __future__ import annotations

import unittest

import torch

from irodori_tts.config import ModelConfig
from irodori_tts.model import TextToLatentRFDiT


class ConditionDropoutOrthogonalityTest(unittest.TestCase):
    def _model(self) -> TextToLatentRFDiT:
        cfg = ModelConfig(
            latent_dim=4,
            latent_patch_size=1,
            model_dim=8,
            num_layers=0,
            num_heads=2,
            mlp_ratio=2.0,
            text_vocab_size=16,
            text_dim=8,
            text_layers=0,
            text_heads=2,
            use_speaker_condition=True,
            speaker_dim=8,
            speaker_layers=0,
            speaker_heads=2,
            speaker_patch_size=1,
            speaker_condition_vocab_size=8,
            timestep_embed_dim=8,
            adaln_rank=4,
        )
        model = TextToLatentRFDiT(cfg)
        model.eval()
        assert model.speaker_condition_embedding is not None
        with torch.no_grad():
            # token ごとの差が目視でも追いやすいよう、embedding を単調な値で固定
            token_values = torch.arange(
                cfg.speaker_condition_vocab_size * cfg.speaker_dim,
                dtype=torch.float32,
            )
            model.speaker_condition_embedding.weight.copy_(
                token_values.reshape(cfg.speaker_condition_vocab_size, cfg.speaker_dim) / 100.0
            )
        return model

    def _duration_model(self) -> TextToLatentRFDiT:
        torch.manual_seed(1234)
        cfg = ModelConfig(
            latent_dim=4,
            latent_patch_size=1,
            model_dim=8,
            num_layers=1,
            num_heads=2,
            mlp_ratio=2.0,
            text_vocab_size=16,
            text_dim=8,
            text_layers=0,
            text_heads=2,
            use_speaker_condition=True,
            speaker_dim=8,
            speaker_layers=0,
            speaker_heads=2,
            speaker_patch_size=1,
            speaker_condition_vocab_size=8,
            timestep_embed_dim=8,
            adaln_rank=4,
            use_duration_predictor=True,
            duration_aux_dim=3,
            duration_hidden_dim=8,
            duration_layers=1,
            duration_dropout=0.0,
            duration_attention_heads=2,
        )
        model = TextToLatentRFDiT(cfg)
        model.eval()
        assert model.speaker_condition_embedding is not None
        with torch.no_grad():
            # 条件 token の差分が DiT 出力へ伝わったかを、乱数初期化に依存せず追える値へ固定
            token_values = torch.arange(
                cfg.speaker_condition_vocab_size * cfg.speaker_dim,
                dtype=torch.float32,
            )
            model.speaker_condition_embedding.weight.copy_(
                token_values.reshape(cfg.speaker_condition_vocab_size, cfg.speaker_dim) / 10.0
            )
            # 本体は学習初期化で出力層がゼロなので、テストでは speaker context の差を観測できる重みにする
            output_values = torch.arange(
                cfg.patched_latent_dim * cfg.model_dim,
                dtype=torch.float32,
            )
            model.out_proj.weight.copy_(
                output_values.reshape(cfg.patched_latent_dim, cfg.model_dim) / 100.0
            )
        return model

    def _reference_inputs(self) -> tuple[torch.Tensor, torch.Tensor]:
        ref_latent = torch.tensor(
            [
                [
                    [0.1, 0.2, 0.3, 0.4],
                    [0.5, 0.6, 0.7, 0.8],
                    [0.9, 1.0, 1.1, 1.2],
                ]
            ],
            dtype=torch.float32,
        )
        ref_mask = torch.ones((1, 3), dtype=torch.bool)
        return ref_latent, ref_mask

    def _encode(
        self,
        *,
        is_speaker_dropped: bool,
        is_token_dropped: bool,
        speaker_uncond_mode: str = "mask",
    ) -> tuple[TextToLatentRFDiT, torch.Tensor, torch.Tensor]:
        model = self._model()
        ref_latent, ref_mask = self._reference_inputs()
        token_ids = torch.tensor([[2, 5]], dtype=torch.long)
        token_dropout_mask = torch.full((1, 2), is_token_dropped, dtype=torch.bool)

        # `encode_speaker_condition()` の戻り値だけを見て、モデル入力直前の列構造を固定
        state, mask = model.encode_speaker_condition(
            batch_size=1,
            dtype=torch.float32,
            device=torch.device("cpu"),
            ref_latent=ref_latent,
            ref_mask=ref_mask,
            speaker_uncond_mode=speaker_uncond_mode,
            speaker_condition_dropout=torch.tensor([is_speaker_dropped], dtype=torch.bool),
            condition_token_ids=token_ids,
            condition_token_dropout_mask=token_dropout_mask,
        )
        return model, state, mask

    def _forward_duration_path(
        self,
        model: TextToLatentRFDiT,
        *,
        condition_token_ids: torch.Tensor | None,
    ) -> torch.Tensor:
        ref_latent, ref_mask = self._reference_inputs()
        x_t = torch.tensor(
            [
                [
                    [0.10, 0.20, 0.30, 0.40],
                    [0.50, 0.60, 0.70, 0.80],
                ]
            ],
            dtype=torch.float32,
        )
        text_input_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)

        # duration_features あり分岐を通し、DiT 直前の2回目 speaker dropout を全サンプルに適用
        v_pred, _duration_pred = model(
            x_t=x_t,
            t=torch.tensor([0.25], dtype=torch.float32),
            text_input_ids=text_input_ids,
            text_mask=torch.ones_like(text_input_ids, dtype=torch.bool),
            ref_latent=ref_latent,
            ref_mask=ref_mask,
            speaker_condition_dropout=torch.tensor([True], dtype=torch.bool),
            condition_token_ids=condition_token_ids,
            duration_features=torch.tensor([[0.1, 0.2, 0.3]], dtype=torch.float32),
            duration_has_speaker=torch.tensor([True], dtype=torch.bool),
        )
        return v_pred

    def _expected_token_state(self, model: TextToLatentRFDiT) -> torch.Tensor:
        assert model.speaker_condition_embedding is not None
        token_ids = torch.tensor([[2, 5]], dtype=torch.long)
        return model.speaker_condition_embedding(token_ids)

    def test_speaker_drop_keeps_condition_token_mask(self) -> None:
        _model, _state, mask = self._encode(
            is_speaker_dropped=True,
            is_token_dropped=False,
        )

        # reference 部分は mean token 1列 + reference 3列で、条件 token は末尾2列
        self.assertFalse(mask[:, :4].any().item())
        self.assertTrue(mask[:, 4:].all().item())

    def test_token_dropout_keeps_reference_mask(self) -> None:
        _model, _state, mask = self._encode(
            is_speaker_dropped=False,
            is_token_dropped=True,
        )

        # token dropout は条件 token だけを落とし、reference 由来の列は残す
        self.assertTrue(mask[:, :4].all().item())
        self.assertFalse(mask[:, 4:].any().item())

    def test_both_dropouts_make_all_speaker_tokens_unconditional(self) -> None:
        _model, _state, mask = self._encode(
            is_speaker_dropped=True,
            is_token_dropped=True,
        )

        self.assertFalse(mask.any().item())

    def test_no_dropout_keeps_all_speaker_tokens_enabled(self) -> None:
        _model, _state, mask = self._encode(
            is_speaker_dropped=False,
            is_token_dropped=False,
        )

        self.assertTrue(mask.all().item())

    def test_noise_mode_does_not_overwrite_condition_token_state(self) -> None:
        torch.manual_seed(1234)
        model, state, mask = self._encode(
            is_speaker_dropped=True,
            is_token_dropped=False,
            speaker_uncond_mode="noise",
        )
        expected_token_state = self._expected_token_state(model)

        # noise モードでも置換対象は reference 部分だけで、append 後の条件 token は embedding の値を保つ
        self.assertTrue(mask[:, :4].all().item())
        self.assertTrue(mask[:, 4:].all().item())
        self.assertTrue(torch.allclose(state[:, 4:], expected_token_state))

    def test_condition_token_state_is_independent_from_speaker_dropout(self) -> None:
        dropped_model, dropped_state, _dropped_mask = self._encode(
            is_speaker_dropped=True,
            is_token_dropped=False,
        )
        kept_model, kept_state, _kept_mask = self._encode(
            is_speaker_dropped=False,
            is_token_dropped=False,
        )

        # speaker dropout の有無だけを変えても、条件 token 部分の state は同じ値になる
        self.assertTrue(
            torch.allclose(dropped_state[:, 4:], self._expected_token_state(dropped_model))
        )
        self.assertTrue(torch.allclose(kept_state[:, 4:], self._expected_token_state(kept_model)))
        self.assertTrue(torch.allclose(dropped_state[:, 4:], kept_state[:, 4:]))

    def test_second_dropout_keeps_trailing_condition_token_mask(self) -> None:
        model = self._model()
        speaker_state = torch.ones((2, 5, 8), dtype=torch.float32)
        speaker_mask = torch.ones((2, 5), dtype=torch.bool)

        _state, mask = model._apply_speaker_condition_dropout(
            speaker_state=speaker_state,
            speaker_mask=speaker_mask,
            dropout_mask=torch.tensor([True, False], dtype=torch.bool),
            uncond_state=None,
            uncond_mask=None,
            uncond_mode="mask",
            num_trailing_condition_tokens=2,
        )

        # dropout 対象サンプルは reference 側3列だけを落とし、末尾の条件 token 2列を残す
        self.assertEqual(mask[0].tolist(), [False, False, False, True, True])
        self.assertEqual(mask[1].tolist(), [True, True, True, True, True])

    def test_second_dropout_noise_mode_keeps_trailing_condition_tokens(self) -> None:
        model = self._model()
        speaker_state = torch.arange(2 * 5 * 8, dtype=torch.float32).reshape(2, 5, 8)
        speaker_mask = torch.tensor(
            [
                [True, True, True, True, False],
                [True, False, True, True, True],
            ],
            dtype=torch.bool,
        )
        uncond_state = torch.full_like(speaker_state, -1.0)
        uncond_mask = torch.zeros_like(speaker_mask)

        state, mask = model._apply_speaker_condition_dropout(
            speaker_state=speaker_state,
            speaker_mask=speaker_mask,
            dropout_mask=torch.tensor([True, False], dtype=torch.bool),
            uncond_state=uncond_state,
            uncond_mask=uncond_mask,
            uncond_mode="noise",
            num_trailing_condition_tokens=2,
        )

        # noise モードでも置換範囲は reference 側に閉じ、条件 token の値と mask は入力値を保つ
        self.assertTrue(torch.allclose(state[0, :3], uncond_state[0, :3]))
        self.assertTrue(torch.allclose(state[0, 3:], speaker_state[0, 3:]))
        self.assertEqual(mask[0].tolist(), [False, False, False, True, False])
        self.assertTrue(torch.allclose(state[1], speaker_state[1]))
        self.assertEqual(mask[1].tolist(), speaker_mask[1].tolist())

    def test_duration_predictor_path_keeps_condition_tokens_visible_to_dit(self) -> None:
        model = self._duration_model()

        without_token = self._forward_duration_path(
            model,
            condition_token_ids=None,
        )
        with_token = self._forward_duration_path(
            model,
            condition_token_ids=torch.tensor([[2, 5]], dtype=torch.long),
        )
        with_other_token = self._forward_duration_path(
            model,
            condition_token_ids=torch.tensor([[3, 6]], dtype=torch.long),
        )

        # 修正前は2回目の speaker dropout が条件 token も落とすため、token なしと完全一致していた
        self.assertFalse(torch.allclose(with_token, without_token))
        self.assertFalse(torch.allclose(with_other_token, with_token))


if __name__ == "__main__":
    unittest.main()
