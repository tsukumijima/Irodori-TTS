from __future__ import annotations

import hashlib
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any, cast

import torch
import torch.nn.functional as functional
from safetensors import safe_open
from safetensors.torch import load_file as load_safetensors_file
from torch import nn


class SeedRetryPredecodeSelector(nn.Module):
    """
    DACVAE の波形化前状態から再試行と最終採用を判定する。
    """

    CONTRACT_VERSION = "irodori-seed-retry-predecode-selector-v1"
    FEATURE_COUNT = 800
    TEMPORAL_POSITIONS = 64
    DECODER_BLOCK_INDEX = 4

    def __init__(
        self,
        *,
        effective_weight: torch.Tensor,
        effective_bias: torch.Tensor,
        trigger_threshold: torch.Tensor,
        pair_margin_threshold: torch.Tensor,
    ) -> None:
        """
        線形品質スコアと較正済みしきい値を初期化する。

        Args:
            effective_weight (torch.Tensor): 800特徴へ適用する線形重み
            effective_bias (torch.Tensor): 候補単体の品質スコアへ加える切片
            trigger_threshold (torch.Tensor): 第1候補から再試行を始めるしきい値
            pair_margin_threshold (torch.Tensor): 第2候補を採用する品質差のしきい値

        Raises:
            ValueError: テンソル形状がモデル契約と異なる場合
        """

        super().__init__()

        if tuple(effective_weight.shape) != (self.FEATURE_COUNT,):
            raise ValueError(
                f"effective_weight must have shape ({self.FEATURE_COUNT},), "
                f"got {tuple(effective_weight.shape)}"
            )
        scalar_tensors = {
            "effective_bias": effective_bias,
            "trigger_threshold": trigger_threshold,
            "pair_margin_threshold": pair_margin_threshold,
        }
        for name, value in scalar_tensors.items():
            if value.numel() != 1:
                raise ValueError(f"{name} must contain one value, got {tuple(value.shape)}")

        self.register_buffer("effective_weight", effective_weight.float())
        self.register_buffer("effective_bias", effective_bias.reshape(()).float())
        self.register_buffer("trigger_threshold", trigger_threshold.reshape(()).float())
        self.register_buffer("pair_margin_threshold", pair_margin_threshold.reshape(()).float())
        self.codec_out_proj: nn.Module | None = None
        self.codec_decoder_blocks = nn.ModuleList()

    def bind_codec_model(self, codec_model: Any) -> None:
        """
        判定に必要な DACVAE の射影と先頭ブロックをセレクターへ保持する。

        TensorRT ランタイムは起動後に完全復号用の eager 重みを解放するため、判定器が使う
        block 4までの参照は解放処理より前に独立した寿命へ移す。

        Args:
            codec_model (Any): ロード済み DACVAE のモデル本体

        Raises:
            ValueError: 必要な復号ブロックが不足する場合
        """

        codec_out_proj = cast(nn.Module, codec_model.quantizer.out_proj)
        decoder_blocks = [
            cast(nn.Module, block)
            for block in codec_model.decoder.model[: self.DECODER_BLOCK_INDEX + 1]
        ]
        if len(decoder_blocks) != self.DECODER_BLOCK_INDEX + 1:
            raise ValueError(
                "DACVAE decoder blocks are incomplete: "
                f"expected {self.DECODER_BLOCK_INDEX + 1}, got {len(decoder_blocks)}"
            )

        # 共有 Module への参照を保持し、TensorRT 側が親属性を外しても重みを存続させる
        self.codec_out_proj = codec_out_proj
        self.codec_decoder_blocks = nn.ModuleList(decoder_blocks)
        codec_out_proj.eval()
        self.codec_decoder_blocks.eval()

    @classmethod
    def from_safetensors(
        cls,
        path: str | Path,
        *,
        device: torch.device,
    ) -> SeedRetryPredecodeSelector:
        """
        safetensors の重みと契約メタデータから判定器を復元する。

        Args:
            path (str | Path): 判定器チェックポイントのパス
            device (torch.device): 線形判定を実行するデバイス

        Returns:
            SeedRetryPredecodeSelector: 推論モードへ移した判定器

        Raises:
            ValueError: 契約版または特徴定義が現行実装と一致しない場合
        """

        checkpoint_path = Path(path)
        # safetensors の型スタブにコンテキスト管理と metadata() がないため、実装どおりの型へ限定して補う
        checkpoint_context = cast(
            AbstractContextManager[Any],
            safe_open(checkpoint_path, framework="pt", device="cpu"),
        )
        with checkpoint_context as checkpoint_handle:
            metadata = cast(Any, checkpoint_handle).metadata()
        if metadata.get("contract_version") != cls.CONTRACT_VERSION:
            raise ValueError(
                "Seed retry predecode selector contract mismatch: "
                f"{metadata.get('contract_version')!r}"
            )
        expected_metadata = {
            "feature_count": str(cls.FEATURE_COUNT),
            "decoder_block_index": str(cls.DECODER_BLOCK_INDEX),
            "temporal_positions": str(cls.TEMPORAL_POSITIONS),
            "channels": "96",
        }
        for name, expected_value in expected_metadata.items():
            if metadata.get(name) != expected_value:
                raise ValueError(
                    f"Seed retry predecode selector {name} mismatch: "
                    f"{metadata.get(name)!r}, expected {expected_value!r}"
                )

        # strict なコンストラクタ検査を通し、未知の重みや欠損を黙って受理しない
        tensors = load_safetensors_file(checkpoint_path, device="cpu")
        expected_tensor_names = {
            "effective_weight",
            "effective_bias",
            "trigger_threshold",
            "pair_margin_threshold",
        }
        if set(tensors) != expected_tensor_names:
            raise ValueError(
                "Seed retry predecode selector tensor mismatch: "
                f"{sorted(tensors)}, expected {sorted(expected_tensor_names)}"
            )
        selector = cls(
            effective_weight=tensors["effective_weight"],
            effective_bias=tensors["effective_bias"],
            trigger_threshold=tensors["trigger_threshold"],
            pair_margin_threshold=tensors["pair_margin_threshold"],
        )
        return selector.to(device=device, dtype=torch.float32).eval()

    @torch.inference_mode()
    def encode_latent(
        self,
        latent: torch.Tensor,
    ) -> torch.Tensor:
        """
        生成潜在を波形化直前の block 4系列へ変換する。

        Args:
            latent (torch.Tensor): `(batch, time, 32)` の生成または参照潜在

        Returns:
            torch.Tensor: `(batch, 96, 64)` の時間系列

        Raises:
            RuntimeError: DACVAE の部分復号器が未接続の場合
            ValueError: 潜在の形または DACVAE の出力チャネル数が契約と異なる場合
        """

        if latent.ndim != 3 or latent.shape[2] != 32:
            raise ValueError(f"latent must have shape (batch, time, 32), got {tuple(latent.shape)}")
        if self.codec_out_proj is None or len(self.codec_decoder_blocks) == 0:
            raise RuntimeError("DACVAE predecode encoder is not bound.")
        codec_parameter = next(self.codec_out_proj.parameters())

        # DACVAE の通常復号と同じ射影から始め、波形を作る最終層の手前で止める
        state = self.codec_out_proj(
            latent.transpose(1, 2)
            .contiguous()
            .to(
                device=codec_parameter.device,
                dtype=codec_parameter.dtype,
            )
        )
        for block_index, block in enumerate(self.codec_decoder_blocks):
            state = block(state)
            if block_index >= self.DECODER_BLOCK_INDEX:
                break
        if state.shape[1] != 96:
            raise ValueError(f"DACVAE block 4 must have 96 channels, got {tuple(state.shape)}")
        return functional.adaptive_avg_pool1d(state, self.TEMPORAL_POSITIONS).float()

    @torch.inference_mode()
    def score(
        self,
        generated_state: torch.Tensor,
        reference_state: torch.Tensor | None,
    ) -> torch.Tensor:
        """
        候補単体の復号前品質スコアを返す。

        Args:
            generated_state (torch.Tensor): `(batch, 96, 64)` の候補特徴
            reference_state (torch.Tensor | None): 同じ形の参照特徴 (参照なしは None)

        Returns:
            torch.Tensor: 値が大きいほど採用しやすい候補品質スコア

        Raises:
            ValueError: 生成または参照特徴の形が契約と異なる場合
        """

        expected_shape = (generated_state.shape[0], 96, self.TEMPORAL_POSITIONS)
        if tuple(generated_state.shape) != expected_shape:
            raise ValueError(
                f"generated_state must have shape {expected_shape}, "
                f"got {tuple(generated_state.shape)}"
            )

        # 候補単体の絶対値と時間変化を分け、局所的な崩れを平均値だけへ潰さない
        temporal_difference = generated_state[:, :, 1:] - generated_state[:, :, :-1]
        candidate_features = torch.cat(
            (
                generated_state.mean(dim=2),
                generated_state.std(dim=2),
                temporal_difference.square().mean(dim=2).sqrt(),
                generated_state.mean(dim=1),
                generated_state.std(dim=1),
            ),
            dim=1,
        )

        # 参照音声がある場合だけ絶対距離と正規化後の時間対応を追加する
        if reference_state is not None:
            if tuple(reference_state.shape) != expected_shape:
                raise ValueError(
                    f"reference_state must have shape {expected_shape}, "
                    f"got {tuple(reference_state.shape)}"
                )
            difference = generated_state - reference_state
            normalized_generated = (
                generated_state - generated_state.mean(dim=2, keepdim=True)
            ) / generated_state.std(dim=2, keepdim=True).clamp_min(1e-4)
            normalized_reference = (
                reference_state - reference_state.mean(dim=2, keepdim=True)
            ) / reference_state.std(dim=2, keepdim=True).clamp_min(1e-4)
            reference_features = torch.cat(
                (
                    difference.abs().mean(dim=2),
                    difference.square().mean(dim=2).sqrt(),
                    (normalized_generated - normalized_reference).abs().mean(dim=2),
                    (normalized_generated * normalized_reference).mean(dim=2),
                ),
                dim=1,
            )
        else:
            reference_features = torch.zeros(
                (generated_state.shape[0], 96 * 4),
                device=generated_state.device,
                dtype=generated_state.dtype,
            )
        features = torch.cat((candidate_features, reference_features), dim=1)
        return (features * self.get_buffer("effective_weight").unsqueeze(0)).sum(
            dim=1
        ) + self.get_buffer("effective_bias")

    def should_retry(self, score: torch.Tensor) -> torch.Tensor:
        """
        第1候補のスコアから完全 RF の再試行を始めるか返す。

        Args:
            score (torch.Tensor): 第1候補の品質スコア

        Returns:
            torch.Tensor: 候補ごとの再試行判断
        """

        return score <= self.get_buffer("trigger_threshold")

    def should_adopt_retry(
        self,
        *,
        base_score: torch.Tensor,
        retry_score: torch.Tensor,
    ) -> torch.Tensor:
        """
        第2候補が第1候補を十分に上回る場合だけ置き換える。

        Args:
            base_score (torch.Tensor): 第1候補の品質スコア
            retry_score (torch.Tensor): 第2候補の品質スコア

        Returns:
            torch.Tensor: 候補ごとの第2候補採用判断
        """

        return (retry_score - base_score) >= self.get_buffer("pair_margin_threshold")

    @classmethod
    def derive_retry_seed(
        cls,
        *,
        base_seed: int,
        chunk_index: int,
        attempt_index: int,
    ) -> int:
        """
        基準 seed とチャンク位置から再現可能な再試行 seed を作る。

        Args:
            base_seed (int): 第1試行で使用した符号なし63bit整数
            chunk_index (int): リクエスト内の0始まりチャンク番号
            attempt_index (int): 1始まりの再試行番号

        Returns:
            int: Irodori-TTS が受理する符号なし63bit整数

        Raises:
            ValueError: 入力値が負数の場合
        """

        if base_seed < 0 or chunk_index < 0 or attempt_index <= 0:
            raise ValueError(
                "Retry seed inputs must be non-negative and attempt_index must be positive: "
                f"base_seed: {base_seed}, chunk_index: {chunk_index}, "
                f"attempt_index: {attempt_index}"
            )

        # Python の hash() に依存せず、プロセスを跨いで同じ seed を再現する
        payload = b"irodori-seed-retry-v1"
        for value in (base_seed, chunk_index, attempt_index):
            payload += int(value).to_bytes(8, byteorder="big", signed=False)
        return int.from_bytes(
            hashlib.blake2b(payload, digest_size=8).digest(),
            byteorder="big",
            signed=False,
        ) & ((1 << 63) - 1)
