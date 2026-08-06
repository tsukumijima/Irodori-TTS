from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors import safe_open

import prepare_speaker_inversion_base
from irodori_tts.speaker_inversion import speaker_inversion_checkpoint_sha256


class SpeakerInversionBaseRuntime:
    """
    base 作成コマンドが利用する推論ランタイムの最小契約を提供する。

    Args:
        local_tokens (int): 返す局所話者 token 数
    """

    def __init__(self, local_tokens: int) -> None:
        self.local_tokens = int(local_tokens)
        self.model_cfg = SimpleNamespace(speaker_patch_size=4)

    def encode_speaker_inversion_base(
        self,
        request: object,
        *,
        log_fn: object,
    ) -> SimpleNamespace:
        """
        指定した token 数の基準状態を返す。

        Args:
            request (object): base 作成コマンドが組み立てた推論要求
            log_fn (object): 推論ログの出力先

        Returns:
            SimpleNamespace: 保存前と平均 token 追加後の話者状態
        """

        del request, log_fn
        state = torch.arange(self.local_tokens * 8, dtype=torch.float32).reshape(
            1,
            self.local_tokens,
            8,
        )
        condition_state = torch.cat([state.mean(dim=1, keepdim=True), state], dim=1)
        return SimpleNamespace(state=state, condition_state=condition_state)


def _run_prepare_command(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    actual_local_tokens: int,
    expected_local_tokens: int,
) -> tuple[Path, Path]:
    """
    一時 checkpoint と参照音声を使って base 作成コマンドを実行する。

    Args:
        monkeypatch (pytest.MonkeyPatch): コマンド引数と推論ランタイムの差し替え
        tmp_path (Path): 入出力を置く一時ディレクトリ
        actual_local_tokens (int): 推論ランタイムが返す局所話者 token 数
        expected_local_tokens (int): コマンドへ指定する期待 token 数

    Returns:
        tuple[Path, Path]: checkpoint と出力 base のパス
    """

    checkpoint_path = tmp_path / "model.safetensors"
    checkpoint_path.write_bytes(b"checkpoint contents")
    reference_path = tmp_path / "reference.wav"
    reference_path.write_bytes(b"reference placeholder")
    output_path = tmp_path / "voice.speaker-base.safetensors"
    runtime = SpeakerInversionBaseRuntime(actual_local_tokens)
    monkeypatch.setattr(
        prepare_speaker_inversion_base.InferenceRuntime,
        "from_key",
        staticmethod(lambda _key: runtime),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prepare_speaker_inversion_base.py",
            "--checkpoint",
            str(checkpoint_path),
            "--ref-wav",
            str(reference_path),
            "--expected-local-tokens",
            str(expected_local_tokens),
            "--output",
            str(output_path),
        ],
    )

    prepare_speaker_inversion_base.main()
    return checkpoint_path, output_path


def test_prepare_command_saves_verified_base(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """
    token 数を確認した base と checkpoint 識別情報を保存する。

    Args:
        monkeypatch (pytest.MonkeyPatch): 推論ランタイムとコマンド引数の差し替え
        tmp_path (Path): 入出力を置く一時ディレクトリ
    """

    checkpoint_path, output_path = _run_prepare_command(
        monkeypatch,
        tmp_path,
        actual_local_tokens=6,
        expected_local_tokens=6,
    )

    with safe_open(str(output_path), framework="pt", device="cpu") as handle:
        metadata = handle.metadata()
        embedding = handle.get_tensor("speaker_pre_norm_embedding")
    assert embedding.shape == (6, 8)
    assert metadata["local_tokens"] == "6"
    assert metadata["speaker_patch_size"] == "4"
    assert "checkpoint" not in metadata
    assert metadata["checkpoint_sha256"] == speaker_inversion_checkpoint_sha256(checkpoint_path)


def test_prepare_command_rejects_unexpected_token_count(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """
    参照が必要な token 数を生成できない場合は base を保存しない。

    Args:
        monkeypatch (pytest.MonkeyPatch): 推論ランタイムとコマンド引数の差し替え
        tmp_path (Path): 入出力を置く一時ディレクトリ
    """

    with pytest.raises(ValueError, match="expected 6, got 5"):
        _run_prepare_command(
            monkeypatch,
            tmp_path,
            actual_local_tokens=5,
            expected_local_tokens=6,
        )

    assert not (tmp_path / "voice.speaker-base.safetensors").exists()


def test_prepare_command_rejects_output_suffix_before_loading_model(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """
    不正な保存名をモデル読込より前に拒否する。

    Args:
        monkeypatch (pytest.MonkeyPatch): コマンド引数とモデル読込を差し替えるフィクスチャ
        tmp_path (Path): 入力 checkpoint を置く一時ディレクトリ
    """

    checkpoint_path = tmp_path / "model.safetensors"
    checkpoint_path.write_bytes(b"checkpoint contents")
    monkeypatch.setattr(
        prepare_speaker_inversion_base.InferenceRuntime,
        "from_key",
        staticmethod(lambda _key: pytest.fail("runtime must not be loaded")),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prepare_speaker_inversion_base.py",
            "--checkpoint",
            str(checkpoint_path),
            "--ref-wav",
            str(tmp_path / "reference.wav"),
            "--output",
            str(tmp_path / "invalid.safetensors"),
        ],
    )

    with pytest.raises(ValueError, match=r"speaker-base\.safetensors"):
        prepare_speaker_inversion_base.main()
