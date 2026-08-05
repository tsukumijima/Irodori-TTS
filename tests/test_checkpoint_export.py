from pathlib import Path

import pytest

from irodori_tts.checkpoint_export import CheckpointPublisher


def _write_tokenizer(directory: Path, *, model_content: str) -> None:
    """
    公開処理の比較に必要な最小 tokenizer を作成する。

    Args:
        directory (Path): tokenizer の出力先
        model_content (str): tokenizer model の内容
    """

    directory.mkdir(parents=True)
    (directory / "tokenizer_config.json").write_text('{"version": 1}', encoding="utf-8")
    (directory / "tokenizer.model").write_text(model_content, encoding="utf-8")


def test_publish_reuses_matching_tokenizer_without_force(tmp_path: Path) -> None:
    """
    同じ tokenizer を使う別名 checkpoint は上書き指定なしで公開できる。
    """

    output_checkpoint = tmp_path / "model-int4.safetensors"
    tokenizer_directory = tmp_path / "tokenizer"
    staged_tokenizer = tmp_path / "staging" / "tokenizer"
    _write_tokenizer(tokenizer_directory, model_content="same")
    _write_tokenizer(staged_tokenizer, model_content="same")
    staged_checkpoint = tmp_path / "staging" / output_checkpoint.name
    staged_checkpoint.write_bytes(b"checkpoint")

    published_tokenizer = CheckpointPublisher.publish(
        staged_checkpoint=staged_checkpoint,
        output_checkpoint=output_checkpoint,
        staged_tokenizer=staged_tokenizer,
        temporary_directory=tmp_path / "staging",
        force=False,
    )

    assert output_checkpoint.read_bytes() == b"checkpoint"
    assert published_tokenizer == tokenizer_directory
    assert (tokenizer_directory / "tokenizer.model").read_text(encoding="utf-8") == "same"


def test_publish_rejects_different_tokenizer_without_force(tmp_path: Path) -> None:
    """
    内容が異なる tokenizer は既存ファイルを保ったまま拒否する。
    """

    output_checkpoint = tmp_path / "model-int4.safetensors"
    tokenizer_directory = tmp_path / "tokenizer"
    staged_tokenizer = tmp_path / "staging" / "tokenizer"
    _write_tokenizer(tokenizer_directory, model_content="existing")
    _write_tokenizer(staged_tokenizer, model_content="different")
    staged_checkpoint = tmp_path / "staging" / output_checkpoint.name
    staged_checkpoint.write_bytes(b"checkpoint")

    with pytest.raises(FileExistsError, match="Tokenizer already exists"):
        CheckpointPublisher.publish(
            staged_checkpoint=staged_checkpoint,
            output_checkpoint=output_checkpoint,
            staged_tokenizer=staged_tokenizer,
            temporary_directory=tmp_path / "staging",
            force=False,
        )

    assert output_checkpoint.exists() is False
    assert (tokenizer_directory / "tokenizer.model").read_text(encoding="utf-8") == "existing"


def test_publish_restores_previous_tokenizer_when_checkpoint_publish_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    チェックポイントの公開失敗時に既存 tokenizer を復元できることを確認する。
    """

    output_checkpoint = tmp_path / "model.safetensors"
    tokenizer_directory = tmp_path / "tokenizer"
    temporary_directory = tmp_path / "staging"
    staged_tokenizer = temporary_directory / "tokenizer"
    _write_tokenizer(tokenizer_directory, model_content="existing")
    _write_tokenizer(staged_tokenizer, model_content="replacement")
    staged_checkpoint = temporary_directory / output_checkpoint.name
    staged_checkpoint.write_bytes(b"checkpoint")
    original_replace = Path.replace

    def fail_checkpoint_publish(path: Path, target: Path) -> Path:
        if path == staged_checkpoint:
            raise OSError("injected checkpoint publish failure")
        return original_replace(path, target)

    monkeypatch.setattr(Path, "replace", fail_checkpoint_publish)

    with pytest.raises(OSError, match="injected checkpoint publish failure"):
        CheckpointPublisher.publish(
            staged_checkpoint=staged_checkpoint,
            output_checkpoint=output_checkpoint,
            staged_tokenizer=staged_tokenizer,
            temporary_directory=temporary_directory,
            force=True,
        )

    assert output_checkpoint.exists() is False
    assert (tokenizer_directory / "tokenizer.model").read_text(encoding="utf-8") == "existing"
    assert (staged_tokenizer / "tokenizer.model").read_text(encoding="utf-8") == "replacement"
