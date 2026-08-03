from __future__ import annotations

import shutil
from pathlib import Path


class CheckpointPublisher:
    """
    Publish a checkpoint and its bundled tokenizer as one coordinated operation.
    """

    @staticmethod
    def copy_bundled_tokenizer(source_checkpoint: Path, destination: Path) -> Path | None:
        """Copy a tokenizer bundled beside a source checkpoint into a staging directory.

        Args:
            source_checkpoint (Path): Checkpoint whose sibling tokenizer directory is copied.
            destination (Path): Staging directory for the tokenizer.

        Returns:
            Path | None: Staged tokenizer directory, or ``None`` when no tokenizer is bundled.
        """

        source = source_checkpoint.parent / "tokenizer"
        if not (source / "tokenizer_config.json").is_file():
            return None
        shutil.copytree(source, destination)
        return destination

    @staticmethod
    def publish(
        *,
        staged_checkpoint: Path,
        output_checkpoint: Path,
        staged_tokenizer: Path | None,
        temporary_directory: Path,
    ) -> Path | None:
        """Publish fully prepared checkpoint artifacts and restore the tokenizer on failure.

        Args:
            staged_checkpoint (Path): Completed checkpoint file in the staging directory.
            output_checkpoint (Path): Final checkpoint path.
            staged_tokenizer (Path | None): Completed tokenizer directory when required.
            temporary_directory (Path): Directory retaining the previous tokenizer during publish.

        Returns:
            Path | None: Published tokenizer directory, or ``None`` when absent.

        Raises:
            OSError: Filesystem publication fails.
        """

        tokenizer_directory = output_checkpoint.parent / "tokenizer"
        previous_tokenizer = temporary_directory / "previous-tokenizer"
        has_previous_tokenizer = staged_tokenizer is not None and tokenizer_directory.exists()
        if has_previous_tokenizer:
            tokenizer_directory.replace(previous_tokenizer)

        try:
            if staged_tokenizer is not None:
                staged_tokenizer.replace(tokenizer_directory)
            staged_checkpoint.replace(output_checkpoint)
        except OSError:
            # モデル公開に失敗した場合は新 tokenizer を一時領域へ戻し、従来の組を復元する
            if staged_tokenizer is not None and tokenizer_directory.exists():
                tokenizer_directory.replace(staged_tokenizer)
            if has_previous_tokenizer:
                previous_tokenizer.replace(tokenizer_directory)
            raise
        return tokenizer_directory if staged_tokenizer is not None else None
