import json
from pathlib import Path

import pytest
import torch

from irodori_tts.dataset import LatentTextDataset


def test_subset_indices_reject_multidimensional_tensor(tmp_path: Path) -> None:
    """
    多次元の index Tensor を標本列として解釈しない。

    Args:
        tmp_path (Path): 存在しない manifest パスを置く一時ディレクトリ
    """

    with pytest.raises(ValueError, match="one-dimensional"):
        LatentTextDataset(
            tmp_path / "manifest.jsonl",
            latent_dim=4,
            subset_indices=torch.tensor([[0, 1]]),
        )


def test_blank_speaker_ids_remain_unlabeled(tmp_path: Path) -> None:
    """
    空白だけの話者 ID を独立した話者として登録しない。

    Args:
        tmp_path (Path): manifest を置く一時ディレクトリ
    """

    manifest_path = tmp_path / "manifest.jsonl"
    rows = (
        {"text": "blank", "latent_path": "blank.pt", "speaker_id": "   ", "num_frames": 1},
        {
            "text": "named",
            "latent_path": "named.pt",
            "speaker_id": "speaker-a",
            "num_frames": 1,
        },
    )
    manifest_path.write_text(
        "".join(f"{json.dumps(row)}\n" for row in rows),
        encoding="utf-8",
    )

    dataset = LatentTextDataset(manifest_path, latent_dim=4)

    assert dataset.manifest_index.speakers == ["speaker-a"]
    assert dataset.manifest_index.speaker_codes.tolist() == [-1, 0]
