import json
from pathlib import Path

import pytest
import torch

from irodori_tts.dataset import LatentTextDataset


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


def test_disabled_speaker_condition_does_not_load_another_reference(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Speaker Inversion 用データセットでは同話者の別 latent を読み込まない。

    Args:
        tmp_path (Path): manifest と latent を置く一時ディレクトリ
        monkeypatch (pytest.MonkeyPatch): 別参照の読込を失敗させるフィクスチャ
    """

    manifest_path = tmp_path / "manifest.jsonl"
    rows: list[dict[str, str | int]] = []
    for sample_index in range(2):
        latent_path = tmp_path / f"latent_{sample_index}.pt"
        torch.save(torch.full((4, 2), float(sample_index)), latent_path)
        rows.append(
            {
                "text": f"sample {sample_index}",
                "latent_path": latent_path.name,
                "speaker_id": "speaker",
                "num_frames": 4,
            }
        )
    manifest_path.write_text(
        "".join(f"{json.dumps(row)}\n" for row in rows),
        encoding="utf-8",
    )
    dataset = LatentTextDataset(
        manifest_path,
        latent_dim=2,
        enable_speaker_condition=False,
    )

    def fail_reference_load(_path: str) -> torch.Tensor:
        """
        別サンプルの参照 latent 読込を検出する。

        Args:
            _path (str): 読込対象として渡された未使用のパス

        Raises:
            AssertionError: Speaker Inversion 用データセットが別参照を読んだ場合
        """

        raise AssertionError("unexpected reference load")

    monkeypatch.setattr(dataset, "_load_ref_latent", fail_reference_load)

    item = dataset[0]

    torch.testing.assert_close(item["ref_latent"], item["latent"])
    assert item["has_speaker"] is False
