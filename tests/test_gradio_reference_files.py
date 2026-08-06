from __future__ import annotations

from dataclasses import dataclass

import pytest

from irodori_tts.gradio_reference_files import GradioReferenceFiles


@dataclass
class GradioUploadValue:
    """
    Gradio のファイルオブジェクトと同じパス属性を提供する。

    Args:
        path (str | None): 優先して利用するファイルパス
        name (str | None): path がない場合に利用するファイル名
    """

    path: str | None = None
    name: str | None = None


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (" /tmp/string.wav ", "/tmp/string.wav"),
        ({"path": " /tmp/dict-path.wav ", "name": "/tmp/ignored.wav"}, "/tmp/dict-path.wav"),
        ({"path": " ", "name": " /tmp/dict-name.wav "}, "/tmp/dict-name.wav"),
        (GradioUploadValue(path=" /tmp/object-path.wav "), "/tmp/object-path.wav"),
        (GradioUploadValue(name=" /tmp/object-name.wav "), "/tmp/object-name.wav"),
        (None, None),
        (" ", None),
        ({"path": " ", "name": ""}, None),
        (object(), None),
    ],
)
def test_coerce_gradio_file_path_accepts_supported_upload_values(
    value: object,
    expected: str | None,
) -> None:
    """
    Gradio が返す各形式から有効なパスだけを取得する。

    Args:
        value (object): 変換するアップロード値
        expected (str | None): 期待するファイルパス
    """

    assert GradioReferenceFiles.coerce_file_path(value) == expected


def test_resolve_gradio_reference_wavs_preserves_order() -> None:
    """
    複数アップロードの順序を保ち、空の値だけを除外する。
    """

    assert GradioReferenceFiles.resolve_reference_wavs(
        [
            {"path": "/tmp/first.wav"},
            None,
            GradioUploadValue(name="/tmp/second.wav"),
            "   ",
            "/tmp/third.wav",
        ]
    ) == [
        "/tmp/first.wav",
        "/tmp/second.wav",
        "/tmp/third.wav",
    ]


def test_filter_gradio_reference_wavs_reports_unsupported_checkpoint() -> None:
    """
    話者条件を持たない checkpoint では参照を除外して理由を返す。
    """

    reference_wavs, notification = GradioReferenceFiles.filter_reference_wavs(
        ["/tmp/reference.wav"],
        supports_speaker_condition=False,
    )

    assert reference_wavs == []
    assert notification is not None
    assert "gradio-caption" not in notification
    assert "does not support speaker conditioning" in notification


def test_filter_gradio_reference_wavs_keeps_supported_reference() -> None:
    """
    話者条件を利用できる checkpoint では参照と順序を維持する。
    """

    expected = ["/tmp/first.wav", "/tmp/second.wav"]

    reference_wavs, notification = GradioReferenceFiles.filter_reference_wavs(
        expected,
        supports_speaker_condition=True,
    )

    assert reference_wavs == expected
    assert notification is None
