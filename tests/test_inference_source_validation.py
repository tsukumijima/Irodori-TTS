import pytest

from irodori_tts.inference_runtime import _split_hf_checkpoint_source


def test_hugging_face_checkpoint_source_requires_owner_and_repository() -> None:
    """所有者を欠く Hugging Face の参照を repository ID として受理しない。"""

    with pytest.raises(ValueError, match="owner/repo"):
        _split_hf_checkpoint_source("model")


def test_hugging_face_checkpoint_source_preserves_optional_subfolder() -> None:
    """正しい repository ID と1階層のサブフォルダを分離する。"""

    assert _split_hf_checkpoint_source("owner/model") == ("owner/model", None)
    assert _split_hf_checkpoint_source("owner/model/checkpoints") == (
        "owner/model",
        "checkpoints",
    )
