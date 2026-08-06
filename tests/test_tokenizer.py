from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

from irodori_tts.tokenizer import PretrainedTextTokenizer


def _build_tokenizer(*, add_bos: bool) -> PretrainedTextTokenizer:
    """
    ラッパーに必要な特殊トークンを持つ tokenizer を構築する。

    Args:
        add_bos (bool): BOS トークンを先頭へ追加するか。

    Returns:
        PretrainedTextTokenizer: 決定論的な単語単位 tokenizer のラッパー。
    """

    backend = Tokenizer(
        models.WordLevel(
            {
                "<pad>": 0,
                "<bos>": 1,
                "<unk>": 2,
                "one": 3,
                "two": 4,
                "three": 5,
            },
            unk_token="<unk>",
        )
    )
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    return PretrainedTextTokenizer(
        PreTrainedTokenizerFast(
            tokenizer_object=backend,
            pad_token="<pad>",
            bos_token="<bos>",
            unk_token="<unk>",
        ),
        add_bos=add_bos,
    )


def test_batch_encode_preserves_empty_text_with_bos() -> None:
    """
    BOS 有効時の空文字列を BOS と右側パディングで表現できることを確認する。
    """

    batch, mask = _build_tokenizer(add_bos=True).batch_encode(["", "one two"])

    assert batch.tolist() == [[1, 0, 0], [1, 3, 4]]
    assert mask.tolist() == [[True, False, False], [True, True, True]]


def test_batch_encode_preserves_empty_text_without_bos() -> None:
    """
    BOS 無効時の空文字列をパディングだけで表現できることを確認する。
    """

    batch, mask = _build_tokenizer(add_bos=False).batch_encode(["", "one"])

    assert batch.tolist() == [[0], [3]]
    assert mask.tolist() == [[False], [True]]


def test_batch_encode_respects_explicit_max_length() -> None:
    """
    本文の切り詰め前に BOS の領域を確保できることを確認する。
    """

    batch, mask = _build_tokenizer(add_bos=True).batch_encode(
        ["one two three"],
        max_length=2,
    )

    assert batch.tolist() == [[1, 3]]
    assert mask.tolist() == [[True, True]]
