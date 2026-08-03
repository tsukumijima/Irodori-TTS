from __future__ import annotations

from torch import nn

from irodori_tts.optim import _partition_adamw_params, _partition_muon_params


class DummyOptimizerModel(nn.Module):
    """
    事前学習済みバックボーンを含む最小のオプティマイザ分割対象。
    """

    def __init__(self) -> None:
        super().__init__()
        self.pretrained_text_backbone = nn.Linear(3, 2)
        self.output = nn.Linear(2, 1)


def test_adamw_partitions_pretrained_text_backbone() -> None:
    """
    AdamW が事前学習済みバックボーンを専用学習率の対象へ分離する。
    """

    model = DummyOptimizerModel()

    partitions = _partition_adamw_params(model)

    assert model.pretrained_text_backbone.weight in partitions.pretrained_decay
    assert model.pretrained_text_backbone.bias in partitions.pretrained_no_decay
    assert model.output.weight in partitions.decay


def test_muon_partitions_pretrained_text_backbone_into_auxiliary_adamw() -> None:
    """
    Muon が事前学習済みバックボーンを補助 AdamW の専用分割へ配置する。
    """

    model = DummyOptimizerModel()

    partitions = _partition_muon_params(model)

    assert model.pretrained_text_backbone.weight in partitions.pretrained_decay
    assert model.pretrained_text_backbone.bias in partitions.pretrained_no_decay
    assert model.output.weight in partitions.muon_decay
