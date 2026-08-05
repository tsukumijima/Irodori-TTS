from __future__ import annotations

from typing import Any

import pytest
import torch
from torch import nn

from irodori_tts.config import TrainConfig
from irodori_tts.optim import _partition_adamw_params, _partition_muon_params, build_optimizer


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

    assert id(model.pretrained_text_backbone.weight) in {
        id(param) for param in partitions.pretrained_decay
    }
    assert id(model.pretrained_text_backbone.bias) in {
        id(param) for param in partitions.pretrained_no_decay
    }
    assert id(model.output.weight) in {id(param) for param in partitions.decay}


def test_muon_partitions_pretrained_text_backbone_into_auxiliary_adamw() -> None:
    """
    Muon が事前学習済みバックボーンを補助 AdamW の専用分割へ配置する。
    """

    model = DummyOptimizerModel()

    partitions = _partition_muon_params(model)

    assert id(model.pretrained_text_backbone.weight) in {
        id(param) for param in partitions.pretrained_decay
    }
    assert id(model.pretrained_text_backbone.bias) in {
        id(param) for param in partitions.pretrained_no_decay
    }
    assert id(model.output.weight) in {id(param) for param in partitions.muon_decay}


def test_adamw_rejects_model_without_trainable_parameters() -> None:
    """
    AdamW が学習対象のないモデルを明示的に拒否する。
    """

    model = nn.Linear(2, 1)
    model.requires_grad_(False)

    with pytest.raises(ValueError, match="No trainable parameters found for optimizer=adamw"):
        build_optimizer(model, TrainConfig(optimizer="adamw"))


def _pretrained_groups(optimizer: Any) -> list[dict[str, Any]]:
    return [
        group
        for group in optimizer.param_groups
        if str(group.get("group_name", "")).startswith("pretrained_text_encoder")
    ]


def test_build_optimizer_adamw_applies_pretrained_learning_rate() -> None:
    """
    公開経路 build_optimizer が事前学習済みバックボーンへ専用学習率を載せる。
    """

    model = DummyOptimizerModel()
    train_cfg = TrainConfig(
        optimizer="adamw",
        learning_rate=1e-4,
        weight_decay=0.01,
        pretrained_text_encoder_learning_rate=2e-5,
    )

    optimizer = build_optimizer(model, train_cfg)
    pretrained_groups = _pretrained_groups(optimizer)

    assert pretrained_groups
    for group in pretrained_groups:
        assert group["lr"] == pytest.approx(2e-5)
    decay_groups = [group for group in pretrained_groups if group["weight_decay"] > 0.0]
    no_decay_groups = [group for group in pretrained_groups if group["weight_decay"] == 0.0]
    assert decay_groups
    assert no_decay_groups
    assert all(group["weight_decay"] == pytest.approx(0.01) for group in decay_groups)


def test_build_optimizer_muon_places_pretrained_in_auxiliary_adamw() -> None:
    """
    Muon 経路でも事前学習済みバックボーンが補助 AdamW の専用学習率を受け取る。
    """

    if not hasattr(torch.optim, "Muon"):
        pytest.skip("torch.optim.Muon is unavailable in this environment.")

    model = DummyOptimizerModel()
    train_cfg = TrainConfig(
        optimizer="muon",
        learning_rate=1e-4,
        weight_decay=0.01,
        pretrained_text_encoder_learning_rate=3e-5,
    )

    optimizer = build_optimizer(model, train_cfg)
    pretrained_groups = _pretrained_groups(optimizer)

    assert pretrained_groups
    for group in pretrained_groups:
        assert group["lr"] == pytest.approx(3e-5)
    assert any(group["weight_decay"] == pytest.approx(0.01) for group in pretrained_groups)
    assert any(group["weight_decay"] == 0.0 for group in pretrained_groups)
