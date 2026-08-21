from __future__ import annotations

from typing import Any

import pytest
from torch import nn

from irodori_tts.config import TrainConfig
from irodori_tts.optim import build_optimizer


class DummyOptimizerModel(nn.Module):
    """
    事前学習済みバックボーンを含む最小のオプティマイザ分割対象。
    """

    def __init__(self) -> None:
        super().__init__()
        self.pretrained_text_backbone = nn.Linear(3, 2)
        self.output = nn.Linear(2, 1)


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
    assert id(model.pretrained_text_backbone.weight) in {
        id(param) for group in optimizer.param_groups for param in group["params"]
    }


def test_build_optimizer_muon_places_pretrained_in_auxiliary_adamw() -> None:
    """
    Muon 経路でも事前学習済みバックボーンが補助 AdamW の専用学習率を受け取る。
    """

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
    assert id(model.output.weight) in {
        id(param)
        for group in optimizer.param_groups
        if str(group.get("group_name", "")).startswith("main_muon")
        for param in group["params"]
    }
