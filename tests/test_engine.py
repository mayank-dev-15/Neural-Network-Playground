import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from core.engine import TrainingEngine, TrainingConfig


def _make_dummy_data():
    x = torch.randn(64, 10)
    y = torch.randint(0, 2, (64,))
    ds = TensorDataset(x, y)
    loader = DataLoader(ds, batch_size=16)
    return loader


def test_engine_init():
    config = TrainingConfig()
    engine = TrainingEngine(config)
    assert engine is not None
    assert engine.config.epochs == 10


def test_build_optimizer():
    config = TrainingConfig(optimizer="adam", learning_rate=0.01)
    engine = TrainingEngine(config)
    model = nn.Linear(10, 2)
    opt = engine.build_optimizer(model)
    assert opt is not None


def test_train_epoch():
    config = TrainingConfig(epochs=1)
    engine = TrainingEngine(config)
    model = nn.Linear(10, 2)
    engine.model = model
    engine.optimizer = engine.build_optimizer(model)
    loader = _make_dummy_data()
    loss, acc = engine.train_epoch(loader)
    assert isinstance(loss, float)
    assert 0.0 <= acc <= 1.0


def test_validate():
    config = TrainingConfig()
    engine = TrainingEngine(config)
    model = nn.Linear(10, 2)
    engine.model = model
    loader = _make_dummy_data()
    loss, acc = engine.validate(loader)
    assert isinstance(loss, float)
    assert 0.0 <= acc <= 1.0


def test_full_training():
    config = TrainingConfig(epochs=2, early_stopping_patience=10)
    engine = TrainingEngine(config)
    model = nn.Linear(10, 2)
    train_loader = _make_dummy_data()
    val_loader = _make_dummy_data()
    history = engine.train(model, train_loader, val_loader)
    assert len(history) > 0
    assert history[0].epoch == 1


if __name__ == '__main__':
    test_engine_init()
    test_build_optimizer()
    test_train_epoch()
    test_validate()
    test_full_training()
    print("All engine tests passed.")
