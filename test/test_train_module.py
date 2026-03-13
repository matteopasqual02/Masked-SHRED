import importlib
import sys
import types

import numpy as np  # type: ignore
import pytest  # type: ignore
import torch  # type: ignore
import torch.nn as nn  # type: ignore
from torch.utils.data import TensorDataset  # type: ignore


def _stub_module(name, **symbols):
    module = types.ModuleType(name)
    for key, value in symbols.items():
        setattr(module, key, value)
    return module


@pytest.fixture()
def train_module(monkeypatch):
    # src.train imports these paths at module import time.
    monkeypatch.setitem(sys.modules, "src.shred", _stub_module("src.shred"))
    monkeypatch.setitem(sys.modules, "src.tshred", _stub_module("src.tshred"))
    monkeypatch.setitem(sys.modules, "src.shred.SHRED", _stub_module("src.shred.SHRED", SHRED=type("SHRED", (nn.Module,), {})))
    monkeypatch.setitem(sys.modules, "src.shred.SHRED_GRUD", _stub_module("src.shred.SHRED_GRUD", SHRED_GRUD=type("SHRED_GRUD", (nn.Module,), {})))
    monkeypatch.setitem(sys.modules, "src.tshred.TSHRED", _stub_module("src.tshred.TSHRED", TSHRED=type("TSHRED", (nn.Module,), {})))

    if "src.train" in sys.modules:
        del sys.modules["src.train"]

    return importlib.import_module("src.train")


def test_get_optimizer_supported_and_invalid(train_module):
    for name in ["AdamW", "Adam", "SGD", "RMSprop", "Adagrad"]:
        params = nn.Linear(3, 2).parameters()
        opt = train_module._get_optimizer(params, name, lr=1e-3, weight_decay=1e-2)
        assert isinstance(opt, torch.optim.Optimizer)

    with pytest.raises(ValueError):
        train_module._get_optimizer(nn.Linear(3, 2).parameters(), "BAD", lr=1e-3, weight_decay=1e-2)


def test_get_lr_scheduler_variants(train_module):
    opt = torch.optim.Adam(nn.Linear(2, 2).parameters(), lr=1e-3)

    assert train_module._get_lr_scheduler(opt, "Classical") is not None
    assert train_module._get_lr_scheduler(opt, "StepLR") is not None
    assert train_module._get_lr_scheduler(opt, "ExponentialLR") is not None
    assert train_module._get_lr_scheduler(opt, "CosineAnnealingLR") is not None

    with pytest.raises(ValueError):
        train_module._get_lr_scheduler(opt, "Transformer")

    with pytest.raises(ValueError):
        train_module._get_lr_scheduler(opt, "Transformer", embed_dim=8, warmup_steps=0)

    sched = train_module._get_lr_scheduler(opt, "Transformer", embed_dim=8, warmup_steps=4)
    assert sched is not None

    with pytest.raises(ValueError):
        train_module._get_lr_scheduler(opt, "unknown")


def test_evaluate_returns_expected_metrics_shape(train_module):
    class IdentityModel(nn.Module):
        def forward(self, x):
            return x

    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    y = x.clone()
    dataset = TensorDataset(x, y)

    mse, mre = train_module.evaluate(IdentityModel(), dataset, batch_size=1)
    assert np.isclose(mse, 0.0)
    assert np.isclose(mre, 0.0)


def test_fit_runs_and_returns_losses(train_module):
    torch.manual_seed(0)

    model = nn.Linear(2, 2)
    x = torch.randn(20, 2)
    y = x @ torch.tensor([[1.2, -0.4], [0.5, 0.7]]) + 0.1

    train_dataset = TensorDataset(x[:14], y[:14])
    val_dataset = TensorDataset(x[14:], y[14:])

    train_losses, val_losses = train_module.fit(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        batch_size=4,
        num_epochs=5,
        lr=1e-2,
        optimizer="Adam",
        lr_scheduler="StepLR",
        scheduler_step_size=2,
        scheduler_gamma=0.5,
        patience=10,
        verbose=False,
    )

    assert isinstance(train_losses, np.ndarray)
    assert isinstance(val_losses, np.ndarray)
    assert len(train_losses) >= 1
    assert len(val_losses) >= 1
