import importlib
import sys
import types

import numpy as np  # type: ignore
import pytest  # type: ignore
import torch  # type: ignore
import torch.nn as nn  # type: ignore
from sklearn.preprocessing import MinMaxScaler  # type: ignore
from torch.utils.data import TensorDataset  # type: ignore

from src.datamanager import TimeSeriesDataset, TimeSeriesDatasetWithMask


class _FlatPredictor(nn.Module):
    def __init__(self, input_dim=2, output_dim=2):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x, mask=None):
        if x.ndim == 3:
            x = x[:, -1, :]
        return self.linear(x)


class _LaggedPredictor(nn.Module):
    def __init__(self, feature_dim=2):
        super().__init__()
        self.linear = nn.Linear(feature_dim, feature_dim)
        self.decoder_3D = True

    def forward(self, x, mask=None):
        if x.ndim == 2:
            x = x.unsqueeze(1)
        outputs = [self.linear(x[:, step, :]) for step in range(x.shape[1])]
        return torch.stack(outputs, dim=0)


class _DummySequenceEncoder(nn.Module):
    def __init__(self, *args, hidden_size=8, **kwargs):
        super().__init__()
        self.hidden_size = hidden_size

    def forward(self, x, mask=None):
        if x.ndim == 2:
            batch_size = x.shape[0]
            seq_len = 1
        else:
            batch_size = x.shape[0]
            seq_len = x.shape[1]
        return torch.ones(seq_len, batch_size, self.hidden_size, dtype=x.dtype, device=x.device)


class _LinearDecoder(nn.Module):
    def __init__(self, input_size=8, output_size=5, hidden_sizes=None, dropout=0.0, **kwargs):
        super().__init__()
        self.layer = nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.layer(x)


class _DummyDataset:
    def __init__(self, x, has_mask=True):
        self.X = x
        self.M = (~torch.isnan(x)).float() if has_mask else None


class _DummyDM:
    def __init__(self, width=2):
        self.lags = 2
        self._dataset_ids = ["setA"]
        self._dataset_lengths = {"setA": width}
        self._Vt_registry = {"setA": np.eye(width, dtype=float)}
        self._preSVD_scaler_registry = {"setA": MinMaxScaler().fit(np.array([[0.0] * width, [1.0] * width]))}
        self._dataset_spatial_shape = {"setA": (width,)}
        self.sensor_scaler = MinMaxScaler().fit(np.array([[0.0, 0.0], [1.0, 1.0]]))
        self.data_scaler = MinMaxScaler().fit(np.array([[0.0] * width, [1.0] * width]))


@pytest.fixture()
def train_module():
    if "src.train" in sys.modules:
        del sys.modules["src.train"]
    return importlib.import_module("src.train")


@pytest.fixture()
def engine_module(monkeypatch):
    monkeypatch.setitem(sys.modules, "src.shred", types.ModuleType("src.shred"))
    monkeypatch.setitem(sys.modules, "src.tshred", types.ModuleType("src.tshred"))
    monkeypatch.setitem(sys.modules, "src.shred.SHRED", types.ModuleType("src.shred.SHRED"))
    monkeypatch.setitem(sys.modules, "src.tshred.TSHRED", types.ModuleType("src.tshred.TSHRED"))
    monkeypatch.setattr(sys.modules["src.shred.SHRED"], "SHRED", type("SHRED", (nn.Module,), {}), raising=False)
    monkeypatch.setattr(sys.modules["src.tshred.TSHRED"], "TSHRED", type("TSHRED", (nn.Module,), {}), raising=False)

    if "src.engine" in sys.modules:
        del sys.modules["src.engine"]
    return importlib.import_module("src.engine")


@pytest.fixture()
def mshred_module(monkeypatch):
    encoder_stubs = {
        "src.mSHRED.encoders.transformer": ("Transformer",),
        "src.mSHRED.encoders.transformer_hidden_mask": ("TransformerHiddenMask",),
        "src.mSHRED.encoders.transformer_attn_mask": ("TransformerAttnMask",),
        "src.mSHRED.encoders.transformer_learned_mask": ("TransformerLearnedMask",),
        "src.mSHRED.encoders.xlstm.xlstm": ("XLSTM",),
        "src.mSHRED.encoders.LSTM": ("LSTM",),
        "src.mSHRED.encoders.GRUD": ("GRUD",),
        "src.mSHRED.encoders.attnGRUD": ("AttnGRUD",),
    }
    for module_name, class_names in encoder_stubs.items():
        module = types.ModuleType(module_name)
        for class_name in class_names:
            setattr(module, class_name, _DummySequenceEncoder)
        monkeypatch.setitem(sys.modules, module_name, module)

    decoder_stubs = {
        "src.mSHRED.decoders.MLP": ("MLP",),
        "src.mSHRED.decoders.CNN": ("CNN",),
    }
    for module_name, class_names in decoder_stubs.items():
        module = types.ModuleType(module_name)
        for class_name in class_names:
            setattr(module, class_name, _LinearDecoder)
        monkeypatch.setitem(sys.modules, module_name, module)

    if "src.mSHRED.mSHRED" in sys.modules:
        del sys.modules["src.mSHRED.mSHRED"]
    return importlib.import_module("src.mSHRED.mSHRED")


def test_train_helpers_fit_variants_and_schedulers(train_module):
    true = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    pred = torch.tensor([[1.0, 1.0], [3.0, 5.0]])

    assert torch.is_tensor(train_module.mre(true, pred))
    assert torch.is_tensor(train_module.mse(true, pred))

    for name in ["AdamW", "Adam", "SGD", "RMSprop", "Adagrad"]:
        linear_model = nn.Linear(2, 2)
        params = linear_model.parameters()
        opt = train_module._get_optimizer(params, name, lr=1e-3, weight_decay=1e-2)
        assert isinstance(opt, torch.optim.Optimizer)

    with pytest.raises(ValueError):
        train_module._get_optimizer(nn.Linear(2, 2).parameters(), "bad", lr=1e-3, weight_decay=1e-2)

    opt = torch.optim.Adam(nn.Linear(2, 2).parameters(), lr=1e-3)
    assert train_module._get_lr_scheduler(opt, "Classical") is not None
    assert train_module._get_lr_scheduler(opt, "StepLR") is not None
    assert train_module._get_lr_scheduler(opt, "ExponentialLR") is not None
    assert train_module._get_lr_scheduler(opt, "CosineAnnealingLR") is not None
    with pytest.raises(ValueError):
        train_module._get_lr_scheduler(opt, "Transformer")
    with pytest.raises(ValueError):
        train_module._get_lr_scheduler(opt, "Transformer", embed_dim=4, warmup_steps=0)
    assert train_module._get_lr_scheduler(opt, "Transformer", embed_dim=4, warmup_steps=2) is not None
    with pytest.raises(ValueError):
        train_module._get_lr_scheduler(opt, "unknown")


def test_train_and_evaluate_variants_cover_masked_and_unmasked_paths(train_module):
    torch.manual_seed(0)

    x = torch.randn(12, 2)
    y = x @ torch.tensor([[1.2, -0.4], [0.5, 0.7]]) + 0.1
    train_dataset = TensorDataset(x[:8], y[:8])
    val_dataset = TensorDataset(x[8:], y[8:])

    flat_model = _FlatPredictor(2, 2)
    train_losses, val_losses = train_module.fit(
        model=flat_model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        batch_size=4,
        num_epochs=3,
        lr=1e-2,
        optimizer="Adam",
        lr_scheduler=None,
        patience=5,
        verbose=False,
    )
    assert train_losses.size > 0
    assert val_losses.size > 0

    no_val_model = _FlatPredictor(2, 2)
    train_losses_no_val, val_losses_no_val = train_module.fit(
        model=no_val_model,
        train_dataset=train_dataset,
        val_dataset=None,
        batch_size=4,
        num_epochs=2,
        lr=1e-2,
        optimizer="Adam",
        lr_scheduler="Transformer",
        scheduler_embed_dim=4,
        scheduler_warmup_steps=2,
        patience=5,
        verbose=False,
    )
    assert train_losses_no_val.size > 0
    assert val_losses_no_val.size == 0

    mse, mre, nrmse = train_module.evaluate(no_val_model, val_dataset, batch_size=4)
    assert isinstance(mse, float)
    assert isinstance(mre, float)
    assert isinstance(nrmse, float)

    masked_x = torch.randn(12, 3, 2)
    masked_x[0, 0, 0] = torch.nan
    masked_y = torch.randn(12, 2)
    masked_dataset = TimeSeriesDatasetWithMask(masked_x.clone(), masked_y.clone())

    masked_model = _FlatPredictor(2, 2)
    train_losses_masked, val_losses_masked = train_module.fit(
        model=masked_model,
        train_dataset=masked_dataset,
        val_dataset=masked_dataset,
        batch_size=4,
        num_epochs=2,
        lr=1e-2,
        optimizer="Adam",
        lr_scheduler=None,
        patience=5,
        verbose=False,
        masking_type="mask_data",
    )
    assert train_losses_masked.size > 0
    assert val_losses_masked.size > 0

    fit2_model = _FlatPredictor(2, 2)
    train_losses_fit2, val_losses_fit2 = train_module.fit2(
        model=fit2_model,
        train_dataset=masked_dataset,
        val_dataset=masked_dataset,
        batch_size=4,
        num_epochs=2,
        lr=1e-2,
        optimizer="Adam",
        lr_scheduler=None,
        patience=5,
        verbose=False,
        masking_type="mask_data",
    )
    assert train_losses_fit2.size > 0
    assert val_losses_fit2.size > 0

    eval2_mse, eval2_mre = train_module.evaluate2(fit2_model, masked_dataset, masking_type="mask_data")
    assert isinstance(eval2_mse, (float, torch.Tensor))
    assert isinstance(eval2_mre, (float, torch.Tensor))

    eval2_unmasked_mse, eval2_unmasked_mre = train_module.evaluate2(fit2_model, TimeSeriesDataset(masked_dataset.X, masked_dataset.Y))
    assert isinstance(eval2_unmasked_mse, (float, torch.Tensor))
    assert isinstance(eval2_unmasked_mre, (float, torch.Tensor))

    sparse_model = _FlatPredictor(2, 2)
    sparse_train_losses, sparse_val_losses = train_module.sparse_fit(
        model=sparse_model,
        train_dataset=masked_dataset,
        val_dataset=masked_dataset,
        batch_size=4,
        num_epochs=2,
        lr=1e-2,
        optimizer="Adam",
        lr_scheduler=None,
        patience=5,
        verbose=False,
        sparse_factor=1,
        sparse_log=True,
    )
    assert sparse_train_losses.size > 0
    assert sparse_val_losses.size > 0


def test_lagged_and_engine_and_mshred_paths(train_module, engine_module, mshred_module, monkeypatch):
    torch.manual_seed(0)

    lag_x = torch.randn(10, 3, 2)
    lag_x[0, 0, 0] = torch.nan
    lag_y = torch.randn(10, 3, 2)
    lag_masked_dataset = TimeSeriesDatasetWithMask(lag_x.clone(), lag_y.clone())
    lag_dataset = TimeSeriesDataset(lag_x.clone(), lag_y.clone())

    lag_model = _LaggedPredictor(2)
    lag_train_losses, lag_val_losses = train_module.lagged_fit(
        model=lag_model,
        train_dataset=lag_masked_dataset,
        val_dataset=lag_masked_dataset,
        batch_size=2,
        num_epochs=2,
        lr=1e-2,
        optimizer="Adam",
        lr_scheduler=None,
        patience=5,
        verbose=False,
        masking_type="mask_data",
        w_factor=0.5,
        w_keeped_steps=2,
    )
    assert lag_train_losses.size > 0
    assert lag_val_losses.size > 0

    lag_eval_mse, lag_eval_mre = train_module.lagged_evaluate(
        model=lag_model,
        dataset=lag_masked_dataset,
        batch_size=2,
        masking_type="mask_data",
    )
    assert isinstance(lag_eval_mse, float)
    assert isinstance(lag_eval_mre, float)

    lag_train_losses2, lag_val_losses2 = train_module.lagged_fit(
        model=lag_model,
        train_dataset=lag_dataset,
        val_dataset=lag_dataset,
        batch_size=2,
        num_epochs=2,
        lr=1e-2,
        optimizer="Adam",
        lr_scheduler=None,
        patience=5,
        verbose=False,
        masking_type=None,
        w_factor=0.5,
        w_keeped_steps=2,
    )
    assert lag_train_losses2.size > 0
    assert lag_val_losses2.size > 0

    lag_eval_mse2, lag_eval_mre2 = train_module.lagged_evaluate(
        model=lag_model,
        dataset=lag_dataset,
        batch_size=2,
        masking_type=None,
    )
    assert isinstance(lag_eval_mse2, float)
    assert isinstance(lag_eval_mre2, float)

    # Engine coverage.
    engine = engine_module.SHREDEngine(_DummyDM(width=2), _FlatPredictor(2, 2))
    sensor_measurements = np.array(
        [[0.0, 0.0], [0.2, 0.3], [0.4, 0.6], [0.6, 0.9], [1.0, 1.2]],
        dtype=float,
    )
    reconstruction_input = _DummyDataset(
        torch.tensor(
            [[
                [0.0, 0.0],
                [0.2, 0.3],
            ], [
                [0.2, 0.3],
                [0.4, 0.6],
            ], [
                [0.4, 0.6],
                [0.6, 0.9],
            ], [
                [0.6, 0.9],
                [1.0, 1.2],
            ], [
                [1.0, 1.2],
                [1.0, 1.2],
            ]],
            dtype=torch.float32,
        ),
        has_mask=True,
    )
    recon = engine.sensor_to_reconstruction_dataset(reconstruction_input)
    assert set(recon.keys()) == {"setA"}
    assert recon["setA"].shape == (5, 2)

    lagged_recon = engine.sensor_to_reconstruction_dataset_lagged(reconstruction_input)
    assert set(lagged_recon.keys()) == {"setA"}
    assert lagged_recon["setA"].shape == (5, 2)

    decoded = engine._decode_output(np.ones((5, 2), dtype=float))
    assert decoded["setA"].shape == (5, 2)

    with pytest.raises(ValueError):
        engine._decode_output(np.ones((5, 3), dtype=float))

    # mSHRED coverage for the current architecture variants.
    def build_mshred_args(encoder_name, decoder_name, decoder_3d=False):
        args = {
            "encoder": encoder_name,
            "decoder": decoder_name,
            "input_size": 3,
            "hidden_size": 8,
            "output_size": 5,
            "input_length": 4,
            "n_heads": 1,
            "encoder_depth": 1,
            "dim_feedforward": 16,
            "dropout": 0.0,
            "num_layers": 1,
            "xlstm_type": "slstm",
            "decoder_hidden_sizes": [6],
            "double_channels": False,
        }
        if encoder_name in {"grud", "att_grud"}:
            args["X_mean"] = torch.zeros(1, 1, 4, 3)
        if decoder_3d:
            args["decoder_3D"] = True
        return args

    encoder_variants = [
        "transformer",
        "transformer_hidden_mask",
        "transformer_attn_mask",
        "transformer_learned_mask",
        "xlstm",
        "lstm",
        "grud",
        "att_grud",
    ]
    mask_aware = {"transformer_hidden_mask", "transformer_attn_mask", "transformer_learned_mask"}
    for encoder_name in encoder_variants:
        for decoder_name in ["mlp", "cnn"]:
            if "src.mSHRED.mSHRED" in sys.modules:
                del sys.modules["src.mSHRED.mSHRED"]
            model_module = mshred_module
            model_cls = model_module.mSHRED
            model = model_cls(build_mshred_args(encoder_name, decoder_name))
            x = torch.randn(2, 4, 3)
            if encoder_name in mask_aware:
                y = model(x, mask=torch.ones_like(x))
            else:
                y = model(x)
            assert y.shape == (2, 5)

    if "src.mSHRED.mSHRED" in sys.modules:
        del sys.modules["src.mSHRED.mSHRED"]
    model_module = importlib.import_module("src.mSHRED.mSHRED")
    model_cls = model_module.mSHRED
    model = model_cls(build_mshred_args("transformer", "cnn", decoder_3d=True))
    y = model(torch.randn(2, 4, 3))
    assert isinstance(y, list)
    assert len(y) == 4
    assert y[0].shape == (2, 5)
