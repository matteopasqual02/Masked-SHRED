import sys
import types

import numpy as np  # type: ignore
import pytest  # type: ignore
import torch  # type: ignore
import torch.nn as nn  # type: ignore
from sklearn.preprocessing import MinMaxScaler  # type: ignore

from src.mSHRED.mSRED import mSHRED


class _EchoEncoder(nn.Module):
    def __init__(self, *args, hidden_size=8, **kwargs):
        super().__init__()
        self.hidden_size = hidden_size

    def forward(self, x):
        # Accept (B, T, D) and produce (B, H) to match decoders.
        return torch.ones(x.shape[0], self.hidden_size, dtype=x.dtype, device=x.device)


class _LinearDecoder(nn.Module):
    def __init__(self, *args, input_size=8, output_size=4, **kwargs):
        super().__init__()
        self.layer = nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.layer(x)


def _stub_module(name, **symbols):
    module = types.ModuleType(name)
    for key, value in symbols.items():
        setattr(module, key, value)
    return module


@pytest.fixture()
def stubbed_linkage_modules(monkeypatch):
    # mSRED imports these modules dynamically inside __init__.
    monkeypatch.setitem(sys.modules, "src.mSHRED.encoders.transformer", _stub_module("src.mSHRED.encoders.transformer", Transformer=_EchoEncoder))
    monkeypatch.setitem(sys.modules, "src.mSHRED.encoders.xlstm.xlstm", _stub_module("src.mSHRED.encoders.xlstm.xlstm", XLSTM=_EchoEncoder))
    monkeypatch.setitem(sys.modules, "src.mSHRED.encoders.LSTM", _stub_module("src.mSHRED.encoders.LSTM", LSTM=_EchoEncoder))
    monkeypatch.setitem(sys.modules, "src.mSHRED.encoders.GRUD", _stub_module("src.mSHRED.encoders.GRUD", GRUD=_EchoEncoder))
    monkeypatch.setitem(sys.modules, "src.mSHRED.encoders.attnGRUD", _stub_module("src.mSHRED.encoders.attnGRUD", attnGRUD=_EchoEncoder))
    monkeypatch.setitem(sys.modules, "src.mSHRED.decoders.MLP", _stub_module("src.mSHRED.decoders.MLP", MLP=_LinearDecoder))
    monkeypatch.setitem(sys.modules, "src.mSHRED.decoders.CNN", _stub_module("src.mSHRED.decoders.CNN", CNN=_LinearDecoder))


@pytest.mark.parametrize("encoder_name", ["transformer", "xlstm", "lstm", "grud", "attnGRUD"])
@pytest.mark.parametrize("decoder_name", ["mlp", "cnn"])
def test_mshred_dynamic_linkage_all_pairs(stubbed_linkage_modules, encoder_name, decoder_name):
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
    }
    model = mSHRED(args)

    x = torch.randn(6, args["input_length"], args["input_size"])
    y = model(x)

    assert y.shape == (6, args["output_size"])


def test_engine_sensor_to_reconstruction_and_evaluate(monkeypatch):
    # Stub the external model classes imported by src.engine at import time.
    monkeypatch.setitem(sys.modules, "src.shred", _stub_module("src.shred"))
    monkeypatch.setitem(sys.modules, "src.tshred", _stub_module("src.tshred"))
    monkeypatch.setitem(sys.modules, "src.shred.SHRED", _stub_module("src.shred.SHRED", SHRED=type("SHRED", (nn.Module,), {})))
    monkeypatch.setitem(sys.modules, "src.tshred.TSHRED", _stub_module("src.tshred.TSHRED", TSHRED=type("TSHRED", (nn.Module,), {})))

    import importlib

    engine_module = importlib.import_module("src.engine")
    SHREDEngine = engine_module.SHREDEngine

    class DummyModel(nn.Module):
        def forward(self, x):
            # Return two latent features for each timestep.
            return torch.stack((x[:, -1, 0], x[:, -1, 1]), dim=1)

    class DummyDM:
        def __init__(self):
            raw_sensor = np.array(
                [[0.0, 0.0], [0.2, 0.3], [0.4, 0.6], [0.6, 0.9], [1.0, 1.2]],
                dtype=float,
            )
            raw_state = np.array(
                [[0.0, 0.0], [0.2, 0.3], [0.4, 0.6], [0.6, 0.9], [1.0, 1.2]],
                dtype=float,
            )

            self.lags = 2
            self.sensor_scaler = MinMaxScaler().fit(raw_sensor)
            self.data_scaler = MinMaxScaler().fit(raw_state)
            self._dataset_ids = ["setA"]
            self._dataset_lengths = {"setA": 2}
            self._Vt_registry = {}
            self._preSVD_scaler_registry = {}
            self._dataset_spatial_shape = {"setA": (2,)}

    dm = DummyDM()
    model = DummyModel()
    engine = SHREDEngine(datamanager=dm, model=model)

    sensor_measurements = np.array(
        [[0.0, 0.0], [0.2, 0.3], [0.4, 0.6], [0.6, 0.9], [1.0, 1.2]],
        dtype=float,
    )
    rec = engine.sensor_to_reconstruction(sensor_measurements)

    assert set(rec.keys()) == {"setA"}
    assert rec["setA"].shape == (5, 2)

    # Build a simple target dictionary and ensure evaluate returns metrics table.
    eval_df = engine.evaluate(sensor_measurements, {"setA": rec["setA"]})
    assert list(eval_df.index) == ["setA"]
    assert all(col in eval_df.columns for col in ["MSE", "RMSE", "MAE", "R2"])
