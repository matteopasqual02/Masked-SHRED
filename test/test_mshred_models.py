import numpy as np      #type: ignore
import pytest           #type: ignore
import torch            #type: ignore

from src.mSHRED.encoders.attnGRUD import AttnGRUD, FilterLinear
from src.mSHRED.mSHRED import mSHRED


def test_filterlinear_applies_mask_to_weights():
    mask = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    layer = FilterLinear(in_features=2, out_features=2, filter_square_matrix=mask, bias=False)

    with torch.no_grad():
        layer.weight.fill_(1.0)

    x = torch.tensor([[3.0, 5.0]])
    out = layer(x)

    np.testing.assert_allclose(out.detach().cpu().numpy(), np.array([[3.0, 5.0]]), rtol=1e-6)


def test_attngrud_forward_output_shapes():
    batch_size = 4
    steps = 6
    features = 3
    hidden_size = 5

    x_mean = torch.zeros(1, 1, steps, features)
    model_all = AttnGRUD(
        input_size=features,
        cell_size=hidden_size,
        hidden_size=hidden_size,
        X_mean=x_mean,
        output_last=False,
    )
    model_last = AttnGRUD(
        input_size=features,
        cell_size=hidden_size,
        hidden_size=hidden_size,
        X_mean=x_mean,
        output_last=True,
    )

    # Input layout expected by GRUD-style encoders: (B, 4, T, D)
    sample = torch.randn(batch_size, 4, steps, features)

    out_all = model_all(sample)
    out_last = model_last(sample)

    assert out_all.shape == (steps, batch_size, hidden_size)
    assert out_last.shape == (steps, batch_size, hidden_size)


def test_mshred_transformer_mlp_forward_and_freeze_unfreeze():
    if not hasattr(torch, "gelu"):
        torch.gelu = torch.nn.functional.gelu  # type: ignore[attr-defined]

    args = {
        "encoder": "transformer",
        "decoder": "mlp",
        "input_size": 3,
        "hidden_size": 8,
        "output_size": 4,
        "input_length": 5,
        "n_heads": 1,
        "encoder_depth": 2,
        "dim_feedforward": 16,
        "dropout": 0.0,
        "num_layers": 1,
        "xlstm_type": "slstm",
        "decoder_hidden_sizes": [6],
        "double_channels": False,
    }
    model = mSHRED(args)

    x = torch.randn(7, args["input_length"], args["input_size"])
    y = model(x)
    assert y.shape == (7, args["output_size"])

    model.freeze()
    assert not model.training
    assert all(not p.requires_grad for p in model.parameters())

    model.unfreeze()
    assert model.training
    assert all(p.requires_grad for p in model.parameters())


def test_mshred_invalid_encoder_raises():
    args = {
        "encoder": "not-implemented",
        "decoder": "mlp",
        "input_size": 3,
        "hidden_size": 8,
        "output_size": 4,
        "input_length": 5,
        "n_heads": 1,
        "encoder_depth": 1,
        "dim_feedforward": 16,
        "dropout": 0.0,
        "num_layers": 1,
        "xlstm_type": "slstm",
        "decoder_hidden_sizes": [6],
        "double_channels": False,
    }

    with pytest.raises(NotImplementedError):
        mSHRED(args)


def test_mshred_invalid_decoder_raises():
    args = {
        "encoder": "transformer",
        "decoder": "not-implemented",
        "input_size": 3,
        "hidden_size": 8,
        "output_size": 4,
        "input_length": 5,
        "n_heads": 1,
        "encoder_depth": 1,
        "dim_feedforward": 16,
        "dropout": 0.0,
        "num_layers": 1,
        "xlstm_type": "slstm",
        "decoder_hidden_sizes": [6],
        "double_channels": False,
    }

    with pytest.raises(NotImplementedError):
        mSHRED(args)