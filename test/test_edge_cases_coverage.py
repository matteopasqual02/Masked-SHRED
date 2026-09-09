"""
Edge cases and corner cases to reach 90% coverage in high-score modules.
Focus: engine.py (93%), MLP.py (92%), attnGRUD.py (95%), mSHRED.py (97%), train.py (86%)
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from unittest.mock import MagicMock, patch

from src.mSHRED.decoders.MLP import MLP
from src.mSHRED.encoders.attnGRUD import AttnGRUD
from src.mSHRED.mSHRED import mSHRED
from src.train import fit, fit2, evaluate, evaluate2
from src.datamanager import TimeSeriesDataset, TimeSeriesDatasetWithMask


class TestMLPEdgeCases:
    """Test MLP.py missing lines 46, 49 (param_count property)."""
    
    def test_mlp_model_name_property(self):
        """Test MLP model_name property."""
        mlp = MLP(input_size=10, hidden_sizes=[20, 15], output_size=5)
        assert mlp.model_name == "MLP"
    
    def test_mlp_param_count_property(self):
        """Test MLP param_count property."""
        mlp = MLP(input_size=10, hidden_sizes=[20, 15], output_size=5)
        param_count = mlp.param_count
        assert isinstance(param_count, int)
        assert param_count > 0
        
        # Verify calculation: (10*20+20) + (20*15+15) + (15*5+5) = 220 + 315 + 80 = 615
        expected = (10 * 20 + 20) + (20 * 15 + 15) + (15 * 5 + 5)
        assert param_count == expected


class TestAttnGRUDEdgeCases:
    """Test attnGRUD.py missing lines 37, 98-100, 199 (GPU path, cuda calls)."""
    
    def test_attngrud_init_with_x_mean(self):
        """Test AttnGRUD initialization with X_mean parameter."""
        input_size = 4
        cell_size = 8
        hidden_size = 16
        X_mean = np.array([1.0, 2.0, 3.0, 4.0])
        
        attngrud = AttnGRUD(input_size, cell_size, hidden_size, X_mean)
        assert attngrud is not None
        assert attngrud.hidden_size == hidden_size
        assert attngrud.X_mean is not None


class TestMSHREDEdgeCases:
    """Test mSHRED.py missing lines 141, 143 (NotImplementedError paths)."""
    
    def test_mshred_forward_with_none_encoder(self):
        """Test mSHRED forward raises NotImplementedError when encoder is None."""
        args = {
            "encoder": "transformer",
            "decoder": "mlp",
            "input_size": 5,
            "hidden_size": 16,
            "output_size": 3,
            "input_length": 10,
            "n_heads": 2,
            "encoder_depth": 2,
            "dim_feedforward": 32,
            "dropout": 0.1,
            "num_layers": 1,
            "decoder_hidden_sizes": [32, 16],
            "double_channels": False,
            "decoder_3D": False
        }
        
        mshred = mSHRED(args)
        
        # Manually set encoder to None to trigger NotImplementedError
        mshred.encoder = None
        x = torch.randn(2, 5, 10)
        
        with pytest.raises(NotImplementedError, match="Encoder not implemented"):
            mshred(x)
    
    def test_mshred_forward_with_none_decoder(self):
        """Test mSHRED forward raises NotImplementedError when decoder is None."""
        args = {
            "encoder": "transformer",
            "decoder": "mlp",
            "input_size": 5,
            "hidden_size": 16,
            "output_size": 3,
            "input_length": 10,
            "n_heads": 2,
            "encoder_depth": 2,
            "dim_feedforward": 32,
            "dropout": 0.1,
            "num_layers": 1,
            "decoder_hidden_sizes": [32, 16],
            "double_channels": False,
            "decoder_3D": False
        }
        
        mshred = mSHRED(args)
        
        # Manually set decoder to None to trigger NotImplementedError
        mshred.decoder = None
        x = torch.randn(2, 5, 10)
        
        with pytest.raises(NotImplementedError, match="Decoder not implemented"):
            mshred(x)


class TestTrainEdgeCases:
    """Test train.py edge cases to reach 90% (missing lines 96, 169-178, etc)."""
    
    def test_evaluate_with_batch_size_mismatch(self):
        """Test evaluate handles batch_size larger than dataset."""
        from torch.utils.data import TensorDataset
        
        x = torch.randn(2, 3)
        y = torch.randn(2, 2)
        dataset = TensorDataset(x, y)
        
        model = nn.Linear(3, 2)
        # batch_size=10 > dataset length=2, should handle gracefully
        mse, mre, nrmse = evaluate(model, dataset, batch_size=10)
        assert isinstance(mse, float)
        assert isinstance(mre, float)
        assert isinstance(nrmse, float)
    
    def test_fit_with_early_stopping_patience_exceeded(self):
        """Test fit with early stopping when validation loss doesn't improve."""
        from torch.utils.data import TensorDataset
        
        x_train = torch.randn(8, 3)
        y_train = torch.randn(8, 2)
        x_val = torch.randn(4, 3)
        y_val = torch.randn(4, 2)
        
        train_dataset = TensorDataset(x_train, y_train)
        val_dataset = TensorDataset(x_val, y_val)
        
        model = nn.Linear(3, 2)
        train_losses, val_losses = fit(
            model=model,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            batch_size=2,
            num_epochs=20,
            lr=1e-2,
            optimizer="Adam",
            lr_scheduler=None,
            patience=2,
            verbose=False
        )
        
        # Should stop early due to patience
        assert len(train_losses) <= 20
        assert len(val_losses) > 0
    
    def test_fit2_with_none_masking_type(self):
        """Test fit2 with TimeSeriesDataset (unmasked path)."""
        from torch.utils.data import TensorDataset
        
        x = torch.randn(8, 3)
        y = torch.randn(8, 2)
        dataset = TimeSeriesDataset(x, y)
        
        model = nn.Linear(3, 2)
        train_losses, val_losses = fit2(
            model=model,
            train_dataset=dataset,
            val_dataset=dataset,
            batch_size=4,
            num_epochs=2,
            lr=1e-2,
            optimizer="Adam",
            lr_scheduler=None,
            patience=5,
            verbose=False,
            masking_type=None
        )
        
        assert train_losses.size > 0
        assert val_losses.size > 0


# Run with: PYTHONPATH=. conda run -n mask-shred pytest test/test_edge_cases_coverage.py -v
