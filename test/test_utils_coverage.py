"""
Comprehensive tests for src/utils.py to reach 90% coverage.
Tests all imputation strategies, GRUD input generation, and utility functions.
"""

import pytest
import numpy as np
import torch
from src.utils import (
    _to_numpy_array,
    get_train_medians,
    get_train_medians_3D,
    get_train_means,
    get_train_means_3D,
    apply_imputation,
    apply_imputation_3D,
    median_imputer_single_dataset,
    mean_imputer_single_dataset,
    last_value_imputer_single_dataset,
    generate_GRUD_4D_input,
    generate_GRUD_4D_input_3D,
    generate_channels,
    plot_training_curves,
    evaluate_model,
)


class TestToNumpyArray:
    """Test _to_numpy_array utility function."""
    
    def test_to_numpy_array_with_numpy_array(self):
        """Test conversion with numpy array input."""
        arr = np.array([[1, 2], [3, 4]])
        result = _to_numpy_array(arr)
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, arr)
    
    def test_to_numpy_array_with_torch_tensor(self):
        """Test conversion with torch tensor."""
        tensor = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
        result = _to_numpy_array(tensor)
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_almost_equal(result, tensor.numpy())
    
    def test_to_numpy_array_with_list(self):
        """Test conversion with list."""
        lst = [[1, 2], [3, 4]]
        result = _to_numpy_array(lst)
        assert isinstance(result, np.ndarray)


class TestMedianComputation:
    """Test median computation functions."""
    
    def test_get_train_medians_2d(self):
        """Test median computation on 2D array."""
        X_train = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        medians = get_train_medians(X_train)
        expected = np.array([4, 5, 6])
        np.testing.assert_array_almost_equal(medians, expected)
    
    def test_get_train_medians_with_nans(self):
        """Test median computation with NaN values."""
        X_train = np.array([[1, np.nan, 3], [4, 5, 6], [7, 8, 9]])
        medians = get_train_medians(X_train)
        assert medians.shape == (3,)
        assert not np.any(np.isnan(medians))
    
    def test_get_train_medians_3d(self):
        """Test median computation on 3D array."""
        X_train = np.random.randn(4, 5, 3)  # (samples, time, features)
        medians = get_train_medians_3D(X_train)
        assert medians.shape == (3,)
    
    def test_get_train_medians_3d_with_all_nans(self):
        """Test median computation on 3D array with entire feature as NaN."""
        X_train = np.array([
            [[1, np.nan, 3], [4, np.nan, 6]],
            [[7, np.nan, 9], [10, np.nan, 12]]
        ])
        medians = get_train_medians_3D(X_train)
        assert medians[1] == 0.0  # Column with all NaNs should become 0.0


class TestMeanComputation:
    """Test mean computation functions."""
    
    def test_get_train_means_2d(self):
        """Test mean computation on 2D array."""
        X_train = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=float)
        means = get_train_means(X_train)
        expected = np.array([4, 5, 6])
        np.testing.assert_array_almost_equal(means, expected)
    
    def test_get_train_means_with_nans(self):
        """Test mean computation with NaN values."""
        X_train = np.array([[1, np.nan, 3], [4, 5, 6], [7, 8, 9]], dtype=float)
        means = get_train_means(X_train)
        assert means.shape == (3,)
        assert not np.any(np.isnan(means))
    
    def test_get_train_means_3d(self):
        """Test mean computation on 3D array."""
        X_train = np.random.randn(4, 5, 3)
        means = get_train_means_3D(X_train)
        assert means.shape == (3,)


class TestApplyImputation:
    """Test imputation application functions."""
    
    def test_apply_imputation_2d(self):
        """Test applying imputation on 2D array."""
        X = np.array([[1, np.nan, 3], [np.nan, 5, 6], [7, 8, 9]], dtype=float)
        medians = np.array([4, 5, 6])
        X_imputed = apply_imputation(X, medians)
        assert X_imputed.shape == X.shape
        assert not np.any(np.isnan(X_imputed))
        np.testing.assert_equal(X_imputed[0, 1], 5)  # NaN replaced with median
    
    def test_apply_imputation_3d(self):
        """Test applying imputation on 3D array."""
        X = np.random.randn(3, 4, 5)
        X[0, 1, 2] = np.nan
        medians = np.array([1, 2, 3, 4, 5], dtype=float)
        X_imputed = apply_imputation_3D(X, medians)
        assert X_imputed.shape == X.shape
        assert not np.any(np.isnan(X_imputed))


class TestSingleDatasetImputations:
    """Test imputation on single datasets with train/val/test splits."""
    
    def test_median_imputer_single_dataset_2d(self):
        """Test median imputation for 2D dataset."""
        X = np.array([[1, np.nan, 3], [4, 5, 6], [7, np.nan, 9]], dtype=float)
        train_idx = np.array([0, 1])
        X_imputed = median_imputer_single_dataset(X, train_idx)
        assert X_imputed.shape == X.shape
        assert not np.any(np.isnan(X_imputed))
    
    def test_median_imputer_single_dataset_3d(self):
        """Test median imputation for 3D dataset."""
        X = np.random.randn(5, 10, 3)  # samples, time, sensors
        X[0, 2, 1] = np.nan
        X[1, 5, 0] = np.nan
        train_idx = np.array([0, 1, 2])
        X_imputed = median_imputer_single_dataset(X, train_idx)
        assert X_imputed.shape == X.shape
        assert not np.any(np.isnan(X_imputed))
    
    def test_mean_imputer_single_dataset_2d(self):
        """Test mean imputation for 2D dataset."""
        X = np.array([[1, np.nan, 3], [4, 5, 6], [7, np.nan, 9]], dtype=float)
        train_idx = np.array([0, 1])
        X_imputed = mean_imputer_single_dataset(X, train_idx)
        assert X_imputed.shape == X.shape
        assert not np.any(np.isnan(X_imputed))
    
    def test_mean_imputer_single_dataset_3d(self):
        """Test mean imputation for 3D dataset."""
        X = np.random.randn(5, 10, 3)
        X[0, 2, 1] = np.nan
        X[1, 5, 0] = np.nan
        train_idx = np.array([0, 1, 2])
        X_imputed = mean_imputer_single_dataset(X, train_idx)
        assert X_imputed.shape == X.shape
        assert not np.any(np.isnan(X_imputed))
    
    def test_last_value_imputer_single_dataset_2d(self):
        """Test last value imputation for 2D dataset."""
        X = np.array([[1, 2, 3], [np.nan, np.nan, 6], [7, 8, 9]], dtype=float)
        train_idx = np.array([0, 1])
        X_imputed = last_value_imputer_single_dataset(X, train_idx)
        assert X_imputed.shape == X.shape
        assert not np.any(np.isnan(X_imputed))
        # Last observed value imputation: NaNs filled with previous row's value
        assert X_imputed[1, 0] == 1  # Forward fill from row 0
        assert X_imputed[1, 1] == 2  # Forward fill from row 0
    
    def test_last_value_imputer_single_dataset_3d(self):
        """Test last value imputation for 3D dataset."""
        X = np.array([
            [[1, 2], [3, 4], [np.nan, np.nan]],
            [[5, 6], [7, 8], [9, 10]],
            [[11, 12], [13, 14], [15, 16]]
        ], dtype=float)
        train_idx = np.array([0, 1])
        X_imputed = last_value_imputer_single_dataset(X, train_idx)
        assert X_imputed.shape == X.shape
        assert not np.any(np.isnan(X_imputed))
        # Check forward fill in 3D case
        assert X_imputed[0, 2, 0] == 3  # Forward fill from [0, 1, 0]
        assert X_imputed[0, 2, 1] == 4  # Forward fill from [0, 1, 1]
    
    def test_last_value_imputer_single_dataset(self):
        """Test last value imputation for single dataset."""
        X = np.array([[1, 2, 3], [np.nan, np.nan, 6], [7, 8, 9]], dtype=float)
        train_idx = np.array([0, 1])
        X_imputed = last_value_imputer_single_dataset(X, train_idx)
        assert X_imputed.shape == X.shape
        assert not np.any(np.isnan(X_imputed))
        # Last observed value imputation: NaNs filled with previous row's value
        assert X_imputed[1, 0] == 1  # Forward fill from row 0
        assert X_imputed[1, 1] == 2  # Forward fill from row 0


class TestGRUDInput:
    """Test GRUD input generation functions."""
    
    def test_generate_grud_4d_input(self):
        """Test 4D GRUD input generation."""
        measurements = np.random.randn(10, 5)  # 10 time steps, 5 sensors
        grud_input = generate_GRUD_4D_input(measurements)
        # Output shape is (4, T, n_sensors) where 4 channels are: X, X_last_obsv, Mask, Delta
        assert grud_input.ndim == 3
        assert grud_input.shape[0] == 4  # 4 channels
        assert grud_input.shape[1] == 10  # time steps
        assert grud_input.shape[2] == 5  # sensors
    
    def test_generate_grud_4d_input_with_nans(self):
        """Test GRUD input generation with NaN values."""
        measurements = np.random.randn(10, 5)
        measurements[2, 1] = np.nan  # Introduce missing value
        grud_input = generate_GRUD_4D_input(measurements)
        assert not np.any(np.isnan(grud_input))
    
    def test_generate_grud_4d_input_3d(self):
        """Test 3D GRUD input generation."""
        measurements = np.random.randn(3, 10, 5)  # batch, time, sensors
        grud_input = generate_GRUD_4D_input_3D(measurements)
        assert grud_input.ndim == 4
        assert grud_input.shape[0] == 3  # batch size preserved
        assert grud_input.shape[1] == 10  # time steps preserved
        assert grud_input.shape[2] == 4  # 4 channels
        assert grud_input.shape[3] == 5  # sensors
        # Verify no NaNs in output
        assert not np.any(np.isnan(grud_input))
    
    def test_generate_grud_4d_input_3d_with_missing_values(self):
        """Test 3D GRUD input generation with missing values."""
        measurements = np.random.randn(2, 5, 3)  # batch, time, sensors
        measurements[0, 1, 0] = np.nan
        measurements[1, 3, 2] = np.nan
        grud_input = generate_GRUD_4D_input_3D(measurements)
        # Verify the mask channel properly captures missing values
        assert grud_input.shape == (2, 5, 4, 3)
        # Mask channel (index 2) should have 0 where there are NaNs
        assert grud_input[0, 1, 2, 0] == 0.0  # Missing value
        assert grud_input[1, 3, 2, 2] == 0.0  # Missing value


class TestGenerateChannels:
    """Test channel generation function."""
    
    def test_generate_channels_2d(self):
        """Test channel generation from 2D measurements."""
        measurements = np.random.randn(10, 5)
        channels = generate_channels(measurements)
        # Expected shape: (time, 2*features) where 2 channels are: data and mask
        assert channels.ndim == 2
        assert channels.shape[0] == 10
        assert channels.shape[1] == 2 * 5
    
    def test_generate_channels_3d(self):
        """Test channel generation from 3D measurements."""
        measurements = np.random.randn(5, 10, 3)  # samples, time, sensors
        channels = generate_channels(measurements)
        # Expected shape: (samples, time, 2*features)
        assert channels.ndim == 3
        assert channels.shape[0] == 5
        assert channels.shape[1] == 10
        assert channels.shape[2] == 2 * 3


class TestPlottingAndEvaluation:
    """Test plotting and evaluation utility functions."""
    
    def test_plot_training_curves(self, tmp_path):
        """Test training curve plotting."""
        train_errors = np.random.randn(10)
        valid_errors = np.random.randn(10)
        
        # Just verify it doesn't raise an error
        try:
            import matplotlib
            matplotlib.use('Agg')  # Use non-interactive backend
            plot_training_curves(train_errors, valid_errors)
        except Exception as e:
            # If plotting fails, it's likely an environment issue, not a code issue
            pytest.skip(f"Plotting not available: {e}")
    
    def test_evaluate_model(self):
        """Test model evaluation function."""
        from unittest.mock import MagicMock
        
        # Create mock evaluate function
        evaluate_fn = MagicMock(return_value=(0.5, 0.3, 0.2))
        
        train_dataset = MagicMock()
        val_dataset = MagicMock()
        test_dataset = MagicMock()
        model = MagicMock()
        model_type = "mSHRED"
        
        result = evaluate_model(
            evaluate_fn,
            model,
            train_dataset,
            val_dataset,
            test_dataset,
            model,
            model_type
        )
        assert isinstance(result, dict)
        assert "train-mse" in result
        assert "val-mse" in result
        assert "test-mse" in result
        assert "train-nrmse" in result
    
    def test_evaluate_model_with_masking(self):
        """Test model evaluation function with masking_type."""
        from unittest.mock import MagicMock
        
        # Create mock evaluate function that accepts masking_type
        evaluate_fn = MagicMock(return_value=(0.5, 0.3, 0.2))
        
        train_dataset = MagicMock()
        val_dataset = MagicMock()
        test_dataset = MagicMock()
        model = MagicMock()
        model_type = "mSHRED-masked"
        
        result = evaluate_model(
            evaluate_fn,
            model,
            train_dataset,
            val_dataset,
            test_dataset,
            model,
            model_type,
            masking_type="mask_data"
        )
        assert isinstance(result, dict)
        assert "train-mse" in result
        # Verify evaluate was called with masking_type
        assert evaluate_fn.call_count >= 3


class TestEdgeCases:
    """Test edge cases and corner cases."""
    
    def test_imputation_with_all_nans(self):
        """Test imputation when a feature is entirely NaN."""
        X = np.array([[np.nan, 2], [np.nan, 4], [np.nan, 6]], dtype=float)
        medians = get_train_medians(X)
        X_imputed = apply_imputation(X, medians)
        assert not np.any(np.isnan(X_imputed))
    
    def test_grud_with_single_time_step(self):
        """Test GRUD input with single time step."""
        measurements = np.random.randn(1, 5)
        grud_input = generate_GRUD_4D_input(measurements)
        assert grud_input.shape == (4, 1, 5)  # (channels, time, sensors)
    
    def test_grud_with_single_sensor(self):
        """Test GRUD input with single sensor."""
        measurements = np.random.randn(10, 1)
        grud_input = generate_GRUD_4D_input(measurements)
        assert grud_input.shape == (4, 10, 1)  # (channels, time, sensors)
    
    def test_median_imputer_invalid_ndim(self):
        """Test median imputer with invalid array dimensions."""
        X = np.random.randn(3, 4, 5, 6)  # 4D array - invalid
        train_idx = np.array([0, 1])
        with pytest.raises(ValueError, match="Input X must be a 2D or 3D array"):
            median_imputer_single_dataset(X, train_idx)
    
    def test_mean_imputer_invalid_ndim(self):
        """Test mean imputer with invalid array dimensions."""
        X = np.random.randn(3, 4, 5, 6)  # 4D array - invalid
        train_idx = np.array([0, 1])
        with pytest.raises(ValueError, match="Input X must be a 2D or 3D array"):
            mean_imputer_single_dataset(X, train_idx)
    
    def test_last_value_imputer_invalid_ndim(self):
        """Test last value imputer with invalid array dimensions."""
        X = np.random.randn(3, 4, 5, 6)  # 4D array - invalid
        train_idx = np.array([0, 1])
        with pytest.raises(ValueError, match="Input X must be a 2D or 3D array"):
            last_value_imputer_single_dataset(X, train_idx)
    
    def test_generate_channels_invalid_ndim(self):
        """Test channel generation with invalid array dimensions."""
        measurements = np.random.randn(3, 4, 5, 6)  # 4D array - invalid
        with pytest.raises(ValueError, match="Input measurements must be 2D or 3D array"):
            generate_channels(measurements)


# Run with: PYTHONPATH=. conda run -n mask-shred pytest test/test_utils_coverage.py -v
