import numpy as np
import pytest
import torch

from src.datamanager import (
    DataManager,
    generate_lagged_sensor_measurements,
    TimeSeriesDataset,
)


class TestDataManagerWithMedianImputer:
    """Test DataManager with median imputation"""

    @pytest.fixture
    def sample_temporal_data(self):
        """Create sample temporal data for testing"""
        np.random.seed(42)
        # Shape: (time_steps, spatial_x, spatial_y)
        time_steps = 100
        spatial_x, spatial_y = 10, 10
        data = np.random.randn(time_steps, spatial_x, spatial_y).astype(np.float32)
        return data

    @pytest.fixture
    def sample_measurements_with_nans(self):
        """Create sample measurements with NaN values"""
        np.random.seed(42)
        measurements = np.random.randn(100, 5).astype(np.float32)
        # Add some NaN values
        nan_mask = np.random.rand(100, 5) < 0.1
        measurements[nan_mask] = np.nan
        return measurements

    def test_datamanager_init(self):
        """Test DataManager initialization"""
        dm = DataManager(lags=10, train_size=0.7, val_size=0.15, test_size=0.15)
        assert dm.lags == 10
        assert dm.train_size == 0.7
        assert dm.val_size == 0.15
        assert dm.test_size == 0.15

    def test_datamanager_init_invalid_split(self):
        """Test DataManager rejects invalid splits"""
        with pytest.raises(ValueError):
            DataManager(train_size=0.7, val_size=0.2, test_size=0.2)

    def test_add_data_without_measurements(self, sample_temporal_data):
        """Test adding data without sensor measurements"""
        dm = DataManager(lags=5, train_size=0.8, val_size=0.1, test_size=0.1)
        dm.add_data(sample_temporal_data, id="spatial_data", compress=False)

        assert "spatial_data" in dm._dataset_ids
        assert dm.data is not None
        assert dm.data.shape[0] == sample_temporal_data.shape[0]

    def test_add_data_with_measurements_no_imputation(
        self, sample_temporal_data, sample_measurements_with_nans
    ):
        """Test adding data with measurements but no imputation"""
        dm = DataManager(lags=5, train_size=0.8, val_size=0.1, test_size=0.1)
        
        # Create measurements without NaNs for this test
        measurements_clean = np.random.randn(100, 5).astype(np.float32)
        
        dm.add_data(
            sample_temporal_data,
            id="data_with_measurements",
            measurements=measurements_clean,
            compress=False,
            imputer=None,
        )

        assert dm.sensor_measurements is not None
        assert dm.sensor_measurements.shape == (100, 5)
        assert dm.sensor_summary_df is not None
        assert len(dm.sensor_summary_df) == 5

    def test_add_data_with_median_imputation(
        self, sample_temporal_data, sample_measurements_with_nans
    ):
        """Test adding data with median imputation"""
        dm = DataManager(lags=5, train_size=0.8, val_size=0.1, test_size=0.1)

        dm.add_data(
            sample_temporal_data,
            id="data_with_imputation",
            measurements=sample_measurements_with_nans,
            compress=False,
            imputer="median",
        )

        # Check that imputation happened
        assert dm.sensor_measurements is not None
        assert dm.sensor_measurements.shape == (100, 5)
        
        # Check that no NaNs remain
        assert not np.isnan(dm.sensor_measurements).any()
        
        # Check that sensor summary is created
        assert dm.sensor_summary_df is not None
        assert len(dm.sensor_summary_df) == 5

    def test_imputed_values_are_medians(self, sample_temporal_data):
        """Test that imputed values match expected medians from training data"""
        np.random.seed(42)
        measurements = np.random.randn(100, 3).astype(np.float32)
        
        # Add specific NaNs in test set
        measurements[85:, 0] = np.nan  # NaNs in test set only
        measurements[90:, 1] = np.nan
        
        dm = DataManager(lags=5, train_size=0.8, val_size=0.1, test_size=0.1)
        dm.add_data(
            sample_temporal_data,
            id="test_imputed_values",
            measurements=measurements,
            compress=False,
            imputer="median",
        )

        # Get train medians
        train_measurements = measurements[:80]
        train_median_col0 = np.nanmedian(train_measurements[:, 0])
        
        # Check that imputed values in test set match the train median
        imputed_values = dm.sensor_measurements[85:, 0]
        # Should be close to median (allowing for torch median rounding differences)
        assert np.allclose(imputed_values, train_median_col0, rtol=0.1)

    def test_multiple_datasets_with_imputation(self, sample_temporal_data):
        """Test adding multiple datasets with imputation"""
        dm = DataManager(lags=5, train_size=0.8, val_size=0.1, test_size=0.1)

        measurements1 = np.random.randn(100, 3).astype(np.float32)
        measurements1[85:, 0] = np.nan
        
        measurements2 = np.random.randn(100, 2).astype(np.float32)
        measurements2[85:, 1] = np.nan

        dm.add_data(
            sample_temporal_data,
            id="dataset1",
            measurements=measurements1,
            compress=False,
            imputer="median",
        )

        dm.add_data(
            sample_temporal_data,
            id="dataset2",
            measurements=measurements2,
            compress=False,
            imputer="median",
        )

        # Check that both datasets were added
        assert len(dm._dataset_ids) == 2
        assert "dataset1" in dm._dataset_ids
        assert "dataset2" in dm._dataset_ids
        
        # Check combined sensor measurements
        assert dm.sensor_measurements is not None
        assert dm.sensor_measurements.shape == (100, 5)  # 3 + 2 sensors
        
        # Check no NaNs in final data
        assert not np.isnan(dm.sensor_measurements).any()

    def test_sensor_measurements_train_val_test_split(self, sample_temporal_data):
        """Test that sensor measurements are correctly split by imputation"""
        measurements = np.random.randn(100, 2).astype(np.float32)
        measurements[80:, 0] = np.nan  # NaNs in val/test
        
        dm = DataManager(lags=5, train_size=0.8, val_size=0.1, test_size=0.1)
        dm.add_data(
            sample_temporal_data,
            id="split_test",
            measurements=measurements,
            compress=False,
            imputer="median",
        )

        # Check that the imputation was applied after split
        assert dm.train_sensor_measurements is not None
        assert dm.val_sensor_measurements is not None
        assert dm.test_sensor_measurements is not None
        
        assert dm.train_sensor_measurements.shape == (80, 2)
        assert dm.val_sensor_measurements.shape == (10, 2)
        assert dm.test_sensor_measurements.shape == (10, 2)

    def test_prepare_with_imputed_data(self, sample_temporal_data):
        """Test prepare() method with imputed sensor measurements"""
        measurements = np.random.randn(100, 3).astype(np.float32)
        measurements[85:, :] = np.nan
        
        dm = DataManager(lags=5, train_size=0.8, val_size=0.1, test_size=0.1)
        dm.add_data(
            sample_temporal_data,
            id="prepare_test",
            measurements=measurements,
            compress=False,
            imputer="median",
        )

        train_ds, val_ds, test_ds = dm.prepare(verbose=False)

        # Check that datasets are created
        assert isinstance(train_ds, TimeSeriesDataset)
        assert isinstance(val_ds, TimeSeriesDataset)
        assert isinstance(test_ds, TimeSeriesDataset)

        # Check dataset sizes (considering lags)
        assert len(train_ds) == 80
        assert len(val_ds) == 10
        assert len(test_ds) == 10

        # Check shapes
        X_train, Y_train = train_ds[0]
        assert X_train.shape == (5, 3)  # (lags, num_sensors)

    def test_prepare_without_sensor_measurements_raises(self, sample_temporal_data):
        """Test that prepare() raises error when no sensor measurements"""
        dm = DataManager(lags=5)
        dm.add_data(sample_temporal_data, id="no_sensors", compress=False)

        with pytest.raises(ValueError):
            dm.prepare()

    def test_lagged_sensor_measurements(self, sample_temporal_data):
        """Test lagged sensor measurements generation"""
        measurements = np.random.randn(100, 2).astype(np.float32)
        measurements[85:, :] = np.nan
        
        dm = DataManager(lags=10, train_size=0.8, val_size=0.1, test_size=0.1)
        dm.add_data(
            sample_temporal_data,
            id="lag_test",
            measurements=measurements,
            compress=False,
            imputer="median",
        )

        train_ds, val_ds, test_ds = dm.prepare(verbose=False)

        # Check lagged shape
        X_train, _ = train_ds[0]
        assert X_train.shape == (10, 2)  # (lags, sensors)

        # The early timesteps are left-padded with zeros, while the last row
        # carries the current observation.
        X_early = train_ds[0][0]
        assert torch.all(X_early[:-1] == 0)

    def test_sensor_measurements_consistency_across_calls(self, sample_temporal_data):
        """Test that sensor measurements are consistent across multiple calls"""
        measurements = np.random.randn(100, 2).astype(np.float32)
        measurements[85:, :] = np.nan
        
        dm = DataManager(lags=5, train_size=0.8, val_size=0.1, test_size=0.1)
        dm.add_data(
            sample_temporal_data,
            id="consistency_test",
            measurements=measurements,
            compress=False,
            imputer="median",
            seed=42,
        )

        measurements_copy = dm.sensor_measurements.copy()
        
        # Prepare should not modify measurements
        dm.prepare(verbose=False)
        
        np.testing.assert_array_equal(dm.sensor_measurements, measurements_copy)

    def test_imputation_only_uses_training_data_medians(self):
        """Test that imputation medians come only from training data"""
        np.random.seed(42)
        data = np.random.randn(100, 5, 5)
        
        # Create measurements where training dist is different from overall dist
        measurements = np.zeros((100, 2))
        measurements[:80] = np.random.normal(loc=0, scale=1, size=(80, 2))  # Train: mean 0, std 1
        measurements[80:] = np.random.normal(loc=10, scale=1, size=(20, 2))  # Val/test: mean 10
        measurements[85:, :] = np.nan  # Add NaNs in val/test
        
        dm = DataManager(lags=5, train_size=0.8, val_size=0.1, test_size=0.1)
        dm.add_data(
            data,
            id="median_source_test",
            measurements=measurements,
            compress=False,
            imputer="median",
        )

        # Imputed values should be close to train medians (around 0), not val/test medians (around 10)
        imputed_vals = dm.sensor_measurements[85:]
        assert np.all(np.abs(imputed_vals) < 5)  # Much closer to 0 than 10


class TestGenerateLaggedSensorMeasurements:
    """Test lagged sensor measurements generation"""

    def test_lagged_shape(self):
        """Test correct shape of lagged measurements"""
        measurements = np.random.randn(100, 3)
        lags = 10
        
        lagged = generate_lagged_sensor_measurements(measurements, lags)
        
        assert lagged.shape == (100, 10, 3)

