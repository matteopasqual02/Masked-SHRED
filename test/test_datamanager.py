from itertools import count

import numpy as np      #type: ignore
import pytest           #type: ignore
import torch            #type: ignore

from src.datamanager import (
    DataManager,
    generate_lagged_sensor_measurements,
    get_data,
    get_sensor_measurements,
    parse_compress,
)


def test_parse_compress_variants():
    assert parse_compress(None) == 0
    assert parse_compress(False) == 0
    assert parse_compress(True) == 50
    assert parse_compress(0) == 0
    assert parse_compress(7) == 7


def test_parse_compress_invalid_raises():
    with pytest.raises(ValueError):
        parse_compress(-1)
    with pytest.raises(ValueError):
        parse_compress("bad")


def test_get_data_from_supported_sources(tmp_path):
    arr = np.arange(12, dtype=float).reshape(3, 4)
    tensor = torch.tensor(arr)

    npy_path = tmp_path / "sample.npy"
    np.save(npy_path, arr)

    npz_path = tmp_path / "sample.npz"
    np.savez(npz_path, arr)

    np.testing.assert_allclose(get_data(arr), arr)
    np.testing.assert_allclose(get_data(tensor), arr)
    np.testing.assert_allclose(get_data(str(npy_path)), arr)
    np.testing.assert_allclose(get_data(str(npz_path)), arr)

    with pytest.raises(ValueError):
        get_data(123)


def test_generate_lagged_sensor_measurements_shape_and_padding():
    sensor_measurements = np.array(
        [
            [1.0, 10.0],
            [2.0, 20.0],
            [3.0, 30.0],
        ]
    )
    lags = 2

    lagged = generate_lagged_sensor_measurements(sensor_measurements, lags)

    assert lagged.shape == (3, 2, 2)
    np.testing.assert_allclose(lagged[0], np.array([[0.0, 0.0], [1.0, 10.0]]))
    np.testing.assert_allclose(lagged[1], np.array([[1.0, 10.0], [2.0, 20.0]]))


def test_datamanager_add_data_prepare_pipeline():
    rng = np.random.default_rng(0)
    data = rng.normal(size=(30, 3, 2)).astype(np.float32)

    dm = DataManager(lags=4, train_size=0.6, val_size=0.2, test_size=0.2)
    dm.add_data(data=data, id="flow", random=2, compress=False, seed=0)

    train_dataset, val_dataset, test_dataset = dm.prepare()

    assert len(train_dataset) == 18
    assert len(val_dataset) == 6
    assert len(test_dataset) == 6

    x0, y0 = train_dataset[0]
    assert x0.shape == (4, 2)
    assert y0.shape == (6,)

    assert dm.sensor_summary_df is not None
    assert dm.sensor_measurements_df is not None
    assert dm.sensor_scaler is not None
    assert dm.data_scaler is not None


def test_datamanager_duplicate_dataset_id_raises():
    data = np.zeros((10, 2, 2), dtype=np.float32)
    dm = DataManager()

    dm.add_data(data=data, id="dup", random=1, compress=False, seed=1)
    with pytest.raises(ValueError):
        dm.add_data(data=data, id="dup", random=1, compress=False, seed=1)


def test_get_sensor_measurements_mixed_sources():
    data = np.arange(5 * 2 * 2, dtype=float).reshape(5, 2, 2)
    stationary = [(0, 0)]
    provided_measurements = np.stack(
        [
            np.linspace(0.0, 1.0, 5),
            np.linspace(1.0, 2.0, 5),
        ],
        axis=1,
    )

    out = get_sensor_measurements(
        data=data,
        id="case",
        sensor_number=count(start=0),
        random=1,
        stationary=stationary,
        mobile=None,
        measurements=provided_measurements,
        seed=0,
    )

    # 1 random + 1 stationary + 2 provided sensors.
    assert out["sensor_summary"].shape[0] == 4
    assert out["sensor_measurements"].shape == (5, 4)
    assert out["sensor_measurements_df"].shape == (5, 4)


def test_get_sensor_measurements_invalid_mobile_raises():
    data = np.zeros((4, 2, 2), dtype=float)
    # Wrong length: mobile trajectory must provide one coordinate per timestep.
    mobile = [[(0, 0), (0, 1)]]

    with pytest.raises(ValueError):
        get_sensor_measurements(
            data=data,
            id="bad",
            sensor_number=count(start=0),
            random=None,
            stationary=None,
            mobile=mobile,
            measurements=None,
            seed=None,
        )