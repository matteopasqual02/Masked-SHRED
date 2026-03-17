from typing import Union, List, Tuple, Optional     #type: ignore
import numpy as np                                  #type: ignore
import torch                                        #type: ignore
import pandas as pd                                 #type: ignore

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score   #type: ignore

from src.datamanager import DataManager, generate_lagged_sensor_measurements, generate_GRUD_4D_lags
from src.mSHRED.mSRED import mSHRED
from src.utils import generate_GRUD_4D_input

class SHREDEngine:

    def __init__(self, datamanager:DataManager, model: mSHRED):
        # initialize the SHRED engine with a data manager and a trained model
        self.dm = datamanager
        self.model = model
        # ensure model is in eval mode
        self.model.eval()

    def sensor_to_reconstruction(self, sensor_measurements):
        # convert raw sensor measurements into latent-space embeddings
        
        # 1) Pull out raw numpy array
        if isinstance(sensor_measurements, pd.DataFrame):
            sensor_measurements = sensor_measurements.values
        elif torch.is_tensor(sensor_measurements):
            sensor_measurements = sensor_measurements.detach().cpu().numpy()
        elif isinstance(sensor_measurements, np.ndarray):
            sensor_measurements = sensor_measurements
        else:
            raise TypeError(f"Unsupported type {type(sensor_measurements)} for sensor_measurements")
        
        # 2) Scale using the DataManager's fitted scaler (shape -> (T, n_sensors))
        scaled_sensor_measurements = self.dm.sensor_scaler.transform(sensor_measurements)  
        
        # 3) Build lagged windows based on encoder type.
        # Standard encoders expect (T, lags, n_sensors), while GRUD expects (T, 4, lags, n_sensors).
        lags = self.dm.lags
        encoder_name = self.model.encoder.__class__.__name__.lower()
        if encoder_name == "grud":
            grud_input = generate_GRUD_4D_input(scaled_sensor_measurements)
            lagged = generate_GRUD_4D_lags(grud_input, lags)
        else:
            lagged = generate_lagged_sensor_measurements(scaled_sensor_measurements, lags)
        
        # 4) To torch on same device as model:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        X = torch.tensor(lagged, dtype=torch.float32, device=device)
        
        # 5) Run through sequence to get latent:
        with torch.no_grad():
            output = self.model(X)

        # 6) Convert to numpy and inverse scale using the DataManager's fitted scaler
        output = self.dm.data_scaler.inverse_transform(output.detach().cpu().numpy())

        # 7) If SVD was applied, inverse transform using the Vt matrix
        results = {}
        start_index = 0
        for id in self.dm._dataset_ids:
            length = self.dm._dataset_lengths.get(id)
            Vt = self.dm._Vt_registry.get(id)
            preSVD_scaler = self.dm._preSVD_scaler_registry.get(id)
            spatial_shape = self.dm._dataset_spatial_shape.get(id)
            dataset = output[:,start_index:start_index+length]
            if Vt is not None:
                dataset = dataset @ Vt
            if preSVD_scaler is not None:
                dataset = preSVD_scaler.inverse_transform(dataset)
            original_shape = (dataset.shape[0],) + spatial_shape
            results[id] = dataset.reshape(original_shape)
            start_index = length + start_index

        # Return a dictionary mapping dataset IDs to their corresponding reconstructed physical states
        return results


    def evaluate(self,sensor_measurements, Y):
        
        # 1) Get the model's reconstruction in raw space
        recon_dict = self.sensor_to_reconstruction(sensor_measurements)   # dict[id] -> (T, *spatial_shape)
        
        # 2) Compute stats
        records = []
        for id, y_true in Y.items():
            y_pred = recon_dict[id]
            y_true_flat = y_true.reshape(y_true.shape[0], -1)
            y_pred_flat = y_pred.reshape(y_pred.shape[0], -1)

            mse   = mean_squared_error(y_true_flat, y_pred_flat)
            rmse  = np.sqrt(mse)
            mae   = mean_absolute_error(y_true_flat, y_pred_flat)
            r2    = r2_score(y_true_flat, y_pred_flat)

            records.append({
                "dataset": id, "MSE": mse, "RMSE": rmse,
                "MAE": mae, "R2": r2
            })
        return pd.DataFrame.from_records(records).set_index("dataset")

        