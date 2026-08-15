import glob                 #type: ignore
from pathlib import Path    #type: ignore
import warnings             #type: ignore
import pandas as pd         #type: ignore
import modelskill as ms     #type: ignore
import mikeio               #type: ignore
import numpy as np          #type: ignore
import matplotlib.pyplot as plt  #type: ignore

obs_fldr = "datasets/observations/" 
df_stn = pd.read_csv(obs_fldr + "stations.csv", index_col=0)
df_stn_cur = pd.read_csv(obs_fldr + "stations.csv", index_col=0)

def get_wl_point_obs():
    # Get water level point observations as list of PointObservation objects.
    # Some datasets use a filtered column name while Oresund CSVs use raw "water_level".
    q = ms.Quantity(name="water level", unit="meter")
    wllist = []
    for i, row in df_stn.iterrows():
        csv_path = Path(obs_fldr) / f"{i}_wl.csv"
        if not csv_path.exists():
            continue

        df = pd.read_csv(csv_path, index_col=0)
        df.index = pd.to_datetime(df.index, format="ISO8601")

        if "water_level_filtered" in df.columns:
            series = df["water_level_filtered"]
        elif "water_level" in df.columns:
            series = df["water_level"]
        else:
            raise ValueError(f"Station {i} has no recognized water level column in {csv_path.name}")

        o = ms.PointObservation(series, x=row['Longitude'], y=row['Latitude'], name=i, quantity=q)
        wllist.append(o)
    return wllist

def get_u_v_point_obs():
    # Get current point observations as two lists of PointObservation objects
    qu = ms.Quantity(name="u-current", unit="m/s")
    qv = ms.Quantity(name="v-current", unit="m/s")
    ulist = []
    vlist = []
    for i, row in df_stn_cur.iterrows():
        if not Path(obs_fldr + f"{i}_u_v.csv").exists():
            continue
        df = pd.read_csv(obs_fldr + f"{i}_u_v.csv", index_col=0) 
        df.index = pd.to_datetime(df.index, format="ISO8601")   
        x, y = row['Longitude'], row['Latitude']
        ou = ms.PointObservation(df["u"], x=x, y=y, name=i, quantity=qu)
        ulist.append(ou)
        ov = ms.PointObservation(df["v"], x=x, y=y, name=i, quantity=qv)
        vlist.append(ov)
    return ulist, vlist

def get_altimetry_obs(quality=None):
    # Get altimetry observations as list of TrackObservation objects
    sat_files = sorted(glob.glob(obs_fldr + "Altimetry_wl_*.csv"))
    missions = [f.split("_")[2].split(".")[0] for f in sat_files]
    q = ms.Quantity(name="Surface Elevation", unit="meter")
    altlist = []
    for m in missions:
        df = pd.read_csv(obs_fldr + f"Altimetry_wl_{m}.csv", index_col=0)
        df.index = pd.to_datetime(df.index, format="ISO8601")
        if quality is not None:
            df = df[df.quality_water_level == quality]
        
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*duplicate timestamps.*")
            o = ms.TrackObservation(df, item="water_level", x_item="longitude", y_item="latitude", name=m, quantity=q)
        altlist.append(o)
    return altlist

def extract_water_level_measurements(stations, ds_time, obs_fldr, window="25min"):
    # Extract water level measurements for given stations and simulation timestamps.
    # For each station, attempts to read filtered water level data aligned with simulation timestamps.
    # If unavailable, falls back to raw water level data and takes the last measurement within 
    # a specified time window before each simulation timestamp.
    
    measurements = np.full((ds_time.shape[0], len(stations)), np.nan)
    ds_times = pd.DatetimeIndex(ds_time.values)
    window_delta = pd.Timedelta(window)
    missing_files = []

    # Build once and reuse in merge_asof fallback.
    target_times = pd.DataFrame({"time": ds_times})

    # Loop through each station and attempt to read corresponding water level data
    for i, station in enumerate(stations):

        # Try both naming conventions used in this dataset
        candidates = [
            obs_fldr + station.name + "_wl.csv",
            obs_fldr + "Altimetry_wl_" + station.name + ".csv",
        ]

        # Find the first existing file among the candidates
        csv_path = next((p for p in candidates if Path(p).exists()), None)
        if csv_path is None:
            missing_files.append(station.name)
            continue

        # Read station data from the identified CSV file
        stn_data = pd.read_csv(csv_path, index_col=0, parse_dates=True)

        # Keep only the last sample for duplicated timestamps to avoid ambiguous alignment.
        stn_data = stn_data[~stn_data.index.duplicated(keep="last")]

        # If filtered data exists, align exactly on simulation timestamps
        if "water_level_filtered" in stn_data.columns:
            measurements[:, i] = stn_data["water_level_filtered"].reindex(ds_times).to_numpy()
            continue

        # Fallback for raw water level data: take last value in a time window
        if "water_level" in stn_data.columns:
            wl_df = stn_data[["water_level"]].sort_index().rename(columns={"water_level": "wl"})
            merged = pd.merge_asof(
                target_times,
                wl_df,
                left_on="time",
                right_index=True,
                direction="backward",
                tolerance=window_delta,
            )
            measurements[:, i] = merged["wl"].to_numpy()

        # If neither filtered nor raw data could be aligned, mark as missing
        else:
            missing_files.append(station.name)

        print(f"Extracted measurements for station {station.name} from {csv_path}")

    # Report any stations for which measurement files were missing or incompatible
    if missing_files:
        print("Missing or incompatible measurement files for:", sorted(set(missing_files)))

    # Return the measurements array
    return measurements

def plot_shred_reconstruction(test_reconstruction, ds, geom, plot_time=None, items=None, cmaps=None, title_suffix=""):
    
    # Set default items and colormaps if not provided
    if items is None:
        items = ["Surface elevation", "U velocity", "V velocity"]
    elif isinstance(items, str):
        items = [items]
    else:
        items = list(items)
    if cmaps is None:
        cmaps = {
            "Surface elevation": "terrain",
            "U velocity": "twilight_shifted",
            "V velocity": "twilight_shifted",
        }

    # Validate that all requested items exist in both input structures
    missing_in_reconstruction = [item for item in items if item not in test_reconstruction]
    if missing_in_reconstruction:
        raise KeyError(
            f"Missing items in test_reconstruction: {missing_in_reconstruction}. "
            f"Available keys: {list(test_reconstruction.keys())}"
        )

    # get the corresponding time values for the test set
    times_test = ds.time[-test_reconstruction[items[0]].shape[0]:]

    # Fill NaN locations in reconstructions for plotting
    reconstructions = {}
    for item in items:
        full = np.full((test_reconstruction[item].shape[0], ds[item].values.shape[1]), np.nan)
        valid_cols = ~np.isnan(ds[item].values).any(axis=0)
        full[:, valid_cols] = test_reconstruction[item]
        reconstructions[item] = full

    # Create datasets for reconstructions and errors
    ds_shred = mikeio.Dataset([
        mikeio.DataArray(data=reconstructions[item], time=times_test, item=ds[item].item, geometry=geom)
        for item in items
    ])
    ds_error = mikeio.Dataset([
        mikeio.DataArray(
            data=np.sqrt((ds[item].sel(time=times_test).values - ds_shred[item].values) ** 2),
            time=times_test,
            item=ds[item].item,
            geometry=geom,
        )
        for item in items
    ])

    metrics = {}
    for item in items:
        target = ds[item].sel(time=times_test).values
        error = target - ds_shred[item].values
        mse = np.nanmean(error ** 2)

        # Mean Relative Error: ||target - prediction|| / ||target|| over last axis,
        # then averaged over all remaining axes.
        eps = 1e-12
        numerator = np.sqrt(np.nansum((target - ds_shred[item].values) ** 2, axis=-1))
        denominator = np.sqrt(np.nansum(target ** 2, axis=-1))
        rel_error = numerator / np.maximum(denominator, eps)
        mre = np.nanmean(rel_error)
        metrics[item] = {"mse": mse, "mre": mre}

    # Plotting
    if plot_time is None:
        plot_time = ds_shred.time[-1]

    fig, axs = plt.subplots(len(items), 3, figsize=(15, 4 * len(items)), squeeze=False)
    for row, item in enumerate(items):
        reference  = ds[item].sel(time=plot_time)
        prediction = ds_shred[item].sel(time=plot_time)
        error      = ds_error[item].sel(time=plot_time)

        vmin = min(np.nanmin(reference.values), np.nanmin(prediction.values))
        vmax = max(np.nanmax(reference.values), np.nanmax(prediction.values))

        cmap = cmaps.get(item, "viridis")
        reference.plot( ax=axs[row, 0], title=item,    cmap=cmap, vmin=vmin, vmax=vmax)
        prediction.plot(ax=axs[row, 1], title="SHRED", cmap=cmap, vmin=vmin, vmax=vmax)
        error.plot(
            ax=axs[row, 2],
            title=f"Error\nMSE={metrics[item]['mse']:.6f}\nMRE={metrics[item]['mre']:.6f}",
            cmap="gray",
        )

    plt.tight_layout()

    # Save each figure to Notebooks/HYDRO/outputs.
    output_dir = Path(__file__).resolve().parent / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"shred_{title_suffix}.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    print(f"Saved plot: {output_path}")

    plt.show()

    return ds_shred, ds_error

def error_mshred(test_reconstruction, ds, items=None):

    if items is None:
        items = ["Surface elevation", "U velocity", "V velocity"]
    elif isinstance(items, str):
        items = [items]

    ntimes = test_reconstruction[items[0]].shape[0]
    n_items = len(items)

    mse_matrix = np.zeros((ntimes, n_items))
    mre_matrix = np.zeros((ntimes, n_items))

    times_test = ds.time[-ntimes:]

    for j, item in enumerate(items):

        target = ds[item].sel(time=times_test).values

        # rebuild full field with NaNs
        full = np.full(
            (ntimes, ds[item].values.shape[1]),
            np.nan
        )

        valid_cols = ~np.isnan(ds[item].values).any(axis=0)
        full[:, valid_cols] = test_reconstruction[item]

        pred = full

        eps = 1e-12

        for t in range(ntimes):

            error = target[t] - pred[t]

            # MSE at timestep t
            mse_matrix[t, j] = np.nanmean(error**2)

            # MRE at timestep t
            numerator = np.sqrt(np.nansum(error**2))
            denominator = np.sqrt(np.nansum(target[t]**2))

            mre_matrix[t, j] = numerator / max(denominator, eps)

    return mse_matrix, mre_matrix

def align_ds(test_idx, ds, permuted_indices, lenmonths, geom):
    # Build test indices in original (chronological) ds based on the month permutation used for training
    perm_time_idx = np.concatenate([np.arange(m * lenmonths, (m + 1) * lenmonths) for m in permuted_indices])
    test_time_idx = np.concatenate([np.arange(i * lenmonths, (i + 1) * lenmonths) for i in test_idx])
    flat_test_idx = perm_time_idx[test_time_idx]

    # Use the exact column mask used during preprocessing/training (based on Surface elevation)
    common_valid_cols = ~np.isnan(ds["Surface elevation"].values).any(axis=0)

    # mikeio requires monotonic increasing time; create a test reference dataset with aligned values
    items = ["Surface elevation", "U velocity", "V velocity"]
    ref_time = ds.time[:len(flat_test_idx)]
    ds_ref_arrays = []
    for item in items:
        values = ds[item].values[flat_test_idx].copy()
        values[:, ~common_valid_cols] = np.nan
        ds_ref_arrays.append(
            mikeio.DataArray(
                data=values,
                time=ref_time,
                item=ds[item].item,
                geometry=geom,
            )
        )
    return mikeio.Dataset(ds_ref_arrays)