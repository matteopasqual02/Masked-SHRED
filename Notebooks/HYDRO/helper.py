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
df_stn_cur = pd.read_csv(obs_fldr + "current_stations.csv", index_col=0)

def get_wl_point_obs():
    # Get water level point observations as list of PointObservation objects
    q = ms.Quantity(name="Surface Elevation", unit="meter")
    wllist = []
    for i, row in df_stn.iterrows():
        df = pd.read_csv(obs_fldr + f"{i}_wl.csv", index_col=0)
        df.index = pd.to_datetime(df.index, format="ISO8601")
        o = ms.PointObservation(df.water_level_filtered, x=row['Longitude'], y=row['Latitude'], name=i, quantity=q)
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

def plot_shred_reconstruction(test_reconstruction, ds, geom, plot_time=None, items=None, cmaps=None):
    
    # Set default items and colormaps if not provided
    if items is None:
        items = ["Surface elevation", "U velocity", "V velocity"]
    if cmaps is None:
        cmaps = {
            "Surface elevation": "terrain",
            "U velocity": "twilight_shifted",
            "V velocity": "twilight_shifted",
        }

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
        mikeio.DataArray(data=reconstructions[item], time=times_test, item=ds.items[i], geometry=geom)
        for i, item in enumerate(items)
    ])
    ds_error = mikeio.Dataset([
        mikeio.DataArray(data=np.abs(ds[item].sel(time=times_test).values - ds_shred[item].values),
        time=times_test, item=ds.items[i], geometry=geom)
        for i, item in enumerate(items)
    ])

    # Plotting
    if plot_time is None:
        plot_time = ds_shred.time[-1]

    fig, axs = plt.subplots(len(items), 3, figsize=(15, 10))
    for row, item in enumerate(items):
        reference  = ds[item].sel(time=plot_time)
        prediction = ds_shred[item].sel(time=plot_time)
        error      = ds_error[item].sel(time=plot_time)

        vmin = min(np.nanmin(reference.values), np.nanmin(prediction.values))
        vmax = max(np.nanmax(reference.values), np.nanmax(prediction.values))

        reference.plot( ax=axs[row, 0], title=item,    cmap=cmaps[item], vmin=vmin, vmax=vmax)
        prediction.plot(ax=axs[row, 1], title="SHRED", cmap=cmaps[item], vmin=vmin, vmax=vmax)
        error.plot(     ax=axs[row, 2], title="Error", cmap="gray")

    plt.tight_layout()
    plt.show()

    return ds_shred, ds_error