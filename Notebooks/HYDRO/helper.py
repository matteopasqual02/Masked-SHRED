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

def extract_water_level_measurements(stations, ds_time, obs_fldr, window="25min"):
    # Extract water level measurements for given stations and simulation timestamps.
    # For each station, attempts to read filtered water level data aligned with simulation timestamps.
    # If unavailable, falls back to raw water level data and takes the last measurement within 
    # a specified time window before each simulation timestamp.
    
    measurements = np.full((ds_time.shape[0], len(stations)), np.nan)   # Initialize measurements array with NaNs
    ds_times = pd.DatetimeIndex(ds_time.values)                         # Convert simulation timestamps to pandas DatetimeIndex for easier time-based indexing  
    window_delta = pd.Timedelta(window)                                 # Convert window string to pandas Timedelta for time-based filtering
    missing_files = []                                                  # List to track stations with missing or incompatible files

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

        # If filtered data exists, align exactly on simulation timestamps
        if "water_level_filtered" in stn_data.columns:
            measurements[:, i] = stn_data["water_level_filtered"].reindex(
                ds_time.strftime("%Y-%m-%d %H:%M:%S")
            ).values
            continue

        # Fallback for raw water level data: take last value in a time window
        if "water_level" in stn_data.columns:
            wl = stn_data["water_level"].sort_index()
            for j, t in enumerate(ds_times):
                window_vals = wl[(wl.index > t - window_delta) & (wl.index <= t)]
                if len(window_vals) > 0:
                    measurements[j, i] = window_vals.iloc[-1]

        # If neither filtered nor raw data could be aligned, mark as missing
        else:
            missing_files.append(station.name)

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

import numpy as np
import pandas as pd
import mikeio
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.ticker import MaxNLocator

PAPER_RC = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "axes.linewidth": 0.9,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}
C_TEXT = "#222222"


def plot_shred_reconstruction_multi(
    test_reconstruction,
    ds,
    geom,
    time_idxs=(-10000, -5000, -1),
    items=None,
    cmaps=None,
    title="",
    fig_title=None,
    font_scale=1.4,
    error_cmap="Greys",
    map_height=3.0,
    row_gap=0.7,           # spazio verticale tra le righe (contiene il separatore + unita' colorbar)
    gap=0.9,               # GT -> SHRED (prima 0.4)
    cbar_gap=0.2,          # mappa -> colorbar (prima 0.08)
    err_gap=0.35,          # colorbar condivisa -> mappa errore (prima 0.12)
    group_gap=0.9,         # tra i blocchi di variabile (prima 0.45)
    title_pad=0.75,        # spazio extra sotto i titoli delle subplot
    cbar_nbins=6,
    cbar_tick_pt=18,       # numeri sulla colorbar (prima 11)
    cbar_unit_pt=20,       # unita' [m], [m/s] sopra la colorbar (prima 12.5)
    cbar_width=0.35,       # spessore barra in pollici (prima 0.13)
    cbar_text_space=0.9,  # spazio per i numeri a destra della barra (prima 0.6)
    cbar_height_frac=1.1,  # altezza colorbar rispetto alla mappa (prima 0.9)
    symmetric_vel=False,
    sep_lw=0.8,
):
    """
    Una riga per timestep; in ogni riga 3 blocchi (una variabile ciascuno):
    [GT | SHRED | cbar condivisa | Errore | cbar errore].
    Le colorbar sono identiche per tutte le righe di una stessa variabile:
    range = [min, max] su tutti i timestep selezionati (GT e SHRED insieme);
    errore = [0, picco massimo] su tutti i timestep selezionati.
    """
    fs = lambda x: x * font_scale

    if items is None:
        items = ["Surface elevation", "U velocity", "V velocity"]
    items = [items] if isinstance(items, str) else list(items)

    if cmaps is None:
        cmaps = {"Surface elevation": "terrain",
                 "U velocity": "twilight_shifted",
                 "V velocity": "twilight_shifted"}

    missing = [i for i in items if i not in test_reconstruction]
    if missing:
        raise KeyError(f"Missing items: {missing}. Available: {list(test_reconstruction.keys())}")

    times_test = ds.time[-test_reconstruction[items[0]].shape[0]:]

    # --- ricostruzione (una sola volta) ---
    reconstructions = {}
    for item in items:
        full = np.full((test_reconstruction[item].shape[0], ds[item].values.shape[1]), np.nan)
        valid_cols = ~np.isnan(ds[item].values).any(axis=0)
        full[:, valid_cols] = test_reconstruction[item]
        reconstructions[item] = full

    ds_shred = mikeio.Dataset([
        mikeio.DataArray(data=reconstructions[item], time=times_test,
                         item=ds[item].item, geometry=geom)
        for item in items
    ])
    ds_error = mikeio.Dataset([
        mikeio.DataArray(
            data=np.abs(ds[item].sel(time=times_test).values - ds_shred[item].values),
            time=times_test, item=ds[item].item, geometry=geom,
        )
        for item in items
    ])

    n_test = len(ds_shred.time)
    plot_times = []
    for ti in time_idxs:
        if not -n_test <= ti < n_test:
            raise IndexError(f"time_idx={ti} fuori range: il test set ha {n_test} timestep.")
        plot_times.append(ds_shred.time[ti])

    n_rows, n_vars = len(plot_times), len(items)
    col_titles = ["Ground Truth", "Masked SHRED", "Absolute error"]
    unit_map = {"meter": "m", "meter_per_sec": "m/s"}
    units = []
    for it in items:
        try:
            u = ds[it].unit.name
        except Exception:
            u = ""
        units.append(unit_map.get(u, u))

    # --- Limiti globali per variabile (su tutti i timestep selezionati) ---
    var_lims = {}
    err_lims = {}
    for item in items:
        ref_vals = np.concatenate([
            ds[item].sel(time=t).values.ravel() for t in plot_times
        ])
        pred_vals = np.concatenate([
            ds_shred[item].sel(time=t).values.ravel() for t in plot_times
        ])
        vmin_g = min(np.nanmin(ref_vals), np.nanmin(pred_vals))
        vmax_g = max(np.nanmax(ref_vals), np.nanmax(pred_vals))
        if symmetric_vel and "velocity" in item.lower():
            lim = max(abs(vmin_g), abs(vmax_g))
            vmin_g, vmax_g = -lim, lim
        var_lims[item] = (float(vmin_g), float(vmax_g))

        err_vals = np.concatenate([
            ds_error[item].sel(time=t).values.ravel() for t in plot_times
        ])
        err_lims[item] = (0.0, float(np.nanmax(err_vals)))

    # Margini (pollici)
    lm, tm, rm = 0.85, 1.4, 0.1
    cw, cb_text = cbar_width, cbar_text_space
    mh = map_height
    row_pitch = mh + row_gap
    fig_h = tm + n_rows * row_pitch - row_gap / 2

    with plt.rc_context(PAPER_RC):
        fig = plt.figure(figsize=(10, fig_h))
        fig.patch.set_facecolor("white")

        axs = np.empty((n_rows, n_vars, 3), dtype=object)
        cax_s = np.empty((n_rows, n_vars), dtype=object)
        cax_e = np.empty((n_rows, n_vars), dtype=object)
        colorbars = []

        for r, plot_time in enumerate(plot_times):
            for v, item in enumerate(items):
                for c in range(3):
                    axs[r, v, c] = fig.add_axes([0.05, 0.05, 0.1, 0.1])
                cax_s[r, v] = fig.add_axes([0.05, 0.05, 0.02, 0.1])
                cax_e[r, v] = fig.add_axes([0.05, 0.05, 0.02, 0.1])

                reference = ds[item].sel(time=plot_time)
                prediction = ds_shred[item].sel(time=plot_time)
                error = ds_error[item].sel(time=plot_time)

                # limiti globali per variabile (uguali in tutte le righe)
                vmin, vmax = var_lims[item]
                err_vmax = err_lims[item][1]
                cmap = cmaps.get(item, "viridis")

                reference.plot(ax=axs[r, v, 0], title="", cmap=cmap,
                               vmin=vmin, vmax=vmax, add_colorbar=False)
                prediction.plot(ax=axs[r, v, 1], title="", cmap=cmap,
                                vmin=vmin, vmax=vmax, add_colorbar=False)
                error.plot(ax=axs[r, v, 2], title="", cmap=error_cmap,
                           vmin=0.0, vmax=err_vmax, add_colorbar=False)

                for c in range(3):
                    ax = axs[r, v, c]
                    ax.set_axis_off()
                    ax.set_xlabel("")
                    ax.set_ylabel("")

                sm = ScalarMappable(norm=Normalize(vmin=vmin, vmax=vmax), cmap=cmap)
                sm.set_array([])
                colorbars.append(fig.colorbar(sm, cax=cax_s[r, v]))

                sm_e = ScalarMappable(norm=Normalize(vmin=0.0, vmax=err_vmax), cmap=error_cmap)
                sm_e.set_array([])
                colorbars.append(fig.colorbar(sm_e, cax=cax_e[r, v]))

                u = units[v]
                for cax in (cax_s[r, v], cax_e[r, v]):
                    cax.set_title(f"[{u}]" if u else "", fontsize=cbar_unit_pt, pad=6)

        for cb in colorbars:
            cb.locator = MaxNLocator(nbins=cbar_nbins)
            cb.update_ticks()

        # --- Layout in pollici ---
        ax0 = axs[0, 0, 0]
        x0, x1 = ax0.get_xlim()
        y0, y1 = ax0.get_ylim()
        asp = ax0.get_aspect()
        asp = 1.0 if isinstance(asp, str) else float(asp)
        ratio = abs((x1 - x0) / (y1 - y0)) / asp
        mw = ratio * mh

        # offset interni a un blocco-variabile
        o_gt = 0.0
        o_sh = o_gt + mw + gap
        o_cb = o_sh + mw + cbar_gap
        o_er = o_cb + cw + cb_text + err_gap
        o_ecb = o_er + mw + cbar_gap
        gw = o_ecb + cw + cb_text           # larghezza di un blocco-variabile

        fig_w = lm + n_vars * gw + (n_vars - 1) * group_gap + rm
        fig.set_size_inches(fig_w, fig_h)
        fx = lambda val: val / fig_w
        fy = lambda val: val / fig_h
        cb_h = cbar_height_frac * mh
        cb_off = (mh - cb_h) / 2            # colorbar centrata verticalmente sulla mappa

        y_cols_title = fig_h - tm + 0.45 + title_pad          # titoli colonna (sopra riga 0 e sopra le unita' colorbar)
        y_group_title = y_cols_title + 0.50 + title_pad / 2    # nome variabile

        for v, item in enumerate(items):
            xg = lm + v * (gw + group_gap)

            for r in range(n_rows):
                y_top = fig_h - tm - r * row_pitch
                y_bot = y_top - mh

                for c, off in enumerate([o_gt, o_sh, o_er]):
                    axs[r, v, c].set_aspect("auto")
                    axs[r, v, c].set_position([fx(xg + off), fy(y_bot), fx(mw), fy(mh)])

                cax_s[r, v].set_position([fx(xg + o_cb), fy(y_bot + cb_off), fx(cw), fy(cb_h)])
                cax_e[r, v].set_position([fx(xg + o_ecb), fy(y_bot + cb_off), fx(cw), fy(cb_h)])

            # titoli colonna (solo riga 0)
            for c, off in enumerate([o_gt, o_sh, o_er]):
                fig.text(fx(xg + off + mw / 2), fy(y_cols_title), col_titles[c],
                         ha="center", va="bottom",
                         fontsize=fs(22), fontweight="bold", color="black")

            # nome variabile sopra il blocco
            label = f"{item} [{units[v]}]" if units[v] else item
            fig.text(fx(xg + gw / 2), fy(y_group_title), label,
                     ha="center", va="bottom",
                     fontsize=fs(30), fontweight="bold", color=C_TEXT)

            # separatore verticale tra blocchi-variabile
            if v < n_vars - 1:
                x_sep = xg + gw + group_gap / 2
                fig.add_artist(plt.Line2D(
                    [fx(x_sep), fx(x_sep)],
                    [fy(0.1), fy(y_group_title + 0.3)],
                    color="0.25", lw=sep_lw, ls=(0, (4, 3)),
                ))

        # etichetta data/ora a sinistra di ogni riga + separatori orizzontali
        for r, plot_time in enumerate(plot_times):
            y_top = fig_h - tm - r * row_pitch
            y_mid = y_top - mh / 2
            ts = pd.Timestamp(plot_time)
            fig.text(fx(lm - 0.15), fy(y_mid),
                     f"{ts:%Y-%m-%d}\n{ts:%H:%M}",
                     rotation=90, va="center", ha="right",
                     fontsize=fs(22), fontweight="bold", color=C_TEXT)

            if r < n_rows - 1:
                y_sep = y_top - mh - row_gap / 2
                fig.add_artist(plt.Line2D(
                    [0.005, 0.995], [fy(y_sep), fy(y_sep)],
                    color="0.25", lw=sep_lw, ls=(0, (4, 3)),
                ))

        # stile colorbar
        main_axes = set(axs.ravel())
        for cax in fig.axes:
            if cax in main_axes:
                continue
            cax.tick_params(labelsize=cbar_tick_pt, width=1.2, length=5)
            for spine in cax.spines.values():
                spine.set_linewidth(0.8)
                spine.set_color("0.4")

        try:
            base_dir = Path(__file__).resolve().parent
        except NameError:
            base_dir = Path.cwd()
        output_path = base_dir / f"{title}.png"
        fig.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.02, facecolor="white")
        print(f"Saved plot: {output_path}")

        plt.show()

    del fig, axs, ds_shred, ds_error, reconstructions

