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

def extract_water_level_measurements(stations, ds_time, obs_fldr, window="25min", verbose=False):
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

        if verbose:
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
            title=f"Error",
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

import matplotlib as mpl
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
C_BAND = "#f2f4f7"
C_TEXT = "#222222"

def plot_shred_reconstruction_better(
    test_reconstruction,
    ds,
    geom,
    plot_time=None,
    items=None,
    cmaps=None,
    title="",
    fig_title=None,
    font_scale=0.5,
    error_cmap="Reds",
    time_idx=-1,
    show_time_title=False,
    map_height=3.3,    # -20% rispetto a 4.1
    error_scale=1.0,   # moltiplicatore per la mappa dell'errore (1.0 = stessa dimensione delle altre)
    row_pitch=3.7,     # distanza verticale tra le righe (pollici)
    gap=0.25,          # distanza tra oggetti consecutivi (pollici)
    cbar_nbins=6,      # numero massimo di tick per colorbar (uguale per tutte)
):
    """
    Reference vs SHRED reconstruction vs absolute error, one row per item.
    Ground Truth e SHRED condividono un'unica colorbar per riga.
    La colorbar dell'errore è creata con lo stesso metodo (ScalarMappable),
    quindi ha lo stesso stile delle altre.
    """

    fs = lambda x: x * font_scale

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

    missing_in_reconstruction = [item for item in items if item not in test_reconstruction]
    if missing_in_reconstruction:
        raise KeyError(
            f"Missing items in test_reconstruction: {missing_in_reconstruction}. "
            f"Available keys: {list(test_reconstruction.keys())}"
        )

    times_test = ds.time[-test_reconstruction[items[0]].shape[0]:]

    reconstructions = {}
    for item in items:
        full = np.full((test_reconstruction[item].shape[0], ds[item].values.shape[1]), np.nan)
        valid_cols = ~np.isnan(ds[item].values).any(axis=0)
        full[:, valid_cols] = test_reconstruction[item]
        reconstructions[item] = full

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

    n_test = len(ds_shred.time)
    if plot_time is None:
        if not -n_test <= time_idx < n_test:
            raise IndexError(
                f"time_idx={time_idx} fuori range: il test set ha {n_test} timestep "
                f"(validi da {-n_test} a {n_test - 1})."
            )
        plot_time = ds_shred.time[time_idx]

    col_titles = ["Ground Truth", "Masked SHRED", "Absolute error"]
    n_rows = len(items)

    # Margini e dimensioni fisse (pollici)
    lm = 0.45
    tm = 0.9
    cw = 0.16
    cb_text = 0.55
    rm = 0.1
    fig_h = tm + n_rows * row_pitch

    with plt.rc_context(PAPER_RC):
        fig = plt.figure(figsize=(10, fig_h))
        fig.patch.set_facecolor("white")

        axs = np.empty((n_rows, 3), dtype=object)
        cax_shared = []
        cax_error = []
        colorbars = []
        units = []
        unit_map = {"meter": "m", "meter_per_sec": "m/s"}

        for row, item in enumerate(items):
            for col in range(3):
                axs[row, col] = fig.add_axes([0.05, 0.05, 0.1, 0.1])
            cax_shared.append(fig.add_axes([0.05, 0.05, 0.02, 0.1]))
            cax_error.append(fig.add_axes([0.05, 0.05, 0.02, 0.1]))

            reference = ds[item].sel(time=plot_time)
            prediction = ds_shred[item].sel(time=plot_time)
            error = ds_error[item].sel(time=plot_time)

            vmin = min(np.nanmin(reference.values), np.nanmin(prediction.values))
            vmax = max(np.nanmax(reference.values), np.nanmax(prediction.values))
            cmap = cmaps.get(item, "viridis")

            err_vmin = 0.0
            err_vmax = float(np.nanmax(error.values))

            try:
                unit = ds[item].unit.name
            except Exception:
                unit = ""
            units.append(unit_map.get(unit, unit))

            reference.plot(ax=axs[row, 0], title="", cmap=cmap, vmin=vmin, vmax=vmax,
                           add_colorbar=False)
            prediction.plot(ax=axs[row, 1], title="", cmap=cmap, vmin=vmin, vmax=vmax,
                            add_colorbar=False)
            error.plot(ax=axs[row, 2], title="", cmap=error_cmap,
                       vmin=err_vmin, vmax=err_vmax, add_colorbar=False)

            for col in range(3):
                ax = axs[row, col]
                ax.set_axis_off()
                ax.set_xlabel("")
                ax.set_ylabel("")
                if row == 0:
                    ax.set_title(col_titles[col], fontsize=fs(20), fontweight="bold", pad=8)

            # colorbar condivisa (GT + SHRED)
            sm = ScalarMappable(norm=Normalize(vmin=vmin, vmax=vmax), cmap=cmap)
            sm.set_array([])
            cb = fig.colorbar(sm, cax=cax_shared[row])
            colorbars.append(cb)

            # colorbar errore: stesso metodo, stesso stile
            sm_err = ScalarMappable(norm=Normalize(vmin=err_vmin, vmax=err_vmax), cmap=error_cmap)
            sm_err.set_array([])
            cb_err = fig.colorbar(sm_err, cax=cax_error[row])
            colorbars.append(cb_err)

        # Tick uniformi su tutte le colorbar
        for c in colorbars:
            c.locator = MaxNLocator(nbins=cbar_nbins)
            c.update_ticks()

        # --- Layout in pollici -------------------------------------------------
        # proporzione larghezza/altezza reale della mappa
        ax0 = axs[0, 0]
        x0, x1 = ax0.get_xlim()
        y0, y1 = ax0.get_ylim()
        asp = ax0.get_aspect()
        asp = 1.0 if isinstance(asp, str) else float(asp)
        ratio = abs((x1 - x0) / (y1 - y0)) / asp

        mh = map_height
        mw = ratio * mh

        # mappa errore: dimensione propria
        mh_e = mh * error_scale
        mw_e = mw * error_scale

        x_gt = lm
        x_sh = x_gt + mw + gap
        x_cb = x_sh + mw + gap
        x_er = x_cb + cw + cb_text + gap
        x_ecb = x_er + mw_e + gap
        fig_w = x_ecb + cw + cb_text + rm

        fig.set_size_inches(fig_w, fig_h)

        fx = lambda v: v / fig_w
        fy = lambda v: v / fig_h
        cb_h = 0.9 * mh

        for row in range(n_rows):
            y_top = fig_h - tm - row * row_pitch
            y_bot = y_top - mh
            y_mid = y_bot + mh / 2

            # mappe GT e SHRED
            for col, xx in enumerate([x_gt, x_sh]):
                axs[row, col].set_position([fx(xx), fy(y_bot), fx(mw), fy(mh)])

            # mappa errore: stessa linea centrale, dimensione propria
            ax_e = axs[row, 2]
            ax_e.set_aspect("auto")  # evita che matplotlib la rimpicciolisca nel box
            ax_e.set_position([fx(x_er), fy(y_mid - mh_e / 2), fx(mw_e), fy(mh_e)])
            ax_e.set_anchor("C")

            # colorbar condivisa e colorbar errore (stessa altezza, centrate sulla riga)
            cax_shared[row].set_position([fx(x_cb), fy(y_mid - 0.45 * mh), fx(cw), fy(cb_h)])
            cax_error[row].set_position([fx(x_ecb), fy(y_mid - 0.45 * mh_e), fx(cw), fy(0.9 * mh_e)])

            fig.text(
                fx(lm - 0.1), fy(y_mid),
                f"{items[row]}\n[{units[row]}]" if units[row] else items[row],
                rotation=90, va="center", ha="right",
                fontsize=fs(20), fontweight="bold", color=C_TEXT,
            )

        # Titoli di colonna: allineati in alto sopra la prima riga
        for col, xx, w in [(0, x_gt, mw), (1, x_sh, mw), (2, x_er, mw_e)]:
            axs[0, col].set_title("")
            fig.text(
                fx(xx + w / 2), fy(fig_h - tm + 0.12), col_titles[col],
                ha="center", va="bottom",
                fontsize=fs(20), fontweight="bold", color="black",
            )

        # Stile di tutte le colorbar
        main_axes = set(axs.ravel())
        for cax in fig.axes:
            if cax in main_axes:
                continue
            cax.tick_params(labelsize=fs(13), width=0.8, length=3)
            cax.yaxis.label.set_size(fs(14))
            cax.yaxis.label.set_fontfamily("serif")
            for spine in cax.spines.values():
                spine.set_linewidth(0.6)
                spine.set_color("0.5")

        main_title = fig_title if fig_title is not None else title
        if show_time_title:
            main_title = f"{main_title}\n{pd.Timestamp(plot_time):%Y-%m-%d %H:%M}".strip()
        if main_title:
            fig.suptitle(main_title, fontsize=fs(26), fontweight="bold",
                         color=C_TEXT, x=0.5, y=1 - 0.05 / fig_h, va="top")

        try:
            base_dir = Path(__file__).resolve().parent
        except NameError:
            base_dir = Path.cwd()
        output_path = base_dir / f"{title}.png"
        fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"Saved plot: {output_path}")

        plt.show()

    del fig, axs, ds_shred, ds_error, reconstructions

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


def plot_shred_reconstruction_best(
    test_reconstruction,
    ds,
    geom,
    plot_time=None,
    items=None,
    cmaps=None,
    title="",
    fig_title=None,
    font_scale=0.5,
    error_cmap="Reds",
    time_idx=-1,
    map_height=3.3,
    error_scale=1.0,
    row_pitch=3.45,        # distanza verticale tra le righe (pollici)
    gap=0.15,              # GT -> SHRED
    cbar_gap=0.10,         # SHRED -> colorbar condivisa
    group_gap=0.55,        # gruppo (GT+SHRED+cbar) -> errore
    cbar_nbins=6,
    cbar_tick_pt=8.5,      # font tick colorbar (pt effettivi)
    cbar_unit_pt=9.5,      # font unità sopra colorbar
    symmetric_vel=True,    # limiti simmetrici attorno a 0 per le velocità
    show_datetime=True,    # riga "Date: ... Time: ..." sotto il titolo
):
    """
    Ground Truth vs Masked SHRED vs Absolute error, una riga per variabile.
    GT e SHRED condividono una colorbar (subito a destra di SHRED).
    """
    fs = lambda x: x * font_scale

    if items is None:
        items = ["Surface elevation", "U velocity", "V velocity"]
    elif isinstance(items, str):
        items = [items]
    else:
        items = list(items)

    if cmaps is None:
        cmaps = {
            "Surface elevation": "viridis",
            "U velocity": "RdBu_r",
            "V velocity": "RdBu_r",
        }

    missing = [i for i in items if i not in test_reconstruction]
    if missing:
        raise KeyError(
            f"Missing items in test_reconstruction: {missing}. "
            f"Available keys: {list(test_reconstruction.keys())}"
        )

    times_test = ds.time[-test_reconstruction[items[0]].shape[0]:]

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
    if plot_time is None:
        if not -n_test <= time_idx < n_test:
            raise IndexError(
                f"time_idx={time_idx} fuori range: il test set ha {n_test} timestep "
                f"(validi da {-n_test} a {n_test - 1})."
            )
        plot_time = ds_shred.time[time_idx]

    col_titles = ["Ground Truth", "Masked SHRED", "Absolute error"]
    n_rows = len(items)

    # Margini e dimensioni fisse (pollici)
    lm, tm, cw, cb_text, rm = 0.45, 1.15, 0.16, 0.55, 0.1
    fig_h = tm + n_rows * row_pitch

    with plt.rc_context(PAPER_RC):
        fig = plt.figure(figsize=(10, fig_h))
        fig.patch.set_facecolor("white")

        axs = np.empty((n_rows, 3), dtype=object)
        cax_shared, cax_error, colorbars, units = [], [], [], []
        unit_map = {"meter": "m", "meter_per_sec": "m/s"}

        for row, item in enumerate(items):
            for col in range(3):
                axs[row, col] = fig.add_axes([0.05, 0.05, 0.1, 0.1])
            cax_shared.append(fig.add_axes([0.05, 0.05, 0.02, 0.1]))
            cax_error.append(fig.add_axes([0.05, 0.05, 0.02, 0.1]))

            reference = ds[item].sel(time=plot_time)
            prediction = ds_shred[item].sel(time=plot_time)
            error = ds_error[item].sel(time=plot_time)

            vmin = min(np.nanmin(reference.values), np.nanmin(prediction.values))
            vmax = max(np.nanmax(reference.values), np.nanmax(prediction.values))
            cmap = cmaps.get(item, "viridis")

            # velocità: limiti simmetrici => zero al centro della colormap divergente
            if symmetric_vel and "velocity" in item.lower():
                lim = max(abs(vmin), abs(vmax))
                vmin, vmax = -lim, lim

            norm = Normalize(vmin=vmin, vmax=vmax)          # solo per le colorbar
            err_vmax = float(np.nanmax(error.values))
            err_norm = Normalize(vmin=0.0, vmax=err_vmax)

            try:
                unit = ds[item].unit.name
            except Exception:
                unit = ""
            units.append(unit_map.get(unit, unit))

            # mikeio accetta solo vmin/vmax (niente norm=)
            reference.plot(ax=axs[row, 0], title="", cmap=cmap,
                           vmin=vmin, vmax=vmax, add_colorbar=False)
            prediction.plot(ax=axs[row, 1], title="", cmap=cmap,
                            vmin=vmin, vmax=vmax, add_colorbar=False)
            error.plot(ax=axs[row, 2], title="", cmap=error_cmap,
                       vmin=0.0, vmax=err_vmax, add_colorbar=False)

            for col in range(3):
                ax = axs[row, col]
                ax.set_axis_off()
                ax.set_xlabel("")
                ax.set_ylabel("")

            # colorbar condivisa (GT + SHRED)
            sm = ScalarMappable(norm=norm, cmap=cmap)
            sm.set_array([])
            colorbars.append(fig.colorbar(sm, cax=cax_shared[row]))

            # colorbar errore
            sm_err = ScalarMappable(norm=err_norm, cmap=error_cmap)
            sm_err.set_array([])
            colorbars.append(fig.colorbar(sm_err, cax=cax_error[row]))

            # unità in cima a entrambe le colorbar
            u = units[row]
            for cax in (cax_shared[row], cax_error[row]):
                cax.set_title(f"[{u}]" if u else "", fontsize=cbar_unit_pt, pad=6)

        # Tick uniformi su tutte le colorbar
        for c in colorbars:
            c.locator = MaxNLocator(nbins=cbar_nbins)
            c.update_ticks()

        # --- Layout in pollici ---
        ax0 = axs[0, 0]
        x0, x1 = ax0.get_xlim()
        y0, y1 = ax0.get_ylim()
        asp = ax0.get_aspect()
        asp = 1.0 if isinstance(asp, str) else float(asp)
        ratio = abs((x1 - x0) / (y1 - y0)) / asp

        mh = map_height
        mw = ratio * mh
        mh_e = mh * error_scale
        mw_e = mw * error_scale

        x_gt = lm
        x_sh = x_gt + mw + gap
        x_cb = x_sh + mw + cbar_gap
        x_er = x_cb + cw + cb_text + group_gap
        x_ecb = x_er + mw_e + cbar_gap
        fig_w = x_ecb + cw + cb_text + rm

        fig.set_size_inches(fig_w, fig_h)

        fx = lambda v: v / fig_w
        fy = lambda v: v / fig_h
        cb_h = 0.9 * mh

        for row in range(n_rows):
            y_top = fig_h - tm - row * row_pitch
            y_bot = y_top - mh
            y_mid = y_bot + mh / 2

            for col, xx in enumerate([x_gt, x_sh]):
                axs[row, col].set_position([fx(xx), fy(y_bot), fx(mw), fy(mh)])

            ax_e = axs[row, 2]
            ax_e.set_aspect("auto")
            ax_e.set_position([fx(x_er), fy(y_mid - mh_e / 2), fx(mw_e), fy(mh_e)])
            ax_e.set_anchor("C")

            cax_shared[row].set_position([fx(x_cb), fy(y_mid - 0.45 * mh), fx(cw), fy(cb_h)])
            cax_error[row].set_position([fx(x_ecb), fy(y_mid - 0.45 * mh_e), fx(cw), fy(0.9 * mh_e)])

            fig.text(
                fx(lm - 0.1), fy(y_mid),
                f"{items[row]}\n[{units[row]}]" if units[row] else items[row],
                rotation=90, va="center", ha="right",
                fontsize=fs(20), fontweight="bold", color=C_TEXT,
            )

        # Separatore verticale tra gruppo (GT+SHRED) ed errore
        x_sep = x_cb + cw + cb_text + group_gap / 2
        fig.add_artist(plt.Line2D(
            [fx(x_sep), fx(x_sep)],
            [fy(0.15), fy(fig_h - tm + 0.55)],
            color="0.75", lw=0.8, ls="--",
        ))

        # Titoli di colonna
        for col, xx, w in [(0, x_gt, mw), (1, x_sh, mw), (2, x_er, mw_e)]:
            axs[0, col].set_title("")
            fig.text(
                fx(xx + w / 2), fy(fig_h - tm + 0.12), col_titles[col],
                ha="center", va="bottom",
                fontsize=fs(20), fontweight="bold", color="black",
            )

        # Stile colorbar (font in punti effettivi)
        main_axes = set(axs.ravel())
        for cax in fig.axes:
            if cax in main_axes:
                continue
            cax.tick_params(labelsize=cbar_tick_pt, width=0.8, length=3)
            for spine in cax.spines.values():
                spine.set_linewidth(0.6)
                spine.set_color("0.5")

        # Titolo + riga data/ora
        main_title = fig_title if fig_title is not None else title
        ts = pd.Timestamp(plot_time)

        y_title = fig_h - 0.05
        if main_title:
            fig.text(
                0.5, fy(y_title), main_title,
                ha="center", va="top",
                fontsize=fs(26), fontweight="bold", color=C_TEXT,
            )

        if show_datetime:
            y_dt = y_title - (fs(26) / 72) * 1.35 - 0.08
            fig.text(
                0.5, fy(y_dt),
                f"Date: {ts:%Y-%m-%d}    Time: {ts:%H:%M}",
                ha="center", va="top",
                fontsize=fs(20), fontweight="normal", color=C_TEXT,
            )

        try:
            base_dir = Path(__file__).resolve().parent
        except NameError:
            base_dir = Path.cwd()
        output_path = base_dir / f"{title}.png"
        fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"Saved plot: {output_path}")

        plt.show()

    del fig, axs, ds_shred, ds_error, reconstructions
    
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
    font_scale=0.5,
    error_cmap="Greys",
    map_height=3.0,
    row_gap=0.55,          # spazio verticale tra le righe (contiene il separatore)
    gap=0.08,              # GT -> SHRED
    cbar_gap=0.08,         # mappa -> colorbar
    err_gap=0.12,          # colorbar condivisa -> mappa errore
    group_gap=0.45,        # tra i blocchi di variabile (contiene il separatore)
    cbar_nbins=5,
    cbar_tick_pt=7.5,
    cbar_unit_pt=8.5,
    symmetric_vel=False,
    sep_lw=0.8,
):
    """
    Una riga per timestep; in ogni riga 3 blocchi (una variabile ciascuno):
    [GT | SHRED | cbar condivisa | Errore | cbar errore].
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

    # Margini (pollici)
    lm, tm, cw, cb_text, rm = 0.55, 1.05, 0.13, 0.45, 0.1
    mh = map_height
    row_pitch = mh + row_gap
    fig_h = tm + n_rows * row_pitch - row_gap / 2 + 0.1

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

                vmin = min(np.nanmin(reference.values), np.nanmin(prediction.values))
                vmax = max(np.nanmax(reference.values), np.nanmax(prediction.values))
                cmap = cmaps.get(item, "viridis")
                if symmetric_vel and "velocity" in item.lower():
                    lim = max(abs(vmin), abs(vmax))
                    vmin, vmax = -lim, lim

                err_vmax = float(np.nanmax(error.values))

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
                    cax.set_title(f"[{u}]" if u else "", fontsize=cbar_unit_pt, pad=5)

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
        cb_h = 0.9 * mh

        y_cols_title = fig_h - tm + 0.08          # titoli colonna (sopra riga 0)
        y_group_title = y_cols_title + 0.22       # nome variabile

        for v, item in enumerate(items):
            xg = lm + v * (gw + group_gap)

            for r in range(n_rows):
                y_top = fig_h - tm - r * row_pitch
                y_bot = y_top - mh
                y_mid = y_bot + mh / 2

                for c, off in enumerate([o_gt, o_sh, o_er]):
                    axs[r, v, c].set_aspect("auto")
                    axs[r, v, c].set_position([fx(xg + off), fy(y_bot), fx(mw), fy(mh)])

                cax_s[r, v].set_position([fx(xg + o_cb), fy(y_mid - 0.45 * mh), fx(cw), fy(cb_h)])
                cax_e[r, v].set_position([fx(xg + o_ecb), fy(y_mid - 0.45 * mh), fx(cw), fy(cb_h)])

            # titoli colonna (solo riga 0)
            for c, off in enumerate([o_gt, o_sh, o_er]):
                fig.text(fx(xg + off + mw / 2), fy(y_cols_title), col_titles[c],
                         ha="center", va="bottom",
                         fontsize=fs(18), fontweight="bold", color="black")

            # nome variabile sopra il blocco
            label = f"{item} [{units[v]}]" if units[v] else item
            fig.text(fx(xg + gw / 2), fy(y_group_title), label,
                     ha="center", va="bottom",
                     fontsize=fs(24), fontweight="bold", color=C_TEXT)

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
            fig.text(fx(lm - 0.12), fy(y_mid),
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
            cax.tick_params(labelsize=cbar_tick_pt, width=0.8, length=3)
            for spine in cax.spines.values():
                spine.set_linewidth(0.6)
                spine.set_color("0.5")

        # titolo unico
        main_title = fig_title if fig_title is not None else title
        if main_title:
            fig.text(0.5, fy(fig_h - 0.05), main_title,
                     ha="center", va="top",
                     fontsize=fs(30), fontweight="bold", color=C_TEXT)

        try:
            base_dir = Path(__file__).resolve().parent
        except NameError:
            base_dir = Path.cwd()
        output_path = base_dir / f"{title}.png"
        fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"Saved plot: {output_path}")

        plt.show()

    del fig, axs, ds_shred, ds_error, reconstructions