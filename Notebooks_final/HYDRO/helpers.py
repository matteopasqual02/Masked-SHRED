"""Data loading, evaluation, and plotting helpers for the HYDRO notebooks."""

import glob
import json
import matplotlib as mpl
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import warnings
import numpy as np
import pandas as pd
import mikeio
import modelskill as ms
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.text import Text
from matplotlib.ticker import MaxNLocator
from pathlib import Path
from types import SimpleNamespace

from src.engine import SHREDEngine
from src.utils import evaluate_reconstruction_metrics


_OBSERVATIONS_DIR = Path("datasets/observations")


def get_wl_point_obs(obs_fldr=_OBSERVATIONS_DIR):
    """Load water-level station observations as ModelSkill point observations."""
    obs_fldr = Path(obs_fldr)
    stations = pd.read_csv(obs_fldr / "stations.csv", index_col=0)
    quantity = ms.Quantity(name="Surface Elevation", unit="meter")
    observations = []
    for station_name, row in stations.iterrows():
        data = pd.read_csv(obs_fldr / f"{station_name}_wl.csv", index_col=0)
        data.index = pd.to_datetime(data.index, format="ISO8601")
        observations.append(
            ms.PointObservation(
                data["water_level_filtered"],
                x=row["Longitude"],
                y=row["Latitude"],
                name=station_name,
                quantity=quantity,
            )
        )
    return observations


def get_altimetry_obs(obs_fldr=_OBSERVATIONS_DIR, quality=None):
    """Load satellite altimetry observations as ModelSkill track observations."""
    obs_fldr = Path(obs_fldr)
    quantity = ms.Quantity(name="Surface Elevation", unit="meter")
    observations = []
    for csv_path in sorted(obs_fldr.glob("Altimetry_wl_*.csv")):
        data = pd.read_csv(csv_path, index_col=0)
        data.index = pd.to_datetime(data.index, format="ISO8601")
        if quality is not None:
            data = data[data["quality_water_level"] == quality]
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*duplicate timestamps.*")
            observations.append(
                ms.TrackObservation(
                    data,
                    item="water_level",
                    x_item="longitude",
                    y_item="latitude",
                    name=csv_path.stem.removeprefix("Altimetry_wl_"),
                    quantity=quantity,
                )
            )
    return observations


def extract_water_level_measurements(stations, ds_time, obs_fldr=_OBSERVATIONS_DIR, window="25min"):
    """Align station observations to simulation timestamps.

    Filtered station files are matched exactly. Raw files use the latest value
    in ``window`` before each simulation timestamp.
    """
    obs_fldr = Path(obs_fldr)
    simulation_times = pd.DatetimeIndex(ds_time.values)
    measurements = np.full((len(simulation_times), len(stations)), np.nan)
    window_delta = pd.Timedelta(window)
    missing_files = []

    for station_index, station in enumerate(stations):
        candidates = (
            obs_fldr / f"{station.name}_wl.csv",
            obs_fldr / f"Altimetry_wl_{station.name}.csv",
        )
        csv_path = next((path for path in candidates if path.exists()), None)
        if csv_path is None:
            missing_files.append(station.name)
            continue

        data = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        if "water_level_filtered" in data:
            measurements[:, station_index] = data["water_level_filtered"].reindex(
                simulation_times
            ).to_numpy()
        elif "water_level" in data:
            water_level = data["water_level"].sort_index()
            for time_index, timestamp in enumerate(simulation_times):
                values = water_level[
                    (water_level.index > timestamp - window_delta)
                    & (water_level.index <= timestamp)
                ]
                if not values.empty:
                    measurements[time_index, station_index] = values.iloc[-1]
        else:
            missing_files.append(station.name)

    if missing_files:
        print("Missing or incompatible measurement files for:", sorted(set(missing_files)))
    return measurements


def plot_shred_reconstruction(
    test_reconstruction, ds, geom, plot_time=None, items=None, cmaps=None, title_suffix=""
):
    """Plot reconstructed fields and return the reconstruction and error datasets."""
    if items is None:
        items = ["Surface elevation", "U velocity", "V velocity"]
    elif isinstance(items, str):
        items = [items]
    else:
        items = list(items)
    cmaps = cmaps or {
        "Surface elevation": "terrain",
        "U velocity": "twilight_shifted",
        "V velocity": "twilight_shifted",
    }

    missing = [item for item in items if item not in test_reconstruction]
    if missing:
        raise KeyError(f"Missing items in test_reconstruction: {missing}")

    times_test = ds.time[-test_reconstruction[items[0]].shape[0]:]
    reconstructions = {}
    for item in items:
        valid_columns = ~np.isnan(ds[item].values).any(axis=0)
        full = np.full((len(times_test), ds[item].values.shape[1]), np.nan)
        full[:, valid_columns] = test_reconstruction[item]
        reconstructions[item] = full

    ds_shred = mikeio.Dataset(
        [
            mikeio.DataArray(
                data=reconstructions[item],
                time=times_test,
                item=ds[item].item,
                geometry=geom,
            )
            for item in items
        ]
    )
    ds_error = mikeio.Dataset(
        [
            mikeio.DataArray(
                data=np.abs(ds[item].sel(time=times_test).values - ds_shred[item].values),
                time=times_test,
                item=ds[item].item,
                geometry=geom,
            )
            for item in items
        ]
    )

    if plot_time is None:
        plot_time = ds_shred.time[-1]
    fig, axes = plt.subplots(len(items), 3, figsize=(15, 4 * len(items)), squeeze=False)
    for row, item in enumerate(items):
        reference = ds[item].sel(time=plot_time)
        prediction = ds_shred[item].sel(time=plot_time)
        error = ds_error[item].sel(time=plot_time)
        vmin = min(np.nanmin(reference.values), np.nanmin(prediction.values))
        vmax = max(np.nanmax(reference.values), np.nanmax(prediction.values))
        reference.plot(ax=axes[row, 0], title=item, cmap=cmaps.get(item, "viridis"), vmin=vmin, vmax=vmax)
        prediction.plot(ax=axes[row, 1], title="SHRED", cmap=cmaps.get(item, "viridis"), vmin=vmin, vmax=vmax)
        error.plot(ax=axes[row, 2], title="Absolute error", cmap="gray")
    plt.tight_layout()
    if title_suffix:
        output_dir = Path(__file__).resolve().parent / "outputs"
        output_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_dir / f"shred_{title_suffix}.png", dpi=200, bbox_inches="tight")
    plt.show()
    return ds_shred, ds_error


# ------------------------------------------------------------------
# Shared style and plotting constants
# ------------------------------------------------------------------
PAPER_RC = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "axes.linewidth": 0.9,
    "xtick.direction": "out",
    "xtick.major.width": 0.9,
    "xtick.major.size": 4,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}

C_TEXT = "#222222"
C_DATA = "#1f4e79"      # valid measurements
C_NAN = "#c8553d"       # missing data
C_BAND = "#f2f4f7"      # alternating row band
C_MARK = "#c8553d"      # map sensor markers

SWE_MODEL_NAMES = (
    "T0A", "T0B", "T1A", "T1B", "T1C", "T2A", "T2B",
    "T3A", "T3B", "T3C", "T3D", "T3E", "T3F", "T3G",
)


# ------------------------------------------------------------------
# Data preprocessing
# ------------------------------------------------------------------
def ROM_data_preprocessing(sensors_data, H,  Utheta, Uphi, augmentation_factor=2):
    np.random.seed(42)  # For reproducibility

    for _ in range(augmentation_factor - 1):
        H = np.vstack((H, H))  # Duplicate the data
        H = H + np.random.normal(0, 0.001, H.shape)  # Add small Gaussian noise for augmentation
        Utheta = np.vstack((Utheta, Utheta))  # Duplicate the data
        Utheta = Utheta + np.random.normal(0, 0.001, Utheta.shape)  # Add small Gaussian noise for augmentation
        Uphi = np.vstack((Uphi, Uphi))  # Duplicate the data
        Uphi = Uphi + np.random.normal(0, 0.001, Uphi.shape)  # Add small Gaussian noise for augmentation
        sensors_data = np.vstack((sensors_data, sensors_data))  # Duplicate the data
        sensors_data = sensors_data + np.random.normal(0, 0.001, sensors_data.shape)  # Add small Gaussian noise for augmentation
        
    H = H.reshape(16, -1, 131072)
    Utheta = Utheta.reshape(16, -1, 131072)
    Uphi = Uphi.reshape(16, -1, 131072)
    sensors_data = sensors_data.reshape(16, -1, 10)

    permutation = np.random.permutation(sensors_data.shape[0])
    sensors_data = sensors_data[permutation]
    H = H[permutation]
    Utheta = Utheta[permutation]
    Uphi = Uphi[permutation]

    print("\nSensors data shape:\t", sensors_data.shape)
    print("H shape:\t\t", H.shape)  
    print("Utheta shape:\t\t", Utheta.shape)
    print("Uphi shape:\t\t", Uphi.shape)

    return sensors_data, H, Utheta, Uphi


def eval_all(
    manager,
    mshred,
    train_dataset,
    val_dataset,
    test_dataset,
    H,
    U,
    V,
    model_name=None,
    output_path="reconstruction_metrics_all_models.json",
):
    """Evaluate one HYDRO model on the manager's actual data splits."""
    engine = SHREDEngine(manager, mshred)
    split_datasets = {"train": train_dataset, "val": val_dataset, "test": test_dataset}
    fields = {
        "Surface elevation": H,
        "U velocity": U,
        "V velocity": V,
    }
    split_indices = {
        "train": manager.train_indices,
        "val": manager.val_indices,
        "test": manager.test_indices,
    }
    reconstructions = {
        split: engine.sensor_to_reconstruction_dataset(dataset)
        for split, dataset in split_datasets.items()
    }
    ground_truths = {
        split: {
            field: values[indices].reshape(-1, values.shape[-1])
            for field, values in fields.items()
        }
        for split, indices in split_indices.items()
    }
    metrics = evaluate_reconstruction_metrics(reconstructions, ground_truths)
    for split in metrics:
        field_metrics = metrics[split]
        metrics[split]["Unify velocity field"] = {
            metric: float(
                np.mean(
                    [
                        field_metrics["U velocity"][metric],
                        field_metrics["V velocity"][metric],
                    ]
                )
            )
            for metric in ("mse", "mae", "nmae")
        }
        metrics[split]["Unify all"] = {
            metric: float(
                np.mean([field_metrics[field][metric] for field in fields])
            )
            for metric in ("mse", "mae", "nmae")
        }

    records = {split: dict(values) for split, values in metrics.items()}
    if model_name is not None:
        records["model"] = model_name
    output_path = Path(output_path)
    if output_path.exists():
        with output_path.open("r", encoding="utf-8") as json_file:
            all_metrics = json.load(json_file)
    else:
        all_metrics = {}
    if model_name is not None:
        all_metrics[model_name] = metrics
        with output_path.open("w", encoding="utf-8") as json_file:
            json.dump(all_metrics, json_file, indent=2)
    return records



def eval_all_lagged(
    manager,
    mshred,
    train_dataset,
    val_dataset,
    test_dataset,
    H,
    U,
    V,
    model_name=None,
    output_path="reconstruction_metrics_all_models.json",
):
    """Evaluate one HYDRO model on the manager's actual data splits."""
    engine = SHREDEngine(manager, mshred)
    split_datasets = {"train": train_dataset, "val": val_dataset, "test": test_dataset}
    fields = {
        "Surface elevation": H,
        "U velocity": U,
        "V velocity": V,
    }
    split_indices = {
        "train": manager.train_indices,
        "val": manager.val_indices,
        "test": manager.test_indices,
    }
    reconstructions = {
        split: engine.sensor_to_reconstruction_dataset_lagged(dataset)
        for split, dataset in split_datasets.items()
    }
    ground_truths = {
        split: {
            field: values[indices].reshape(-1, values.shape[-1])
            for field, values in fields.items()
        }
        for split, indices in split_indices.items()
    }
    metrics = evaluate_reconstruction_metrics(reconstructions, ground_truths)
    for split in metrics:
        field_metrics = metrics[split]
        metrics[split]["Unify velocity field"] = {
            metric: float(
                np.mean(
                    [
                        field_metrics["U velocity"][metric],
                        field_metrics["V velocity"][metric],
                    ]
                )
            )
            for metric in ("mse", "mae", "nmae")
        }
        metrics[split]["Unify all"] = {
            metric: float(
                np.mean([field_metrics[field][metric] for field in fields])
            )
            for metric in ("mse", "mae", "nmae")
        }

    records = {split: dict(values) for split, values in metrics.items()}
    if model_name is not None:
        records["model"] = model_name
    output_path = Path(output_path)
    if output_path.exists():
        with output_path.open("r", encoding="utf-8") as json_file:
            all_metrics = json.load(json_file)
    else:
        all_metrics = {}
    if model_name is not None:
        all_metrics[model_name] = metrics
        with output_path.open("w", encoding="utf-8") as json_file:
            json.dump(all_metrics, json_file, indent=2)
    return records



# ------------------------------------------------------------------
# Sphere rendering helpers
# ------------------------------------------------------------------
def _sphere_geometry(Lx, Ly, n_pix=600, elev_deg=20.0, azim_deg=0.0,
                     light=(-0.45, 0.55, 0.70)):
    """Orthographic sphere: pixel -> (lat, lon) -> bilinear sampling indices + shading."""
    t, az = np.deg2rad(elev_deg), np.deg2rad(azim_deg)
    xs = np.linspace(-1, 1, n_pix)
    X, Y = np.meshgrid(xs, xs)                      # row index grows with y (origin='lower')
    R2 = X**2 + Y**2
    mask = R2 <= 1.0
    x, y = X[mask], Y[mask]
    z = np.sqrt(np.clip(1 - x**2 - y**2, 0, None))

    # screen -> body frame (polar axis = body y), tilt toward viewer by elev
    px = x
    py = y * np.cos(t) + z * np.sin(t)
    pz = -y * np.sin(t) + z * np.cos(t)
    lat = np.arcsin(np.clip(py, -1, 1))
    lon = (np.arctan2(px, pz) + az) % (2 * np.pi)

    r = np.clip((np.pi / 2 - lat) / np.pi * (Lx - 1), 0, Lx - 1)
    c = lon / (2 * np.pi) * Ly
    r0 = np.floor(r).astype(int); r1 = np.minimum(r0 + 1, Lx - 1); wr = r - r0
    c0 = np.floor(c).astype(int) % Ly; c1 = (c0 + 1) % Ly; wc = c - np.floor(c)

    L = np.asarray(light, float); L /= np.linalg.norm(L)
    shade = 0.62 + 0.38 * np.clip(x * L[0] + y * L[1] + z * L[2], 0, 1)

    return dict(n_pix=n_pix, mask=mask, r0=r0, r1=r1, c0=c0, c1=c1, wr=wr, wc=wc,
                shade=shade, t=t, az=az, Lx=Lx, Ly=Ly)

def _project(geo, lat, lon):
    """(lat, lon) in radians -> screen x, y and visibility."""
    t, az = geo["t"], geo["az"]
    d = lon - az
    px, py, pz = np.cos(lat) * np.sin(d), np.sin(lat), np.cos(lat) * np.cos(d)
    x = px
    y = py * np.cos(t) - pz * np.sin(t)
    z = py * np.sin(t) + pz * np.cos(t)
    return x, y, z > 0

def _render_sphere(field, geo, cmap, norm):
    g = geo
    f = ((1 - g["wr"]) * (1 - g["wc"]) * field[g["r0"], g["c0"]]
         + (1 - g["wr"]) * g["wc"] * field[g["r0"], g["c1"]]
         + g["wr"] * (1 - g["wc"]) * field[g["r1"], g["c0"]]
         + g["wr"] * g["wc"] * field[g["r1"], g["c1"]])
    rgba = mpl.colormaps[cmap](norm(f)) if isinstance(cmap, str) else cmap(norm(f))
    rgba[:, :3] *= g["shade"][:, None]
    img = np.zeros((g["n_pix"], g["n_pix"], 4))
    img[g["mask"]] = rgba
    return img

def _draw_sphere(ax, field, geo, cmap, norm, sensors=None, show_equator=True):
    ax.imshow(_render_sphere(field, geo, cmap, norm), origin="lower",
              extent=(-1, 1, -1, 1), interpolation="bilinear")
    if show_equator:
        lon = np.linspace(0, 2 * np.pi, 721)
        x, y, vis = _project(geo, np.zeros_like(lon), lon)
        x, y = np.where(vis, x, np.nan), np.where(vis, y, np.nan)
        ax.plot(x, y, color="k", lw=1.0, zorder=3)
    if sensors is not None:
        lat = np.pi / 2 - sensors[:, 0] / (geo["Lx"] - 1) * np.pi
        lon = sensors[:, 1] / geo["Ly"] * 2 * np.pi
        x, y, vis = _project(geo, lat, lon)
        ax.plot(x[vis], y[vis], "o", ms=3.5, mfc="#a020f0", mec="#3a0a5a",
                mew=0.6, ls="none", zorder=4)
    ax.set_xlim(-1.03, 1.03)
    ax.set_ylim(-1.03, 1.03)
    ax.set_aspect("equal")
    ax.set_axis_off()

# ------------------------------------------------------------------
# SHRED reconstruction plots
# ------------------------------------------------------------------
def plot_shred_reconstruction_multi(
    test_reconstruction,          # dict: item -> (n_test, Lx*Ly)
    ground_truth,                 # dict: item -> (n_test, Lx*Ly), same test split / order
    grid_shape,                   # (Lx, Ly)
    time_idxs=(0, 100, -1),
    items=None,
    cmaps=None,
    title="",
    fig_title=None,
    sensors_coordinates=None,     # (n_sensors, 2) grid indices [row, col], optional
    elev_deg=20.0,
    azim_deg=0.0,
    n_pix=600,
    show_equator=True,
    font_scale=1.0,
    error_cmap="Greys",
    map_height=2.2,
    row_gap=0.35,
    gap=0.15,
    cbar_gap=0.06,
    err_gap=0.08,
    group_gap=0.35,
    cbar_nbins=5,
    cbar_tick_pt=14,
    cbar_unit_pt=15,
    symmetric_vel=False,
    sep_lw=0.8,
):
    """
    One row per time index; each row has 3 blocks (one per variable):
    [GT sphere | SHRED sphere | shared cbar | Error sphere | error cbar].
    """
    fs = lambda x: x * font_scale
    Lx, Ly = grid_shape

    if items is None:
        items = ["Surface elevation", "U theta", "U phi"]
    items = [items] if isinstance(items, str) else list(items)
    if cmaps is None:
        cmaps = {"Surface elevation": "gist_earth", "U theta": "RdBu_r", "U phi": "RdBu_r"}

    for name, d in (("test_reconstruction", test_reconstruction),
                    ("ground_truth", ground_truth)):
        missing = [i for i in items if i not in d]
        if missing:
            raise KeyError(f"Missing items in {name}: {missing}. Available: {list(d.keys())}")

    to_np = lambda a: a.detach().cpu().numpy() if hasattr(a, "detach") else np.asarray(a)
    gt, pred, err = {}, {}, {}
    for item in items:
        g = to_np(ground_truth[item]).reshape(-1, Lx, Ly)
        p = to_np(test_reconstruction[item]).reshape(-1, Lx, Ly)
        if g.shape != p.shape:
            raise ValueError(f"{item}: ground truth {g.shape} vs reconstruction {p.shape}")
        gt[item], pred[item], err[item] = g, p, np.abs(g - p)

    n_test = gt[items[0]].shape[0]
    plot_idx = []
    for ti in time_idxs:
        if not -n_test <= ti < n_test:
            raise IndexError(f"time_idx={ti} out of range: the test set has {n_test} timesteps.")
        plot_idx.append(ti % n_test)

    n_rows, n_vars = len(plot_idx), len(items)
    col_titles = ["Ground Truth", "Masked SHRED", "Absolute error"]
    units = [""] * n_vars

    sens = None if sensors_coordinates is None else np.asarray(sensors_coordinates, float)
    geo = _sphere_geometry(Lx, Ly, n_pix=n_pix, elev_deg=elev_deg, azim_deg=azim_deg)

    # --- Global limits per variable ---
    var_lims, err_lims = {}, {}
    for item in items:
        ref_vals, pred_vals = gt[item][plot_idx], pred[item][plot_idx]
        vmin_g = min(np.nanmin(ref_vals), np.nanmin(pred_vals))
        vmax_g = max(np.nanmax(ref_vals), np.nanmax(pred_vals))
        if symmetric_vel and item.startswith("U"):
            lim = max(abs(vmin_g), abs(vmax_g))
            vmin_g, vmax_g = -lim, lim
        var_lims[item] = (float(vmin_g), float(vmax_g))
        err_lims[item] = (0.0, float(np.nanmax(err[item][plot_idx])))

    # --- Geometry (inches) ---
    lm, tm, cw, cb_text, rm = 0.75, 1.05, 0.12, 0.55, 0.1
    mh = map_height
    mw = mh                                   # spheres are square
    row_pitch = mh + row_gap
    fig_h = tm + n_rows * row_pitch - row_gap / 2

    o_gt = 0.0
    o_sh = o_gt + mw + gap
    o_cb = o_sh + mw + cbar_gap
    o_er = o_cb + cw + cb_text + err_gap
    o_ecb = o_er + mw + cbar_gap
    gw = o_ecb + cw + cb_text
    fig_w = lm + n_vars * gw + (n_vars - 1) * group_gap + rm
    fx = lambda val: val / fig_w
    fy = lambda val: val / fig_h
    cb_h = 0.9 * mh

    y_cols_title = fig_h - tm + 0.25
    y_group_title = y_cols_title + 0.38

    with plt.rc_context(PAPER_RC):
        fig = plt.figure(figsize=(fig_w, fig_h))
        fig.patch.set_facecolor("white")
        colorbars, main_axes = [], set()

        for v, item in enumerate(items):
            xg = lm + v * (gw + group_gap)
            vmin, vmax = var_lims[item]
            err_vmax = err_lims[item][1]
            cmap = cmaps.get(item, "viridis")
            norm = Normalize(vmin=vmin, vmax=vmax)
            norm_e = Normalize(vmin=0.0, vmax=err_vmax)
            u = units[v]

            for r, ti in enumerate(plot_idx):
                y_top = fig_h - tm - r * row_pitch
                y_bot = y_top - mh
                y_mid = y_bot + mh / 2

                ax_gt = fig.add_axes([fx(xg + o_gt), fy(y_bot), fx(mw), fy(mh)])
                ax_sh = fig.add_axes([fx(xg + o_sh), fy(y_bot), fx(mw), fy(mh)])
                ax_er = fig.add_axes([fx(xg + o_er), fy(y_bot), fx(mw), fy(mh)])
                cax_s = fig.add_axes([fx(xg + o_cb), fy(y_mid - 0.45 * mh), fx(cw), fy(cb_h)])
                cax_e = fig.add_axes([fx(xg + o_ecb), fy(y_mid - 0.45 * mh), fx(cw), fy(cb_h)])

                _draw_sphere(ax_gt, gt[item][ti], geo, cmap, norm, sens, show_equator)
                _draw_sphere(ax_sh, pred[item][ti], geo, cmap, norm, sens, show_equator)
                _draw_sphere(ax_er, err[item][ti], geo, error_cmap, norm_e, None, show_equator)
                main_axes.update((ax_gt, ax_sh, ax_er))

                sm = ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
                colorbars.append(fig.colorbar(sm, cax=cax_s))
                sm_e = ScalarMappable(norm=norm_e, cmap=error_cmap); sm_e.set_array([])
                colorbars.append(fig.colorbar(sm_e, cax=cax_e))
                for cax in (cax_s, cax_e):
                    cax.set_title(f"[{u}]" if u else "", fontsize=cbar_unit_pt, pad=5)

            for c, off in enumerate([o_gt, o_sh, o_er]):
                fig.text(fx(xg + off + mw / 2), fy(y_cols_title), col_titles[c],
                         ha="center", va="bottom",
                         fontsize=fs(20), fontweight="bold", color="black")

            label = f"{item} [{u}]" if u else item
            fig.text(fx(xg + gw / 2), fy(y_group_title), label,
                     ha="center", va="bottom",
                     fontsize=fs(26), fontweight="bold", color=C_TEXT)

            if v < n_vars - 1:
                x_sep = xg + gw + group_gap / 2
                fig.add_artist(plt.Line2D(
                    [fx(x_sep), fx(x_sep)], [fy(0.1), fy(y_group_title + 0.3)],
                    color="0.25", lw=sep_lw, ls=(0, (4, 3))))

        for r, ti in enumerate(plot_idx):
            y_top = fig_h - tm - r * row_pitch
            y_mid = y_top - mh / 2
            fig.text(fx(lm - 0.12), fy(y_mid), f"Test step\n{ti}",
                     rotation=90, va="center", ha="right",
                     fontsize=fs(20), fontweight="bold", color=C_TEXT)
            if r < n_rows - 1:
                y_sep = y_top - mh - row_gap / 2
                fig.add_artist(plt.Line2D(
                    [0.005, 0.995], [fy(y_sep), fy(y_sep)],
                    color="0.25", lw=sep_lw, ls=(0, (4, 3))))

        for cb in colorbars:
            cb.locator = MaxNLocator(nbins=cbar_nbins)
            cb.update_ticks()
        for cax in fig.axes:
            if cax in main_axes:
                continue
            cax.tick_params(labelsize=cbar_tick_pt, width=0.8, length=3)
            for spine in cax.spines.values():
                spine.set_linewidth(0.6)
                spine.set_color("0.5")

        try:
            base_dir = Path(__file__).resolve().parent
        except NameError:
            base_dir = Path.cwd()
        output_path = base_dir / f"{title}.png"
        fig.savefig(output_path, dpi=600, bbox_inches="tight", pad_inches=0.02, facecolor="white")
        print(f"Saved plot: {output_path}")
        plt.show()

    del fig, gt, pred, err

# ------------------------------------------------------------------
# Sensor availability plots
# ------------------------------------------------------------------
def plot_data_availability(sensor_data, title_prefix, sensor_labels=None):
    np.random.seed(42)  # For reproducibility
    """Plot valid measurements over time for each sensor."""
    if hasattr(sensor_data, "detach"):
        sensor_data = sensor_data.detach().cpu().numpy()
    availability = ~np.isnan(np.asarray(sensor_data))

    if availability.ndim == 3:
        availability = availability.reshape(-1, availability.shape[-1])
    elif availability.ndim != 2:
        raise ValueError(
            "sensor_data must have shape (ntimes, nsensors) or "
            "(nbatches, ntimes, nsensors)"
        )

    ntimes, nsensors = availability.shape
    if sensor_labels is None:
        sensor_labels = np.arange(nsensors)
    sensor_labels = np.asarray(sensor_labels)
    if sensor_labels.shape != (nsensors,):
        raise ValueError("sensor_labels must contain one label per sensor")

    time = np.arange(ntimes)
    fig_height = max(3.0, 0.7 * nsensors)
    fig, ax = plt.subplots(figsize=(14, fig_height))

    for sensor_idx in range(nsensors):
        line = np.where(availability[:, sensor_idx], sensor_idx + 1.0, np.nan)
        ax.plot(
            time,
            line,
            linewidth=4,
            solid_capstyle="butt",
            label=f"S{sensor_labels[sensor_idx]}",
        )

    ax.set_xlabel("Time step")
    ax.set_ylabel("Sensor")
    ax.set_yticks(np.arange(1, nsensors + 1))
    ax.set_yticklabels([f"S{label}" for label in sensor_labels])
    ax.set_ylim(0.5, nsensors + 0.5)
    ax.grid(True, axis="x", alpha=0.25)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), title="Sensors")
    ax.set_title(title_prefix)
    fig.tight_layout()
    plt.show()
    return fig, ax

def plot_station_availability(
    wllist,
    ntimes,
    ax=None,
    row_h=0.27,
    sig_offset=0.13,
    mask_offset=-0.33,
    mask_h=0.12,
    marker_size=1.4,
    c_data=C_DATA,
    c_nan=C_NAN,
    figsize=(30, 10),
    font_scale=1.0,
    time_pad_frac=0.03,
    xlim=None,
    show_title=True,
    panel_label=None,
    savepath=None,
    dpi=600,
):
    """
    Sensor availability against the time-step axis.
      - sensor stream: blue dots
      - missing data: red bars

    Returns: fig, ax, coverage_pct_by_name
    """
    fs = lambda x: x * font_scale

    time = np.arange(ntimes)
    coverage_pct_by_name = {}
    series_by_name = {}

    for obs in wllist:
        s = np.asarray(obs.values, dtype=float).ravel()
        series_by_name[obs.name] = s
        coverage_pct_by_name[obs.name] = 100.0 * np.isfinite(s).sum() / ntimes

    # Lowest coverage at bottom, highest at top
    wllist_sorted = sorted(wllist, key=lambda o: coverage_pct_by_name[o.name])
    n = len(wllist_sorted)

    if xlim is not None:
        t0, t1 = xlim
    else:
        pad = (ntimes - 1) * time_pad_frac
        t0, t1 = -pad, (ntimes - 1) + pad

    standalone = ax is None

    with plt.rc_context(PAPER_RC):
        if standalone:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.figure

        for i, obs in enumerate(wllist_sorted):
            s = series_by_name[obs.name]
            mask = np.isfinite(s)

            y_sig = i + sig_offset
            y_mask = i + mask_offset

            if i % 2 == 0:
                ax.axhspan(i - 0.5, i + 0.5, color=C_BAND, zorder=-2, lw=0)

            ax.axhline(y_sig, color="0.86", lw=0.6, zorder=0)

            # Sensor stream
            if mask.any():
                vals = s[mask] - s[mask].mean()
                scale = np.percentile(np.abs(vals), 99)
                if not np.isfinite(scale) or scale <= 0:
                    scale = 1.0
                y = y_sig + np.clip(vals / scale, -1, 1) * row_h
                ax.plot(
                    time[mask], y,
                    linestyle="none", marker="o", ms=marker_size, mew=0,
                    color=c_data, zorder=2, rasterized=True,
                )

            # Missing data
            if (~mask).any():
                miss = (~mask).astype(int)
                edges = np.diff(np.concatenate([[0], miss, [0]]))
                starts = np.where(edges == 1)[0]
                ends = np.where(edges == -1)[0]

                min_w = (t1 - t0) * 0.0008
                widths = np.maximum(ends - starts, min_w)

                ax.broken_barh(
                    list(zip(starts, widths)),
                    (y_mask - mask_h / 2, mask_h),
                    facecolors=c_nan, edgecolors="none",
                    zorder=1, rasterized=True,
                )

        ax.set_ylim(-0.6, n - 0.4)
        ax.set_xlim(t0, t1)

        ax.tick_params(axis="x", labelsize=fs(20), pad=4)
        ax.set_xlabel("Time step", fontsize=fs(20), labelpad=6, fontweight="bold")

        # Left labels: sensor names
        ax.set_yticks(range(n))
        ax.set_yticklabels([o.name for o in wllist_sorted], fontsize=fs(20))
        ax.tick_params(axis="y", length=0, pad=8)

        # Right labels: temporal coverage
        ax2 = ax.twinx()
        ax2.set_ylim(ax.get_ylim())
        ax2.set_yticks(range(n))
        ax2.set_yticklabels(
            [f"{coverage_pct_by_name[o.name]:.1f}" for o in wllist_sorted],
            fontsize=fs(20),
        )
        ax2.tick_params(axis="y", length=0, pad=8)
        ax2.set_ylabel("Temporal coverage (%)", fontsize=fs(20), labelpad=12, fontweight="bold")

        ax.grid(axis="x", color="0.9", lw=0.6)
        ax.set_axisbelow(True)

        for a in (ax, ax2):
            for side in ("top", "left", "right"):
                a.spines[side].set_visible(False)

        if panel_label:
            ax.text(-0.30, 1.02, panel_label, transform=ax.transAxes,
                    fontsize=fs(20), fontweight="bold", va="bottom", ha="left")

        if standalone and show_title:
            ax.set_title("Sensor availability", fontsize=fs(20), pad=14, fontweight="bold")

        ax.legend(
            handles=[
                Line2D([0], [0], marker="o", ls="none", color=c_data, ms=7, label="Measurement"),
                Patch(facecolor=c_nan, edgecolor="none", label="Missing value"),
            ],
            loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2,
            frameon=False, fontsize=fs(20), handletextpad=0.5, columnspacing=2.5,
        )

        if standalone:
            fig.tight_layout()
            if savepath:
                fig.savefig(savepath, bbox_inches="tight", dpi=dpi)
            plt.show()

    return fig, ax, coverage_pct_by_name

# ------------------------------------------------------------------
# Sensor map labels
# ------------------------------------------------------------------
def _place_map_labels(ax, stations, fontsize=20, offset=(9, 6), pad=3.0, max_iter=500):
    """Sensor names on the map, pushed apart vertically; leader line only if moved."""
    fig = ax.figure
    pts = [(st.name, float(st.x), float(st.y)) for st in stations]

    for txt in list(ax.texts):
        txt.remove()

    halo = [pe.withStroke(linewidth=3.0, foreground="white")]
    k = fig.dpi / 72.0

    anns = [
        ax.annotate(
            name, xy=(x, y), xytext=offset, textcoords="offset points",
            fontsize=fontsize, ha="left", va="bottom", zorder=6, path_effects=halo,
            arrowprops=dict(arrowstyle="-", color="0.4", lw=0.7,
                            shrinkA=2, shrinkB=6, relpos=(0, 0.5)),
        )
        for name, x, y in pts
    ]

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    n = len(anns)

    anchor = np.array([ax.transData.transform((x, y)) for _, x, y in pts])
    size = np.array([[b.width, b.height]
                     for b in (Text.get_window_extent(a, renderer) for a in anns)])

    off0 = np.tile(np.array(offset, float) * k, (n, 1))
    off = off0.copy()
    marker_r = 8.0 * k

    def box(i):
        x0, y0 = anchor[i] + off[i]
        return x0, y0, x0 + size[i, 0], y0 + size[i, 1]

    for _ in range(max_iter):
        moved = False
        for i in range(n):
            xi0, yi0, xi1, yi1 = box(i)

            # label-label overlap
            for j in range(i + 1, n):
                xj0, yj0, xj1, yj1 = box(j)
                ov_x = min(xi1, xj1) - max(xi0, xj0) + pad * k
                ov_y = min(yi1, yj1) - max(yi0, yj0) + pad * k
                if ov_x > 0 and ov_y > 0:
                    step = ov_y / 2 + 0.5
                    if yi0 + yi1 >= yj0 + yj1:
                        off[i, 1] += step
                        off[j, 1] -= step
                    else:
                        off[i, 1] -= step
                        off[j, 1] += step
                    moved = True
                    xi0, yi0, xi1, yi1 = box(i)

            # label-marker overlap
            for j in range(n):
                if j == i:
                    continue
                mx, my = anchor[j]
                ov_x = min(xi1, mx + marker_r) - max(xi0, mx - marker_r)
                ov_y = min(yi1, my + marker_r) - max(yi0, my - marker_r)
                if ov_x > 0 and ov_y > 0:
                    direction = 1.0 if (yi0 + yi1) / 2 >= my else -1.0
                    off[i, 1] += direction * (ov_y + 1.0)
                    moved = True
                    xi0, yi0, xi1, yi1 = box(i)

        if not moved:
            break

    for a, o, o_init in zip(anns, off, off0):
        a.xyann = (o[0] / k, o[1] / k)
        a.arrow_patch.set_visible(bool(np.hypot(*(o - o_init)) > 6 * k))

# ------------------------------------------------------------------
# Combined overview figure
# ------------------------------------------------------------------
def plot_overview_and_availability(
    wllist,
    H0,
    ntimes,
    figsize=(30, 9),
    font_scale=1.0,
    width_ratios=(0.3, 0.7),
    wspace=0.1,
    panel_labels=False,
    title="Sensor availability",
    elev_deg=20.0,
    azim_deg=90.0,
    n_pix=600,
    show_equator=True,
    savepath=None,
    dpi=600,
    **avail_kwargs,
):
    """
    Left : sensor locations on a sphere coloured by the height field at t = 0.
    Right: sensor availability timeline.
    """
    fs = lambda x: x * font_scale
    Lx, Ly = H0.shape

    with plt.rc_context(PAPER_RC):
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(1, 2, width_ratios=width_ratios, wspace=wspace)
        axs = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])]

        # LEFT: sphere
        geo = _sphere_geometry(Lx, Ly, n_pix=n_pix, elev_deg=elev_deg, azim_deg=azim_deg)
        norm = Normalize(vmin=float(np.nanmin(H0)), vmax=float(np.nanmax(H0)))
        _draw_sphere(axs[0], H0, geo, "gist_earth", norm, sensors=None, show_equator=show_equator)

        # project sensors (row = colatitude, col = longitude); keep the visible ones
        rows = np.array([o.y for o in wllist])
        cols = np.array([o.x for o in wllist])
        lat = np.pi / 2 - rows / (Lx - 1) * np.pi
        lon = cols / Ly * 2 * np.pi
        px, py, vis = _project(geo, lat, lon)

        visible = [SimpleNamespace(name=o.name, x=float(px[i]), y=float(py[i]))
                   for i, o in enumerate(wllist) if vis[i]]
        hidden = [o.name for i, o in enumerate(wllist) if not vis[i]]
        if hidden:
            print(f"Sensors on the far side of the sphere (not drawn): {hidden}")

        axs[0].scatter(
            [s.x for s in visible], [s.y for s in visible],
            s=fs(110), c=C_MARK, edgecolors="white", linewidths=2.2, zorder=5,
        )

        # RIGHT: availability
        _, _, cov = plot_station_availability(
            wllist, ntimes, ax=axs[1], font_scale=font_scale, **avail_kwargs
        )
        axs[1].set_title("")

        # Map labels (visible sensors only)
        _place_map_labels(axs[0], visible, fontsize=fs(20))

        if panel_labels:
            axs[0].text(0.0, 1.02, "(a)", transform=axs[0].transAxes,
                        fontsize=fs(22), fontweight="bold", va="bottom", ha="left")
            axs[1].text(-0.10, 1.02, "(b)", transform=axs[1].transAxes,
                        fontsize=fs(22), fontweight="bold", va="bottom", ha="left")

        if title:
            fig.suptitle(title, fontsize=fs(25), fontweight="bold", x=0.5, y=1.0)

        if savepath:
            fig.savefig(savepath, bbox_inches="tight", dpi=dpi, facecolor="white")

        plt.show()

    return fig, axs, cov
