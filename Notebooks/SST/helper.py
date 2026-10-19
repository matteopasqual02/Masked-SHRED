## This file also contains a utility function for plotting training curves: 
# - plot_training_curves: plot the training and validation MSE loss curves over epochs using matplotlib
import mikeio
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import modelskill as ms
import torch

import sys
from pathlib import Path
import os

def plot_training_curves(train_errors, valid_errors):

    # PLOT TRAINING CURVES
    plt.figure(figsize=(10, 5))
    plt.plot(train_errors, label='Train MSE')
    plt.plot(valid_errors, label='Validation MSE')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Training and Validation MSE Loss Curves')
    plt.legend()
    plt.grid()
    plt.show()

def evaluate_model(
    evaluate,
    mshred,
    train_dataset,
    val_dataset,
    test_dataset,
    model,
    type,
    masking_type=None,
):
    
    # MSE and MRE
    if masking_type is None:
        train_mse, train_mre, train_nrmse = evaluate(mshred, train_dataset)
        val_mse, val_mre, val_nrmse = evaluate(mshred, val_dataset)
        test_mse, test_mre, test_nrmse = evaluate(mshred, test_dataset)
    else:
        train_mse, train_mre, train_nrmse = evaluate(mshred, train_dataset, masking_type=masking_type)
        val_mse, val_mre, val_nrmse = evaluate(mshred, val_dataset, masking_type=masking_type)
        test_mse, test_mre, test_nrmse = evaluate(mshred, test_dataset, masking_type=masking_type)
        
    test_error = {
        "model": model,
        "type": type,
        "train-mse": float(train_mse),
        "val-mse": float(val_mse),
        "test-mse": float(test_mse),
        "train-mre": float(train_mre),
        "val-mre": float(val_mre),
        "test-mre": float(test_mre),
        "train-nrmse": float(train_nrmse),
        "val-nrmse": float(val_nrmse),
        "test-nrmse": float(test_nrmse),
    }
    return test_error

import numpy as np
import matplotlib.pyplot as plt


def plot_sst_with_buoys(
    X_raw,
    S_lon,
    S_lat,
    frame_idx=-1,
    shape=(180, 360),
    cmap="jet",
    figsize=(10, 4),
    title="Last Frame of SST Data",
    ax=None,
    show=True,
):
    """
    Plotta un frame SST con la posizione dei sensori/boe sovrapposta.

    Parametri
    ---------
    X_raw : ndarray (n_frames, n_lat * n_lon)
        Dati SST grezzi, un frame per riga.
    S_lon, S_lat : array-like
        Coordinate geografiche delle boe (lon in [-180, 180] o [0, 360), lat in [-90, 90]).
    frame_idx : int
        Indice del frame da mostrare (default: l'ultimo).
    shape : tuple
        Forma della griglia (n_lat, n_lon).
    cmap : str
        Colormap di matplotlib.
    figsize : tuple
        Dimensione della figura (usata solo se ax è None).
    title : str
        Titolo del grafico.
    ax : matplotlib Axes, opzionale
        Se fornito, il plot viene disegnato su questo asse.
    show : bool
        Se True chiama plt.show().

    Ritorna
    -------
    fig, ax : Figure e Axes di matplotlib
    """
    # Frame SST scelto
    sst = np.asarray(X_raw)[frame_idx, :].reshape(shape)

    # Longitudine: [-180, 180] -> [0, 360)
    lon_plot = np.asarray(S_lon) % 360
    lat_plot = np.asarray(S_lat)

    # Figura / asse
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    im = ax.imshow(
        sst,
        cmap=cmap,
        origin="upper",
        extent=[0, 360, -90, 90],
        aspect="auto",
    )

    ax.scatter(
        lon_plot,
        lat_plot,
        c="white",
        edgecolors="black",
        s=45,
        linewidths=1.2,
        label="Buoys",
        zorder=3,
    )

    ax.set_xlim(0, 360)
    ax.set_ylim(-90, 90)
    ax.set_xlabel("Longitude (°)")
    ax.set_ylabel("Latitude (°)")
    ax.set_title(title)
    ax.set_xticks(np.arange(0, 361, 50))
    ax.set_yticks(np.arange(-90, 91, 30))

    fig.colorbar(im, ax=ax, label="SST (°C)")
    ax.legend()
    fig.tight_layout()

    if show:
        plt.show()

    return fig, ax


def plot_reconstruction(test_reconstruction, X_raw, sst_locs, index=-1 , title="None", save_fig=False):
    title_save = title
    #compare the last step of the reconstruction with the original data
    last_step_hat = test_reconstruction['Surface elevation'][index]
    last_step_hat_raw = np.zeros(X_raw.shape[1])
    last_step_hat_raw[sst_locs] = last_step_hat

    X_raw_last_step = X_raw[index]

    import copy

    X_orig = X_raw_last_step.reshape(180, 360)
    X_hat = last_step_hat_raw.reshape(180, 360)
    X_err = np.abs(X_orig - X_hat)

    # Mask land — adjust the condition to match how land is actually encoded
    # (NaN is most common; swap for `== 0` or another sentinel if that's your case)
    land_mask = np.isnan(X_orig)

    X_orig_m = np.ma.masked_where(land_mask, X_orig)
    X_hat_m  = np.ma.masked_where(land_mask, X_hat)
    X_err_m  = np.ma.masked_where(land_mask, X_err)

    vmin = min(X_orig_m.min(), X_hat_m.min())
    vmax = max(X_orig_m.max(), X_hat_m.max())

    from mpl_toolkits.axes_grid1 import make_axes_locatable

    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 11,
        'axes.titlesize': 12,
    })

    # Black land for the two temperature maps; error map keeps its own cmap as-is
    cmap_temp = copy.copy(plt.get_cmap('RdYlBu_r'))
    cmap_temp.set_bad(color='black')

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

    panels = [
        (X_orig_m, f'Ground Truth {1424 + index}', cmap_temp, vmin, vmax, '°C'),
        (X_hat_m,  'M-SHRED',      cmap_temp, vmin, vmax, '°C'),
        (X_err_m,  'Absolute Error', 'inferno', None, None, '°C'),
    ]

    for ax, (data, title, cmap, vmn, vmx, unit) in zip(axes, panels):
        im = ax.imshow(data, cmap=cmap, vmin=vmn, vmax=vmx,
                        aspect='equal', origin='upper')
        ax.set_title(title, fontweight='semibold', pad=8)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.8)

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="4%", pad=0.08)
        cbar = fig.colorbar(im, cax=cax)
        cbar.set_label(unit, fontsize=9, labelpad=4)
        cbar.ax.tick_params(labelsize=8, length=3)
        cbar.outline.set_linewidth(0.6)

    plt.tight_layout()

    if save_fig:
        if not os.path.exists('outputs'):
            os.makedirs('outputs')
        #plt.savefig(f'outputs/sst_reconstruction_{title}.pdf', bbox_inches='tight')
        plt.savefig(f'outputs/sst_reconstruction_{title_save}.png', dpi=300, bbox_inches='tight')

    plt.show()


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.text import Text

PAPER_RC = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "axes.linewidth": 0.9,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}
C_DATA = "#1f4e79"
C_NAN = "#c8553d"
C_BAND = "#f2f4f7"
C_LAND = "#e9e6df"

def _render_globe(ax, field, lon0, lat0, cmap, bad_color, n=900):
    """Rende un campo (n_lat, n_lon) su un globo in proiezione ortografica."""
    n_lat, n_lon = field.shape
    g = np.linspace(-1, 1, n)
    X, Y = np.meshgrid(g, -g)                       # y verso l'alto
    rho = np.hypot(X, Y)
    inside = rho <= 1.0
    rho_s = np.where(rho == 0, 1e-12, rho)
    c = np.arcsin(np.clip(rho, 0, 1))
    p0, l0 = np.radians(lat0), np.radians(lon0)

    lat = np.arcsin(np.clip(np.cos(c) * np.sin(p0) + Y * np.sin(c) * np.cos(p0) / rho_s, -1, 1))
    lon = l0 + np.arctan2(X * np.sin(c),
                          rho_s * np.cos(p0) * np.cos(c) - Y * np.sin(p0) * np.sin(c))

    rows = np.clip(np.floor(90 - np.degrees(lat)).astype(int), 0, n_lat - 1)
    cols = np.clip(np.floor(np.degrees(lon) % 360).astype(int), 0, n_lon - 1)
    import matplotlib.colors as mcolors
    import matplotlib.cm as mcm

    img = np.ma.masked_where(np.ma.getmaskarray(field)[rows, cols], field[rows, cols])
    cm = plt.get_cmap(cmap).copy()
    cm.set_bad(bad_color)                           # terra / dati mancanti
    norm = mcolors.Normalize(vmin=field.min(), vmax=field.max())
    rgba = cm(norm(img))
    rgba[~inside, 3] = 0.0                          # fuori dal globo: trasparente
    ax.imshow(rgba, extent=[-1, 1, -1, 1], origin="upper",
              interpolation="nearest", zorder=1)

    th = np.linspace(0, 2 * np.pi, 400)             # contorno sottile del globo
    ax.plot(np.cos(th), np.sin(th), color="k", lw=0.8, zorder=4)
    return mcm.ScalarMappable(norm=norm, cmap=cm)


def _ortho_xy(lon, lat, lon0, lat0):
    """Proiezione ortografica diretta; ritorna x, y, visibile."""
    lo, la = np.radians(np.asarray(lon, float)), np.radians(np.asarray(lat, float))
    l0, p0 = np.radians(lon0), np.radians(lat0)
    cosc = np.sin(p0) * np.sin(la) + np.cos(p0) * np.cos(la) * np.cos(lo - l0)
    x = np.cos(la) * np.sin(lo - l0)
    y = np.cos(p0) * np.sin(la) - np.sin(p0) * np.cos(la) * np.cos(lo - l0)
    return x, y, cosc > 0


def plot_sst_overview_and_availability(
    X_raw,
    S_lon,
    S_lat,
    S_names=None,
    time=None,                 # array di date (n_frames,); se None usa l'indice del frame
    X_buoys=None,              # (n_frames, n_buoys); se None viene estratto da X_raw
    frame_idx=-1,              # frame SST mostrato nella mappa
    shape=(180, 360),
    missing_value=0.0,         # valore da considerare "mancante" (oltre a NaN); None per ignorarlo
    cmap="jet",
    figsize=(15, 4.5),         # formato paper: più largo che alto
    font_scale=0.72,
    width_ratios=(0.36, 0.64),
    wspace=0.12,
    row_h=0.27, sig_offset=0.13, mask_offset=-0.33, mask_h=0.12,
    marker_size=1.0,
    title="SST buoy availability",
    panel_labels=True,
    view_lon=None,             # longitudine centrale del globo; None = scelta automatica
    view_lat=20.0,             # latitudine centrale del globo
    rotate_deg=30.0,           # rotazione extra del globo verso destra (gradi); negativo = verso sinistra
    show_equator=True,
    show_map_labels=False,     # numeri/nomi delle boe sul globo (tolti di default)
    show_coverage=True,        # percentuali di copertura a destra
    savepath=None,
    dpi=300,
    show=True,
):
    """
    Sinistra: globo SST (frame scelto, proiezione ortografica) con le boe.
    Destra: disponibilità temporale per boa (serie demediata e scalata +
    barre rosse per i dati mancanti + copertura %).

    Ritorna
    -------
    fig, axs, coverage_pct_by_name
    """
    import matplotlib
    import matplotlib.patheffects as pe

    fs = lambda x: x * font_scale
    X_raw = np.asarray(X_raw)
    n_frames = X_raw.shape[0]
    n_lat, n_lon = shape

    lon = np.asarray(S_lon, float) % 360
    lat = np.asarray(S_lat, float)
    n = len(lon)
    names = list(S_names) if S_names is not None else [f"B{i+1}" for i in range(n)]

    # ---- serie temporali delle boe ----
    if X_buoys is None:
        rows = np.clip(np.floor(90 - lat).astype(int), 0, n_lat - 1)   # riga 0 = lat 90
        cols = np.clip(np.floor(lon).astype(int), 0, n_lon - 1)
        X_buoys = X_raw[:, rows * n_lon + cols]
    X_buoys = np.asarray(X_buoys, float)

    valid = ~np.isnan(X_buoys)
    if missing_value is not None:
        valid &= X_buoys != missing_value
    coverage = 100.0 * valid.sum(axis=0) / n_frames
    cov_by_name = dict(zip(names, coverage))
    order = np.argsort(coverage, kind="stable")        # copertura più bassa in basso

    # ---- asse tempo ----
    if time is not None:
        t = pd.DatetimeIndex(time)
        x = mdates.date2num(t)
        is_dt = True
    else:
        x = np.arange(n_frames, dtype=float)
        is_dt = False
    dx = np.median(np.diff(x))
    span = x[-1] - x[0]
    x0_lim, x1_lim = x[0] - 0.03 * span, x[-1] + 0.03 * span

    # ---- vista del globo: longitudine che mostra più boe ----
    if view_lon is None:
        cand = np.arange(0, 360, 5)
        counts = [_ortho_xy(lon, lat, c, view_lat)[2].sum() for c in cand]
        view_lon = float(cand[int(np.argmax(counts))])
    # globo ruotato verso destra: la superficie scorre a destra, il centro si sposta a ovest
    view_lon = (view_lon - rotate_deg) % 360

    with plt.rc_context(PAPER_RC):
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(1, 2, width_ratios=width_ratios, wspace=wspace)
        ax_map = fig.add_subplot(gs[0, 0])
        ax = fig.add_subplot(gs[0, 1])

        # ================= sinistra: globo =================
        sst = X_raw[frame_idx, :].reshape(shape).astype(float)
        bad = np.isnan(sst)
        if missing_value is not None:
            bad |= sst == missing_value
        sst = np.ma.masked_where(bad, sst)

        im = _render_globe(ax_map, sst, view_lon, view_lat, cmap, C_LAND)

        if show_equator:
            el = np.linspace(0, 360, 721)
            ex, ey, ev = _ortho_xy(el, np.zeros_like(el), view_lon, view_lat)
            ax_map.plot(np.where(ev, ex, np.nan), np.where(ev, ey, np.nan),
                        color="k", lw=1.0, zorder=3)

        bx, by, bv = _ortho_xy(lon, lat, view_lon, view_lat)
        ax_map.scatter(bx[bv], by[bv], s=fs(110), c=C_NAN, edgecolors="white",
                       linewidths=1.3, zorder=5)
        if show_map_labels:
            for i in np.where(bv)[0]:
                ax_map.annotate(names[i], (bx[i], by[i]), xytext=(4, 8),
                                textcoords="offset points", fontsize=fs(14),
                                path_effects=[pe.withStroke(linewidth=3, foreground="white")],
                                zorder=6)

        ax_map.set_xlim(-1.02, 1.02)
        ax_map.set_ylim(-1.02, 1.02)
        ax_map.set_aspect("equal")
        ax_map.axis("off")
        cb = fig.colorbar(im, ax=ax_map, orientation="horizontal",
                          pad=0.02, fraction=0.04, shrink=0.85)
        cb.set_label("SST (°C)", fontsize=fs(16))
        cb.ax.tick_params(labelsize=fs(14))

        # ================= destra: disponibilità =================
        for r, i in enumerate(order):
            m = valid[:, i]
            y_sig, y_mask = r + sig_offset, r + mask_offset

            if r % 2 == 0:
                ax.axhspan(r - 0.5, r + 0.5, color=C_BAND, zorder=-2, lw=0)
            ax.axhline(y_sig, color="0.86", lw=0.6, zorder=0)

            if m.any():
                v = X_buoys[m, i]
                v = v - v.mean()
                sc = np.percentile(np.abs(v), 99)
                sc = sc if np.isfinite(sc) and sc > 0 else 1.0
                y = y_sig + np.clip(v / sc, -1, 1) * row_h
                ax.plot(x[m], y, ls="none", marker="o", ms=marker_size,
                        mew=0, color=C_DATA, zorder=2, rasterized=True)

            if (~m).any():
                miss = (~m).astype(int)
                edges = np.diff(np.concatenate([[0], miss, [0]]))
                starts = np.where(edges == 1)[0]
                ends = np.where(edges == -1)[0]            # esclusivo
                xs0 = x[starts]
                xs1 = x[ends - 1] + dx
                widths = np.maximum(xs1 - xs0, (x1_lim - x0_lim) * 0.0008)
                ax.broken_barh(list(zip(xs0, widths)),
                               (y_mask - mask_h / 2, mask_h),
                               facecolors=C_NAN, edgecolors="none",
                               zorder=1, rasterized=True)

        ax.set_ylim(-0.6, n - 0.4)
        ax.set_xlim(x0_lim, x1_lim)

        if is_dt:
            ax.xaxis_date()
            loc = mdates.AutoDateLocator(minticks=6, maxticks=12)
            ax.xaxis.set_major_locator(loc)
            ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))
            xlabel = "Time"
        else:
            xlabel = "Frame"
        ax.tick_params(axis="x", labelsize=fs(16), pad=4)
        ax.set_xlabel(xlabel, fontsize=fs(16), labelpad=6, fontweight="bold")

        ax.set_yticks(range(n))
        ax.set_yticklabels([names[i] for i in order], fontsize=fs(16))
        ax.tick_params(axis="y", length=0, pad=8)

        ax2 = ax.twinx()
        ax2.set_ylim(ax.get_ylim())
        ax2.set_yticks(range(n))
        if show_coverage:
            ax2.set_yticklabels([f"{coverage[i]:.1f}" for i in order], fontsize=fs(16))
            ax2.set_ylabel("Temporal coverage (%)", fontsize=fs(16),
                           labelpad=12, fontweight="bold")
        else:
            ax2.set_yticklabels([])
        ax2.tick_params(axis="y", length=0, pad=8)

        ax.grid(axis="x", color="0.9", lw=0.6)
        ax.set_axisbelow(True)
        for a in (ax, ax2):
            for side in ("top", "left", "right"):
                a.spines[side].set_visible(False)

        ax.legend(
            handles=[
                Line2D([0], [0], marker="o", ls="none", color=C_DATA, ms=7,
                       label="Measurement"),
                Patch(facecolor=C_NAN, edgecolor="none", label="Missing value"),
            ],
            loc="upper center", bbox_to_anchor=(0.5, -0.10), ncol=2,
            frameon=False, fontsize=fs(16), handletextpad=0.5, columnspacing=2.5,
        )

        if panel_labels:
            ax_map.text(0.0, 1.02, "(a)", transform=ax_map.transAxes,
                        fontsize=fs(18), fontweight="bold", va="bottom", ha="left")
            ax.text(-0.10, 1.02, "(b)", transform=ax.transAxes,
                    fontsize=fs(18), fontweight="bold", va="bottom", ha="left")
        if title:
            fig.suptitle(title, fontsize=fs(22), fontweight="bold", x=0.5, y=1.0)

        if savepath:
            fig.savefig(savepath, bbox_inches="tight", dpi=dpi)
        if show:
            plt.show()

    return fig, [ax_map, ax], cov_by_name