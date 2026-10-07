"""
viz.py

Plotting functions for the methane plume injection notebook.

Each function produces one figure.  All take pre-computed data as
arguments -- no I/O, no computation beyond what's needed for layout.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import rasterio

from trans_enhmap import _get_utm_transformer, _project_grid

# Default plume colour palette
PLUME_COLORS = [
    "#F4A6C1", "#87CEEB", "#C9A0DC", "#FFF176", "#FFCC80",
    "#A5D6A7", "#E0C09F", "#EF9A9A", "#80DEEA", "#CE93D8",
]


# ------------------------------------------------------------------
# 1. Scene overview
# ------------------------------------------------------------------

def plot_scene_overview(
    scene_row: dict,
    reading_window: rasterio.windows.Window,
    patch_h: int,
    patch_w: int,
    emit_wvl: np.ndarray,
    plume_mask_dilated: np.ndarray,
) -> None:
    """Full EMIT scene in RGB with plume mask, tile grid, and selected tile.

    Parameters
    ----------
    scene_row : dict-like
        Scene metadata row from tacoreader.
    reading_window :
        Rasterio window of the selected patch.
    patch_h, patch_w :
        Tile dimensions in pixels.
    emit_wvl : (n_bands,)
        EMIT central wavelengths [nm] (for RGB band selection).
    plume_mask_dilated : (H_scene, W_scene)
        Dilated plume mask (as returned by ``get_emit_background``).
    """
    _vsi = scene_row["internal:gdal_vsi"]
    scene_dir = _vsi.rsplit("/", 1)[0]

    # RGB band indices
    rgb_bands = [
        int(np.abs(emit_wvl - 640).argmin()),
        int(np.abs(emit_wvl - 550).argmin()),
        int(np.abs(emit_wvl - 460).argmin()),
    ]

    with rasterio.open(f"{scene_dir}/radiance.tif") as src:
        rgb = np.stack(
            [src.read(b + 1) for b in rgb_bands], axis=-1
        ).astype(np.float32)

    # Stretch to [0, 1]
    rgb = np.clip(rgb, 0, None)
    for ch in range(3):
        band = rgb[:, :, ch]
        valid = band[band > 0]
        if len(valid) > 0:
            p2, p98 = np.percentile(valid, [2, 98])
            if p98 > p2:
                rgb[:, :, ch] = np.clip((band - p2) / (p98 - p2), 0, 1)

    # Plot
    fig, ax = plt.subplots(
        figsize=(12, 12 * rgb.shape[0] / rgb.shape[1])
    )
    ax.imshow(rgb)

    mask_rgba = np.zeros((*plume_mask_dilated.shape, 4))
    mask_rgba[plume_mask_dilated] = [1, 0, 0, 0.5]
    ax.imshow(mask_rgba)
    ax.contour(plume_mask_dilated, levels=[0.5], colors="red", linewidths=1.5)

    H, W = plume_mask_dilated.shape
    n_rows = H // patch_h
    n_cols = W // patch_w
    for r in range(n_rows + 1):
        ax.axhline(r * patch_h, color="yellow", linewidth=0.5, alpha=0.6)
    for c in range(n_cols + 1):
        ax.axvline(c * patch_w, color="yellow", linewidth=0.5, alpha=0.6)

    y0 = reading_window.row_off
    x0 = reading_window.col_off
    selected = mpatches.Rectangle(
        (x0, y0), patch_w, patch_h,
        linewidth=2.5, edgecolor="lime", facecolor="lime", alpha=0.2,
    )
    ax.add_patch(selected)

    ax.set_xlabel("Cross-track (px)")
    ax.set_ylabel("Down-track (px)")
    plt.tight_layout()
    plt.show()


# ------------------------------------------------------------------
# 2. Raw bank plume maps
# ------------------------------------------------------------------

def plot_bank_plumes(
    plume_metas: list[dict],
    wind_speeds: list[float],
    wind_dirs: list[float],
    raas: list[float],
    plume_raws: list[np.ndarray] | None = None,
    colors: list[str] | None = None,
) -> None:
    """Plot raw enhancement maps from the plume bank.

    Parameters
    ----------
    plume_metas : list[dict]
        Bank metadata dicts.
    wind_speeds, wind_dirs, raas :
        Per-plume wind parameters.
    plume_raws : list of (H_bank, W_bank), optional
        Pre-loaded bank arrays.  If None, reads from ``gdal_vsi``.
    colors :
        Colour per plume.  Defaults to ``PLUME_COLORS``.
    """
    if colors is None:
        colors = PLUME_COLORS

    n_plumes = len(plume_metas)
    ncols = min(n_plumes, 3)
    nrows = int(np.ceil(n_plumes / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(6 * ncols, 5 * nrows), squeeze=False,
    )

    for i, meta in enumerate(plume_metas):
        if plume_raws is not None:
            plume_ppb = plume_raws[i]
        else:
            with rasterio.open(meta["gdal_vsi"]) as src:
                plume_ppb = src.read(1).astype(np.float32)

        ax = axes[i // ncols, i % ncols]
        im = ax.imshow(plume_ppb, cmap="plasma")
        fig.colorbar(im, ax=ax, label="ppb", shrink=0.8)
        ax.set_title(f"Plume {i + 1}", fontweight="bold")
        for spine in ax.spines.values():
            spine.set_edgecolor(colors[i % len(colors)])
            spine.set_linewidth(4)

    for j in range(n_plumes, nrows * ncols):
        axes[j // ncols, j % ncols].set_visible(False)

    plt.tight_layout()
    plt.show()


# ------------------------------------------------------------------
# 3. Enhancement summary (wind + combined + masks)
# ------------------------------------------------------------------

def plot_enhancement_summary(
    enhancement_total: np.ndarray,
    enhancements_ref: list[np.ndarray],
    q_targets: np.ndarray,
    wind_speeds: list[float],
    wind_dirs: list[float],
    emit_lat: np.ndarray,
    emit_lon: np.ndarray,
    patch_h: int,
    patch_w: int,
    use_wind_field: bool = False,
    wind_u: np.ndarray | None = None,
    wind_v: np.ndarray | None = None,
    fixed_wind_speed: float | None = None,
    fixed_wind_dir: float | None = None,
    mask_threshold: float = 20.0,
    colors: list[str] | None = None,
) -> None:
    """Three-panel figure: wind field, combined enhancement, coloured masks.

    Parameters
    ----------
    enhancement_total : (H, W)
        Combined enhancement [ppb].
    enhancements_ref : list of (H, W)
        Per-plume reference enhancements (for masks).
    q_targets : (n_plumes,)
        Emission rates [kg/h].
    wind_speeds, wind_dirs :
        Per-plume wind parameters.
    emit_lat, emit_lon : (H, W)
        Patch WGS-84 coordinates.
    patch_h, patch_w :
        Patch dimensions.
    use_wind_field :
        Whether per-pixel wind was used.
    wind_u, wind_v : (H, W), optional
        U/V wind components (when ``use_wind_field=True``).
    fixed_wind_speed, fixed_wind_dir :
        Scalar wind values (when ``use_wind_field=False``).
    mask_threshold :
        Enhancement threshold [ppb] for plume masks.
    colors :
        Colour per plume.
    """
    from matplotlib.colors import LinearSegmentedColormap

    if colors is None:
        colors = PLUME_COLORS

    n_plumes = len(enhancements_ref)
    plasma = plt.get_cmap("plasma")
    yellow_red = LinearSegmentedColormap.from_list(
        "yellow_red", plasma(np.linspace(0.5, 0.95, 256)),
    )

    fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(20, 5))

    # --- Panel 1: Wind field ---
    ax0.imshow(np.zeros((patch_h, patch_w)), cmap="gray", vmin=0, vmax=1)

    if use_wind_field and wind_u is not None and wind_v is not None:
        wu, wv = wind_u.copy(), wind_v.copy()
    else:
        wd_rad = np.radians(
            fixed_wind_dir if fixed_wind_dir is not None else wind_dirs[0]
        )
        ws = fixed_wind_speed if fixed_wind_speed is not None else wind_speeds[0]
        wu = np.full((patch_h, patch_w), ws * np.cos(wd_rad))
        wv = np.full((patch_h, patch_w), ws * np.sin(wd_rad))

    wspeed_grid = np.hypot(wu, wv)

    lat_c = float(emit_lat[patch_h // 2, patch_w // 2])
    lon_c = float(emit_lon[patch_h // 2, patch_w // 2])
    transformer, _ = _get_utm_transformer(lat_c, lon_c)
    east, north, _ = _project_grid(emit_lat, emit_lon, transformer, lon_c, lat_c)

    deast_drow = np.gradient(east, axis=0)
    deast_dcol = np.gradient(east, axis=1)
    dnorth_drow = np.gradient(north, axis=0)
    dnorth_dcol = np.gradient(north, axis=1)

    det = deast_dcol * dnorth_drow - deast_drow * dnorth_dcol
    arrow_col = (dnorth_drow * wu - deast_drow * wv) / det
    arrow_row = (-dnorth_dcol * wu + deast_dcol * wv) / det

    stride = max(max(patch_h, patch_w) // 12, 4)
    rows_q = np.arange(stride // 2, patch_h, stride)
    cols_q = np.arange(stride // 2, patch_w, stride)
    cc, rr = np.meshgrid(cols_q, rows_q)

    U = arrow_col[rr, cc]
    V = arrow_row[rr, cc]
    speed = wspeed_grid[rr, cc]

    magnitude = np.hypot(U, V)
    max_mag = magnitude.max()
    if max_mag > 0:
        scale_factor = stride * 0.55 / max_mag
        U = U * scale_factor
        V = V * scale_factor

    q = ax0.quiver(
        cc, rr, U, V,
        speed.ravel(),
        angles="xy", scale_units="xy", scale=1,
        cmap=yellow_red, clim=(speed.min(), speed.max()),
        linewidth=0.4,
        headwidth=4, headlength=5, headaxislength=4,
        zorder=5,
    )
    fig.colorbar(
        q, ax=ax0, label=r"Wind speed ($\mathrm{m \, s^{-1}}$)", shrink=0.8,
    )
    ax0.set_xlim(-0.5, patch_w - 0.5)
    ax0.set_ylim(patch_h - 0.5, -0.5)
    ax0.set_xlabel("Cross-track (px)")
    ax0.set_ylabel("Down-track (px)")
    ax0.set_title("Wind field")

    # --- Panel 2: Combined enhancement ---
    im = ax1.imshow(enhancement_total, cmap="plasma")
    ax1.set_title("Combined enhancement")
    ax1.set_xlabel("Cross-track (px)")
    ax1.set_ylabel("Down-track (px)")
    fig.colorbar(im, ax=ax1, label="ppb", shrink=0.8)

    # --- Panel 3: Coloured masks ---
    ax2.imshow(np.zeros((patch_h, patch_w)), cmap="gray", vmin=0, vmax=1)
    for i in range(n_plumes):
        mask_i = enhancements_ref[i] >= mask_threshold
        rgba = np.zeros((patch_h, patch_w, 4))
        r, g, b = mcolors.to_rgb(colors[i % len(colors)])
        rgba[mask_i] = [r, g, b, 1.0]
        ax2.imshow(rgba, interpolation="nearest")
        ax2.plot(
            [], [], "s", color=colors[i % len(colors)], markersize=10,
            label=f"Plume {i + 1} -- {q_targets[i]:.0f} kg/h",
        )

    ax2.legend(
        loc="center left", bbox_to_anchor=(1.02, 0.5),
        fontsize=8, framealpha=0.9,
    )
    ax2.set_title("Plume masks")
    ax2.set_xlabel("Cross-track (px)")
    ax2.set_ylabel("Down-track (px)")

    plt.tight_layout(rect=[0, 0, 0.90, 1])
    plt.show()


# ------------------------------------------------------------------
# 4. Spectral comparison at peak pixel
# ------------------------------------------------------------------

def plot_spectral_comparison(
    radiance_original: np.ndarray,
    radiance_modified: np.ndarray,
    enhancement: np.ndarray,
    emit_wvl: np.ndarray,
    band_indices: np.ndarray | None = None,
) -> None:
    """Compare original vs. modified spectra at the peak enhancement pixel.

    Parameters
    ----------
    radiance_original, radiance_modified : (n_bands, H, W)
    enhancement : (H, W)
    emit_wvl : (n_bands,)
    band_indices : optional
        CH4 band indices.  If None, derived from config.
    """
    from inject import get_methane_band_indices

    if band_indices is None:
        band_indices = get_methane_band_indices(emit_wvl)

    active_mask = enhancement > 0
    peak_idx = np.argmax(enhancement[active_mask])
    rows, cols = np.where(active_mask)
    r, c = rows[peak_idx], cols[peak_idx]

    swir_mask = (emit_wvl[band_indices] >= 2100) & (emit_wvl[band_indices] <= 2500)
    swir_bands = band_indices[swir_mask]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))

    ax1.plot(
        emit_wvl[band_indices], radiance_original[band_indices, r, c],
        label="Original", color="steelblue",
    )
    ax1.plot(
        emit_wvl[band_indices], radiance_modified[band_indices, r, c],
        label="With plume", color="tomato", linestyle="--",
    )
    ax1.set_xlabel("Wavelength (nm)")
    ax1.set_ylabel(
        r"Radiance ($\mathrm{\mu W \, cm^{-2} \, sr^{-1} \, nm^{-1}}$)"
    )
    ax1.set_title("Full CH4 window")
    ax1.legend()

    ax2.plot(
        emit_wvl[swir_bands], radiance_original[swir_bands, r, c],
        label="Original", color="steelblue",
    )
    ax2.plot(
        emit_wvl[swir_bands], radiance_modified[swir_bands, r, c],
        label="With plume", color="tomato", linestyle="--",
    )
    ax2.set_xlabel("Wavelength (nm)")
    ax2.set_title("SWIR zoom (2100-2500 nm)")
    ax2.legend()

    fig.suptitle(
        f"Peak enhancement pixel: {enhancement[r, c]:.0f} ppb",
        fontweight="bold",
    )
    plt.tight_layout()
    plt.show()


# ------------------------------------------------------------------
# 5. Retrieval comparison (before / after)
# ------------------------------------------------------------------

def plot_retrieval_comparison(
    enhancement_total: np.ndarray,
    enhancements_ref: list[np.ndarray],
    original_retrievals: dict[str, np.ndarray],
    modified_retrievals: dict[str, np.ndarray],
    patch_h: int,
    patch_w: int,
    mask_threshold: float = 20.0,
    colors: list[str] | None = None,
) -> None:
    """Grid of retrieval maps: ground truth + before/after for each method.

    Parameters
    ----------
    enhancement_total : (H, W)
    enhancements_ref : list of (H, W)
    original_retrievals : dict  name -> (H, W) array (pre-injection)
    modified_retrievals : dict  name -> (H, W) array (post-injection)
    patch_h, patch_w : int
    mask_threshold : float
    colors : list[str]
    """
    if colors is None:
        colors = PLUME_COLORS

    n_plumes = len(enhancements_ref)
    methods = list(modified_retrievals.keys())
    n_methods = len(methods)

    # Shared colour scale
    all_vals = []
    for name in methods:
        if name in modified_retrievals:
            all_vals.append(modified_retrievals[name].ravel())
        if name in original_retrievals:
            all_vals.append(original_retrievals[name].ravel())
    vmin = float(np.nanpercentile(np.concatenate(all_vals), 1))
    vmax = float(np.nanpercentile(np.concatenate(all_vals), 99))

    fig = plt.figure(figsize=(5 * (1 + n_methods), 10))
    gs = gridspec.GridSpec(
        2, 1 + n_methods, figure=fig, wspace=0.35, hspace=0.25,
    )

    # Enhancement (top-left)
    ax_enh = fig.add_subplot(gs[0, 0])
    im_enh = ax_enh.imshow(enhancement_total, cmap="plasma")
    ax_enh.set_title("Enhancement (ground truth)")
    ax_enh.axis("off")
    fig.colorbar(im_enh, ax=ax_enh, label="ppb", shrink=0.8)

    # Mask (bottom-left)
    ax_mask = fig.add_subplot(gs[1, 0])
    ax_mask.imshow(np.zeros((patch_h, patch_w)), cmap="gray", vmin=0, vmax=1)
    for i in range(n_plumes):
        mask_i = enhancements_ref[i] >= mask_threshold
        rgba = np.zeros((patch_h, patch_w, 4))
        r, g, b = mcolors.to_rgb(colors[i % len(colors)])
        rgba[mask_i] = [r, g, b, 1.0]
        ax_mask.imshow(rgba, interpolation="nearest")
    ax_mask.set_title("Plume masks")
    ax_mask.axis("off")
    cbar = fig.colorbar(im_enh, ax=ax_mask, shrink=0.8)
    cbar.ax.set_visible(False)

    # Before (top row)
    for i, name in enumerate(methods):
        ax = fig.add_subplot(gs[0, 1 + i])
        if name in original_retrievals:
            im = ax.imshow(
                original_retrievals[name], cmap="plasma", vmin=vmin, vmax=vmax,
            )
            fig.colorbar(im, ax=ax, label=r"ppm$\cdot$m", shrink=0.8)
        else:
            ax.text(
                0.5, 0.5, "not available", transform=ax.transAxes,
                ha="center", va="center", fontsize=12, color="gray",
            )
        ax.set_title(f"{name.upper()} -- before")
        ax.axis("off")

    # After (bottom row)
    for i, name in enumerate(methods):
        ax = fig.add_subplot(gs[1, 1 + i])
        im = ax.imshow(
            modified_retrievals[name], cmap="plasma", vmin=vmin, vmax=vmax,
        )
        ax.set_title(f"{name.upper()} -- after")
        ax.axis("off")
        fig.colorbar(im, ax=ax, label=r"ppm$\cdot$m", shrink=0.8)

    plt.show()
