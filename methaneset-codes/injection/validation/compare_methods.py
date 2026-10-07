#!/usr/bin/env python
"""
compare_methods.py

Head-to-head comparison of two reprojection strategies for mapping the
methane enhancement bank onto the EMIT sensor grid:

    Option 1 — "prefilter"
        Pre-filter the bank with a uniform (box) kernel matching the EMIT
        pixel size (~3 px), then point-sample one value per EMIT pixel.

    Option 2 — "footprint"  (with variable n_sub)
        For each EMIT pixel, project its four corners into bank
        coordinates, sample n_sub × n_sub points inside the resulting
        quadrilateral, and average.

Both options share the same affine-based coordinate transformation.

Metrics
-------
    η        = IME_EMIT / IME_bank        (mass conservation: ideal = 1.0)
    dilution = peak_EMIT / peak_bank      (peak preservation: ideal = 1.0)
    pearson_r = Pearson r vs reference     (spatial fidelity: ideal = 1.0)
    ssim      = Structural Similarity vs ref (spatial fidelity: ideal = 1.0)
    time_s   = wall-clock time per map_to_emit call

    Reference for spatial metrics: footprint with n_sub = N_SUB_REF (16).

Sweep
-----
    - methods   : prefilter, footprint with n_sub ∈ {1, 2, 3, 4, 6, 8}
    - scenes    : N_SCENES random EMIT scenes
    - wind_speed: all values in the bank
    - wind_dir  : 0°–330° every WIND_DIR_STEP°

For each (scene, wind_speed) pair, the SAME plume is used for every
method and wind_dir, so differences are purely methodological.
"""

import sys
from pathlib import Path

PIPELINE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PIPELINE_DIR))

import time
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import rasterio
import tacoreader
import matplotlib.pyplot as plt
import warnings
from rasterio.errors import NotGeoreferencedWarning
from rasterio.windows import Window
from scipy.ndimage import map_coordinates, uniform_filter

from config import (
    DIR_EMIT_SCENES,
    DIR_PLUME_BANK,
    ENHANCEMENT_PIXEL_RES,
)
from bank import MethaneBank

# Import Option 2 and shared helpers from trans_enhmap
from plume_injection_test.injection.trans_enhmap_prefilter import (
    map_to_emit as map_to_emit_footprint,
    compute_pixel_areas,
    _get_utm_transformer,
    _project_grid,
    _inpaint_nan,
    _build_bank_affine,
)

warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)

# ── Configuration ────────────────────────────────────────────────────
PATCH_H = 200
PATCH_W = 200
N_SCENES = 10
WIND_DIR_STEP = 30       # degrees (12 directions for tractability)
SOURCE_TYPE = "multi"
OUTPUT_DIR = Path("method_comparison")
N_WORKERS = 10
SEED = 42
N_SUB_REF = 16           # reference n_sub for spatial fidelity metrics

# Methods to compare: name → kwargs passed to the dispatch function.
METHODS = OrderedDict([
    ("prefilter",    {"type": "prefilter", "emit_gsd": 60.0}),
    ("footprint_n1", {"type": "footprint", "n_sub": 1}),
    ("footprint_n2", {"type": "footprint", "n_sub": 2}),
    ("footprint_n3", {"type": "footprint", "n_sub": 3}),
    ("footprint_n4", {"type": "footprint", "n_sub": 4}),
    ("footprint_n6", {"type": "footprint", "n_sub": 6}),
    ("footprint_n8", {"type": "footprint", "n_sub": 8}),
])


# ── Spatial fidelity metrics ────────────────────────────────────────

def _spatial_pearson(test: np.ndarray, ref: np.ndarray) -> float:
    """Pearson correlation between two enhancement maps.

    Only pixels where *ref* > 0 are considered, so background zeros
    (which dominate the map) don't inflate the correlation.

    Returns NaN if fewer than 2 valid points.
    """
    mask = ref > 0
    if mask.sum() < 2:
        return float("nan")
    a = test[mask].ravel()
    b = ref[mask].ravel()
    # np.corrcoef returns NaN when std is 0 (constant array)
    r = np.corrcoef(a, b)[0, 1]
    return float(r)


def _spatial_ssim(test: np.ndarray, ref: np.ndarray) -> float:
    """Structural Similarity Index between two enhancement maps.

    Uses the standard SSIM formula with an 11×11 uniform window and
    constants derived from the dynamic range of the reference map.
    Implemented directly to avoid a hard dependency on scikit-image.

    Both maps are normalised to [0, 1] by the maximum of *ref* so that
    the constants (C1, C2) are scale-independent.
    """
    vmax = float(ref.max())
    if vmax == 0:
        return float("nan")

    # Normalise to [0, 1]
    a = test.astype(np.float64) / vmax
    b = ref.astype(np.float64) / vmax

    # Constants (Wang et al., 2004) for data_range = 1
    C1 = (0.01) ** 2
    C2 = (0.03) ** 2
    win = 11

    mu_a = uniform_filter(a, size=win, mode="constant")
    mu_b = uniform_filter(b, size=win, mode="constant")
    mu_a_sq = mu_a * mu_a
    mu_b_sq = mu_b * mu_b
    mu_ab = mu_a * mu_b

    # np.maximum(0, ...) guards against floating-point noise where
    # E[X²] − E[X]² goes slightly negative for near-constant regions.
    sigma_a_sq = np.maximum(0.0, uniform_filter(a * a, size=win, mode="constant") - mu_a_sq)
    sigma_b_sq = np.maximum(0.0, uniform_filter(b * b, size=win, mode="constant") - mu_b_sq)
    sigma_ab = uniform_filter(a * b, size=win, mode="constant") - mu_ab

    num = (2.0 * mu_ab + C1) * (2.0 * sigma_ab + C2)
    den = (mu_a_sq + mu_b_sq + C1) * (sigma_a_sq + sigma_b_sq + C2)

    ssim_map = num / den

    # Average only over the plume region (ref > 0) so background
    # doesn't inflate the score.
    mask = ref > 0
    if mask.sum() == 0:
        return float("nan")
    return float(ssim_map[mask].mean())


# ── Option 1 implementation ─────────────────────────────────────────
# Self-contained here; reuses shared helpers from trans_enhmap.

def map_to_emit_prefilter(
    enhancement, source_row, source_col,
    emit_lat, emit_lon, emit_source_row, emit_source_col,
    pixel_res=ENHANCEMENT_PIXEL_RES, wind_dir=0.0, emit_gsd=60.0,
    _prefiltered=None,
):
    """Option 1: pre-filter + single-point cubic sampling.

    If *_prefiltered* is provided it is used directly, skipping the
    (wind-dir-independent) inpainting and uniform-filter steps.
    """
    H, W = emit_lat.shape

    if _prefiltered is not None:
        enh = _prefiltered
    else:
        enh = _inpaint_nan(enhancement)
        kernel = max(3, int(np.round(emit_gsd / pixel_res)))
        if kernel % 2 == 0:
            kernel += 1
        enh = uniform_filter(enh, size=kernel, mode="constant", cval=0.0)

    # Affine: bank → UTM, inverse: UTM → bank
    src_lat = float(emit_lat[emit_source_row, emit_source_col])
    src_lon = float(emit_lon[emit_source_row, emit_source_col])
    transformer, _ = _get_utm_transformer(src_lat, src_lon)
    src_east, src_north = transformer.transform(src_lon, src_lat)

    fwd = _build_bank_affine(
        src_east, src_north, source_col, source_row, pixel_res, wind_dir,
    )
    inv = ~fwd

    # Project EMIT centres to bank pixel coords
    east, north, valid = _project_grid(
        emit_lat, emit_lon, transformer, src_lon, src_lat,
    )
    col_coords = np.where(valid, inv.a * east + inv.b * north + inv.c, 0.0)
    row_coords = np.where(valid, inv.d * east + inv.e * north + inv.f, 0.0)

    # Sample
    coords = np.array([row_coords.ravel(), col_coords.ravel()])
    result = map_coordinates(
        enh, coords, order=3, mode="constant", cval=0.0,
    ).reshape(H, W).astype(np.float32)

    result[~valid] = 0.0
    np.clip(result, 0.0, None, out=result)
    return result


# ── Dispatch ─────────────────────────────────────────────────────────

def _run_method(method_cfg, plume_ppb, source_row, source_col,
                lat, lon, patch_h, patch_w, pixel_res, wind_dir,
                enh_inpainted=None, enh_prefiltered=None):
    """Call the appropriate map_to_emit variant and return (result, dt).

    *enh_inpainted* and *enh_prefiltered* are optional pre-computed
    arrays that avoid repeating the (wind-dir-independent) inpainting
    and PSF convolution at every call.
    """
    t0 = time.perf_counter()

    if method_cfg["type"] == "prefilter":
        result = map_to_emit_prefilter(
            plume_ppb, source_row, source_col,
            lat, lon, patch_h // 2, patch_w // 2,
            pixel_res=pixel_res, wind_dir=wind_dir,
            emit_gsd=method_cfg["emit_gsd"],
            _prefiltered=enh_prefiltered,
        )
    else:
        # Pass the inpainted array so _inpaint_nan inside
        # map_to_emit_footprint is a near-no-op (one copy + check).
        src = enh_inpainted if enh_inpainted is not None else plume_ppb
        result = map_to_emit_footprint(
            src, source_row, source_col,
            lat, lon, patch_h // 2, patch_w // 2,
            pixel_res=pixel_res, wind_dir=wind_dir,
            n_sub=method_cfg["n_sub"],
        )

    dt = time.perf_counter() - t0
    return result, dt


# ── Worker (module-level for pickling) ───────────────────────────────

def _evaluate_task(args):
    """Process one (scene, wind_speed): run every method at every wind_dir.

    Returns a list of result dicts, one per (method, wind_dir).

    For spatial fidelity, a high-resolution reference (footprint with
    n_sub = N_SUB_REF) is computed once per wind_dir and all methods
    are compared against it.
    """
    (plume_ppb, source_row, source_col,
     lat, lon, pixel_area,
     wind_dirs, patch_h, patch_w,
     methods_config) = args

    # Use nansum/nanmax to handle possible NaN pixels in the bank map
    ime_bank = float(np.nansum(plume_ppb)) * ENHANCEMENT_PIXEL_RES ** 2
    peak_bank = float(np.nanmax(plume_ppb))

    if ime_bank == 0 or peak_bank == 0:
        return [{
            "method": name, "wind_dir": wd,
            "eta": float("nan"), "peak_bank": peak_bank,
            "peak_emit": float("nan"), "dilution": float("nan"),
            "pearson_r": float("nan"), "ssim": float("nan"),
            "time_s": 0.0,
        } for name in methods_config for wd in wind_dirs]

    # ------------------------------------------------------------------
    # Pre-compute wind-dir-independent steps ONCE per task:
    #   1) Inpainting  (shared by all methods, including the reference)
    #   2) Uniform filter (prefilter method only)
    # ------------------------------------------------------------------
    enh_inpainted = _inpaint_nan(plume_ppb)

    enh_prefiltered = None
    prefilter_cfg = methods_config.get("prefilter")
    if prefilter_cfg is not None:
        emit_gsd = prefilter_cfg.get("emit_gsd", 60.0)
        kernel = max(3, int(np.round(emit_gsd / ENHANCEMENT_PIXEL_RES)))
        if kernel % 2 == 0:
            kernel += 1
        enh_prefiltered = uniform_filter(
            enh_inpainted, size=kernel, mode="constant", cval=0.0,
        )

    # Reference method config (not timed — it's the ground truth)
    ref_cfg = {"type": "footprint", "n_sub": N_SUB_REF}

    results = []

    # Iterate wind_dir FIRST so we compute the reference once per direction
    for wd in wind_dirs:
        # Compute reference map for this wind_dir
        ref_map, _ = _run_method(
            ref_cfg, plume_ppb, source_row, source_col,
            lat, lon, patch_h, patch_w,
            ENHANCEMENT_PIXEL_RES, float(wd),
            enh_inpainted=enh_inpainted,
        )

        for method_name, method_cfg in methods_config.items():
            enh, dt = _run_method(
                method_cfg, plume_ppb, source_row, source_col,
                lat, lon, patch_h, patch_w,
                ENHANCEMENT_PIXEL_RES, float(wd),
                enh_inpainted=enh_inpainted,
                enh_prefiltered=enh_prefiltered,
            )
            ime_emit = float(np.nansum(enh * pixel_area))
            peak_emit = float(np.nanmax(enh))

            results.append({
                "method": method_name,
                "wind_dir": wd,
                "eta": ime_emit / ime_bank,
                "peak_bank": peak_bank,
                "peak_emit": peak_emit,
                "dilution": peak_emit / peak_bank,
                "pearson_r": _spatial_pearson(enh, ref_map),
                "ssim": _spatial_ssim(enh, ref_map),
                "time_s": dt,
            })

    return results


# ── Scene loading ────────────────────────────────────────────────────

def load_valid_scenes(data_dir, patch_h, patch_w, n_max):
    """Load up to *n_max* scenes whose centre patch has valid lat/lon."""
    tacoreader.use("pandas")
    df = tacoreader.load(str(data_dir)).data

    scenes = []
    indices = np.arange(len(df))
    np.random.shuffle(indices)

    for idx in indices:
        if len(scenes) >= n_max:
            break
        row = df.iloc[idx]
        scene_dir = Path(row["internal:gdal_vsi"]).parent

        try:
            with rasterio.open(scene_dir / "latlon.tif") as src:
                h, w = src.height, src.width
                if h < patch_h or w < patch_w:
                    continue

                y0 = (h - patch_h) // 2
                x0 = (w - patch_w) // 2
                window = Window(col_off=x0, row_off=y0,
                                width=patch_w, height=patch_h)
                lat = src.read(1, window=window)
                lon = src.read(2, window=window)

            if not (np.isfinite(lat).all() and np.isfinite(lon).all()):
                continue

            pixel_area = compute_pixel_areas(lat, lon)
            scenes.append({
                "id": row["id"], "sza": row["target:sza"],
                "lat": lat, "lon": lon, "pixel_area": pixel_area,
                "mean_lat": float(np.mean(lat)),
            })
            print(f"  [{len(scenes):>2}/{n_max}] {row['id']}  "
                  f"lat={float(np.mean(lat)):+.1f}")
        except Exception:
            continue

    return scenes


# ── Figures ──────────────────────────────────────────────────────────

def generate_figures(df, output_dir):
    """Generate all comparison plots."""
    plt.rcParams.update({
        "font.family": "sans-serif", "font.size": 11,
        "axes.labelsize": 12, "axes.titlesize": 13, "figure.dpi": 150,
    })

    method_names = list(METHODS.keys())
    n_sub_values = [1, 2, 3, 4, 6, 8]

    # Colours: prefilter in red, footprint in blue gradient
    colors = {"prefilter": "#D64541"}
    blues = plt.cm.Blues(np.linspace(0.3, 0.9, len(n_sub_values)))
    for i, n in enumerate(n_sub_values):
        colors[f"footprint_n{n}"] = blues[i]

    # Shared comparison config for multi-panel figures
    compare_methods = ["prefilter", "footprint_n4"]
    cmap_compare = {"prefilter": "#D64541", "footprint_n4": "#4C72B0"}

    # ── Fig 1: η boxplot per method ──────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    data = [df.loc[df["method"] == m, "eta"].dropna().values
            for m in method_names]
    bp = ax.boxplot(data, labels=method_names, patch_artist=True,
                    showfliers=False,
                    medianprops=dict(color="black", linewidth=1.2))
    for patch, name in zip(bp["boxes"], method_names):
        patch.set_facecolor(colors[name])
        patch.set_alpha(0.8)
    ax.axhline(1.0, color="gray", linestyle=":", alpha=0.7)
    ax.set_ylabel("Conservation ratio η")
    ax.set_title("Mass conservation by method")
    ax.tick_params(axis="x", rotation=30)
    plt.tight_layout()
    plt.savefig(output_dir / "fig1_eta_by_method.png",
                dpi=300, bbox_inches="tight")
    plt.close()

    # ── Fig 2: Dilution boxplot per method ───────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    data = [df.loc[df["method"] == m, "dilution"].dropna().values
            for m in method_names]
    bp = ax.boxplot(data, labels=method_names, patch_artist=True,
                    showfliers=False,
                    medianprops=dict(color="black", linewidth=1.2))
    for patch, name in zip(bp["boxes"], method_names):
        patch.set_facecolor(colors[name])
        patch.set_alpha(0.8)
    ax.set_ylabel("Peak dilution (peak_emit / peak_bank)")
    ax.set_title("Peak preservation by method")
    ax.tick_params(axis="x", rotation=30)
    plt.tight_layout()
    plt.savefig(output_dir / "fig2_dilution_by_method.png",
                dpi=300, bbox_inches="tight")
    plt.close()

    # ── Fig 3: η convergence vs n_sub ────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    fp_stats = []
    for n in n_sub_values:
        sub = df[df["method"] == f"footprint_n{n}"]["eta"].dropna()
        fp_stats.append((n, sub.mean(), sub.std()))
    fp_stats = np.array(fp_stats)

    ax.errorbar(fp_stats[:, 0], fp_stats[:, 1], yerr=fp_stats[:, 2],
                fmt="o-", color="#4C72B0", capsize=4, markersize=6,
                label="Footprint integration")

    pf = df[df["method"] == "prefilter"]["eta"].dropna()
    ax.axhline(pf.mean(), color="#D64541", linestyle="--",
               linewidth=1.5, label=f"Prefilter (η = {pf.mean():.4f})")
    ax.axhspan(pf.mean() - pf.std(), pf.mean() + pf.std(),
               color="#D64541", alpha=0.1)

    ax.axhline(1.0, color="gray", linestyle=":", alpha=0.5)
    ax.set_xlabel("n_sub (sub-samples per axis)")
    ax.set_ylabel("Mean η ± std")
    ax.set_title("Conservation convergence with n_sub")
    ax.set_xticks(n_sub_values)
    ax.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(output_dir / "fig3_eta_convergence.png",
                dpi=300, bbox_inches="tight")
    plt.close()

    # ── Fig 4: Dilution convergence vs n_sub ─────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    fp_stats_d = []
    for n in n_sub_values:
        sub = df[df["method"] == f"footprint_n{n}"]["dilution"].dropna()
        fp_stats_d.append((n, sub.mean(), sub.std()))
    fp_stats_d = np.array(fp_stats_d)

    ax.errorbar(fp_stats_d[:, 0], fp_stats_d[:, 1], yerr=fp_stats_d[:, 2],
                fmt="o-", color="#4C72B0", capsize=4, markersize=6,
                label="Footprint integration")

    pf_d = df[df["method"] == "prefilter"]["dilution"].dropna()
    ax.axhline(pf_d.mean(), color="#D64541", linestyle="--",
               linewidth=1.5,
               label=f"Prefilter (d = {pf_d.mean():.4f})")
    ax.axhspan(pf_d.mean() - pf_d.std(), pf_d.mean() + pf_d.std(),
               color="#D64541", alpha=0.1)

    ax.set_xlabel("n_sub (sub-samples per axis)")
    ax.set_ylabel("Mean dilution ± std")
    ax.set_title("Peak dilution convergence with n_sub")
    ax.set_xticks(n_sub_values)
    ax.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(output_dir / "fig4_dilution_convergence.png",
                dpi=300, bbox_inches="tight")
    plt.close()

    # ── Fig 5: η vs wind_dir — prefilter vs footprint_n4 (polar) ────
    fig, ax = plt.subplots(figsize=(6, 6),
                           subplot_kw={"projection": "polar"})
    for m in compare_methods:
        sub = df[df["method"] == m].dropna(subset=["eta"])
        wd_unique = sorted(sub["wind_dir"].unique())
        means = np.array([sub.loc[sub["wind_dir"] == wd, "eta"].mean()
                          for wd in wd_unique])
        theta = np.radians(np.append(wd_unique, wd_unique[0]))
        means_c = np.append(means, means[0])
        ax.plot(theta, means_c, "o-", color=cmap_compare[m], markersize=3,
                linewidth=1.2, label=m)

    ax.set_theta_zero_location("E")
    ax.set_theta_direction(1)
    ax.set_title("η vs. wind direction", pad=20)
    ax.legend(loc="lower right", bbox_to_anchor=(1.3, 0.0), frameon=True)
    plt.tight_layout()
    plt.savefig(output_dir / "fig5_eta_vs_winddir.png",
                dpi=300, bbox_inches="tight")
    plt.close()

    # ── Fig 6: Computation time per method ───────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    time_means = [df.loc[df["method"] == m, "time_s"].mean()
                  for m in method_names]
    time_stds = [df.loc[df["method"] == m, "time_s"].std()
                 for m in method_names]
    bars = ax.bar(method_names, time_means, yerr=time_stds,
                  capsize=3, color=[colors[m] for m in method_names],
                  alpha=0.8, edgecolor="black", linewidth=0.5)
    ax.set_ylabel("Time per call (s)")
    ax.set_title("Computation time by method")
    ax.tick_params(axis="x", rotation=30)

    for bar, t in zip(bars, time_means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{t:.3f}s", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.savefig(output_dir / "fig6_time_by_method.png",
                dpi=300, bbox_inches="tight")
    plt.close()

    # ── Fig 7: η and dilution vs wind_speed ──────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    for m in compare_methods:
        sub = df[df["method"] == m].dropna(subset=["eta"])
        grouped = sub.groupby("wind_speed")["eta"]
        ws_vals = sorted(sub["wind_speed"].unique())
        means = [grouped.get_group(ws).mean() for ws in ws_vals]
        stds = [grouped.get_group(ws).std() for ws in ws_vals]
        ax1.errorbar(ws_vals, means, yerr=stds, fmt="o-",
                     color=cmap_compare[m], capsize=3, markersize=5,
                     label=m, alpha=0.8)
    ax1.axhline(1.0, color="gray", linestyle=":", alpha=0.5)
    ax1.set_xlabel("Wind speed (m/s)")
    ax1.set_ylabel("η")
    ax1.set_title("Mass conservation vs. wind speed")
    ax1.legend(frameon=True)

    for m in compare_methods:
        sub = df[df["method"] == m].dropna(subset=["dilution"])
        grouped = sub.groupby("wind_speed")["dilution"]
        ws_vals = sorted(sub["wind_speed"].unique())
        means = [grouped.get_group(ws).mean() for ws in ws_vals]
        stds = [grouped.get_group(ws).std() for ws in ws_vals]
        ax2.errorbar(ws_vals, means, yerr=stds, fmt="o-",
                     color=cmap_compare[m], capsize=3, markersize=5,
                     label=m, alpha=0.8)
    ax2.set_xlabel("Wind speed (m/s)")
    ax2.set_ylabel("Peak dilution")
    ax2.set_title("Peak preservation vs. wind speed")
    ax2.legend(frameon=True)

    plt.tight_layout()
    plt.savefig(output_dir / "fig7_metrics_vs_windspeed.png",
                dpi=300, bbox_inches="tight")
    plt.close()

    # ── Fig 8: η and dilution vs scene latitude ──────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    for m in compare_methods:
        sub = df[df["method"] == m].dropna(subset=["eta"])
        grouped = sub.groupby("scene_lat")["eta"]
        lat_vals = sorted(sub["scene_lat"].unique())
        means = [grouped.get_group(lt).mean() for lt in lat_vals]
        stds = [grouped.get_group(lt).std() for lt in lat_vals]
        ax1.errorbar(lat_vals, means, yerr=stds, fmt="o-",
                     color=cmap_compare[m], capsize=3, markersize=5,
                     label=m, alpha=0.8)
    ax1.axhline(1.0, color="gray", linestyle=":", alpha=0.5)
    ax1.set_xlabel("Scene latitude (°)")
    ax1.set_ylabel("η")
    ax1.set_title("Mass conservation vs. latitude")
    ax1.legend(frameon=True)

    for m in compare_methods:
        sub = df[df["method"] == m].dropna(subset=["dilution"])
        grouped = sub.groupby("scene_lat")["dilution"]
        lat_vals = sorted(sub["scene_lat"].unique())
        means = [grouped.get_group(lt).mean() for lt in lat_vals]
        stds = [grouped.get_group(lt).std() for lt in lat_vals]
        ax2.errorbar(lat_vals, means, yerr=stds, fmt="o-",
                     color=cmap_compare[m], capsize=3, markersize=5,
                     label=m, alpha=0.8)
    ax2.set_xlabel("Scene latitude (°)")
    ax2.set_ylabel("Peak dilution")
    ax2.set_title("Peak preservation vs. latitude")
    ax2.legend(frameon=True)

    plt.tight_layout()
    plt.savefig(output_dir / "fig8_metrics_vs_latitude.png",
                dpi=300, bbox_inches="tight")
    plt.close()

    # ── Fig 9: Dilution vs peak_bank (plume intensity) ───────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    for m in compare_methods:
        sub = df[df["method"] == m].dropna(subset=["dilution"])
        ax.scatter(sub["peak_bank"], sub["dilution"],
                   color=cmap_compare[m], alpha=0.15, s=12,
                   label=m, edgecolors="none")
    ax.set_xlabel("Peak enhancement in bank (ppb)")
    ax.set_ylabel("Peak dilution")
    ax.set_title("Peak dilution vs. plume intensity")
    ax.legend(frameon=True, markerscale=3)
    plt.tight_layout()
    plt.savefig(output_dir / "fig9_dilution_vs_peak.png",
                dpi=300, bbox_inches="tight")
    plt.close()

    # ── Fig 10: Pearson r convergence vs n_sub ───────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    fp_stats_r = []
    for n in n_sub_values:
        sub = df[df["method"] == f"footprint_n{n}"]["pearson_r"].dropna()
        fp_stats_r.append((n, sub.mean(), sub.std()))
    fp_stats_r = np.array(fp_stats_r)

    ax.errorbar(fp_stats_r[:, 0], fp_stats_r[:, 1], yerr=fp_stats_r[:, 2],
                fmt="o-", color="#4C72B0", capsize=4, markersize=6,
                label="Footprint integration")

    pf_r = df[df["method"] == "prefilter"]["pearson_r"].dropna()
    ax.axhline(pf_r.mean(), color="#D64541", linestyle="--",
               linewidth=1.5,
               label=f"Prefilter (r = {pf_r.mean():.4f})")
    ax.axhspan(pf_r.mean() - pf_r.std(), pf_r.mean() + pf_r.std(),
               color="#D64541", alpha=0.1)

    ax.axhline(1.0, color="gray", linestyle=":", alpha=0.5)
    ax.set_xlabel("n_sub (sub-samples per axis)")
    ax.set_ylabel("Pearson r vs. reference (n_sub=16)")
    ax.set_title("Spatial correlation convergence with n_sub")
    ax.set_xticks(n_sub_values)
    ax.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(output_dir / "fig10_pearson_convergence.png",
                dpi=300, bbox_inches="tight")
    plt.close()

    # ── Fig 11: SSIM convergence vs n_sub ────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    fp_stats_s = []
    for n in n_sub_values:
        sub = df[df["method"] == f"footprint_n{n}"]["ssim"].dropna()
        fp_stats_s.append((n, sub.mean(), sub.std()))
    fp_stats_s = np.array(fp_stats_s)

    ax.errorbar(fp_stats_s[:, 0], fp_stats_s[:, 1], yerr=fp_stats_s[:, 2],
                fmt="o-", color="#4C72B0", capsize=4, markersize=6,
                label="Footprint integration")

    pf_s = df[df["method"] == "prefilter"]["ssim"].dropna()
    ax.axhline(pf_s.mean(), color="#D64541", linestyle="--",
               linewidth=1.5,
               label=f"Prefilter (SSIM = {pf_s.mean():.4f})")
    ax.axhspan(pf_s.mean() - pf_s.std(), pf_s.mean() + pf_s.std(),
               color="#D64541", alpha=0.1)

    ax.axhline(1.0, color="gray", linestyle=":", alpha=0.5)
    ax.set_xlabel("n_sub (sub-samples per axis)")
    ax.set_ylabel("SSIM vs. reference (n_sub=16)")
    ax.set_title("Structural similarity convergence with n_sub")
    ax.set_xticks(n_sub_values)
    ax.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(output_dir / "fig11_ssim_convergence.png",
                dpi=300, bbox_inches="tight")
    plt.close()

    # ── Fig 12: Spatial metrics boxplot per method ───────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))

    data_r = [df.loc[df["method"] == m, "pearson_r"].dropna().values
              for m in method_names]
    bp1 = ax1.boxplot(data_r, labels=method_names, patch_artist=True,
                      showfliers=False,
                      medianprops=dict(color="black", linewidth=1.2))
    for patch, name in zip(bp1["boxes"], method_names):
        patch.set_facecolor(colors[name])
        patch.set_alpha(0.8)
    ax1.axhline(1.0, color="gray", linestyle=":", alpha=0.7)
    ax1.set_ylabel(f"Pearson r vs. reference (n_sub={N_SUB_REF})")
    ax1.set_title("Spatial correlation by method")
    ax1.tick_params(axis="x", rotation=30)

    data_s = [df.loc[df["method"] == m, "ssim"].dropna().values
              for m in method_names]
    bp2 = ax2.boxplot(data_s, labels=method_names, patch_artist=True,
                      showfliers=False,
                      medianprops=dict(color="black", linewidth=1.2))
    for patch, name in zip(bp2["boxes"], method_names):
        patch.set_facecolor(colors[name])
        patch.set_alpha(0.8)
    ax2.axhline(1.0, color="gray", linestyle=":", alpha=0.7)
    ax2.set_ylabel(f"SSIM vs. reference (n_sub={N_SUB_REF})")
    ax2.set_title("Structural similarity by method")
    ax2.tick_params(axis="x", rotation=30)

    plt.tight_layout()
    plt.savefig(output_dir / "fig12_spatial_metrics_by_method.png",
                dpi=300, bbox_inches="tight")
    plt.close()


# ── Example maps ─────────────────────────────────────────────────────

def generate_example_maps(
    plume_ppb, source_row, source_col,
    lat, lon, patch_h, patch_w,
    wind_dir, output_dir,
):
    """Run selected methods on a single (plume, wind_dir) and plot side-by-side.

    Layout:
        Row 1: bank (raw) | prefilter | footprint_n2 | footprint_n4 | footprint_n8
        Row 2: (empty)    | diff vs n8 for each of the above three methods
    """
    show_methods = [
        ("prefilter",    {"type": "prefilter", "emit_gsd": 60.0}),
        ("footprint_n2", {"type": "footprint", "n_sub": 2}),
        ("footprint_n4", {"type": "footprint", "n_sub": 4}),
        ("footprint_n8", {"type": "footprint", "n_sub": 8}),
    ]

    # Pre-compute cached arrays
    enh_inpainted = _inpaint_nan(plume_ppb)
    kernel = max(3, int(np.round(60.0 / ENHANCEMENT_PIXEL_RES)))
    if kernel % 2 == 0:
        kernel += 1
    enh_prefiltered = uniform_filter(
        enh_inpainted, size=kernel, mode="constant", cval=0.0,
    )

    maps = {}
    for name, cfg in show_methods:
        result, _ = _run_method(
            cfg, plume_ppb, source_row, source_col,
            lat, lon, patch_h, patch_w,
            ENHANCEMENT_PIXEL_RES, wind_dir,
            enh_inpainted=enh_inpainted,
            enh_prefiltered=enh_prefiltered,
        )
        maps[name] = result

    ref = maps["footprint_n8"]
    # Use separate vmax for bank (20 m) and EMIT maps (60 m) so that
    # the projected maps are not washed out by the bank's higher peak.
    vmax_bank = float(np.nanmax(plume_ppb))
    vmax_emit = max(float(np.nanmax(m)) for m in maps.values()) or 1.0

    fig, axes = plt.subplots(2, 5, figsize=(22, 8))

    # Row 1: enhancement maps
    im0 = axes[0, 0].imshow(plume_ppb, cmap="inferno", vmin=0,
                             vmax=vmax_bank)
    axes[0, 0].set_title("Bank (raw, 20 m)", fontsize=11)
    fig.colorbar(im0, ax=axes[0, 0], shrink=0.75, label="ppb")

    for col_idx, (name, _) in enumerate(show_methods):
        ax = axes[0, col_idx + 1]
        im = ax.imshow(maps[name], cmap="inferno", vmin=0, vmax=vmax_emit)
        ax.set_title(name, fontsize=11)
        fig.colorbar(im, ax=ax, shrink=0.75, label="ppb")

    # Row 2: difference vs reference (footprint_n8)
    axes[1, 0].set_visible(False)
    diff_names = ["prefilter", "footprint_n2", "footprint_n4"]
    diffs = {n: maps[n] - ref for n in diff_names}
    vabs = max(float(np.abs(d).max()) for d in diffs.values()) or 1.0

    for col_idx, name in enumerate(diff_names):
        ax = axes[1, col_idx + 1]
        im = ax.imshow(diffs[name], cmap="RdBu_r", vmin=-vabs, vmax=vabs)
        ax.set_title(f"{name} − footprint_n8", fontsize=10)
        fig.colorbar(im, ax=ax, shrink=0.75, label="Δ ppb")

    axes[1, 4].set_visible(False)

    plt.suptitle(f"Example projection — wind_dir = {wind_dir:.0f}°",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_dir / "fig13_example_maps.png",
                dpi=300, bbox_inches="tight")
    plt.close()


# ── Main ─────────────────────────────────────────────────────────────

def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    np.random.seed(SEED)
    t0 = time.time()

    # ── 1. Load bank and scenes ──────────────────────────────────────
    print("Loading plume bank...")
    bank = MethaneBank(DIR_PLUME_BANK)
    wind_speeds = bank.wind_speeds
    wind_dirs = np.arange(0, 360, WIND_DIR_STEP)

    print(f"\nLoading up to {N_SCENES} EMIT scenes...")
    scenes = load_valid_scenes(DIR_EMIT_SCENES, PATCH_H, PATCH_W, N_SCENES)
    if not scenes:
        print("FATAL: no valid scenes found.", file=sys.stderr)
        sys.exit(1)
    print(f"  {len(scenes)} valid scenes loaded.\n")

    # ── 2. Build task list ───────────────────────────────────────────
    # Each task = one (scene, wind_speed).  The worker runs ALL methods
    # and ALL wind_dirs on the SAME plume, guaranteeing a fair comparison.
    print("Pre-loading plumes from bank...")
    tasks = []
    task_meta = []

    for si, scene in enumerate(scenes):
        for ws in wind_speeds:
            try:
                meta = bank.query(
                    sza=scene["sza"], wind_speed=ws,
                    source_type=SOURCE_TYPE, meta=True,
                )
            except ValueError:
                continue

            with rasterio.open(meta["gdal_vsi"]) as src:
                plume_ppb = src.read(1).astype(np.float32)

            tasks.append((
                plume_ppb, meta["source_row"], meta["source_col"],
                scene["lat"], scene["lon"], scene["pixel_area"],
                wind_dirs, PATCH_H, PATCH_W,
                dict(METHODS),
            ))
            task_meta.append({
                "scene_idx": si,
                "scene_id": scene["id"],
                "scene_lat": scene["mean_lat"],
                "wind_speed": ws,
                "plume_id": meta["id"],
            })

    n_tasks = len(tasks)
    n_methods = len(METHODS)
    n_wd = len(wind_dirs)
    n_evals = n_tasks * n_methods * n_wd
    print(f"  {n_tasks} tasks × {n_methods} methods × {n_wd} dirs "
          f"= {n_evals} evaluations\n")

    # ── 3. Parallel sweep ────────────────────────────────────────────
    print(f"Running with {N_WORKERS} workers...")
    records = []
    done = 0

    with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
        futures = {
            executor.submit(_evaluate_task, task): i
            for i, task in enumerate(tasks)
        }
        for future in as_completed(futures):
            i = futures[future]
            mi = task_meta[i]
            for row in future.result():
                row.update({
                    "scene_idx": mi["scene_idx"],
                    "scene_id": mi["scene_id"],
                    "scene_lat": mi["scene_lat"],
                    "wind_speed": mi["wind_speed"],
                    "plume_id": mi["plume_id"],
                })
                records.append(row)

            done += 1
            if done % 10 == 0 or done == n_tasks:
                elapsed = time.time() - t0
                print(f"  {done:>4}/{n_tasks} tasks  "
                      f"({100 * done / n_tasks:5.1f}%)  "
                      f"{elapsed:.0f}s elapsed")

    df = pd.DataFrame(records)
    csv_path = OUTPUT_DIR / "method_comparison.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n  {len(df)} results saved to {csv_path}")

    # ── 4. Figures ───────────────────────────────────────────────────
    print("\nGenerating figures...")
    generate_figures(df, OUTPUT_DIR)

    # ── 4b. Example maps (re-run first task at wind_dir=0) ──────────
    if tasks:
        ex = tasks[0]  # (plume_ppb, source_row, source_col, lat, lon, ...)
        generate_example_maps(
            plume_ppb=ex[0], source_row=ex[1], source_col=ex[2],
            lat=ex[3], lon=ex[4],
            patch_h=PATCH_H, patch_w=PATCH_W,
            wind_dir=0.0, output_dir=OUTPUT_DIR,
        )
    print(f"  Figures saved to {OUTPUT_DIR}/")

    # ── 5. Summary table ─────────────────────────────────────────────
    elapsed = time.time() - t0
    print("\n" + "=" * 90)
    print("METHOD COMPARISON SUMMARY")
    print("=" * 90)
    print(f"{'Method':<16} {'η mean':>8} {'η std':>8} "
          f"{'dilution':>10} {'d std':>8} "
          f"{'r':>8} {'SSIM':>8} {'time (s)':>10}")
    print("-" * 90)
    for m in METHODS:
        sub = df[df["method"] == m]
        print(f"{m:<16} "
              f"{sub['eta'].mean():>8.4f} {sub['eta'].std():>8.4f} "
              f"{sub['dilution'].mean():>10.4f} {sub['dilution'].std():>8.4f} "
              f"{sub['pearson_r'].mean():>8.4f} "
              f"{sub['ssim'].mean():>8.4f} "
              f"{sub['time_s'].mean():>10.4f}")
    print("-" * 90)
    print(f"  Reference: footprint n_sub={N_SUB_REF}")
    print(f"  Total evaluations: {len(df)}")
    print(f"  Elapsed time:      {elapsed:.0f} s")
    print("=" * 90)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nFATAL: {e}", file=sys.stderr)
        sys.exit(1)