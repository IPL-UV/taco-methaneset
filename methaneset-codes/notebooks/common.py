"""Shared helpers for the MethaneSET user notebooks.

Self-contained on purpose: the notebooks are opened from Colab, where the repo is
not cloned, so this file is downloaded next to the notebook by the bootstrap cell.
No credentials, no server-specific imports.

Data access resolves the local folder in the METHANESET_DATA environment variable first
and falls back to Hugging Face otherwise.
"""

from __future__ import annotations

import json
import os
import warnings
from pathlib import Path
from typing import Any

# Read over HTTP (Hugging Face) with few requests: without this GDAL probes the
# folder for sidecars (HDR*.txt, RPC*.txt, .msk...) and each probe is a 429 risk.
import os as _os
_os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
_os.environ.setdefault("CPL_VSIL_CURL_USE_HEAD", "NO")

import numpy as np
import rasterio

HF_BASE = "https://huggingface.co/datasets/tacofoundation/methaneset/resolve/main"

# Same hexes as the paper figures (07-paper/assets/paper-figures), complete.
PALETTE = {
    "s2": "#1a9850", "l89": "#d95f02", "emit": "#3730a3",
    "imeo": "#0f766e", "cm": "#8e24aa", "ch4": "#c0392b",
    "ink": "#1f2937", "muted": "#6b7280", "grid": "#e5e7eb",
    "sun": "#f59e0b", "light": "#f3f4f6",
    "free": "#64748b",            # windows without plume (negatives)
    "pos": "#b91c1c", "neg": "#1d4ed8",   # prediction / truth
}

CMAPS = {
    "enh": "plasma",      # enhancements: ch4, mf, rmf, mag1c, bank maps
    "mbmp": "RdBu_r",     # diverging centered on 0, ALWAYS over -MBMP
    "dem": "cividis",
    "prob": "magma",      # model probability
    "wind": "viridis",
}
CMAP_ENH = CMAPS["enh"]   # alias, does not break current notebooks

FIGW = 9.0
ROW_H = {"image": 3.0, "plot": 2.8}
PIXEL_M = {"s2": 10.0, "l89": 30.0, "emit": 60.0, "bank": 20.0}
NODATA_EMIT = -9999.0
WEAK_KGH = 1000.0         # Ruzicka et al. 2023 convention

# 0-based indices over the stored band stack (numpy order; add +1 when passing to rasterio).
SENSORS = {
    "s2": {
        "nbands": 13,
        "rgb": (3, 2, 1),
        "swir": (11, 12),
        "rgn": (3, 2, 7),
        "sim": (1, 2, 3, 11),     # bands used by MARS to score backgrounds
        "px_m": 10.0,
        "names": ("B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A",
                  "B9", "B10", "B11", "B12"),
        "wl_nm": (443, 490, 560, 665, 705, 740, 783, 842, 865,
                  940, 1375, 1610, 2190),
    },
    "l89": {
        "nbands": 11,
        "rgb": (3, 2, 1),
        "swir": (5, 6),
        "rgn": (3, 2, 4),
        "sim": (1, 2, 3, 5),      # bands used by MARS to score backgrounds
        "px_m": 30.0,
        "names": ("B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8",
                  "B9", "B10", "B11"),
        "wl_nm": (443, 482, 561, 654, 865, 1608, 2201, 591,
                  1373, 10895, 12005),
    },
}


def resolve(dataset: str) -> str:
    """Local path if the dataset exists on disk, otherwise the HF URL."""
    base = os.environ.get("METHANESET_DATA")
    local = Path(base) / dataset if base else None
    if local is not None and local.exists():
        print(f"[common] {dataset}: local")
        return str(local)
    print(f"[common] {dataset}: remote (Hugging Face)")
    return f"{HF_BASE}/{dataset}/"


def load(dataset: str, backend: str = "pandas"):
    """tacoreader dataset for one of the seven collections."""
    import tacoreader

    tacoreader.use(backend)
    return tacoreader.load(resolve(dataset))


def read(src, leaf: str | None = None, bands=None, window=None, nan_nodata: bool = True):
    """Read a raster layer as float32 with nodata as NaN (unsigned ints kept as-is).

    `src` is either a tacoreader sample (pass `leaf`, e.g. "target") or a raster path
    inside a TACO sample (the bank/bank-les samples return paths from `ds.read(i)`).
    `bands`: None or tuple/list -> (B, H, W); a single int -> (H, W).
    """
    if leaf is not None:
        src = src.read(leaf)
    with rasterio.open(src) as ds:
        if bands is None:
            a = ds.read(window=window)
        elif isinstance(bands, (int, np.integer)):
            a = ds.read(int(bands), window=window)
        else:
            a = ds.read(list(bands), window=window)
        if not nan_nodata or np.issubdtype(a.dtype, np.unsignedinteger):
            return a
        nodata = ds.nodata
    a = a.astype("float32")
    if nodata is not None:
        a[a == nodata] = np.nan
    return a


def stretch(a: np.ndarray, p: tuple = (2, 98)) -> np.ndarray:
    """Percentile stretch to [0, 1], NaN-safe."""
    a = a.astype("float32")
    lo, hi = np.nanpercentile(a, p)
    if not np.isfinite(lo) or hi <= lo:
        return np.zeros_like(a)
    return np.clip((a - lo) / (hi - lo), 0, 1)


def to_rgb(stack: np.ndarray, idx: tuple = (3, 2, 1)) -> np.ndarray:
    """(B, H, W) -> (H, W, 3), stretched per channel with a mild gamma."""
    ch = [stretch(stack[i]) ** 0.8 for i in idx]
    return np.dstack(ch)


def mbmp(t_swir1, t_swir2, r_swir1, r_swir2, eps: float = 1e-9) -> np.ndarray:
    """Fractional MBMP enhancement (Varon et al. 2021); negative over a plume.

    Inputs are cast to float and, if a reference band is 0 (the uint16 nodata of the
    Sentinel-2 stacks), the pixel is set to NaN instead of blowing up.
    """
    t1, t2, r1, r2 = (np.asarray(x, dtype="float32") for x in (t_swir1, t_swir2, r_swir1, r_swir2))
    out = ((t2 + eps) / (r2 + eps)) / ((t1 + eps) / (r1 + eps)) - 1.0
    out[(r1 <= 0) | (r2 <= 0)] = np.nan
    return out


def mbmp_plus(target_stack: np.ndarray, ref_stack: np.ndarray,
              swir: tuple[int, int] = (11, 12), sim: tuple | None = None) -> np.ndarray:
    """MBMP+ = inverse MBMP minus a Ridge prediction of shared artefacts.

    Inverse pair: B_a_t/B_b_t - B_a_r/B_b_r with 0-based `swir` (S2 default (11, 12) = B11/B12).
    Predictors are the z-scored temporal ratios r_k = B_k_t / B_k_r for every
    band except the two SWIR bands. Ridge(alpha=1.0) is trained on ALL valid
    pixels (no mask leak). `sim` is accepted for API compatibility but ignored
    here; it belongs to `mars_difference`.
    """
    from sklearn.linear_model import Ridge

    target_stack = np.asarray(target_stack, dtype="float32")
    ref_stack = np.asarray(ref_stack, dtype="float32")
    if target_stack.shape != ref_stack.shape:
        raise ValueError("target_stack and ref_stack must have the same shape")

    n_bands = target_stack.shape[0]
    i_a, i_b = [int(s) for s in swir]          # 0-based, e.g. S2 (11, 12) = B11/B12
    eps = 1e-9

    t_flat = target_stack.reshape(n_bands, -1)
    r_flat = ref_stack.reshape(n_bands, -1)

    # inverse MBMP
    safe = (np.abs(t_flat[i_b]) > eps) & (np.abs(r_flat[i_b]) > eps)
    mbmp_inv = np.full(t_flat.shape[1], np.nan, dtype="float32")
    mbmp_inv[safe] = t_flat[i_a, safe] / t_flat[i_b, safe] - r_flat[i_a, safe] / r_flat[i_b, safe]

    # predictors: temporal ratios for all bands except the two SWIR
    exclude = {i_a, i_b}
    pred_idx = [b for b in range(n_bands) if b not in exclude]
    safe_ref = np.where(np.abs(r_flat) > eps, r_flat, np.nan)
    ratio = t_flat / safe_ref
    X = ratio[pred_idx].T                     # (pixels, n_pred)

    valid = np.isfinite(mbmp_inv)
    for pi in pred_idx:
        valid &= np.isfinite(ratio[pi])
    if valid.sum() < 50:
        import warnings
        warnings.warn("mbmp_plus: too few valid pixels for the ridge fit; returning the plain MBMP", stacklevel=2)
        return mbmp_inv.reshape(target_stack.shape[1:])

    mu = np.nanmean(X[valid], axis=0)
    sigma = np.nanstd(X[valid], axis=0)
    sigma = np.where(sigma < 1e-9, 1.0, sigma)
    X_z = (X - mu) / sigma

    ridge = Ridge(alpha=1.0, fit_intercept=True)
    ridge.fit(X_z[valid], mbmp_inv[valid])

    valid_all = valid & np.isfinite(mbmp_inv)
    mbmp_hat = np.full_like(mbmp_inv, np.nan)
    mbmp_hat[valid_all] = ridge.predict(X_z[valid_all])

    mbmp_plus = np.full_like(mbmp_inv, np.nan)
    mbmp_plus[valid_all] = mbmp_inv[valid_all] - mbmp_hat[valid_all]
    return mbmp_plus.reshape(target_stack.shape[1:])


def mbmp_plus_gauss(d: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """MBMP+ Gaussian variant: MBMP minus a Gaussian-smoothed copy of itself."""
    from scipy.ndimage import gaussian_filter

    d = np.asarray(d, dtype="float32")
    return d - gaussian_filter(np.nan_to_num(d, nan=0.0), sigma=sigma)


def mars_difference(target_stack: np.ndarray, bg_stack: np.ndarray,
                    sim_bands: tuple) -> float:
    """MARS background score: mean_k mean|t_k - g_k*b_k| over `sim_bands`.

    g_k = mean(t_k) / mean(b_k) is computed per band and corrects the global
    illumination offset between target and candidate background.
    """
    target_stack = np.asarray(target_stack, dtype="float32")
    bg_stack = np.asarray(bg_stack, dtype="float32")
    idx = tuple(int(s) for s in sim_bands)

    diffs = []
    for k in idx:
        t_k = target_stack[k]
        b_k = bg_stack[k]
        m_t = np.nanmean(t_k)
        m_b = np.nanmean(b_k)
        if not (np.isfinite(m_t) and np.isfinite(m_b) and abs(m_b) > 1e-12):
            diffs.append(np.nan)
            continue
        g_k = m_t / m_b
        diffs.append(np.nanmean(np.abs(t_k - g_k * b_k)))
    return float(np.nanmean(diffs))


def emit_band(nm: float) -> int:
    """1-based EMIT band index closest to `nm` (band 1 = 381 nm, band 285 = 2493 nm)."""
    return int(round((nm - 381.0) / ((2493.0 - 381.0) / 284.0))) + 1


def emit_swir(lo: float = 2122.0, hi: float = 2488.0) -> list[int]:
    """1-based EMIT bands inside the retrieval window (keeps reads small)."""
    step = (2493.0 - 381.0) / 284.0
    return [b for b in range(1, 286) if lo <= 381.0 + (b - 1) * step <= hi]


def bit_layer(mask: np.ndarray, bit: int) -> np.ndarray:
    """Decode one plume out of the EMIT uint64 bitmask (1-based bit, up to 64)."""
    if not 1 <= bit <= 64:
        raise ValueError("bit must be between 1 and 64")
    return (mask & np.uint64(1 << (bit - 1))) != 0


def seg_metrics(pred: np.ndarray, truth: np.ndarray) -> dict:
    """IoU / F1 / precision / recall for two boolean masks."""
    p, t = pred.astype(bool), truth.astype(bool)
    tp = int(np.sum(p & t)); fp = int(np.sum(p & ~t)); fn = int(np.sum(~p & t))
    div = lambda a, b: float(a) / b if b else 0.0  # noqa: E731
    return {"iou": div(tp, tp + fp + fn), "f1": div(2 * tp, 2 * tp + fp + fn),
            "precision": div(tp, tp + fp), "recall": div(tp, tp + fn),
            "tp": tp, "fp": fp, "fn": fn}


def crop_around(mask: np.ndarray, pad: int = 90) -> tuple[slice, slice]:
    """Slices of the bounding box of a mask plus padding."""
    ys, xs = np.nonzero(mask)
    if not len(ys):
        return slice(0, mask.shape[0]), slice(0, mask.shape[1])
    r0 = max(int(ys.min()) - pad, 0); r1 = min(int(ys.max()) + pad, mask.shape[0])
    c0 = max(int(xs.min()) - pad, 0); c1 = min(int(xs.max()) + pad, mask.shape[1])
    return slice(r0, r1), slice(c0, c1)


def style() -> None:
    """Notebook rcParams (keeps figures consistent across the four notebooks)."""
    import matplotlib as mpl

    mpl.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9,
        "axes.titlesize": 9.5, "axes.labelsize": 9,
        "xtick.labelsize": 8, "ytick.labelsize": 8,
        "figure.dpi": 100, "savefig.dpi": 100,
        "figure.facecolor": "white", "savefig.bbox": "tight",
        "axes.grid": False,                      # grid is added by ax_plot()
        "grid.color": PALETTE["grid"], "grid.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#9ca3af", "axes.linewidth": 0.8,
        "axes.labelcolor": PALETTE["ink"],
        "xtick.color": PALETTE["muted"], "ytick.color": PALETTE["muted"],
        "legend.frameon": False, "legend.fontsize": 8,
        "lines.linewidth": 1.3, "image.interpolation": "nearest",
        "figure.autolayout": False,
    })


def panels(n: int, rows: int = 1, kind: str = "image", w: float = FIGW):
    """Return fig, axes already styled. kind='image' removes ticks, spines and grid."""
    import matplotlib.pyplot as plt

    cols = int(np.ceil(n / rows))
    h = rows * ROW_H[kind]
    fig, axes = plt.subplots(rows, cols, figsize=(w, h), constrained_layout=True)
    if n == 1 and rows == 1:
        axes = np.array([axes])
    elif isinstance(axes, np.ndarray):
        axes = axes.ravel()
    else:
        axes = np.array([axes])

    for i, ax in enumerate(axes):
        if i >= n:
            ax.set_visible(False)
        elif kind == "image":
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ("top", "right", "bottom", "left"):
                ax.spines[sp].set_visible(False)
    return fig, axes[:n]


def ax_plot(ax):
    """Grid, ticks and spines for data axes (also restores them if panels() hid them)."""
    from matplotlib.ticker import AutoLocator
    ax.xaxis.set_major_locator(AutoLocator())
    ax.yaxis.set_major_locator(AutoLocator())
    ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("bottom", "left"):
        ax.spines[sp].set_visible(True)
    ax.grid(True, color=PALETTE["grid"], linewidth=0.6)


def panel_label(ax, letter: str):
    """'(a)' at the top-left of an axis: white with a dark outline, readable on any image."""
    import matplotlib.patheffects as pe
    ax.text(0.02, 0.98, f"({letter})", transform=ax.transAxes,
            fontsize=10, color="white", va="top", ha="left", fontweight="bold",
            path_effects=[pe.withStroke(linewidth=2.5, foreground=PALETTE["ink"])])


def imshow_layer(ax, a: np.ndarray, kind: str = "enh", title: str | None = None,
                 cbar: bool = True, label: str | None = None, vmax_pct: float = 99.5,
                 px_m: float | None = None, sym: bool = False):
    """Coherent imshow: CMAPS cmap, percentile vmax (NaN-safe), fine colorbar,
    no ticks, scalebar if px_m. sym=True uses TwoSlopeNorm centered on 0."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    a = np.asarray(a, dtype="float32")
    cmap = CMAPS.get(kind, kind)

    if sym:
        vmax = np.nanpercentile(np.abs(a), vmax_pct)
        if not np.isfinite(vmax) or vmax == 0:
            vmax = 1.0
        norm = TwoSlopeNorm(vcenter=0, vmin=-vmax, vmax=vmax)
        im = ax.imshow(a, cmap=cmap, norm=norm)
    else:
        vmax = np.nanpercentile(a, vmax_pct)
        vmin = np.nanmin(a)
        if not np.isfinite(vmax) or vmax <= vmin:
            vmax = vmin + 1.0
        im = ax.imshow(a, cmap=cmap, vmin=vmin, vmax=vmax)

    if title:
        ax.set_title(title, fontsize=9.5)
    ax.set_xticks([])
    ax.set_yticks([])

    if cbar:
        cbar_obj = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        if label:
            cbar_obj.set_label(label, fontsize=8)
        cbar_obj.ax.tick_params(direction="out", labelsize=7)

    if px_m is not None:
        scalebar(ax, px_m)
    return im


def overlay_mask(ax, mask: np.ndarray, which: str = "imeo",
                 alpha: float = 0.35, lw: float = 0.8):
    """Semi-transparent fill + contour in PALETTE[which].

    Replaces cmap='gray' and loose contour(colors='k'/'w') calls.
    """
    color = PALETTE[which]
    ax.contourf(mask, levels=[0.5, 1], colors=[color], alpha=alpha)
    ax.contour(mask, levels=[0.5], colors=[color], linewidths=lw)


def scalebar(ax, px_m: float, frac: float = 0.25):
    """Draw a scalebar at the bottom-right. Picks a round distance <= frac*width."""
    xlim = ax.get_xlim()
    width_px = abs(xlim[1] - xlim[0])
    max_m = width_px * px_m * frac

    # choose a nice round distance, preferring 1 km when it fits
    choices = [100, 250, 500, 1000, 2000, 5000, 10000, 20000, 50000]
    dist_m = choices[0]
    for d in choices:
        if d <= max_m:
            dist_m = d
    if dist_m >= 1000:
        text = f"{dist_m/1000:g} km"
    else:
        text = f"{dist_m:g} m"

    bar_px = dist_m / px_m
    ylim = ax.get_ylim()
    # axes direction depends on imshow; place inside bottom-right
    x1 = max(xlim[0], xlim[1]) - 0.05 * width_px - bar_px
    x2 = x1 + bar_px
    rng = abs(ylim[1] - ylim[0])
    if ax.yaxis_inverted():          # imshow: the bottom of the image is max(ylim)
        y = max(ylim) - 0.06 * rng
        y_text, va = y - 0.02 * rng, "top"
    else:
        y = min(ylim) + 0.06 * rng
        y_text, va = y + 0.02 * rng, "bottom"

    ax.plot([x1, x2], [y, y], color=PALETTE["ink"], linewidth=2.5, solid_capstyle="butt")
    ax.text((x1 + x2) / 2, y_text, text,
            ha="center", va=va, fontsize=7, color=PALETTE["ink"])


def layers(sample) -> Any:
    """Table of sheet / shape / dtype / nodata for a TACO sample."""
    import pandas as pd

    rows = []
    data = getattr(sample, "_data", None)
    if data is None and hasattr(sample, "data"):
        data = sample.data
    if data is None:
        warnings.warn("layers(): could not access sample rows")
        return pd.DataFrame()

    # pandas backend is the default in the notebooks
    try:
        iter_rows = data.iterrows()
    except AttributeError:
        # fallback for polars/arrow: convert to pandas
        try:
            iter_rows = data.to_pandas().iterrows()
        except Exception:
            warnings.warn("layers(): could not iterate over sample rows")
            return pd.DataFrame()

    for _, row in iter_rows:
        if row.get("type") != "FILE":
            continue
        name = row.get("id", "?")
        path = row.get("internal:gdal_vsi")
        if path is None:
            continue
        try:
            with rasterio.open(path) as src:
                rows.append({
                    "sheet": name,
                    "shape": src.shape,
                    "dtype": str(src.dtypes[0]),
                    "nodata": src.nodata,
                    "count": src.count,
                })
        except Exception as exc:
            rows.append({"sheet": name, "shape": str(exc), "dtype": "", "nodata": None, "count": 0})
    return pd.DataFrame(rows)


def versions():
    """Print versions of tacoreader, rasterio, numpy, torch and the Colab flag."""
    import tacoreader

    try:
        import torch
        torch_ver = torch.__version__
    except Exception:
        torch_ver = "not installed"

    in_colab = bool(os.environ.get("COLAB_GPU")) or _in_colab()
    print(f"tacoreader {tacoreader.__version__}")
    print(f"rasterio   {rasterio.__version__}")
    print(f"numpy      {np.__version__}")
    print(f"torch      {torch_ver}")
    print(f"colab      {in_colab}")


def _in_colab() -> bool:
    try:
        import google.colab  # noqa: F401
        return True
    except Exception:
        return False


def dataset_version(ds) -> str | None:
    """Read `dataset_version` from the COLLECTION.json of a collection.

    `ds` can be a dataset name (str), a tacoreader dataset, or a local/remote path.
    """
    if isinstance(ds, str):
        root = resolve(ds)
    elif hasattr(ds, "collection") and hasattr(ds.collection, "path"):
        root = str(ds.collection.path)
    else:
        root = str(getattr(ds, "path", getattr(ds, "root", "")))

    if not root:
        return None

    base = root.rstrip("/")
    candidates = [f"{base}/COLLECTION.json", f"{base}/.tacocat/COLLECTION.json"]
    if base.startswith("http"):
        import urllib.request
        for url in candidates:
            try:
                with urllib.request.urlopen(url, timeout=30) as resp:
                    meta = json.loads(resp.read().decode("utf-8"))
                return meta.get("dataset_version")
            except Exception:
                continue
        return None
    for url in candidates:
        path = Path(url.replace("file://", ""))
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            return meta.get("dataset_version")
    return None



def _find_col(df, names):
    """Return the first column in `df` matching one of `names`."""
    cols = list(df.columns)
    for name in names:
        if name in cols:
            return name
        matches = [c for c in cols if name in c.lower()]
        if matches:
            return matches[0]
    raise KeyError(f"none of {names} found in dataframe columns: {cols[:20]}")


def bank_query(df, sza: float, raa: float, wind: float, k: int = 1):
    """Nearest-neighbour query in the LES bank geometry space.

    Builds a cKDTree on (sza, sin(raa), cos(raa), wind) so RAA wraps circularly; the returned distance combines the RAA and SZA gaps.
    Returns (indices, angular_distance_deg, wind_error) for the k closest nodes.
    """
    from scipy.spatial import cKDTree

    col_sza = _find_col(df, ["sza"])
    col_raa = _find_col(df, ["raa"])
    col_wind = _find_col(df, ["wind"])

    raa_rad = np.deg2rad(df[col_raa].values)
    X = np.column_stack([
        df[col_sza].values,
        np.sin(raa_rad),
        np.cos(raa_rad),
        df[col_wind].values,
    ])
    # standardise each axis: degrees, unit vectors and m/s do not share a scale
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd = np.where(sd < 1e-9, 1.0, sd)
    tree = cKDTree((X - mu) / sd)

    q_raa_rad = np.deg2rad(raa)
    query = ((np.array([[sza, np.sin(q_raa_rad), np.cos(q_raa_rad), wind]]) - mu) / sd)
    dist, idx = tree.query(query, k=k)

    # cKDTree returns shape (1, k); flatten to 1-D
    idx = np.atleast_1d(idx[0])

    # angular distance: circular RAA gap plus the SZA gap (degrees)
    matched_raa = df[col_raa].values[idx]
    delta = np.abs(matched_raa - raa)
    draa = np.minimum(delta, 360.0 - delta)
    dsza = np.abs(df[col_sza].values[idx] - sza)
    ang_dist = np.sqrt(draa ** 2 + dsza ** 2)

    matched_wind = df[col_wind].values[idx]
    wind_err = np.abs(matched_wind - wind)

    if k == 1:
        return int(idx[0]), float(ang_dist[0]), float(wind_err[0])
    return idx, ang_dist, wind_err
