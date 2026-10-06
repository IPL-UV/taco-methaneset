"""Shared helpers for the MethaneSET user notebooks.

Self-contained on purpose: the notebooks are opened from Colab, where the repo is
not cloned, so this file is downloaded next to the notebook by the bootstrap cell.
No credentials, no server-specific imports.

Data access resolves the local folder in the METHANESET_DATA environment variable first
and falls back to Hugging Face otherwise.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import rasterio

HF_BASE = "https://huggingface.co/datasets/tacofoundation/methaneset/resolve/main"

# Same hexes as the paper figures (07-paper/assets/paper-figures).
PALETTE = {
    "s2": "#1a9850", "l89": "#d95f02", "emit": "#3730a3", "imeo": "#0f766e",
    "cm": "#8e24aa", "ch4": "#c0392b", "ink": "#1f2937", "muted": "#6b7280",
    "grid": "#e5e7eb",
}
CMAP_ENH = "plasma"

# 0-based indices over the stored band stack (see the dataset README).
SENSORS = {
    "s2": {"nbands": 13, "rgb": (3, 2, 1), "swir": (11, 12)},   # B11, B12
    "l89": {"nbands": 11, "rgb": (3, 2, 1), "swir": (5, 6)},    # B6, B7
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
    """Notebook rcParams (low dpi: keeps the executed notebooks light)."""
    import matplotlib as mpl

    mpl.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9,
        "figure.dpi": 80, "savefig.dpi": 80, "figure.facecolor": "white",
        "axes.grid": True, "grid.color": PALETTE["grid"], "grid.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": PALETTE["muted"], "axes.labelcolor": PALETTE["ink"],
        "xtick.color": PALETTE["muted"], "ytick.color": PALETTE["muted"],
        "legend.frameon": False, "figure.autolayout": False,
    })
