"""
retrieval.py

Run matched-filter methane retrieval on a modified EMIT patch.

Reads the full-column radiance from the scene, inserts the modified
patch, runs the requested retrieval methods, and returns cropped
results.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window

# Lazy torch import (only needed when running retrieval)
_torch = None


def _import_torch():
    global _torch
    if _torch is None:
        import torch
        _torch = torch
    return _torch


def load_original_retrievals(
    scene_dir: str,
    reading_window: Window,
    method_names: list[str],
) -> dict[str, np.ndarray]:
    """Load pre-existing retrieval maps from the dataset.

    Parameters
    ----------
    scene_dir:
        GDAL VSI base path of the EMIT scene.
    reading_window:
        Rasterio window for the patch.
    method_names:
        Retrieval method names (e.g. ``["mf", "rmf", "mag1c"]``).

    Returns
    -------
    dict mapping method name -> (H, W) retrieval map.
    Only methods whose TIF exists in the scene are included.
    """
    result = {}
    for name in method_names:
        tif_path = f"{scene_dir}/{name}.tif"
        try:
            with rasterio.open(tif_path) as src:
                result[name] = src.read(1, window=reading_window)
        except Exception:
            pass
    return result


def run_retrievals(
    radiance_modified: np.ndarray,
    scene_dir: str,
    reading_window: Window,
    emit_wvl: np.ndarray,
    patch_h: int,
    patch_w: int,
    swir_range: tuple[float, float] = (2122, 2488),
    methods: dict | None = None,
    batch_size: int = 64,
    device: str | None = None,
    verbose: bool = True,
) -> dict[str, np.ndarray]:
    """Run retrieval methods on the modified scene.

    Parameters
    ----------
    radiance_modified : (n_bands, H, W)
        Modified radiance cube (with plume injected).
    scene_dir:
        GDAL VSI base path of the EMIT scene.
    reading_window:
        Rasterio window of the patch within the scene.
    emit_wvl : (n_bands,)
        EMIT central wavelengths [nm].
    patch_h, patch_w:
        Patch dimensions in pixels.
    swir_range:
        SWIR wavelength range for retrieval [nm].
    methods:
        dict mapping name -> config class.  Defaults to
        ``{"mf": MFConfig, "rmf": RMFConfig, "mag1c": MAG1CConfig}``.
    batch_size:
        Columns per batch for retrieval.
    device:
        ``"cuda"`` or ``"cpu"``.  ``None`` = auto-detect.
    verbose:
        Print progress messages.

    Returns
    -------
    dict mapping method name -> (patch_h, patch_w) retrieval map.
    """
    torch = _import_torch()

    import methanex
    methanex.SAFETENSORS_PATH = Path("ch4_emit.safetensors")

    # Auto-download safetensors if missing
    if not methanex.SAFETENSORS_PATH.exists():
        import urllib.request
        req = urllib.request.Request(
            methanex.SAFETENSORS_URL,
            headers={"User-Agent": "Mozilla/5.0"},
        )
        if verbose:
            print("Downloading CH4 template...")
        with urllib.request.urlopen(req) as resp, \
             open(methanex.SAFETENSORS_PATH, "wb") as f:
            f.write(resp.read())
        if verbose:
            print("Download complete.")

    from methanex import MAG1CConfig, MFConfig, MethaneRetrieval, RMFConfig

    if methods is None:
        methods = {"mf": MFConfig, "rmf": RMFConfig, "mag1c": MAG1CConfig}

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # -- SWIR band indices --
    swir_idx = np.where(
        (emit_wvl >= swir_range[0]) & (emit_wvl <= swir_range[1])
    )[0]

    r0 = reading_window.row_off
    c0 = reading_window.col_off

    # -- Read full-column radiance for covariance context --
    with rasterio.open(f"{scene_dir}/radiance.tif") as src:
        col_window = Window(
            col_off=c0, row_off=0,
            width=patch_w, height=src.height,
        )
        patch_swir = src.read(
            indexes=(swir_idx + 1).tolist(), window=col_window,
        )

    # Insert modified radiance into patch rows
    patch_swir[:, r0:r0 + patch_h, :] = radiance_modified[swir_idx]
    patch_swir = np.moveaxis(patch_swir, 0, -1).astype(np.float64)

    # -- Run retrieval methods --
    engine = MethaneRetrieval(device=device, dtype=torch.float64)
    if verbose:
        print(f"  Device: {device}")

    rad_tensor = torch.from_numpy(patch_swir)
    del patch_swir

    results = {}
    for name, cfg_cls in methods.items():
        if verbose:
            print(f"  Running {name.upper()}...", end=" ", flush=True)
        res = engine.retrieve(
            rad_tensor, cfg_cls(batch_size=batch_size), display_pbar=False,
        )
        results[name] = res.mf.cpu().numpy()[r0:r0 + patch_h, :]
        if verbose:
            print("done.")
        if device == "cuda":
            torch.cuda.empty_cache()

    del rad_tensor
    if verbose:
        print("Retrieval complete.")

    return results
