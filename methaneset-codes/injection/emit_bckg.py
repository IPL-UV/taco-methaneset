"""
emit_bckg.py

Select a plume-free radiance patch from a random EMIT scene.

Strategy -- grid tiling
-----------------------
Each scene is divided into a regular grid of non-overlapping tiles
of size ``patch_height x patch_width``.  A tile is valid when:

    1. It does not intersect the dilated plume mask (IMEO + CM,
       expanded by 5 pixels / 300 m).
    2. It contains no EMIT nodata pixels (-9999).

All valid tiles across all scenes are collected and one is chosen
at random.  This is O(n_tiles) per scene and avoids the
trial-and-error of random placement.
"""

from __future__ import annotations

import numpy as np
import rasterio
import tacoreader
from rasterio.windows import Window
from scipy.ndimage import binary_dilation

from config import EMIT_NODATA


def get_emit_background(
    data_dir: str,
    patch_height: int = 128,
    patch_width: int = 128,
) -> tuple[np.ndarray, dict, Window]:
    """Find a plume-free, nodata-free patch in a randomly chosen EMIT scene.

    The scene is tiled into a regular grid of ``patch_height x patch_width``
    blocks.  Tiles that overlap the dilated plume mask or contain EMIT
    nodata are discarded; one of the remaining tiles is selected at random.

    Parameters
    ----------
    data_dir:
        Root of the EMIT TACO dataset.
    patch_height, patch_width:
        Desired tile size in pixels.

    Returns
    -------
    radiance : np.ndarray, shape ``(n_bands, patch_height, patch_width)``
    metadata : dict-like row with scene-level metadata
        (``sensor:sza_mean``, ``sensor:sun_azimuth_mean``, etc.)
    window : rasterio.windows.Window
        Spatial window used for extraction (needed to read co-located
        lat/lon later).
    plume_mask_dilated : np.ndarray, shape ``(H_scene, W_scene)``
        Dilated plume mask for the full scene (union of IMEO and CM,
        expanded by 5 pixels).
    """
    tacoreader.use("pandas")
    df = tacoreader.load(str(data_dir)).data

    # Shuffle scene order so repeated calls explore different scenes.
    indices = np.arange(len(df))
    np.random.shuffle(indices)

    # Dilation kernel: 11x11 -> 5-pixel margin around any flagged plume
    # pixel.  At EMIT's ~60 m GSD this is ~300 m of safety buffer.
    dilation_kernel = np.ones((11, 11), dtype=bool)

    for idx in indices:
        scene_row = df.iloc[idx]

        # Use string operations instead of Path() to preserve the
        # double slash in GDAL VSI paths (e.g. /vsicurl/https://...).
        # Path() normalises "https://" -> "https:/" which breaks rasterio.
        _vsi = scene_row["internal:gdal_vsi"]
        scene_dir = _vsi.rsplit("/", 1)[0]

        # Union of the two real-plume masks, dilated for safety.
        with rasterio.open(f"{scene_dir}/plume_imeo.tif") as f_imeo, \
             rasterio.open(f"{scene_dir}/plume_cm.tif")   as f_cm:
            plume_mask = (f_imeo.read(1) > 0) | (f_cm.read(1) > 0)

        plume_mask_dilated = binary_dilation(plume_mask, structure=dilation_kernel)

        h, w = plume_mask.shape
        n_rows = h // patch_height
        n_cols = w // patch_width
        if n_rows == 0 or n_cols == 0:
            continue  # Scene smaller than requested tile.

        # -- Build list of valid (mask-free) tile positions ----------------
        valid_tiles = []
        for tr in range(n_rows):
            y = tr * patch_height
            for tc in range(n_cols):
                x = tc * patch_width
                tile_mask = plume_mask[y:y + patch_height, x:x + patch_width]
                if not tile_mask.any():
                    valid_tiles.append((y, x))

        if not valid_tiles:
            continue  # All tiles in this scene overlap plume mask.

        # Shuffle so we try tiles in random order (for nodata rejection).
        np.random.shuffle(valid_tiles)

        # -- Check radiance for nodata ------------------------------------
        with rasterio.open(f"{scene_dir}/radiance.tif") as src:
            for y, x in valid_tiles:
                window = Window(col_off=x, row_off=y,
                                width=patch_width, height=patch_height)
                patch = src.read(window=window)

                if not np.any(patch == EMIT_NODATA):
                    return patch, scene_row, window, plume_mask_dilated

    raise RuntimeError(
        f"No valid {patch_height}x{patch_width} plume-free, nodata-free "
        f"tile found in any of the {len(df)} scenes."
    )
