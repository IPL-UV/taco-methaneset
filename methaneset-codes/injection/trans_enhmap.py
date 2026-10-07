"""
trans_enhmap.py

Map a methane enhancement map from the plume bank onto an EMIT scene.

The plume bank stores enhancement maps on a regular 20 m grid with a known
source position.  EMIT observes at ~60 m with an irregular pushbroom
geometry (each pixel has its own lat/lon).

Strategy -- footprint integration
----------------------------------
Each EMIT pixel covers a quadrilateral footprint on the ground.
The sensor measures the average concentration over that footprint:

    Delta_ij = (1 / A_ij) * integral over footprint of enhancement(x, y) dA

We approximate this integral by:

    1. **Compute EMIT pixel corners** from pixel centres (midpoint of
       four neighbouring centres, linearly extrapolated at patch edges).
    2. **Project the four corners** of each pixel into bank-map
       coordinates via WGS-84 -> UTM -> inverse affine.
    3. **Sample NxN points** uniformly inside each quadrilateral
       via bilinear interpolation of the four corners, evaluate the
       (un-filtered) enhancement map at each point, and average.

This naturally captures the real shape, orientation, and size of each
EMIT pixel's footprint -- including rotation from wind_dir, the non-square
EMIT IFOV (155 x 74 urad), and pushbroom distortion.  No pre-filtering
is needed: the per-pixel integration IS the anti-aliasing.

Coordinate transformation
-------------------------
    EMIT pixel corner (lat, lon)
      -> UTM easting/northing (metres)
        -> bank pixel (col, row) via inverse affine
          -> sample enhancement with cubic interpolation
"""

from __future__ import annotations

import numpy as np
from affine import Affine
from pyproj import Transformer
from scipy.ndimage import map_coordinates, generic_filter

from config import ENHANCEMENT_PIXEL_RES


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_utm_transformer(
    ref_lat: float, ref_lon: float,
) -> tuple[Transformer, int]:
    """Build a WGS-84 -> UTM transformer for the zone of a reference point."""
    zone = int((ref_lon + 180) / 6) + 1
    epsg = 32600 + zone if ref_lat >= 0 else 32700 + zone
    t = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
    return t, epsg


def _project_grid(
    lat: np.ndarray,
    lon: np.ndarray,
    transformer: Transformer,
    ref_lon: float,
    ref_lat: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project a lat/lon grid to UTM.  NaN-safe: invalid pixels are NaN
    in the output and flagged False in the returned mask."""
    valid = np.isfinite(lat) & np.isfinite(lon)
    east, north = transformer.transform(
        np.where(valid, lon, ref_lon),
        np.where(valid, lat, ref_lat),
    )
    east[~valid] = np.nan
    north[~valid] = np.nan
    return east, north, valid


def _inpaint_nan(arr: np.ndarray, max_iter: int = 50) -> np.ndarray:
    """Replace NaN by iterative 3x3 mean of valid neighbours.

    Prevents map_coordinates from seeing NaN (which it cannot handle)
    without the artefact that NaN->0 creates near plume edges.
    """
    out = arr.astype(np.float64, copy=True)
    for _ in range(max_iter):
        nans = np.isnan(out)
        if not nans.any():
            break
        filled = np.where(nans, 0.0, out)
        count = generic_filter(
            (~nans).astype(np.float64), np.sum,
            size=3, mode="constant", cval=0.0,
        )
        total = generic_filter(
            filled, np.sum, size=3, mode="constant", cval=0.0,
        )
        fillable = nans & (count > 0)
        out[fillable] = total[fillable] / count[fillable]
    return out


def _build_bank_affine(
    src_east: float,
    src_north: float,
    source_col: int,
    source_row: int,
    pixel_res: float,
    wind_dir: float,
) -> Affine:
    """Affine that maps bank pixel (col, row) -> UTM (easting, northing).

    Composition (right to left):
        1. Centre on source.
        2. Scale to metres, flip row axis (row-down -> northing-up).
        3. Rotate by wind_dir (CCW from East).
        4. Translate to source UTM position.
    """
    return (
        Affine.translation(src_east, src_north)
        * Affine.rotation(wind_dir)
        * Affine.scale(pixel_res, -pixel_res)
        * Affine.translation(-source_col, -source_row)
    )


def _centers_to_corners(centers: np.ndarray) -> np.ndarray:
    """Convert an (H, W) grid of pixel centres to an (H+1, W+1) grid
    of pixel corners.

    Interior corners are the mean of the four surrounding centres.
    Edge and literal-corner values are linearly extrapolated so that
    every pixel -- including those on the patch boundary -- has four
    well-defined corners.
    """
    H, W = centers.shape
    corners = np.empty((H + 1, W + 1), dtype=np.float64)

    # Interior corners (rows 1..H-1, cols 1..W-1):
    # each is the midpoint of the four centres that share it.
    corners[1:H, 1:W] = 0.25 * (
        centers[:H - 1, :W - 1] + centers[:H - 1, 1:W]
        + centers[1:H, :W - 1] + centers[1:H, 1:W]
    )

    # Edge rows: linearly extrapolate from two nearest interior rows.
    corners[0, 1:W] = 2.0 * corners[1, 1:W] - corners[2, 1:W]
    corners[H, 1:W] = 2.0 * corners[H - 1, 1:W] - corners[H - 2, 1:W]

    # Edge columns: same, from two nearest interior columns.
    corners[1:H, 0] = 2.0 * corners[1:H, 1] - corners[1:H, 2]
    corners[1:H, W] = 2.0 * corners[1:H, W - 1] - corners[1:H, W - 2]

    # Four literal grid corners: extrapolate along the already-filled
    # edge row (which was computed in the step above).
    corners[0, 0] = 2.0 * corners[0, 1] - corners[0, 2]
    corners[0, W] = 2.0 * corners[0, W - 1] - corners[0, W - 2]
    corners[H, 0] = 2.0 * corners[H, 1] - corners[H, 2]
    corners[H, W] = 2.0 * corners[H, W - 1] - corners[H, W - 2]

    return corners


# ---------------------------------------------------------------------------
# Pixel geometry
# ---------------------------------------------------------------------------

def compute_pixel_areas(emit_lat: np.ndarray, emit_lon: np.ndarray) -> np.ndarray:
    """Per-pixel area [m^2] via the Jacobian of the lat/lon -> UTM mapping.

    Parameters
    ----------
    emit_lat, emit_lon : (H, W)

    Returns
    -------
    (H, W) areas in m^2.  NaN where coordinates are invalid.
    """
    clat = float(emit_lat[emit_lat.shape[0] // 2, emit_lat.shape[1] // 2])
    clon = float(emit_lon[emit_lon.shape[0] // 2, emit_lon.shape[1] // 2])
    transformer, _ = _get_utm_transformer(clat, clon)
    east, north, valid = _project_grid(emit_lat, emit_lon, transformer, clon, clat)

    de_dc = np.gradient(east, axis=1)
    dn_dr = np.gradient(north, axis=0)
    de_dr = np.gradient(east, axis=0)
    dn_dc = np.gradient(north, axis=1)

    area = np.abs(de_dc * dn_dr - de_dr * dn_dc)
    area[~valid] = np.nan
    return area


# ---------------------------------------------------------------------------
# Core transformation
# ---------------------------------------------------------------------------

def map_to_emit(
    enhancement: np.ndarray,
    source_row: int,
    source_col: int,
    emit_lat: np.ndarray,
    emit_lon: np.ndarray,
    emit_source_row: int,
    emit_source_col: int,
    pixel_res: float = ENHANCEMENT_PIXEL_RES,
    wind_dir: float = 0.0,
    n_sub: int = 3,
) -> np.ndarray:
    """Map a bank enhancement onto an EMIT scene via footprint integration.

    For each EMIT pixel, the four corners of its footprint are projected
    into bank coordinates.  N_sub x N_sub points are sampled uniformly
    inside the resulting quadrilateral, and the enhancement (un-filtered)
    is evaluated at each via cubic interpolation.  The average of the
    N_sub^2 samples approximates the integral of the enhancement over
    the pixel's real footprint.

    Parameters
    ----------
    enhancement : (H_bank, W_bank)
        Delta-XCH4 [ppb] on the bank's regular 20 m grid.
    source_row, source_col :
        Source position inside the bank map.
    emit_lat, emit_lon : (H, W)
        Per-pixel WGS-84 coordinates of the EMIT patch.
    emit_source_row, emit_source_col :
        EMIT pixel where the source is placed.
    pixel_res :
        Bank pixel size [m].  Default 20.
    wind_dir :
        Wind direction [deg, CCW from East].
    n_sub :
        Sub-samples per axis within each pixel footprint.
        n_sub=4 -> 16 samples per pixel.  Must be >= 1.

    Returns
    -------
    (H, W) array of Delta-XCH4 [ppb] on the EMIT grid.
    """
    H, W = emit_lat.shape
    K = n_sub * n_sub

    # -- 1. Inpaint NaN in the bank map -----------------------------------
    enh = _inpaint_nan(enhancement)

    # -- 2. Compute EMIT pixel corners ------------------------------------
    #    (H, W) centres -> (H+1, W+1) corners via midpoint averaging.
    #    Computed in lat/lon before any projection so the midpoints
    #    reflect the geographic positions, not a projected space.
    corner_lat = _centers_to_corners(emit_lat)
    corner_lon = _centers_to_corners(emit_lon)

    # -- 3. Project corners to bank coordinates ---------------------------
    src_lat = float(emit_lat[emit_source_row, emit_source_col])
    src_lon = float(emit_lon[emit_source_row, emit_source_col])
    transformer, _ = _get_utm_transformer(src_lat, src_lon)
    src_east, src_north = transformer.transform(src_lon, src_lat)

    fwd = _build_bank_affine(
        src_east, src_north, source_col, source_row, pixel_res, wind_dir,
    )
    inv = ~fwd

    east, north, valid = _project_grid(
        corner_lat, corner_lon, transformer, src_lon, src_lat,
    )

    # Inverse affine: UTM -> bank (col, row).
    # NaN propagates naturally through arithmetic: any invalid corner
    # produces NaN bank coordinates.
    c_col = inv.a * east + inv.b * north + inv.c
    c_row = inv.d * east + inv.e * north + inv.f

    # -- 4. Extract the four corners of each pixel ------------------------
    #    Pixel (i, j) has corners at (i, j), (i, j+1), (i+1, j), (i+1, j+1)
    #    in the corner grid.

    nw_r = c_row[:-1, :-1]       # (H, W)
    nw_c = c_col[:-1, :-1]
    ne_r = c_row[:-1, 1:]
    ne_c = c_col[:-1, 1:]
    sw_r = c_row[1:, :-1]
    sw_c = c_col[1:, :-1]
    se_r = c_row[1:, 1:]
    se_c = c_col[1:, 1:]

    # -- 5. Generate sub-pixel sample positions inside each footprint -----
    #    Bilinear interpolation of the four corners parametrises the
    #    quadrilateral interior:
    #      P(s, t) = (1-s)(1-t)*NW + s(1-t)*NE + (1-s)t*SW + s*t*SE
    #
    #    s in (0,1) left->right,  t in (0,1) top->bottom.
    #    Sample points are the centres of n_sub x n_sub equal sub-cells.
    s = (np.arange(n_sub) + 0.5) / n_sub      # e.g. [0.125, 0.375, 0.625, 0.875]
    t = (np.arange(n_sub) + 0.5) / n_sub
    ss, tt = np.meshgrid(s, t)
    ss = ss.ravel()                             # (K,)
    tt = tt.ravel()

    # Bilinear weights -- one per sub-sample, broadcast across all pixels.
    w_nw = (1.0 - ss) * (1.0 - tt)             # (K,)
    w_ne = ss * (1.0 - tt)
    w_sw = (1.0 - ss) * tt
    w_se = ss * tt

    # Bank coordinates for every sub-sample of every pixel: (H, W, K).
    sample_r = (nw_r[..., None] * w_nw + ne_r[..., None] * w_ne
                + sw_r[..., None] * w_sw + se_r[..., None] * w_se)
    sample_c = (nw_c[..., None] * w_nw + ne_c[..., None] * w_ne
                + sw_c[..., None] * w_sw + se_c[..., None] * w_se)

    # Replace NaN with a dummy value for map_coordinates (which cannot
    # handle NaN).  The results at these positions are discarded via
    # pixel_valid below.
    sample_r = np.nan_to_num(sample_r, nan=0.0)
    sample_c = np.nan_to_num(sample_c, nan=0.0)

    # -- 6. Evaluate and average ------------------------------------------
    #    Cubic B-spline interpolation (order=3).  scipy applies a
    #    B-spline prefilter by default (prefilter=True) so the
    #    interpolant passes exactly through the original pixel values.
    #    Anti-aliasing is handled separately by the NxN footprint
    #    averaging, not by the interpolation kernel.
    coords = np.array([sample_r.ravel(), sample_c.ravel()])
    values = map_coordinates(
        enh, coords, order=3, mode="constant", cval=0.0,
    )
    result = values.reshape(H, W, K).mean(axis=2).astype(np.float32)

    # -- 7. Mask invalid pixels -------------------------------------------
    #    A pixel is valid only if all four of its corners have finite
    #    coordinates.  One NaN corner means the footprint is undefined.
    pixel_valid = (
        valid[:-1, :-1] & valid[:-1, 1:]
        & valid[1:, :-1] & valid[1:, 1:]
    )
    result[~pixel_valid] = 0.0
    np.clip(result, 0.0, None, out=result)

    return result
