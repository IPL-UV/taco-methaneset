# run_injection_emit.py

import sys
from pathlib import Path

import numpy as np
import rasterio
import torch
import warnings
from rasterio.errors import NotGeoreferencedWarning
from rasterio.windows import Window

# Allow running this script directly (python scripts/run_injection_emit.py
# from the injection/ root) so `core` is importable.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.methanex import MethaneRetrieval, MFConfig, RMFConfig, MAG1CConfig

from core.config import (
    DIR_EMIT_SCENES,
    DIR_PLUME_BANK,
    MR_BACKGROUND_PPB,
    Q_REF_KGH,
    Q_MIN_KGH,
    Q_MAX_KGH,
    Q_TARGET_KGH,
    ENHANCEMENT_PIXEL_RES,
    USE_WIND_FIELD,
    WIND_SPEED,
    WIND_DIR,
    EMIT_WAVELENGTHS_FILE,
    EMIT_FWHM_FILE,
    PATCH_H,
    PATCH_W,
    N_PLUMES,
    SOURCE,
    MIN_SEPARATION,
    MARGIN,
)
from core.bank import MethaneBank
# TODO: core.lut_emit (planned, not created yet) -> read_luts, air_mass_factor
# from core.lut_emit import read_luts, air_mass_factor
# TODO: core.placement (planned, not created yet) -> get_emit_background
# from core.placement import get_emit_background
from core.reproject import map_to_emit, compute_pixel_areas
from core.inject_emit import get_methane_band_indices, inject_plume
from core.srf import build_srf_matrix
# TODO: core.geometry (planned, not created yet) -> RAA conventions.
# TODO: core.inject_ms (planned, not created yet) -> multispectral family, not used here.
# TODO: the old visualize.py plotting helpers are not ported yet; the
# plot_* calls at the end of main() need a home before this script runs.


warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)


# -- Retrieval configuration -----------------------------------------------
RETRIEVAL_METHODS = {"mf": MFConfig, "rmf": RMFConfig, "mag1c": MAG1CConfig}
RET_BATCH_SIZE = 64
WL_MIN_SWIR, WL_MAX_SWIR = 2122, 2488


# -- Helpers ---------------------------------------------------------------

def generate_source_positions(n, patch_h, patch_w, margin, min_sep):
    """Generate n random positions within the patch with minimum separation.

    Each position is at least `margin` pixels from the patch edge and
    at least `min_sep` pixels from all previously placed positions.

    Returns
    -------
    list of (row, col) tuples.  May be shorter than n if placement fails.
    """
    positions = []
    for _ in range(n):
        for _attempt in range(1000):
            r = np.random.randint(margin, patch_h - margin)
            c = np.random.randint(margin, patch_w - margin)
            if all(np.hypot(r - pr, c - pc) >= min_sep for pr, pc in positions):
                positions.append((r, c))
                break
    return positions


def main() -> None:
    # ==================================================================
    # 1. Background patch
    # ==================================================================
    print("Searching for a plume-free EMIT patch...")
    radiance, scene_row, reading_window = get_emit_background(
        data_dir=DIR_EMIT_SCENES,
        patch_height=PATCH_H,
        patch_width=PATCH_W,
    )
    print(f"  Patch shape : {radiance.shape}")
    print(f"  Scene ID    : {scene_row['id']}")

    sza_mean = scene_row["target:sza"]
    saa_mean = scene_row["target:saa"]
    vza_mean = scene_row["target:vza"]
    _vsi = scene_row["internal:gdal_vsi"]
    scene_dir = _vsi.rsplit("/", 1)[0] 
    print(f"  SZA={sza_mean:.1f} deg  SAA={saa_mean:.1f} deg  VZA={vza_mean:.1f} deg")

    # ==================================================================
    # 2. Generate emission source positions
    # ==================================================================
    n_plumes = N_PLUMES if N_PLUMES is not None else np.random.randint(1, 11)

    emit_positions = generate_source_positions(
        n_plumes, PATCH_H, PATCH_W, MARGIN, MIN_SEPARATION
    )
    if len(emit_positions) < n_plumes:
        print(f"  Warning: only placed {len(emit_positions)} of {n_plumes} plumes "
                f"(separation constraint)")
        n_plumes = len(emit_positions)

    # ==================================================================
    # 3. Resolve wind conditions and query plume bank
    # ==================================================================
    plume_bank = MethaneBank(DIR_PLUME_BANK)

    if USE_WIND_FIELD:
        with rasterio.open(f"{scene_dir}/wind.tif") as src:
            wind_u = src.read(1, window=reading_window)
            wind_v = src.read(2, window=reading_window)

        print(f"\nUsing wind field (wind.tif) for {n_plumes} plume(s)...")

        wind_dirs = []
        wind_speeds = []
        raas = []
        plume_metas = []
        used_ids = []

        for i, (emit_r, emit_c) in enumerate(emit_positions):
            u = float(wind_u[emit_r, emit_c])
            v = float(wind_v[emit_r, emit_c])

            ws = float(np.hypot(u, v))
            wd = float(np.degrees(np.arctan2(v, u))) % 360
            raa = (90 - saa_mean - wd) % 360

            wind_speeds.append(ws)
            wind_dirs.append(wd)
            raas.append(raa)

            meta = plume_bank.query(
                sza=sza_mean, raa=raa,
                wind_speed=ws,
                source_type=SOURCE, meta=True,
                exclude_ids=used_ids,
            )
            plume_metas.append(meta)
            used_ids.append(meta["id"])

            print(f"  [{i+1}] pos=({emit_r},{emit_c})  "
                  f"u={u:.1f} v={v:.1f}  "
                  f"speed={ws:.1f} m/s  dir={wd:.0f} deg  raa={raa:.0f} deg")

    else:
        wind_speed = WIND_SPEED if WIND_SPEED is not None else float(np.random.choice(plume_bank.wind_speeds))
        wind_dir = WIND_DIR if WIND_DIR is not None else np.random.randint(0, 360)
        raa = (90 - saa_mean - wind_dir) % 360

        print(f"\nQuerying plume bank for {n_plumes} plume(s)...")
        print(f"  Wind: {wind_speed} m/s @ {wind_dir} deg  RAA: {raa:.0f} deg")

        if n_plumes == 1:
            plume_metas = [plume_bank.query(
                sza=sza_mean, raa=raa,
                wind_speed=wind_speed,
                source_type=SOURCE, meta=True,
            )]
        else:
            plume_metas = plume_bank.query(
                sza=sza_mean, raa=raa,
                wind_speed=wind_speed,
                source_type=SOURCE, meta=True, n=n_plumes,
            )

        wind_dirs = [wind_dir] * n_plumes
        wind_speeds = [wind_speed] * n_plumes
        raas = [raa] * n_plumes

    n_plumes = len(plume_metas)
    emit_positions = emit_positions[:n_plumes]
    wind_dirs = wind_dirs[:n_plumes]
    wind_speeds = wind_speeds[:n_plumes]
    raas = raas[:n_plumes]

    for i, m in enumerate(plume_metas):
        print(f"  [{i+1}] {m['id']}  wind_speed={m['wind_speed']} m/s")

    # ==================================================================
    # 4. Load, project, and accumulate enhancement maps
    # ==================================================================
    print(f"\nProjecting {n_plumes} enhancement map(s) onto EMIT coordinates...")

    with rasterio.open(f"{scene_dir}/latlon.tif") as src:
        lat = src.read(1, window=reading_window)
        lon = src.read(2, window=reading_window)

    pixel_area = compute_pixel_areas(lat, lon)

    q_targets = np.empty(n_plumes)
    for i in range(n_plumes):
        if Q_TARGET_KGH is not None:
            if Q_TARGET_KGH > Q_MAX_KGH:
                print(f"  Warning: Q_TARGET_KGH={Q_TARGET_KGH} exceeds "
                      f"Q_MAX_KGH={Q_MAX_KGH}, clamping to {Q_MAX_KGH}")
            q_targets[i] = min(Q_TARGET_KGH, Q_MAX_KGH)
        else:
            q_targets[i] = np.random.uniform(Q_MIN_KGH, Q_MAX_KGH)

    enhancement_total = np.zeros((PATCH_H, PATCH_W), dtype=np.float32)
    enhancement_ref_total = np.zeros((PATCH_H, PATCH_W), dtype=np.float32)
    plume_raws = []
    sources_bank = []

    for i, (meta, (emit_r, emit_c)) in enumerate(zip(plume_metas, emit_positions)):
        with rasterio.open(meta["gdal_vsi"]) as src:
            plume_ppb = src.read(1).astype(np.float32)

        source_row = meta["source_row"]
        source_col = meta["source_col"]
        plume_raws.append(plume_ppb)
        sources_bank.append((source_row, source_col))

        # Reproject at Q_ref (no scaling yet) — linear, so equivalent
        # to scaling before reprojection for the injection itself.
        enhancement_i_ref = map_to_emit(
            enhancement=plume_ppb,
            source_row=source_row,
            source_col=source_col,
            emit_lat=lat,
            emit_lon=lon,
            emit_source_row=emit_r,
            emit_source_col=emit_c,
            pixel_res=ENHANCEMENT_PIXEL_RES,
            wind_dir=wind_dirs[i],
        )

        # Scale to Q_target after reprojection
        q_scale = q_targets[i] / Q_REF_KGH
        enhancement_i = enhancement_i_ref * q_scale

        ime_bank = float(plume_ppb.sum()) * ENHANCEMENT_PIXEL_RES ** 2
        ime_emit = float((enhancement_i_ref * pixel_area).sum())
        eta = ime_emit / ime_bank if ime_bank > 0 else 0.0
        q_eff = q_targets[i] * eta

        enhancement_ref_total += enhancement_i_ref
        enhancement_total += enhancement_i
        print(f"  [{i+1}] source at ({emit_r}, {emit_c}), "
              f"Q={q_targets[i]:.0f} kg/h, "
              f"eta={eta:.4f}, "
              f"Q_eff={q_eff:.0f} kg/h, "
              f"wind={wind_speeds[i]:.1f} m/s @ {wind_dirs[i]:.0f} deg, "
              f"active pixels: {(enhancement_i > 0).sum()}")

    # ==================================================================
    # 5. Spectral injection
    # ==================================================================
    print("\nInjecting plume(s) into radiance...")

    emit_wvl  = np.load(EMIT_WAVELENGTHS_FILE)
    emit_fwhm = np.load(EMIT_FWHM_FILE)

    band_indices = get_methane_band_indices(emit_wvl)
    print(f"  CH4 bands selected: {len(band_indices)} "
          f"({emit_wvl[band_indices[0]]:.0f}-{emit_wvl[band_indices[-1]]:.0f} nm)")

    amf = air_mass_factor(sza_mean, vza_mean)
    lut_wvl, lut_t, lut_mr = read_luts(amf)
    print(f"  AMF = {amf:.3f}  |  LUT grid: {len(lut_wvl)} wavelengths, "
          f"{len(lut_mr)} mixing ratios")

    mr_max_lut = float(lut_mr.max())
    max_enhancement = mr_max_lut - MR_BACKGROUND_PPB
    over_mask = enhancement_total > max_enhancement
    clipped = np.sum(over_mask)
    if clipped > 0:
        over_values = enhancement_total[over_mask]
        print(f"  Warning: clamped {clipped} pixels to LUT max "
              f"({max_enhancement:.0f} ppb enhancement, {mr_max_lut:.0f} ppb total)")
        print(f"    Before clamp — min: {over_values.min():.0f}, "
              f"max: {over_values.max():.0f}, mean: {over_values.mean():.0f} ppb")
        print(f"    Excess over limit — max: {(over_values.max() - max_enhancement):.0f}, "
              f"mean: {(over_values.mean() - max_enhancement):.0f} ppb")
        enhancement_total[over_mask] = max_enhancement

    srf_matrix, coverage = build_srf_matrix(emit_wvl, emit_fwhm, lut_wvl, band_indices)

    radiance_original = radiance.copy()

    inject_plume(
        radiance=radiance,
        enhancement=enhancement_total,
        mr_background=MR_BACKGROUND_PPB,
        lut_transmittance=lut_t,
        lut_mixing_ratios=lut_mr,
        srf_matrix=srf_matrix,
        band_indices=band_indices,
        coverage=coverage,
    )
    print("  Injection complete.")

    # ==================================================================
    # 6. Methane retrieval (patch columns only)
    # ==================================================================
    # The retrieval is per-column (independent covariance per crosstrack
    # column), so processing only the patch's columns with all downtrack
    # rows gives identical results to a full-scene run, with ~C_full/PATCH_W
    # speedup (~12× for standard EMIT scenes).
    print("\nRunning methane retrieval on patch columns...")

    r0 = reading_window.row_off
    c0 = reading_window.col_off

    # --- 6a. Load pre-existing retrieval maps from dataset (before) ---
    original_retrievals = {}
    for name in RETRIEVAL_METHODS:
        tif_path = f"{scene_dir}/{name}.tif"
        try:
            with rasterio.open(tif_path) as src:
                original_retrievals[name] = src.read(1, window=reading_window)
            print(f"  Loaded {name}.tif from dataset")
        except Exception:
            print(f"  Warning: {tif_path} not found — skipping {name} (before)")

    # --- 6b. Read only patch columns (all downtrack rows) ---
    swir_idx = np.where((emit_wvl >= WL_MIN_SWIR) & (emit_wvl <= WL_MAX_SWIR))[0]
    print(f"  SWIR bands: {len(swir_idx)} "
          f"({emit_wvl[swir_idx[0]]:.0f}-{emit_wvl[swir_idx[-1]]:.0f} nm)")

    with rasterio.open(f"{scene_dir}/radiance.tif") as src:
        col_window = Window(col_off=c0, row_off=0,
                            width=PATCH_W, height=src.height)
        patch_swir = src.read(indexes=(swir_idx + 1).tolist(),
                              window=col_window)

    # Inject modified radiance into the patch rows
    patch_swir[:, r0:r0 + PATCH_H, :] = radiance[swir_idx]

    patch_swir = np.moveaxis(patch_swir, 0, -1).astype(np.float64)

    engine = MethaneRetrieval(device="cuda", dtype=torch.float64)
    print(f"  Engine ready · template bands: {int(engine.band_mask.sum())} · "
          f"{torch.cuda.get_device_name(0)}")

    rad_tensor = torch.from_numpy(patch_swir)
    del patch_swir

    modified_retrievals = {}
    for name, cfg_cls in RETRIEVAL_METHODS.items():
        print(f"  Running {name.upper()}...", end=" ", flush=True)
        res = engine.retrieve(rad_tensor, cfg_cls(batch_size=RET_BATCH_SIZE),
                              display_pbar=False)
        # Result shape is (D_full, PATCH_W) — crop downtrack to the patch
        modified_retrievals[name] = res.mf.cpu().numpy()[r0:r0 + PATCH_H, :]
        print("done.", flush=True)
        torch.cuda.empty_cache()

    del rad_tensor
    print("  Retrieval complete.")

    # ==================================================================
    # 7. Diagnostics
    # ==================================================================
    print("\nGenerating validation plots...")

    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    plot_enhancement_comparison(
        plume_raws, enhancement_total,
        sources_bank=sources_bank,
        sources_emit=emit_positions,
        save_path=str(output_dir / "plume_comparison.png"),
    )
    plot_orientation_verification(
        enhancement_total, lat, lon,
        source_pixel=emit_positions[0],
        plume_meta=plume_metas[0],
        wind_dir=wind_dirs[0],
        raa=raas[0],
        save_path=str(output_dir / "orientation_verification.png"),
    )
    plot_spectral_comparison(
        radiance_original, radiance, enhancement_total,
        emit_wvl, band_indices,
        save_path=str(output_dir / "validation_spectrum.png"),
    )
    plot_spatial_comparison(
        radiance_original, radiance, enhancement_total, emit_wvl,
        save_path=str(output_dir / "validation_spatial.png"),
    )
    plot_retrieval_comparison(
        original_retrievals, modified_retrievals, enhancement_total,
        plume_mask=(enhancement_ref_total >= 20.0),
        save_path=str(output_dir / "retrieval_comparison.png"),
    )

    # Binary mask at Q_ref (3000 kg/h) with 20 ppb threshold.
    # plume_raws are already at Q_ref (unscaled), and
    # enhancement_ref_total is the reprojected sum at Q_ref.
    plume_bank_combined = enhancement_ref_total
    plot_binary_mask_comparison(
        plume_bank_combined, enhancement_ref_total,
        threshold_ppb=20,
        save_path=str(output_dir / "binary_mask_comparison.png"),
    )

    print(f"  Saved plots to {output_dir}/")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nFATAL: {e}", file=sys.stderr)
        sys.exit(1)