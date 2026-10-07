"""
dilution_analysis.py

Measure the dilution factor (peak_emit / peak_bank) across a
representative sample of plumes and EMIT scenes.  Scenes are
selected stratified by SZA to cover the full geometry range.
Results are saved periodically to avoid data loss.
"""

import sys
import time
from pathlib import Path

# Add parent directory to path so injection modules can be imported
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import rasterio
import tacoreader
from multiprocessing import Pool
from rasterio.windows import Window
from scipy.ndimage import binary_dilation
from trans_enhmap import map_to_emit
from config import DIR_EMIT_SCENES, DIR_PLUME_BANK, ENHANCEMENT_PIXEL_RES
from bank import MethaneBank
from lut import read_luts, air_mass_factor

N_SCENES = 25
N_PLUMES_PER_COMBO = 20
WIND_DIRS = list(range(0, 360, 30))  # 12 directions, every 30 deg
N_WORKERS = 5
PATCH_H = 100
PATCH_W = 100
MAX_PLACEMENT_ATTEMPTS = 1000
N_FALLBACKS = 10

OUTPUT_CSV = Path(__file__).resolve().parent / "dilution_analysis.csv"


def select_scenes_with_fallbacks(data_dir, n_scenes, n_fallbacks):
    """Select scenes stratified by SZA, each with fallback alternatives.

    Returns a list of n_scenes lists, where each inner list contains
    up to n_fallbacks scene dicts sorted by proximity to the target SZA.
    """
    tacoreader.use("pandas")
    taco_df = tacoreader.load(str(data_dir)).data
    n_total = len(taco_df)

    # Extract SZA for all scenes manually (TacoDataFramePandas doesn't
    # support sort_values or reset_index)
    sza_values = np.array([
        float(taco_df.iloc[i]["target:sza"]) for i in range(n_total)
    ])

    # Sorted order by SZA
    sorted_order = np.argsort(sza_values)

    # Pick n_scenes evenly spaced positions in the sorted order
    targets = np.linspace(0, len(sorted_order) - 1, n_scenes, dtype=int)

    scene_groups = []
    for t in targets:
        # Window of fallbacks centered on target, clamped to bounds
        lo = max(0, t - n_fallbacks // 2)
        hi = min(len(sorted_order), lo + n_fallbacks)
        lo = max(0, hi - n_fallbacks)

        target_sza = sza_values[sorted_order[t]]
        window_indices = sorted_order[lo:hi]
        dists = np.abs(sza_values[window_indices] - target_sza)

        group = []
        for idx in window_indices[np.argsort(dists)]:
            row = taco_df.iloc[int(idx)]
            group.append({
                "gdal_vsi": row["internal:gdal_vsi"],
                "sza_mean": float(row["target:sza"]),
                "saa_mean": float(row["target:saa"]),
                "vza_mean": float(row["target:vza"]),
                "id": row["id"],
            })
        scene_groups.append(group)

    szas = [g[0]["sza_mean"] for g in scene_groups]
    print(f"  Selected {n_scenes} SZA targets from {n_total} scenes "
          f"({n_fallbacks} fallbacks each)")
    print(f"  SZA range: {min(szas):.1f} - {max(szas):.1f} deg\n")

    return scene_groups


def find_clean_patch(scene_dict, patch_h, patch_w, max_attempts):
    """Find a plume-free patch in a specific EMIT scene.

    Returns Window or None if no valid patch found.
    """
    scene_dir = Path(scene_dict["gdal_vsi"]).parent
    dilation_kernel = np.ones((11, 11), dtype=bool)

    try:
        with rasterio.open(str(scene_dir / "plume_imeo.tif")) as f_imeo, \
             rasterio.open(str(scene_dir / "plume_cm.tif")) as f_cm:
            plume_mask = (f_imeo.read(1) > 0) | (f_cm.read(1) > 0)
    except Exception as e:
        return None

    plume_mask = binary_dilation(plume_mask, structure=dilation_kernel)

    h, w = plume_mask.shape
    max_y = h - patch_h
    max_x = w - patch_w
    if max_y <= 0 or max_x <= 0:
        return None

    for _ in range(max_attempts):
        y = np.random.randint(0, max_y + 1)
        x = np.random.randint(0, max_x + 1)
        if not plume_mask[y:y + patch_h, x:x + patch_w].any():
            return Window(col_off=x, row_off=y, width=patch_w, height=patch_h)

    return None


def process_scene(args):
    """Project plumes onto one EMIT scene. Runs in a worker process."""
    scene_idx, scene_group = args

    # Each worker loads its own bank (can't share across processes)
    bank = MethaneBank(DIR_PLUME_BANK)

    # Try each scene in the group until a clean patch is found
    scene_dict = None
    window = None
    for candidate in scene_group:
        window = find_clean_patch(candidate, PATCH_H, PATCH_W, MAX_PLACEMENT_ATTEMPTS)
        if window is not None:
            scene_dict = candidate
            break

    if scene_dict is None:
        print(f"  Scene {scene_idx + 1}/{N_SCENES} skipped — "
              f"no clean patch in any of {len(scene_group)} fallbacks", flush=True)
        return []

    scene_dir = Path(scene_dict["gdal_vsi"]).parent
    sza = scene_dict["sza_mean"]
    saa = scene_dict["saa_mean"]
    vza = scene_dict["vza_mean"]

    print(f"  Scene {scene_idx + 1}/{N_SCENES} started "
          f"(SZA={sza:.1f}, SAA={saa:.1f}, {scene_dict['id']})", flush=True)

    with rasterio.open(str(scene_dir / "latlon.tif")) as src:
        lat = src.read(1, window=window)
        lon = src.read(2, window=window)

    amf = air_mass_factor(sza, vza)
    _, _, lut_mr = read_luts(amf)
    max_enh = float(lut_mr.max()) - 1900.0

    results = []

    for ws in bank.wind_speeds:
        for wd in WIND_DIRS:
            raa = (90 - saa - wd) % 360

            try:
                metas = bank.query(
                    wind_speed=ws, sza=sza, raa=raa,
                    source_type="multi",
                    meta=True, n=N_PLUMES_PER_COMBO,
                )
            except ValueError:
                continue

            if not isinstance(metas, list):
                metas = [metas]

            for meta in metas:
                with rasterio.open(meta["gdal_vsi"]) as src:
                    plume = src.read(1).astype(np.float32)

                peak_bank = float(plume.max())
                if peak_bank <= 0:
                    continue

                enh = map_to_emit(
                    enhancement=plume,
                    source_row=meta["source_row"],
                    source_col=meta["source_col"],
                    emit_lat=lat, emit_lon=lon,
                    emit_source_row=PATCH_H // 2, emit_source_col=PATCH_W // 2,
                    pixel_res=ENHANCEMENT_PIXEL_RES,
                    wind_dir=wd,
                )
                peak_emit = float(enh.max())

                results.append({
                    "wind_speed": ws,
                    "wind_dir": wd,
                    "raa": meta["raa"],
                    "sza": meta["sza"],
                    "saa": saa,
                    "amf": amf,
                    "peak_bank": peak_bank,
                    "peak_emit": peak_emit,
                    "dilution": peak_emit / peak_bank,
                    "max_enh_lut": max_enh,
                    "plume_uid": meta["plume_uid"],
                    "scene_id": scene_dict["id"],
                })

    print(f"  Scene {scene_idx + 1}/{N_SCENES} done — "
          f"{len(results)} projections", flush=True)
    return results


if __name__ == "__main__":
    n_ws = 11  # approximate; actual count depends on bank
    print(f"Dilution analysis: {N_SCENES} scenes x "
          f"{n_ws} ws x {len(WIND_DIRS)} wd x "
          f"{N_PLUMES_PER_COMBO} plumes/combo")
    print(f"Workers: {N_WORKERS}\n")

    print("Selecting scenes stratified by SZA...")
    scene_groups = select_scenes_with_fallbacks(
        DIR_EMIT_SCENES, N_SCENES, N_FALLBACKS
    )

    scene_args = [(i, scene_groups[i]) for i in range(len(scene_groups))]

    all_results = []
    BATCH_SIZE = 5
    n_batches = (N_SCENES + BATCH_SIZE - 1) // BATCH_SIZE
    t_start = time.time()

    for batch_start in range(0, N_SCENES, BATCH_SIZE):
        batch = scene_args[batch_start : batch_start + BATCH_SIZE]
        batch_num = batch_start // BATCH_SIZE + 1
        print(f"--- Batch {batch_num}/{n_batches} "
              f"(scenes {batch_start+1}-{batch_start+len(batch)}) ---")

        with Pool(N_WORKERS) as pool:
            batch_results = pool.map(process_scene, batch)

        for scene_results in batch_results:
            all_results.extend(scene_results)

        # Save after each batch
        df = pd.DataFrame(all_results)
        df.to_csv(OUTPUT_CSV, index=False)

        elapsed = time.time() - t_start
        frac_done = batch_num / n_batches
        eta = elapsed / frac_done * (1 - frac_done)
        print(f"  Saved {len(all_results)} samples | "
              f"Elapsed: {elapsed/60:.1f} min | "
              f"ETA: {eta/60:.1f} min\n")

    elapsed = time.time() - t_start
    print(f"Done. {len(all_results)} samples in {OUTPUT_CSV} "
          f"({elapsed/60:.1f} min)")