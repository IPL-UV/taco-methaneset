"""
qmax_analysis.py

Compute Q_max (maximum rescalable flux rate) across EMIT scenes,
wind speeds, and wind directions.

The LUT ceiling depends on the scene's AMF:
    max_enh = max(LUT(amf)) - CH4_BACKGROUND
    Q_max = max_enh * Q_REF / peak_emit

For each wind speed and each RAA bin (30° intervals), reports Q_max
at 100%, 99%, and 95% plume coverage, plus the % of plumes that
can be rescaled to Q_TARGET.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import rasterio
import warnings
from multiprocessing import Pool
from rasterio.errors import NotGeoreferencedWarning

from config import DIR_EMIT_SCENES, DIR_PLUME_BANK, ENHANCEMENT_PIXEL_RES
from bank import MethaneBank
from emit_bckg import get_emit_background
from trans_enhmap import map_to_emit
from lut import read_luts, air_mass_factor

warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)

# ── Config ──────────────────────────────────────────────────────────
Q_REF = 3000.0            # kg/h — bank reference flux rate
CH4_BACKGROUND = 1900.0   # ppb
Q_TARGET = 10000.0        # kg/h — flux rate to check clamping

N_SCENES = 25
N_PLUMES_PER_COMBO = 20   # plumes per (scene, ws, wd) combination
WIND_DIRS = list(range(0, 360, 30))  # 12 directions
N_WORKERS = 5
PATCH_H = 100
PATCH_W = 100
SOURCE_TYPE = "multi"

OUTPUT_CSV = Path(__file__).resolve().parent / "qmax_analysis.csv"


# ── Scene loading ───────────────────────────────────────────────────
def load_scenes():
    """Load N_SCENES clean EMIT patches with lat/lon, using get_emit_background."""
    scenes = []
    seen = set()
    while len(scenes) < N_SCENES:
        _, row, window = get_emit_background(
            DIR_EMIT_SCENES, patch_height=PATCH_H, patch_width=PATCH_W,
        )
        if row["id"] in seen:
            continue
        seen.add(row["id"])
        scene_dir = Path(row["internal:gdal_vsi"]).parent
        with rasterio.open(str(scene_dir / "latlon.tif")) as src:
            lat = src.read(1, window=window)
            lon = src.read(2, window=window)

        sza = float(row["target:sza"])
        vza = float(row["target:vza"])
        amf = air_mass_factor(sza, vza)
        _, _, lut_mr = read_luts(amf)
        max_enh = float(lut_mr.max()) - CH4_BACKGROUND

        scenes.append({
            "scene_id": row["id"],
            "sza": sza,
            "saa": float(row["target:saa"]),
            "vza": vza,
            "amf": amf,
            "max_enh": max_enh,
            "lat": lat,
            "lon": lon,
        })
        print(f"  [{len(scenes)}/{N_SCENES}] SZA={sza:.1f}  "
              f"VZA={vza:.1f}  AMF={amf:.2f}  max_enh={max_enh:.0f} ppb")
    return scenes


# ── Worker ──────────────────────────────────────────────────────────
def process_scene(args):
    """Project plumes onto one EMIT scene. Runs in a worker process."""
    idx, scene = args
    bank = MethaneBank(DIR_PLUME_BANK)
    results = []

    max_enh = scene["max_enh"]

    for ws in bank.wind_speeds:
        for wd in WIND_DIRS:
            raa = (90 - scene["saa"] - wd) % 360

            try:
                metas = bank.query(
                    wind_speed=ws, sza=scene["sza"], raa=raa,
                    source_type=SOURCE_TYPE, meta=True,
                    n=N_PLUMES_PER_COMBO,
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
                    plume, meta["source_row"], meta["source_col"],
                    scene["lat"], scene["lon"],
                    PATCH_H // 2, PATCH_W // 2,
                    ENHANCEMENT_PIXEL_RES, float(wd),
                )
                peak_emit = float(enh.max())
                if peak_emit <= 0:
                    continue

                q_max = max_enh * Q_REF / peak_emit

                results.append({
                    "wind_speed": ws,
                    "wind_dir": wd,
                    "raa": meta["raa"],
                    "sza": meta["sza"],
                    "amf": scene["amf"],
                    "max_enh": max_enh,
                    "peak_bank": peak_bank,
                    "peak_emit": peak_emit,
                    "dilution": peak_emit / peak_bank,
                    "q_max": q_max,
                    "plume_uid": meta["plume_uid"],
                    "scene_id": scene["scene_id"],
                })

    print(f"  Scene {idx + 1}/{N_SCENES} done — {len(results)} samples",
          flush=True)
    return results


# ── Analysis ────────────────────────────────────────────────────────
def analyse(csv_path):
    """Read CSV and print Q_max percentiles by wind speed and by RAA bin."""
    df = pd.read_csv(csv_path)
    print(f"\nTotal samples: {len(df)}")
    print(f"CH4 background = {CH4_BACKGROUND:.0f} ppb  |  Q_REF = {Q_REF:.0f} kg/h")
    print(f"max_enh range: {df['max_enh'].min():.0f} – {df['max_enh'].max():.0f} ppb "
          f"(varies with AMF)\n")

    # ── By wind speed ───────────────────────────────────────────────
    print("=" * 62)
    print("Q_max by wind speed (kg/h)")
    print("=" * 62)
    print(f"{'ws (m/s)':>10}  {'n':>6}  {'100%':>10}  {'99%':>10}  {'95%':>10}")
    print("-" * 62)

    for ws in sorted(df["wind_speed"].unique()):
        sub = df[df["wind_speed"] == ws]["q_max"]
        print(f"{ws:>10.1f}  {len(sub):>6d}  {sub.min():>10.0f}  "
              f"{np.percentile(sub, 1):>10.0f}  {np.percentile(sub, 5):>10.0f}")

    q_all = df["q_max"]
    print("-" * 62)
    print(f"{'Global':>10}  {len(q_all):>6d}  {q_all.min():>10.0f}  "
          f"{np.percentile(q_all, 1):>10.0f}  {np.percentile(q_all, 5):>10.0f}")

    # ── By RAA bin ──────────────────────────────────────────────────
    print(f"\n{'=' * 62}")
    print("Q_max by RAA bin (kg/h)")
    print("=" * 62)
    print(f"{'RAA bin':>12}  {'n':>6}  {'100%':>10}  {'99%':>10}  {'95%':>10}")
    print("-" * 62)

    for raa_lo in np.arange(0, 360, 30):
        raa_hi = raa_lo + 30
        mask = (df["raa"] >= raa_lo) & (df["raa"] < raa_hi)
        sub = df.loc[mask, "q_max"]
        if len(sub) == 0:
            continue
        print(f"{raa_lo:>5.0f}–{raa_hi:<5.0f}  {len(sub):>6d}  "
              f"{sub.min():>10.0f}  {np.percentile(sub, 1):>10.0f}  "
              f"{np.percentile(sub, 5):>10.0f}")

    print("-" * 62)
    print(f"{'Global':>12}  {len(q_all):>6d}  {q_all.min():>10.0f}  "
          f"{np.percentile(q_all, 1):>10.0f}  {np.percentile(q_all, 5):>10.0f}")

    # ── Clamping at Q_TARGET ────────────────────────────────────────
    print(f"\n{'=' * 50}")
    print(f"% of plumes with Q_max >= {Q_TARGET:.0f} kg/h")
    print("=" * 50)

    print(f"\n{'ws (m/s)':>10}  {'n':>6}  {'% valid':>10}")
    print("-" * 32)
    for ws in sorted(df["wind_speed"].unique()):
        sub = df[df["wind_speed"] == ws]["q_max"]
        pct = 100.0 * (sub >= Q_TARGET).sum() / len(sub)
        print(f"{ws:>10.1f}  {len(sub):>6d}  {pct:>9.2f}%")

    pct_all = 100.0 * (q_all >= Q_TARGET).sum() / len(q_all)
    print("-" * 32)
    print(f"{'Global':>10}  {len(q_all):>6d}  {pct_all:>9.2f}%")

    print(f"\n{'RAA bin':>12}  {'n':>6}  {'% valid':>10}")
    print("-" * 34)
    for raa_lo in np.arange(0, 360, 30):
        raa_hi = raa_lo + 30
        mask = (df["raa"] >= raa_lo) & (df["raa"] < raa_hi)
        sub = df.loc[mask, "q_max"]
        if len(sub) == 0:
            continue
        pct = 100.0 * (sub >= Q_TARGET).sum() / len(sub)
        print(f"{raa_lo:>5.0f}–{raa_hi:<5.0f}  {len(sub):>6d}  {pct:>9.2f}%")

    print("-" * 34)
    print(f"{'Global':>12}  {len(q_all):>6d}  {pct_all:>9.2f}%")
    print()


# ── Main ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Loading scenes...")
    scenes = load_scenes()

    n_ws = 11  # approximate
    print(f"\nQ_max analysis: {N_SCENES} scenes × {n_ws} ws × "
          f"{len(WIND_DIRS)} wd × {N_PLUMES_PER_COMBO} plumes/combo")
    print(f"Workers: {N_WORKERS}\n")

    scene_args = list(enumerate(scenes))

    all_results = []
    BATCH_SIZE = 5
    n_batches = (N_SCENES + BATCH_SIZE - 1) // BATCH_SIZE
    t_start = time.time()

    for batch_start in range(0, N_SCENES, BATCH_SIZE):
        batch = scene_args[batch_start: batch_start + BATCH_SIZE]
        batch_num = batch_start // BATCH_SIZE + 1
        print(f"--- Batch {batch_num}/{n_batches} "
              f"(scenes {batch_start + 1}–{batch_start + len(batch)}) ---")

        with Pool(N_WORKERS) as pool:
            batch_results = pool.map(process_scene, batch)

        for scene_results in batch_results:
            all_results.extend(scene_results)

        df = pd.DataFrame(all_results)
        df.to_csv(OUTPUT_CSV, index=False)

        elapsed = time.time() - t_start
        frac_done = batch_num / n_batches
        eta_time = elapsed / frac_done * (1 - frac_done)
        print(f"  Saved {len(all_results)} samples | "
              f"Elapsed: {elapsed / 60:.1f} min | "
              f"ETA: {eta_time / 60:.1f} min\n")

    elapsed = time.time() - t_start
    print(f"Done. {len(all_results)} samples saved to {OUTPUT_CSV} "
          f"({elapsed / 60:.1f} min)\n")

    analyse(OUTPUT_CSV)