#!/usr/bin/env python
"""
analyze_conservation.py

Quantifies the methane conservation ratio (η = IME_EMIT / IME_bank)
across the enhancement map reprojection from the plume bank grid
(regular, 20 m) to the EMIT sensor grid (irregular, ~60 m).

Independent variables
---------------------
  - wind_speed : all values available in the bank
  - wind_dir   : 0°–350° in 10° steps (rotation angle in map_to_emit)
  - scene      : N random EMIT scenes (captures sensor geometry variability)

Protocol
--------
For each (scene, wind_speed), ONE plume is queried from the bank.
That same plume is projected with every wind_dir, so the rotation
effect is isolated from plume shape variability.
A linearity check (η vs. Q) verifies that the ratio is scale-invariant.
"""

import sys
from pathlib import Path

# Adjust PIPELINE_DIR if your pipeline modules (config.py, bank.py, etc.)
# live in a different location relative to this script.
PIPELINE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PIPELINE_DIR))

import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import rasterio
import tacoreader
import matplotlib.pyplot as plt
import warnings
from rasterio.errors import NotGeoreferencedWarning
from rasterio.windows import Window

from config import (
    DIR_EMIT_SCENES,
    DIR_PLUME_BANK,
    ENHANCEMENT_PIXEL_RES,
    Q_REF_KGH,
)
from bank import MethaneBank
from plume_injection_test.injection.trans_enhmap_prefilter import map_to_emit, compute_pixel_areas

warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)

# ── Configuration ────────────────────────────────────────────────────
PATCH_H = 200           # pixels — large enough for full plume
PATCH_W = 200
N_SCENES = 30           # number of EMIT scenes to sample
WIND_DIR_STEP = 10      # degrees
SOURCE_TYPE = "multi"   # plume source type in the bank
OUTPUT_DIR = Path("analysis_output")
N_WORKERS = 10
SEED = 42


# ── Worker function (module-level for pickling) ──────────────────────

def _evaluate_one_task(args):
    """Process one (scene, wind_speed): project the same plume at all wind_dirs.

    Receives pre-loaded arrays so no file I/O happens inside workers.
    Returns a list of (wind_dir, eta) tuples.
    """
    (plume_ppb, source_row, source_col,
     lat, lon, pixel_area,
     wind_dirs, patch_h, patch_w) = args

    ime_bank = float(plume_ppb.sum()) * ENHANCEMENT_PIXEL_RES ** 2
    if ime_bank == 0:
        return [(wd, float("nan")) for wd in wind_dirs]

    results = []
    for wd in wind_dirs:
        enhancement = map_to_emit(
            enhancement=plume_ppb,
            source_row=source_row,
            source_col=source_col,
            emit_lat=lat,
            emit_lon=lon,
            emit_source_row=patch_h // 2,
            emit_source_col=patch_w // 2,
            pixel_res=ENHANCEMENT_PIXEL_RES,
            wind_dir=float(wd),
        )
        # pixel_area may contain NaN at invalid pixels; treat as zero area
        ime_emit = float(np.nansum(enhancement * pixel_area))
        results.append((wd, ime_emit / ime_bank))

    return results


# ── Helpers ──────────────────────────────────────────────────────────

def load_valid_scenes(data_dir, patch_h, patch_w, n_max):
    """Load up to *n_max* scenes whose centre patch has valid lat/lon.

    Returns list of dict with keys: id, sza, saa, vza, lat, lon,
    pixel_area, mean_lat.
    """
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

            with rasterio.open(scene_dir / "latlon.tif") as src:
                lat = src.read(1, window=window)
                lon = src.read(2, window=window)

            if not (np.isfinite(lat).all() and np.isfinite(lon).all()):
                continue

            pixel_area = compute_pixel_areas(lat, lon)

            scenes.append({
                "id": row["id"],
                "sza": row["target:sza"],
                "saa": row["target:saa"],
                "vza": row["target:vza"],
                "lat": lat,
                "lon": lon,
                "pixel_area": pixel_area,
                "mean_lat": float(np.mean(lat)),
            })
            print(f"  [{len(scenes):>2}/{n_max}] {row['id']}  "
                  f"lat={float(np.mean(lat)):+.1f}  "
                  f"sza={row['target:sza']:.1f}")
        except Exception:
            continue

    return scenes


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
    if len(scenes) == 0:
        print("FATAL: no valid scenes found.", file=sys.stderr)
        sys.exit(1)
    print(f"  {len(scenes)} valid scenes loaded.\n")

    # ── 2. Pre-query plumes and build task list ──────────────────────
    # All bank queries and raster reads happen here (main process),
    # so randomness is deterministic and no I/O in workers.
    print("Pre-loading plumes from bank...")
    tasks = []       # arguments for _evaluate_one_task
    task_meta = []   # metadata to attach to results afterwards

    for si, scene in enumerate(scenes):
        for ws in wind_speeds:
            try:
                meta = bank.query(
                    sza=scene["sza"],
                    wind_speed=ws,
                    source_type=SOURCE_TYPE,
                    meta=True,
                )
            except ValueError:
                continue

            with rasterio.open(meta["gdal_vsi"]) as src:
                plume_ppb = src.read(1).astype(np.float32)

            tasks.append((
                plume_ppb, meta["source_row"], meta["source_col"],
                scene["lat"], scene["lon"], scene["pixel_area"],
                wind_dirs, PATCH_H, PATCH_W,
            ))
            task_meta.append({
                "scene_idx": si,
                "scene_id": scene["id"],
                "scene_lat": scene["mean_lat"],
                "wind_speed": ws,
                "plume_id": meta["id"],
            })

    n_tasks = len(tasks)
    n_evals = n_tasks * len(wind_dirs)
    print(f"  {n_tasks} tasks prepared ({n_evals} evaluations total)\n")

    # ── 3. Parallel sweep ────────────────────────────────────────────
    print(f"Running with {N_WORKERS} workers...")
    records = []
    done = 0

    with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
        futures = {
            executor.submit(_evaluate_one_task, task): i
            for i, task in enumerate(tasks)
        }

        for future in as_completed(futures):
            i = futures[future]
            meta_i = task_meta[i]
            wd_etas = future.result()

            for wd, eta in wd_etas:
                records.append({
                    "scene_idx": meta_i["scene_idx"],
                    "scene_id": meta_i["scene_id"],
                    "scene_lat": meta_i["scene_lat"],
                    "wind_speed": meta_i["wind_speed"],
                    "wind_dir": wd,
                    "plume_id": meta_i["plume_id"],
                    "eta": eta,
                })

            done += 1
            if done % 50 == 0 or done == n_tasks:
                elapsed = time.time() - t0
                print(f"  {done:>4}/{n_tasks} tasks  "
                      f"({100 * done / n_tasks:5.1f}%)  "
                      f"{elapsed:.0f}s elapsed")

    df = pd.DataFrame(records)
    df.to_csv(OUTPUT_DIR / "conservation_results.csv", index=False)
    print(f"\n  {len(df)} results saved to {OUTPUT_DIR}/conservation_results.csv")

    # ── 4. Linearity check (single-threaded, 20 evals) ───────────────
    print("\nLinearity check (η vs. Q)...")
    eta_vs_q = []
    q_values = np.linspace(100, 15000, 20)

    if tasks:
        # Reuse the first task's plume and scene
        (ppb_ref, sr, sc, lat_ref, lon_ref, pa_ref,
         _, ph, pw) = tasks[0]

        for q in q_values:
            ppb_scaled = ppb_ref * (q / Q_REF_KGH)
            enhancement = map_to_emit(
                enhancement=ppb_scaled,
                source_row=sr, source_col=sc,
                emit_lat=lat_ref, emit_lon=lon_ref,
                emit_source_row=ph // 2, emit_source_col=pw // 2,
                pixel_res=ENHANCEMENT_PIXEL_RES, wind_dir=0.0,
            )
            ime_bank = float(ppb_scaled.sum()) * ENHANCEMENT_PIXEL_RES ** 2
            ime_emit = float(np.nansum(enhancement * pa_ref))
            eta_vs_q.append(ime_emit / ime_bank if ime_bank > 0 else float("nan"))

    # ── 5. Figures ───────────────────────────────────────────────────
    print("\nGenerating figures...")

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "figure.dpi": 150,
    })

    mean_eta = df["eta"].mean()
    std_eta = df["eta"].std()
    med_eta = df["eta"].median()
    p5 = df["eta"].quantile(0.05)
    p95 = df["eta"].quantile(0.95)

    # ---- Figure 1: global η distribution ----
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(df["eta"], bins=60, edgecolor="black", linewidth=0.3,
            color="#4C72B0", alpha=0.85)
    ax.axvline(mean_eta, color="red", linestyle="--", linewidth=1.2,
               label=f"Mean = {mean_eta:.4f}")
    ax.axvline(1.0, color="gray", linestyle=":", alpha=0.7,
               label="Perfect conservation")
    ax.set_xlabel("Conservation ratio η")
    ax.set_ylabel("Count")
    ax.set_title(
        f"η distribution  (n = {len(df)},  "
        f"σ = {std_eta:.4f},  "
        f"P5/P95 = {p5:.4f} / {p95:.4f})"
    )
    ax.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig1_eta_distribution.png",
                dpi=300, bbox_inches="tight")
    plt.close()

    # ---- Figure 2: η vs. wind speed (boxplot) ----
    ws_unique = sorted(df["wind_speed"].unique())
    ws_groups = [df.loc[df["wind_speed"] == ws, "eta"].values
                 for ws in ws_unique]

    fig, ax = plt.subplots(figsize=(8, 4))
    bp = ax.boxplot(ws_groups,
                    labels=[f"{ws:.1f}" for ws in ws_unique],
                    patch_artist=True, showfliers=False,
                    medianprops=dict(color="black", linewidth=1.2))
    for patch in bp["boxes"]:
        patch.set_facecolor("#4C72B0")
        patch.set_alpha(0.7)
    ax.axhline(1.0, color="gray", linestyle=":", alpha=0.7)
    ax.set_xlabel("Wind speed (m s$^{-1}$)")
    ax.set_ylabel("Conservation ratio η")
    ax.set_title("η vs. wind speed")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig2_eta_vs_windspeed.png",
                dpi=300, bbox_inches="tight")
    plt.close()

    # ---- Figure 3: η vs. wind direction (polar) ----
    wd_unique = sorted(df["wind_dir"].unique())
    wd_means = np.array([df.loc[df["wind_dir"] == wd, "eta"].mean()
                         for wd in wd_unique])
    wd_stds = np.array([df.loc[df["wind_dir"] == wd, "eta"].std()
                        for wd in wd_unique])

    theta = np.radians(np.append(wd_unique, wd_unique[0]))
    means_closed = np.append(wd_means, wd_means[0])
    stds_closed = np.append(wd_stds, wd_stds[0])

    fig, ax = plt.subplots(figsize=(6, 6),
                           subplot_kw={"projection": "polar"})
    ax.plot(theta, means_closed, "o-", color="#4C72B0", markersize=3,
            linewidth=1.2)
    ax.fill_between(theta,
                     means_closed - stds_closed,
                     means_closed + stds_closed,
                     alpha=0.2, color="#4C72B0")
    ax.set_theta_zero_location("E")
    ax.set_theta_direction(1)
    ax.set_title("η vs. wind direction (rotation angle)", pad=20)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig3_eta_vs_winddir.png",
                dpi=300, bbox_inches="tight")
    plt.close()

    # ---- Figure 4: η vs. scene latitude ----
    scene_stats = (df.groupby("scene_idx")
                   .agg(scene_lat=("scene_lat", "first"),
                        eta_mean=("eta", "mean"),
                        eta_std=("eta", "std"))
                   .reset_index())

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.errorbar(scene_stats["scene_lat"], scene_stats["eta_mean"],
                yerr=scene_stats["eta_std"],
                fmt="o", color="#4C72B0", capsize=3, markersize=5,
                elinewidth=1)
    ax.axhline(1.0, color="gray", linestyle=":", alpha=0.7)
    ax.set_xlabel("Scene mean latitude (°)")
    ax.set_ylabel("Mean η per scene")
    ax.set_title("η vs. scene latitude")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig4_eta_vs_latitude.png",
                dpi=300, bbox_inches="tight")
    plt.close()

    # ---- Figure 5: linearity check ----
    if eta_vs_q:
        linearity_delta = max(eta_vs_q) - min(eta_vs_q)
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(q_values, eta_vs_q, "o-", color="#4C72B0", markersize=5)
        ax.axhline(eta_vs_q[0], color="red", linestyle="--", alpha=0.5,
                   label=f"η = {eta_vs_q[0]:.6f}")
        ax.set_xlabel("Emission rate Q (kg h$^{-1}$)")
        ax.set_ylabel("Conservation ratio η")
        ax.set_title("Linearity verification  (η should be constant)")
        ax.legend(frameon=True)
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "fig5_linearity_check.png",
                    dpi=300, bbox_inches="tight")
        plt.close()
    else:
        linearity_delta = float("nan")

    # ── 6. Summary ───────────────────────────────────────────────────
    elapsed = time.time() - t0
    print("\n" + "=" * 60)
    print("CONSERVATION ANALYSIS SUMMARY")
    print("=" * 60)
    print(f"  Samples          : {len(df)}")
    print(f"  Scenes           : {len(scenes)}")
    print(f"  Wind speeds      : {len(wind_speeds)}  "
          f"({wind_speeds[0]:.1f} – {wind_speeds[-1]:.1f} m/s)")
    print(f"  Wind directions  : {len(wind_dirs)}  "
          f"(0° – {wind_dirs[-1]}°, step {WIND_DIR_STEP}°)")
    print(f"  Workers          : {N_WORKERS}")
    print(f"  η mean ± std     : {mean_eta:.4f} ± {std_eta:.4f}")
    print(f"  η median         : {med_eta:.4f}")
    print(f"  η P5 / P95       : {p5:.4f} / {p95:.4f}")
    print(f"  η min / max      : {df['eta'].min():.4f} / {df['eta'].max():.4f}")
    print(f"  Linearity (max Δη): {linearity_delta:.2e}")
    print(f"  Elapsed time     : {elapsed:.0f} s")
    print(f"  Figures saved to   {OUTPUT_DIR}/")
    print("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nFATAL: {e}", file=sys.stderr)
        sys.exit(1)