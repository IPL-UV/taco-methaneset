"""
fig_reprojection.py

2×3 figure for the paper: bank enhancement (top) and EMIT
reprojection (bottom) for three plumes at different wind speeds.
No flux rate rescaling — all at Q_REF = 3000 kg/h.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import rasterio
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import warnings
from rasterio.errors import NotGeoreferencedWarning

from config import DIR_EMIT_SCENES, DIR_PLUME_BANK, ENHANCEMENT_PIXEL_RES
from bank import MethaneBank
from emit_bckg import get_emit_background
from trans_enhmap import map_to_emit, compute_pixel_areas

warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)

PATCH_H = 100
PATCH_W = 100
# Three wind speeds: low, medium, high
TARGET_WIND_SPEEDS = [2.0, 5.0, 10.0]
SOURCE_TYPES = ["area", "multi", "multi"]
OUTPUT_PATH = "plume_reprojection.pdf"

base = plt.get_cmap("plasma")
plasma_claro = LinearSegmentedColormap.from_list(
    "plasma_claro", base(np.linspace(0.15, 1.0, 256))
)


def main():
    bank = MethaneBank(DIR_PLUME_BANK)

    # One different scene per column
    cases = []
    seen_ids = set()

    for i, (ws, src_type) in enumerate(zip(TARGET_WIND_SPEEDS, SOURCE_TYPES)):
        print(f"\nLoading scene {i + 1}/3...")
        while True:
            _, scene_row, window = get_emit_background(
                DIR_EMIT_SCENES, patch_height=PATCH_H, patch_width=PATCH_W,
            )
            if scene_row["id"] not in seen_ids:
                seen_ids.add(scene_row["id"])
                break

        scene_dir = Path(scene_row["internal:gdal_vsi"]).parent
        sza = float(scene_row["target:sza"])
        saa = float(scene_row["target:saa"])
        vza = float(scene_row["target:vza"])

        with rasterio.open(str(scene_dir / "latlon.tif")) as src:
            lat = src.read(1, window=window)
            lon = src.read(2, window=window)
        pixel_area = compute_pixel_areas(lat, lon)

        ws_snap = min(bank.wind_speeds, key=lambda x: abs(x - ws))
        wd = np.random.randint(0, 360)
        raa = (90 - saa - wd) % 360

        meta = bank.query(
            sza=sza, raa=raa, wind_speed=ws_snap,
            source_type=src_type, meta=True,
        )

        with rasterio.open(meta["gdal_vsi"]) as src:
            plume_bank = src.read(1).astype(np.float32)

        enh_emit = map_to_emit(
            plume_bank, meta["source_row"], meta["source_col"],
            lat, lon, PATCH_H // 2, PATCH_W // 2,
            ENHANCEMENT_PIXEL_RES, float(wd),
        )

        ime_bank = float(plume_bank.sum()) * ENHANCEMENT_PIXEL_RES ** 2
        ime_emit = float(np.nansum(enh_emit * pixel_area))
        eta = ime_emit / ime_bank if ime_bank > 0 else float("nan")

        cases.append({
            "scene_id": scene_row["id"],
            "sza": sza,
            "saa": saa,
            "vza": vza,
            "ws": ws_snap,
            "wd": wd,
            "raa": meta["raa"],
            "sza_bank": meta["sza"],
            "source_type": src_type,
            "plume_uid": meta["plume_uid"],
            "source_row": meta["source_row"],
            "source_col": meta["source_col"],
            "plume_bank": plume_bank,
            "enh_emit": enh_emit,
            "eta": eta,
            "peak_bank": float(plume_bank.max()),
            "peak_emit": float(enh_emit.max()),
        })

        print(f"  Scene: {scene_row['id']}")
        print(f"  SZA={sza:.1f}  SAA={saa:.1f}  VZA={vza:.1f}")
        print(f"  ws={ws_snap} m/s  wd={wd}°  raa={meta['raa']:.0f}°  "
              f"η={eta:.4f}  peak: {plume_bank.max():.0f} → {enh_emit.max():.0f} ppb")

    # ── Figure ───────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))

    for col, case in enumerate(cases):
        ax_bank = axes[0, col]
        ax_emit = axes[1, col]

        # -- Top: bank enhancement --
        im_bank = ax_bank.imshow(case["plume_bank"], cmap=plasma_claro,
                                  vmin=0, interpolation="nearest")
        cb = fig.colorbar(im_bank, ax=ax_bank, shrink=0.8)
        cb.set_label("ppb", fontsize=12)
        cb.ax.tick_params(labelsize=12)

        ax_bank.set_title(
            f"Wind: {case['ws']:.1f} m s$^{{-1}}$\n"
            f"SZA: {case['sza_bank']:.0f}°   RAA: {case['raa']:.0f}°",
            fontsize=12,
        )
        ax_bank.set_xticks([])
        ax_bank.set_yticks([])

        # -- Bottom: EMIT reprojection --
        im_emit = ax_emit.imshow(case["enh_emit"], cmap=plasma_claro,
                                  vmin=0, interpolation="nearest")
        cb_emit = fig.colorbar(im_emit, ax=ax_emit, shrink=0.8)
        cb_emit.set_label("ppb", fontsize=12)
        cb_emit.ax.tick_params(labelsize=12)

        ax_emit.set_title(
            f"SZA: {case['sza']:.1f}°  SAA: {case['saa']:.1f}°\n"
            f"Wind dir: {case['wd']}°   "
            f"IME conserved: {case['eta'] * 100:.2f}%",
            fontsize=12,
        )
        ax_emit.set_xticks([])
        ax_emit.set_yticks([])

    # Row labels
    axes[0, 0].annotate("Bank", xy=(-0.15, 0.5),
                         xycoords="axes fraction", fontsize=12,
                         fontweight="bold", rotation=90,
                         ha="center", va="center")
    axes[1, 0].annotate("EMIT", xy=(-0.15, 0.5),
                         xycoords="axes fraction", fontsize=12,
                         fontweight="bold", rotation=90,
                         ha="center", va="center")


    plt.tight_layout(rect=[0.03, 0, 1, 1])
    plt.savefig("plume_reprojection.png", dpi=300, bbox_inches="tight")
    plt.savefig("plume_reprojection.pdf", bbox_inches="tight")
    plt.close()
    print(f"\nSaved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()