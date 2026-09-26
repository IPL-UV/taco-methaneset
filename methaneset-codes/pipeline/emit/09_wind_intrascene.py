"""MODULE 09 (analysis): how much does the wind vary WITHIN an EMIT scene?

Context (Julio, Aug 25): TACO v1 stores ONE wind per granule (the one from the
IMEO catalog). For synthetic injection it matters to know whether that is
enough or whether the wind changes within the ~75 km of the scene (plumes
injected at different corners should curve differently).

Method: hourly ERA5-Land (ECMWF/ERA5_LAND/HOURLY, ~9-11 km) via Earth Engine
(project ee-contrerasnetk), at the scene hour, sampled over the granule bbox.
Figure: wind arrows over the real mf of the scene, plus the single IMEO
catalog arrow for comparison. Stats: intra-scene speed and direction
dispersion.

Usage:
  /data/users/julio/.conda/envs/deep/bin/python code/v2/09_wind_intrascene.py \
        > code/v2/09_wind_intrascene.log 2>&1
"""
import pathlib
import sys

import ee
import numpy as np
import pandas as pd
import rasterio

_theme = pathlib.Path(os.environ.get("METHANESET_THEME", "/data/users/julio/Notes/01-Projects/methanset/assets/slides/theme"))
if _theme.exists():
    sys.path.insert(0, str(_theme))
import figstyle  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

SCENE = "EMIT_L1B_RAD_001_20220810T064957_2222205_033"
DATA_DIR = pathlib.Path("/data/databases/METHANE_DATASETS_TACOv2/methaneset-emit/"
                        "methaneset-emit/DATA") / SCENE
L0 = pathlib.Path("/data/databases/METHANE_DATASETS_TACOv2/methaneset-emit/"
                  "methaneset-emit/METADATA/level0.parquet")
OUT = (pathlib.Path(__file__).resolve().parent.parent.parent
       / "assets" / "images" / "viento-intra-escena.png")


def main():
    ee.Initialize(project="ee-contrerasnetk")

    with rasterio.open(DATA_DIR / "latlon.tif") as s:
        lat, lon = s.read(1), s.read(2)
    with rasterio.open(DATA_DIR / "mf.tif") as s:
        mf = s.read(1)
    ok = mf > -9000
    w, e = float(lon[ok].min()), float(lon[ok].max())
    s_, n = float(lat[ok].min()), float(lat[ok].max())

    ts = SCENE.split("_")[4]
    hour = f"{ts[:4]}-{ts[4:6]}-{ts[6:8]}T{ts[9:11]}:00:00"
    col = (ee.ImageCollection("ECMWF/ERA5_LAND/HOURLY")
           .filterDate(hour, f"{ts[:4]}-{ts[4:6]}-{ts[6:8]}T{int(ts[9:11])+1:02d}:00:00")
           .select(["u_component_of_wind_10m", "v_component_of_wind_10m"]))
    region = ee.Geometry.Rectangle([w - 0.05, s_ - 0.05, e + 0.05, n + 0.05])
    grid = col.getRegion(region, scale=11132).getInfo()
    df = pd.DataFrame(grid[1:], columns=grid[0]).dropna()
    df = df.rename(columns={"u_component_of_wind_10m": "u",
                            "v_component_of_wind_10m": "v"})
    df["speed"] = np.hypot(df.u, df.v)
    df["dir"] = np.degrees(np.arctan2(df.u, df.v)) % 360
    print(f"scene {SCENE} · ERA5-Land hour: {hour} · nodes: {len(df)}")
    print(f"speed: {df.speed.min():.1f} to {df.speed.max():.1f} m/s "
          f"(mean {df.speed.mean():.1f}, std {df.speed.std():.2f})")
    dirs = np.radians(df["dir"])
    R = np.hypot(np.mean(np.sin(dirs)), np.mean(np.cos(dirs)))
    circ_std = np.degrees(np.sqrt(-2 * np.log(R)))
    print(f"direction: circular dispersion {circ_std:.0f} degrees "
          f"(range {df['dir'].min():.0f} to {df['dir'].max():.0f})")

    l0 = pd.read_parquet(L0)
    row = l0[l0.id == SCENE].iloc[0]
    cu, cv = float(row["meteo:wind_u"]), float(row["meteo:wind_v"])
    print(f"IMEO catalog wind (1 value/granule): u={cu:.2f} v={cv:.2f} "
          f"({np.hypot(cu, cv):.1f} m/s)")

    figstyle.apply()
    fig, ax = plt.subplots(figsize=(8.6, 7.2), constrained_layout=True)
    sub = ok & (np.arange(mf.size).reshape(mf.shape) % 7 == 0)
    vmax = np.percentile(mf[ok], 99.5)
    ax.scatter(lon[sub], lat[sub], c=np.clip(mf[sub], 0, vmax), s=2,
               marker="s", cmap="Greys", vmax=vmax, rasterized=True)
    q = ax.quiver(df.longitude, df.latitude, df.u, df.v, df.speed,
                  cmap="viridis", scale=45, width=0.005, zorder=3)
    fig.colorbar(q, ax=ax, label="ERA5-Land wind speed [m/s]", shrink=0.8)
    ax.quiver([w + 0.06], [n - 0.03], [cu], [cv], color=figstyle.PALETTE["red"],
              scale=45, width=0.007, zorder=4)
    ax.text(w + 0.06, n - 0.015, "IMEO catalog\n(one value per granule)",
            fontsize=9, color=figstyle.PALETTE["red"])
    ax.set_xlabel("longitude [deg]"); ax.set_ylabel("latitude [deg]")
    ax.set_aspect(1 / np.cos(np.radians((s_ + n) / 2)))
    ax.grid(False)
    figstyle.save(fig, OUT)
    print("figure ->", OUT)


if __name__ == "__main__":
    main()
