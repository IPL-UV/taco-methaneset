"""MODULE 20: the PER-PLUME wind published by each institution.

Julio's finding (Aug 26, before upload): the TACO stored only OUR
wind (ERA5-Land, wind.tif layer, scene mean in `meteo:wind_*`), but
BOTH catalogs publish the per-plume wind THEY used to invert
the flux. Without that wind the user cannot reproduce or question the
inversion. It is added, always declaring whose wind is whose.

  - IMEO (portal csv): `wind_u`, `wind_v`, `wind_speed` per plume.
  - Carbon Mapper (annotated catalog): `wind_speed_avg_auto` and
    `wind_direction_avg_auto` (speed + DIRECTION, not components).
    Meteorological convention: direction the wind blows FROM, hence
    u = -speed*sin(dir), v = -speed*cos(dir) when deriving components.

The CM snapshot of 2026-08-25 was downloaded without those fields (they were outside
the fetcher KEEP). It is NOT overwritten: the wind is taken from a new
dated snapshot and joined by plume_id to the already frozen plumes.

Output: assets/data/cross/2026-08-25/plume_wind.parquet
        (one row per published plume, from both catalogs)
Usage:
  /data/users/julio/.conda/envs/deep/bin/python code/v2/20_institutional_wind.py \
        > code/v2/20_institutional_wind.log 2>&1
"""
import pathlib
import re

import numpy as np
import pandas as pd

ROOT = pathlib.Path(os.environ.get("METHANESET_ROOT", "/data/users/julio/Notes/01-Projects/methanset"))
DATA = ROOT / "assets" / "data"
CROSS = DATA / "cross" / "2026-08-25"
CM_FROZEN = DATA / "carbonmapper" / "2026-08-25"
LEVEL0 = pathlib.Path("/data/databases/METHANSET_TACOS/methaneset-emit/METADATA/level0.parquet")
TS_G = re.compile(r"_(\d{8})T(\d{6})_")


def newest(source, check):
    ds = sorted(d for d in (DATA / source).iterdir() if (d / check).exists())
    return ds[-1]


def main():
    md = pd.read_parquet(LEVEL0, columns=["id"])
    ts_pub = {TS_G.search(g).group(1) + "t" + TS_G.search(g).group(2): g
              for g in md["id"]}

    # ---------- IMEO: the portal csv already carries the per-plume wind ----------
    imeo_dir = newest("imeo", "unep_methanedata_detected_plumes.csv")
    im = pd.read_csv(imeo_dir / "unep_methanedata_detected_plumes.csv")
    im = im[im.satellite.str.contains("EMIT", na=False)].copy()
    im["granule_ts"] = im.tile.str.extract(r"EMIT_L1B_RAD_001_(\d{8}T\d{6})")[0].str.lower()
    im = im[im.granule_ts.isin(ts_pub)]
    imeo = pd.DataFrame({
        "catalog": "imeo",
        "granule": im.granule_ts.map(ts_pub),
        "plume_key": im.source_name,          # IMEO is identified by EMITTER
        "wind_u": im.wind_u.astype(float),
        "wind_v": im.wind_v.astype(float),
        "wind_speed": im.wind_speed.astype(float),
        "wind_dir": np.nan,
    })
    print(f"IMEO: {len(imeo)} plumes from published scenes · "
          f"with wind {imeo.wind_speed.notna().sum()}")

    # ---------- CM: wind from a new snapshot, joined to the frozen plumes ----------
    cm_new_dir = newest("carbonmapper", "plumes.parquet")
    cm_new = pd.read_parquet(cm_new_dir / "plumes.parquet")
    if "wind_speed_avg_auto" not in cm_new.columns:
        raise SystemExit(f"snapshot {cm_new_dir.name} has no wind: "
                         "run code/fetch_carbonmapper.py (KEEP already patched)")
    print(f"CM: wind snapshot {cm_new_dir.name} ({len(cm_new)} plumes)")

    man = pd.read_parquet(CM_FROZEN / "manifest.parquet")
    src = pd.read_parquet(CM_FROZEN / "sources.parquet")[["plume_id", "source_name"]]
    man = man.merge(src, on="plume_id", how="left")
    man["source_name"] = man.source_name.fillna(man.plume_id)
    man = man[man.granule_ts.isin(ts_pub)]
    man = man.merge(cm_new[["plume_id", "wind_speed_avg_auto",
                            "wind_direction_avg_auto"]], on="plume_id", how="left")
    spd = man.wind_speed_avg_auto.astype(float)
    dr = man.wind_direction_avg_auto.astype(float)
    # meteorological convention: direction the wind blows FROM
    rad = np.radians(dr)
    cm = pd.DataFrame({
        "catalog": "cm",
        "granule": man.granule_ts.map(ts_pub),
        "plume_key": man.source_name,
        "wind_u": -spd * np.sin(rad),
        "wind_v": -spd * np.cos(rad),
        "wind_speed": spd,
        "wind_dir": dr,
    })
    print(f"CM: {len(cm)} plumes from published scenes · "
          f"with wind {cm.wind_speed.notna().sum()} "
          f"({cm.wind_speed.isna().sum()} without data in the new snapshot)")

    out = pd.concat([imeo, cm], ignore_index=True)
    out.to_parquet(CROSS / "plume_wind.parquet", index=False)
    print(f"\n-> {CROSS/'plume_wind.parquet'} ({len(out)} rows)")

    # quick comparison against OUR ERA5-Land (for the note/slide)
    md2 = pd.read_parquet(LEVEL0, columns=["id", "meteo:wind_speed"])
    m = out.merge(md2, left_on="granule", right_on="id", how="left")
    for cat, g in m.groupby("catalog"):
        g = g.dropna(subset=["wind_speed", "meteo:wind_speed"])
        d = g.wind_speed - g["meteo:wind_speed"]
        print(f"{cat}: institutional wind median {g.wind_speed.median():.2f} m/s "
              f"vs our ERA5-Land {g['meteo:wind_speed'].median():.2f} · "
              f"median difference {d.median():+.2f} m/s · "
              f"|dif|>1 m/s in {(d.abs()>1).mean():.0%}")


if __name__ == "__main__":
    main()
