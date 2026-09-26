"""Step 3 · Proposed finetune table with the extra bg + the cloud of the target and the reference.

Joins Cesar's table with the background selection (bg1/2/3) and adds two new columns:
  target:cloud    -> cloud fraction (OmniCloudMask) of Cesar's target
  background:cloud0 -> cloud fraction (OmniCloudMask) of Cesar's reference (the bg0)
"""
from __future__ import annotations
import glob
import pathlib
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import rasterio
import tacoreader

RAW = pathlib.Path("/data/databases/METHANESET_TACOS/methaneset-l89-finetune-bg-raw")
FIN = "/data/databases/METHANE_DATASETS_TACOv2/methaneset-l89-finetune"
RGN = (3, 2, 4)  # Landsat B4,B3,B5

try:
    from omnicloudmask import predict_from_array as _ocm
except Exception:  # noqa: BLE001
    _ocm = None

_DATA = None


def _init(fin):
    tacoreader.use("pandas")
    globals()["_DATA"] = tacoreader.load(fin).data


def _cloud(i):
    row = _DATA.iloc[i]
    if str(row.get("type", "FOLDER")) != "FOLDER":
        return None
    try:
        sub = _DATA.read(i)
        with rasterio.open(sub.read(0)) as s:
            tgt = s.read()
        with rasterio.open(sub.read(1)) as s:
            ref = s.read()
    except Exception:  # noqa: BLE001
        return None

    def frac(a):
        arr = np.stack([a[RGN[0]], a[RGN[1]], a[RGN[2]]]).astype(np.float32)
        try:
            pred = np.asarray(_ocm(arr)).squeeze()
        except Exception:  # noqa: BLE001
            return np.nan
        return float(np.mean(np.isin(pred, (1, 2, 3)))) if pred.size else np.nan

    return {"id": str(row["id"]), "target:cloud": frac(tgt), "background:cloud0": frac(ref)}


def main():
    cesar = tacoreader.load(FIN).data.to_arrow().to_pandas()
    cesar = cesar.drop(columns=[c for c in ("type", "internal:gdal_vsi") if c in cesar.columns])

    sel = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(RAW / "seleccion_*.csv")))], ignore_index=True)
    sel = sel[sel["rank"] <= 3]
    cols_bg = ["background:sensor1", "background:date1", "background:cloud1", "background:difference1", "background:methane1",
               "background:sensor2", "background:date2", "background:cloud2", "background:difference2", "background:methane2",
               "background:sensor3", "background:date3", "background:cloud3", "background:difference3", "background:methane3"]
    rows = []
    for tid, g in sel.groupby("taco_id"):
        g = g.sort_values("rank")
        d = {"id": tid}
        for k, r in enumerate(g.itertuples(), 1):
            d[f"background:sensor{k}"] = r.platform
            dd = str(r.date)
            d[f"background:date{k}"] = f"{dd[:4]}-{dd[4:6]}-{dd[6:8]}" if len(dd) == 8 else dd
            d[f"background:cloud{k}"] = r.cloud
            d[f"background:difference{k}"] = r.difference
            d[f"background:methane{k}"] = r.methane
        rows.append(d)
    bg = pd.DataFrame(rows)
    out = cesar.merge(bg, on="id", how="left")

    # cloud of Cesar's target and reference (OmniCloudMask), in parallel
    import tacoreader as tr
    tr.use("pandas")
    data = tr.load(FIN).data
    idx = [i for i in range(len(data)) if str(data.iloc[i].get("type", "FOLDER")) == "FOLDER"]
    print("computing cloud of target and reference:", len(idx), "sites")
    with ProcessPoolExecutor(max_workers=8, initializer=_init, initargs=(FIN,)) as ex:
        nubes = [r for r in ex.map(_cloud, idx, chunksize=8) if r is not None]
    out = out.merge(pd.DataFrame(nubes), on="id", how="left")
    # drop rows: empty target in bg_prev (qa:target_empty) and without emission lat/lon
    _bp = tacoreader.load("/data/databases/METHANE_DATASETS_TACOv2/methaneset-l89-bg-finetune").data.to_arrow().to_pandas()
    _empty = set(_bp.loc[_bp["qa:target_empty"] == True, "id"])
    out = out[~out["id"].isin(_empty)]
    out = out[out["emission:lat"].notna() & out["emission:lon"].notna()]

    # date of Cesar's reference (bg0), taken from satellite:background_tile
    import re as _re
    def _d0(x):
        m = _re.search(r"_L1TP_\d{6}_(\d{8})_", str(x)) or _re.search(r"_L1C_(\d{8})T", str(x))
        return f"{m.group(1)[:4]}-{m.group(1)[4:6]}-{m.group(1)[6:]}" if m else None
    out["background:date0"] = out["satellite:background_tile"].map(_d0)

    # emission lat/lon from your bg_prev; drop stac:centroid/time_end/time_middle
    bp = tacoreader.load("/data/databases/METHANE_DATASETS_TACOv2/methaneset-l89-bg-finetune").data.to_arrow().to_pandas()
    out = out.merge(bp[["id", "emission:lat", "emission:lon"]].drop_duplicates("id"), on="id", how="left")
    for c in ("stac:time_end", "stac:time_middle", "stac:centroid"):
        if c in cesar.columns:
            cesar = cesar.drop(columns=c)

    cols_bg0 = ["background:date0", "background:cloud0"]
    cols = list(cesar.columns) + cols_bg0 + cols_bg + ["target:cloud", "emission:lat", "emission:lon"]
    out = out[cols]
    out.to_csv(RAW / "tabla_propuesta.csv", index=False)
    print("proposed table:", out.shape)
    print("columns:", len(cols), "| target:cloud notna:", out["target:cloud"].notna().sum(),
          "| reference:cloud notna:", out["background:cloud0"].notna().sum())


if __name__ == "__main__":
    main()
