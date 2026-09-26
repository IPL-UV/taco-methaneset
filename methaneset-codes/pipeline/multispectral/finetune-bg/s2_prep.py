"""Prepares the S2 finetune: proposed table (with sensors from selection_log) + leaf paths.

deep. Output: tabla_propuesta.csv and paths_cesar.csv in the S2 raw.
"""
import pathlib
import re
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import rasterio
import tacoreader

V = "/data/databases/METHANE_DATASETS_TACOv2"
RAW = pathlib.Path("/data/databases/METHANESET_TACOS/methaneset-s2-finetune-bg-raw")
FIN = f"{V}/methaneset-s2-finetune"
BGPREV = f"{V}/methaneset-s2-bg-finetune"
RGN = (3, 2, 7)   # S2 B4,B3,B8
LOG = RAW / "selection_log_methaneset-s2-finetune_S2.csv"
HOJAS = ["target", "reference", "ch4", "plume", "dem"]

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
    cesar = cesar.drop(columns=[c for c in ("type", "internal:gdal_vsi", "stac:time_end",
                                            "stac:time_middle", "stac:centroid") if c in cesar.columns])
    cesar = cesar.rename(columns={"satellite:platform": "target:sensor", "satellite:tile": "target:tile",
                                  "satellite:sza": "target:sza", "satellite:vza": "target:vza"})
    # date0 and tile0 of Cesar's reference
    def _d0(x):
        m = re.search(r"_MSIL1C_(\d{8})T", str(x)) or re.search(r"_L1TP_\d{6}_(\d{8})_", str(x))
        return f"{m.group(1)[:4]}-{m.group(1)[4:6]}-{m.group(1)[6:]}" if m else None
    cesar["background:date0"] = cesar["satellite:background_tile"].map(_d0)
    cesar["background:sensor0"] = cesar["satellite:background_tile"].map(lambda x: str(x).split("_")[0])
    cesar["background:tile0"] = cesar["satellite:background_tile"]
    cesar = cesar.drop(columns=[c for c in ("satellite:background_tile",) if c in cesar.columns])

    sel = pd.read_csv(RAW / "seleccion_methaneset-s2-finetune_S2.csv")
    sel = sel[sel["rank"] <= 3]
    log = pd.read_csv(LOG).set_index("taco_id")

    def sensor_de(tid, fname):
        if tid not in log.index:
            return None
        ids = str(log.loc[tid, "background_gee_ids"]).split("|")
        idx = int(re.search(r"bg(\d+)", fname).group(1))
        return ids[idx].split("/")[-1].split("_")[0] if idx < len(ids) else None

    rows = []
    for tid, g in sel.groupby("taco_id"):
        g = g.sort_values("rank")
        d = {"id": tid}
        for k, r in enumerate(g.itertuples(), 1):
            dd = str(r.date)
            d[f"bg:sensor{k}"] = sensor_de(tid, r.file)
            d[f"bg:date{k}"] = f"{dd[:4]}-{dd[4:6]}-{dd[6:8]}" if len(dd) == 8 else dd
            d[f"bg:cloud{k}"] = r.cloud
            d[f"bg:difference{k}"] = r.difference
            d[f"bg:methane{k}"] = r.methane
        rows.append(d)
    bg = pd.DataFrame(rows)
    out = cesar.merge(bg, on="id", how="left")

    # bg prev lat/lon
    bp = tacoreader.load(BGPREV).data.to_arrow().to_pandas()
    out = out.merge(bp[["id", "emission:lat", "emission:lon"]].drop_duplicates("id"), on="id", how="left")

    # cloud of target and bg0
    tacoreader.use("pandas")
    data = tacoreader.load(FIN).data
    idx = [i for i in range(len(data)) if str(data.iloc[i].get("type", "FOLDER")) == "FOLDER"]
    print("cloud target/reference:", len(idx))
    with ProcessPoolExecutor(max_workers=8, initializer=_init, initargs=(FIN,)) as ex:
        nubes = [r for r in ex.map(_cloud, idx, chunksize=8) if r is not None]
    out = out.merge(pd.DataFrame(nubes), on="id", how="left")

    out.to_csv(RAW / "tabla_propuesta.csv", index=False)
    print("S2 table:", out.shape, "| sensors:", {k: int(out[f'bg:sensor{k}'].notna().sum()) for k in (0, 1, 2, 3)})

    # leaf paths
    filas = {}
    for i in range(len(data)):
        if str(data.iloc[i].get("type", "FOLDER")) != "FOLDER":
            continue
        tid = str(data.iloc[i]["id"])
        sub = data.read(i)
        fila = {"id": tid}
        for j, h in enumerate(HOJAS):
            fila[f"cesar:{h}"] = sub.read(j)
        filas[tid] = fila
    t = pd.DataFrame(filas.values())
    for k in (1, 2, 3):
        g = sel[sel["rank"] == k].set_index("taco_id")
        t[f"bg{k}"] = t["id"].map(lambda i: f"{RAW}/S2/methaneset-s2-finetune/{i}/{g.loc[i,'file']}" if i in g.index else None)
    t.to_csv(RAW / "paths_cesar.csv", index=False)
    print("S2 paths:", t.shape)


if __name__ == "__main__":
    main()
