"""Step 4 · Real sensor of each background by querying GEE (L8/L9 alternate and parity is not enough).

Fills background:sensor0/1/2/3 in tabla_propuesta.csv. sensor0 comes from Cesar's tile;
sensor1/2/3 are queried by (WRS_PATH, WRS_ROW, date) and cached.
"""
import re
from concurrent.futures import ThreadPoolExecutor

import ee
import pandas as pd
import tacoreader

ee.Initialize(project="ee-contrerasnetk")
F = "/data/databases/METHANESET_TACOS/methaneset-l89-finetune-bg-raw/tabla_propuesta.csv"

t = pd.read_csv(F)
info = tacoreader.load("/data/databases/METHANE_DATASETS_TACOv2/methaneset-l89-finetune").data.to_arrow().to_pandas()
tile_by = dict(zip(info["id"], info["satellite:tile"]))

_cache = {}


def sat_of(pathrow, date):
    key = (pathrow, date)
    if key in _cache:
        return _cache[key]
    path, row = int(pathrow[:3]), int(pathrow[3:])
    d = f"{date[:4]}-{date[4:6]}-{date[6:8]}"
    d0 = (pd.Timestamp(d) - pd.Timedelta(days=2)).strftime("%Y-%m-%d")
    d2 = (pd.Timestamp(d) + pd.Timedelta(days=3)).strftime("%Y-%m-%d")
    out = None
    for sat in ("LC08", "LC09"):
        for col in ("C02/T1", "C02/T2"):
            try:
                n = (ee.ImageCollection(f"LANDSAT/{sat}/{col}")
                     .filterDate(d0, d2)
                     .filter(ee.Filter.eq("WRS_PATH", path))
                     .filter(ee.Filter.eq("WRS_ROW", row)).size().getInfo())
            except Exception:  # noqa: BLE001
                n = 0
            if n:
                out = sat
                break
        if out:
            break
    _cache[key] = out
    return out


def pr_of(tid):
    m = re.match(r"LC\d\d_L1TP_(\d{6})_", str(tile_by.get(tid, "")))
    return m.group(1) if m else None


# sensor0 from the tile of Cesar's background
t["background:sensor0"] = [str(tile_by.get(i, " ")).split("_")[0] if str(tile_by.get(i, "")).startswith(("LC08", "LC09", "S2")) else None for i in t["id"]]

tareas = []
for idx in t.index:
    tid = t.at[idx, "id"]
    for k in (1, 2, 3):
        d = t.at[idx, f"background:date{k}"]
        if isinstance(d, str) and "-" in d:
            tareas.append((idx, k, pr_of(tid), d.replace("-", "")))

print("GEE queries:", len(tareas))


def job(x):
    idx, k, pr, d = x
    return idx, k, sat_of(pr, d) if pr else None


with ThreadPoolExecutor(max_workers=16) as ex:
    res = list(ex.map(job, tareas))

for k in (1, 2, 3):
    t[f"background:sensor{k}"] = None
for idx, k, s in res:
    t.at[idx, f"background:sensor{k}"] = s

t.to_csv(F, index=False)
print("sensors filled:", {k: int(t[f'background:sensor{k}'].notna().sum()) for k in (0, 1, 2, 3)})
print(t[["id", "background:sensor0", "background:sensor1", "background:sensor2", "background:sensor3"]].head(5).to_string(index=False))
