"""Step 5a (deep) · Paths of Cesar's leaves and of our bg, to build the TACO.

Writes a CSV with id + path of each leaf (vsi inside the tacozip) and the path of our bg1/2/3.
"""
import re

import pandas as pd
import tacoreader

tacoreader.use("pandas")
FIN = "/data/databases/METHANE_DATASETS_TACOv2/methaneset-l89-finetune"
RAW = "/data/databases/METHANESET_TACOS/methaneset-l89-finetune-bg-raw"
HOJAS = ["target", "reference", "ch4", "plume", "dem"]

d = tacoreader.load(FIN).data
sel = pd.concat([pd.read_csv(f) for f in
                 [f"{RAW}/seleccion_methaneset-l89-finetune_LC08.csv",
                  f"{RAW}/seleccion_methaneset-l89-finetune_LC09.csv"]], ignore_index=True)

filas = {}
for i in range(len(d)):
    if str(d.iloc[i].get("type", "FOLDER")) != "FOLDER":
        continue
    tid = str(d.iloc[i]["id"])
    sub = d.read(i)
    fila = {"id": tid}
    for j, h in enumerate(HOJAS):
        fila[f"cesar:{h}"] = sub.read(j)
    filas[tid] = fila

t = pd.DataFrame(filas.values())
for k in (1, 2, 3):
    g = sel[sel["rank"] == k].set_index("taco_id")
    t[f"bg{k}"] = t["id"].map(lambda i: f"{RAW}/{g.loc[i,'platform']}/methaneset-l89-finetune/{i}/{g.loc[i,'file']}" if i in g.index else None)
t.to_csv(f"{RAW}/paths_cesar.csv", index=False)
print("rows:", len(t), "| columns:", list(t.columns))
print("leaves notna:", {c: int(t[c].notna().sum()) for c in t.columns if c != "id"})
