"""Arma la tabla de reemplazos: qué ranura de qué muestra se sustituye y por qué chip.

Una ranura se sustituye si su nube supera el 35 % de MARS o si cae fuera de ±120 días.
Se rellena con las referencias elegidas por 02_select.py (ordenadas por similitud).

Salida: /tmp/opencode/cambios.csv con fam,id,slot,ruta,sensor,date,cloud
"""

import argparse
import pathlib
import re

import pandas as pd
import pyarrow.parquet as pq

T7 = pathlib.Path("/data/databases/METHANESET_TACOS/_v140/T7")
RAW = pathlib.Path("/data/databases/METHANESET_TACOS/_v140/T12/raw")
CLOUD = pathlib.Path("/data/databases/METHANESET_TACOS/_v140/T11")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/tmp/opencode/cambios.csv")
    a = ap.parse_args()

    filas = []
    resumen = []
    for fam, plataformas in (("l89", ["LC08", "LC09"]), ("s2", ["S2"])):
        l0 = pq.read_table(T7 / fam / ".tacocat" / "level0.parquet").to_pandas().set_index("id")
        nube = pd.read_csv(CLOUD / f"cloud_{fam}.csv").set_index("id")
        sel = pd.concat([pd.read_csv(p) for p in
                         sorted(RAW.glob(f"seleccion_{fam}_*.csv"))]) if list(RAW.glob(f"seleccion_{fam}_*.csv")) else pd.DataFrame()
        if len(sel):
            sel = sel.rename(columns={sel.columns[0]: "id"})
            sel = sel.sort_values([ "id", "rank" ])
        por_sitio = {i: g for i, g in sel.groupby("id")} if len(sel) else {}

        hechas = 0
        for sid, fila in l0.iterrows():
            objetivo = pd.Timestamp(str(fila["stac:time_start"])[:10])
            malas = []
            for k in (1, 2, 3):
                c = float(nube.loc[sid, f"bg{k}:cloud"]) if sid in nube.index else 0.0
                f = pd.Timestamp(str(fila[f"bg:date{k}"])[:10])
                if c > 0.35 or abs((f - objetivo).days) > 120:
                    malas.append(k)
            if not malas:
                continue
            cand = por_sitio.get(sid)
            if cand is None or not len(cand):
                resumen.append((fam, sid[:8], len(malas), 0, "sin selección"))
                continue
            cand = cand.head(len(malas))
            for slot, (_, r) in zip(malas, cand.iterrows()):
                nombre = str(r["file"])
                fecha = re.search(r"_(\d{8})\.tif$", nombre).group(1)
                sensor = nombre.split("__")[1].split("_")[0]
                plat = "S2" if fam == "s2" else sensor
                ruta = RAW / plat / fam / sid / nombre
                if not ruta.exists():
                    resumen.append((fam, sid[:8], len(malas), 0, f"falta {ruta.name}"))
                    continue
                filas.append({"fam": fam, "id": sid, "slot": slot, "ruta": str(ruta),
                              "sensor": sensor, "date": f"{fecha[:4]}-{fecha[4:6]}-{fecha[6:]}",
                              "cloud": float(r["cloud"]), "difference": float(r["difference"]),
                              "granule": nombre})
                hechas += 1
            resumen.append((fam, sid[:8], len(malas), len(cand), ""))

    d = pd.DataFrame(filas)
    d.to_csv(a.out, index=False, float_format="%.6f")
    print("reemplazos:", len(d), "| muestras:", d["id"].nunique() if len(d) else 0)
    if len(d):
        print(d.groupby("fam").size().to_string())
        print("nube de las nuevas: max", round(d["cloud"].max(), 3),
              "| >0.05:", int((d["cloud"] > 0.05).sum()))
    r = pd.DataFrame(resumen, columns=["fam", "id", "malas", "conseguidas", "nota"])
    faltan = r[(r["conseguidas"] < r["malas"])]
    print(f"muestras incompletas: {len(faltan)}")
    if len(faltan):
        print(faltan.head(15).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
