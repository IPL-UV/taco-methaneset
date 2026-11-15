"""T11 · recalcula las fracciones de nube con CloudSEN12 a la resolucion nativa.

Por que existe: las columnas `target:cloud` y `bg:cloud0-3` se calcularon con OmniCloudMask
sobre el chip ya remuestreado a 10 m. Los modelos de nube se entrenaron y validaron a la
resolucion nativa del sensor, y a 10 m sobre Landsat (30 m remuestreado) se disparan: chips
de desierto despejados salen con 0,96-1,00 de nube.

Comprobado: CloudSEN12 `landsat30` a 30 m da 0,00 en cinco chips donde la columna publicada
dice 0,96-1,00 y donde la mascara oficial de MARS (CloudSEN12 sobre el parche nativo) dice
"100 % claro".

Este script no toca nada publicado: escribe un CSV por coleccion.

Uso:
  python v140/t11_cloud.py --coleccion l89 [--limit N]
  python v140/t11_cloud.py --coleccion s2
"""
from __future__ import annotations

import argparse
import pathlib

import numpy as np
import pandas as pd
import rasterio
import torch
import tacoreader
from cloudsen12_models import cloudsen12

BASE = pathlib.Path("/data/databases/METHANESET_TACOS/_v140/T7")
SALIDA = pathlib.Path("/data/databases/METHANESET_TACOS/_v140/T11")
SALIDA.mkdir(parents=True, exist_ok=True)

# Orden de bandas del modelo landsat30 dentro de nuestro chip L89 (B1..B11, 0-indexado).
# Su modelo: B01,B02,B03,B04,B08,B10,B11,B12 -> B1,B2,B3,B4,B5,B9,B6,B7
L89_IDX = [0, 1, 2, 3, 4, 8, 5, 6]
FACTOR = {"l89": 3, "s2": 2}  # 10 m -> 30 m | 20 m


def _modelo(coleccion: str):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nombre = "landsat30" if coleccion == "l89" else "UNetMobV2_V2"
    return cloudsen12.load_model_by_name(nombre, device=dev)


def _a_nativa(x: np.ndarray, coleccion: str) -> np.ndarray:
    """Pasa el chip a la resolucion nativa y a reflectancia 0-1."""
    if coleccion == "s2":
        x = x.astype(np.float32) / 10000.0
    k = FACTOR[coleccion]
    n = (x.shape[1] // k) * k
    return x[:, :n, :n].reshape(x.shape[0], n // k, k, n // k, k).mean(axis=(2, 4)).astype(np.float32)


def _fraccion(modelo, arr: np.ndarray) -> float:
    pred = np.asarray(modelo.predict(arr)).squeeze()
    if pred.size == 0:
        return float("nan")
    return float(np.isin(pred, (1, 2, 3)).mean())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coleccion", required=True, choices=["l89", "s2"])
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()

    modelo = _modelo(a.coleccion)
    ds = tacoreader.load(str(BASE / a.coleccion), cache=False)
    t = ds.data.to_arrow().to_pandas()
    # Las hojas se leen por id: en las 70 muestras de la v1.4.0 el level1 salió ordenado por
    # entrada del zip (ch4 y plume antes de bg1-3), así que la posición no es fiable.
    idx = [int(i) for i in range(len(t))]
    if a.limit:
        idx = idx[: a.limit]
    filas = []
    for n, i in enumerate(idx, 1):
        hojas = ds.read(i)
        sid = str(t.iloc[i]["id"])
        fila = {"id": sid}
        for nombre in ("target", "bg0", "bg1", "bg2", "bg3"):
            with rasterio.open(hojas.read(nombre)) as s:
                x = s.read()
            arr = _a_nativa(x, a.coleccion)
            if a.coleccion == "l89":
                arr = arr[L89_IDX]
            fila[f"{nombre}:cloud"] = _fraccion(modelo, arr)
        filas.append(fila)
        if n % 50 == 0:
            print(f"{n}/{len(idx)}", flush=True)
    out = SALIDA / f"cloud_{a.coleccion}.csv"
    pd.DataFrame(filas).to_csv(out, index=False)
    d = pd.DataFrame(filas)
    cols = [c for c in d.columns if c.endswith(":cloud")]
    print("escrito", out, len(d), "muestras")
    for c in cols:
        print(f"  {c}: mediana {d[c].median():.3f} | >0.35 {int((d[c] > 0.35).sum())}")


if __name__ == "__main__":
    main()
