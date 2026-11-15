"""Recalcula la nube de los chips reparados con CloudSEN12.

Los chips que llegaron vacíos traían una columna de nube medida sobre píxeles en cero
(`target:cloud` = 0 por casualidad y `bg:cloud0` = NaN). Aquí se mide la nube sobre el
chip publicado y se corrige `cloud_<fam>.csv`; después `t21_level0.py` la propaga al
catálogo y a cada zip.

Uso:
  python t24_nube_chips.py --fam s2
  python t24_nube_chips.py --fam l89
"""

import argparse
import pathlib

import numpy as np
import pandas as pd
import rasterio
import torch
from cloudsen12_models import cloudsen12

CHIPS = pathlib.Path("/data/databases/METHANESET_TACOS/_v140/T14")
T11 = pathlib.Path("/data/databases/METHANESET_TACOS/_v140/T11")
IDX_L89 = [0, 1, 2, 3, 4, 8, 5, 6]
FACTOR = {"l89": 3, "s2": 2}


def a_nativa(x: np.ndarray, fam: str) -> np.ndarray:
    if fam == "s2":
        x = x.astype(np.float32) / 10000.0
    k = FACTOR[fam]
    n = (x.shape[1] // k) * k
    return x[:, :n, :n].reshape(x.shape[0], n // k, k, n // k, k).mean(axis=(2, 4)).astype(np.float32)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fam", required=True, choices=["l89", "s2"])
    a = ap.parse_args()

    modelo = cloudsen12.load_model_by_name(
        "landsat30" if a.fam == "l89" else "UNetMobV2_V2",
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    csv = T11 / f"cloud_{a.fam}.csv"
    nube = pd.read_csv(csv).set_index("id")
    l0 = pd.read_parquet(pathlib.Path(f"/data/databases/METHANESET_TACOS/_v140/T16/{a.fam}")
                         / ".tacocat" / "level0.parquet").set_index("id")

    cambios = []
    for p in sorted(CHIPS.glob("*.tif")):
        sid, hoja = p.stem.split("__")
        if sid not in l0.index:
            continue  # chip de la otra familia
        with rasterio.open(p) as s:
            arr = a_nativa(s.read(), a.fam)
        if a.fam == "l89":
            arr = arr[IDX_L89]
        pred = np.asarray(modelo.predict(arr)).squeeze()
        frac = float(np.isin(pred, (1, 2, 3)).mean()) if pred.size else float("nan")
        col = {"target": "target:cloud", "bg0": "bg0:cloud"}.get(hoja, f"{hoja}:cloud")
        antes = nube.loc[sid, col] if sid in nube.index else float("nan")
        if sid in nube.index:
            nube.loc[sid, col] = frac
        cambios.append((sid[:8], hoja, antes, frac))
        print(f"  {sid[:8]} {hoja}: {antes} -> {frac:.3f}", flush=True)

    nube.reset_index().to_csv(csv, index=False, float_format="%.6f")
    d = pd.DataFrame(cambios, columns=["id", "hoja", "antes", "despues"])
    print(f"{len(d)} chips | nube media {d['despues'].mean():.3f} | >0.35 {int((d['despues'] > 0.35).sum())}")
    print("escrito", csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
