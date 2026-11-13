"""T3 · elegir las 3 referencias de cada muestra perdida (v1.4.0).

`02_seleccionar.py` ya hace el trabajo: abre el target, coregistra cada candidata del pool,
puntua diferencia MARS, nube y metano, y devuelve las 3 mejores. Pero carga el TACO con
tacoreader, y el build de Cesar no se puede abrir con la version instalada.

Este envoltorio le da lo que espera: un `.data` que se indexa como un DataFrame y que ademas
sabe devolver el target. El target se lee en memoria desde el zip, porque `/vsizip` no resuelve
estos `.tacozip` con el GDAL de este entorno.

Uso:
  python v140/t3_select.py --coleccion l89 --platform LC09
"""
import argparse
import importlib.util
import io
import pathlib
import sys
import types
import zipfile

import pandas as pd
import tacoreader

BASE = pathlib.Path(__file__).resolve().parent
RAIZ = "/data/databases/METHANE_DATASETS_TACO"
W = pathlib.Path("/data/databases/METHANESET_TACOS/_v140/T2")


class _Leaf:
    def __init__(self, zip_path, member):
        self._zip = zip_path
        self._member = member

    def read(self, _i=0):
        with zipfile.ZipFile(self._zip) as zf:
            return io.BytesIO(zf.read(self._member))


class _DF(pd.DataFrame):
    """DataFrame de level0 con un `read(i)` que devuelve el target de la fila i."""

    _metadata = ["zipmap", "col"]

    @property
    def _constructor(self):
        return _DF

    def read(self, i):
        row = self.iloc[i]
        zip_name = self.zipmap.get(str(row["id"]))
        if zip_name is None:
            raise FileNotFoundError(f"{row['id']} sin source_file")
        return _Leaf(f"{RAIZ}/{self.col}/{zip_name}", f"DATA/{row['id']}/emission")


_CACHE: dict = {}


def _shim(col):
    if col not in _CACHE:
        p = pathlib.Path(f"{RAIZ}/{col}")
        df = pd.read_parquet(p / ".tacocat" / "level0.parquet")
        if "satellite:platform" not in df.columns:
            df = df.rename(columns={"satellite:satellite": "satellite:platform"})
        l1 = pd.read_parquet(p / ".tacocat" / "level1.parquet")
        l1 = l1[l1["internal:relative_path"].str.endswith("/emission")]
        zmap = dict(zip(l1["internal:relative_path"].str.split("/").str[0].astype(str),
                        l1["internal:source_file"].astype(str)))
        out = _DF(df)
        out["type"] = "FOLDER"
        out.zipmap = zmap
        out.col = col
        _CACHE[col] = types.SimpleNamespace(data=out)
        print(f"[shim] {col}: {len(out)} filas, {len(zmap)} targets")
    return _CACHE[col]


_ORIGINAL = tacoreader.load


def _load(path):
    p = str(path).rstrip("/")
    if p.startswith(RAIZ):
        return _shim(p.rsplit("/", 1)[-1])
    return _ORIGINAL(p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coleccion", required=True, choices=["l89", "s2"])
    ap.add_argument("--platform", required=True, choices=["LC08", "LC09", "S2"])
    ap.add_argument("--top", type=int, default=3)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--cloud-max", type=float, default=0.50)
    args = ap.parse_args()

    tacoreader.load = _load
    spec = importlib.util.spec_from_file_location("seleccionar", BASE.parent / "02_seleccionar.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["seleccionar"] = mod
    spec.loader.exec_module(mod)

    sys.argv = [
        "02_seleccionar.py",
        "--taco", f"{RAIZ}/{args.coleccion}",
        "--raw", str(W / "raw"),
        "--platform", args.platform,
        "--top", str(args.top),
        "--workers", str(args.workers),
        "--cloud-max", str(args.cloud_max),
    ]
    mod.main()


if __name__ == "__main__":
    main()
