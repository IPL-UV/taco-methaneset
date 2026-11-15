"""Recupera el satélite real de cada referencia S2 (D1: la columna trae el datatake).

Consulta el catálogo público STAC de Earth Search por tile MGRS (una consulta por tile y
año, porque pide rangos cortos), arma el mapa (tile, fecha) -> plataforma y lo aplica a
cada bg1..bg3.

Uso:
  python t17_satelites.py --taco .../T7/s2 --out /tmp/opencode/satelites_s2.csv
"""

import argparse
import concurrent.futures as cf
import datetime as dt
import json
import pathlib
import urllib.request

import pandas as pd
import pyarrow.parquet as pq

URL = "https://earth-search.aws.element84.com/v1/search"
CAMPOS = {"include": ["id", "properties.datetime", "properties.platform"]}


def consulta(tile: str, d0: str, d1: str, intentos: int = 3) -> list:
    """Una página de resultados para un tile y un rango corto."""
    cuerpo = {
        "collections": ["sentinel-2-l1c"],
        "datetime": f"{d0}T00:00:00Z/{d1}T23:59:59Z",
        "limit": 1000,
        "query": {"grid:code": {"eq": f"MGRS-{tile}"}},
        "fields": CAMPOS,
    }
    for _ in range(intentos):
        req = urllib.request.Request(URL, data=json.dumps(cuerpo).encode(),
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.load(r)
        except Exception:  # noqa: BLE001
            continue
    return {}


def buscar_tile(tile: str, anios: list) -> list:
    salida = []
    for anio in anios:
        d = consulta(tile, f"{anio}-01-01", f"{anio}-12-31")
        for f in d.get("features", []):
            p = f.get("properties", {})
            fecha = str(p.get("datetime") or "")[:10]
            plataforma = str(p.get("platform") or "")
            if fecha and plataforma:
                salida.append((fecha, plataforma))
        siguiente = None
        for enlace in d.get("links", []):
            if enlace.get("rel") == "next":
                siguiente = (enlace.get("body") or {}).get("token")
        vueltas = 0
        while siguiente and vueltas < 10:
            vueltas += 1
            cuerpo = {
                "collections": ["sentinel-2-l1c"],
                "datetime": f"{anio}-01-01T00:00:00Z/{anio}-12-31T23:59:59Z",
                "limit": 1000,
                "query": {"grid:code": {"eq": f"MGRS-{tile}"}},
                "fields": CAMPOS,
                "token": siguiente,
            }
            req = urllib.request.Request(URL, data=json.dumps(cuerpo).encode(),
                                         headers={"Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(req, timeout=60) as r:
                    d = json.load(r)
            except Exception:  # noqa: BLE001
                break
            for f in d.get("features", []):
                p = f.get("properties", {})
                fecha = str(p.get("datetime") or "")[:10]
                plataforma = str(p.get("platform") or "")
                if fecha and plataforma:
                    salida.append((fecha, plataforma))
            siguiente = None
            for enlace in d.get("links", []):
                if enlace.get("rel") == "next":
                    siguiente = (enlace.get("body") or {}).get("token")
    return salida


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--taco", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    l0 = pq.read_table(pathlib.Path(a.taco) / ".tacocat" / "level0.parquet").to_pandas()
    l0["tile"] = l0["target:tile"].astype(str).str.extract(r"_(T\d{2}[A-Z]{3})_")[0]
    fechas = pd.to_datetime(l0["stac:time_start"], errors="coerce")
    d0 = (fechas.min() - dt.timedelta(days=400))
    d1 = fechas.max()
    anios = list(range(d0.year, d1.year + 1))
    tiles = sorted(t for t in l0["tile"].dropna().unique())
    print(f"tiles: {len(tiles)} | años {anios[0]}..{anios[-1]}", flush=True)

    mapa = {}
    import time as _t
    for n, t in enumerate(tiles, 1):
        for fecha, plat in buscar_tile(t, anios):
            mapa[(t, fecha)] = plat
        if n % 10 == 0:
            print(f"  {n}/{len(tiles)} tiles, {len(mapa)} escenas", flush=True)
        _t.sleep(0.3)

    filas = []
    for _, r in l0.iterrows():
        tile = r["tile"]
        for k in (1, 2, 3):
            fecha = str(r[f"bg:date{k}"])[:10]
            filas.append({"id": r["id"], "slot": k, "tile": tile, "fecha": fecha,
                          "plataforma": mapa.get((tile, fecha), "")})
    d = pd.DataFrame(filas)
    d.to_csv(a.out, index=False)
    con = int((d["plataforma"] != "").sum())
    print(f"referencias con satélite: {con}/{len(d)} ({100*con/len(d):.1f} %)")
    if con:
        print(d["plataforma"].value_counts().to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
