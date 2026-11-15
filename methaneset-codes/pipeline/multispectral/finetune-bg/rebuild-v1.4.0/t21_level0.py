"""Reescribe level0 (colección y METADATA interna) con las columnas de calidad finales.

No toca píxeles ni level1: copia las entradas tal cual y sustituye el parquet de level0
dentro de cada zip, recalculando offsets y TACO_HEADER.

Fuentes:
  - cloud_<fam>.csv de T11 (nube objetivo y de bg1..3)
  - cambios.csv (sensor, fecha y nube de las referencias sustituidas)
  - derivadas.parquet (difference y methane recalculados por t16)
"""

import argparse
import io
import pathlib
import struct
import zipfile

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

T7 = pathlib.Path("/data/databases/METHANESET_TACOS/_v140/T7")
T11 = pathlib.Path("/data/databases/METHANESET_TACOS/_v140/T11")
CABECERA = 116

ORDEN = [
    "id",
    "type",
    "target:sensor",
    "detection:case_study",
    "detection:ch4_fluxrate",
    "detection:ch4_fluxrate_std",
    "detection:offshore",
    "detection:sector",
    "detection:wind_source",
    "geoenrich:admin_countries",
    "geoenrich:admin_districts",
    "geoenrich:admin_states",
    "geoenrich:elevation",
    "geoenrich:population",
    "geoenrich:temperature",
    "majortom:code",
    "meteo:wind_u",
    "meteo:wind_v",
    "plume:geometry",
    "quality:last_update",
    "quality:notified",
    "quality:observability",
    "quality:percentage_clear",
    "target:sza",
    "target:tile",
    "target:vza",
    "site:country",
    "site:location_name",
    "split",
    "stac:crs",
    "stac:geotransform",
    "stac:tensor_shape",
    "stac:time_start",
    "bg:sensor0",
    "bg:tile0",
    "bg:date0",
    "bg:cloud0",
    "bg:sensor1",
    "bg:date1",
    "bg:cloud1",
    "bg:difference1",
    "bg:methane1",
    "bg:sensor2",
    "bg:date2",
    "bg:cloud2",
    "bg:difference2",
    "bg:methane2",
    "bg:sensor3",
    "bg:date3",
    "bg:cloud3",
    "bg:difference3",
    "bg:methane3",
    "target:cloud",
    "emission:lat",
    "emission:lon",
    "internal:current_id",
    "internal:parent_id",
    "internal:offset",
    "internal:size",
    "target:saa",
    "target:raa",
    "internal:source_file",
]



def parquet_bytes(df):
    buf = io.BytesIO()
    df.to_parquet(buf, index=False, compression="zstd")
    return buf.getvalue()


def offset_contenido(pos, nombre):
    return pos + 30 + len(nombre)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fam", required=True, choices=["l89", "s2"])
    ap.add_argument("--nuevo", required=True)
    ap.add_argument("--cambios", default="/tmp/opencode/cambios.csv")
    ap.add_argument("--derivadas", default="/tmp/opencode/derivadas.parquet")
    a = ap.parse_args()

    nuevo = pathlib.Path(a.nuevo)
    cambio = pd.read_csv(a.cambios) if pathlib.Path(a.cambios).exists() else pd.DataFrame()
    cambio = cambio[cambio["fam"] == a.fam] if len(cambio) else cambio
    cambio = cambio.set_index([cambio["id"].astype(str), cambio["slot"].astype(int)])
    deriv = pd.DataFrame()
    if pathlib.Path(a.derivadas).exists():
        deriv = pq.read_table(a.derivadas).to_pandas()
        if "id" in deriv.columns:
            deriv = deriv.set_index("id")
        else:
            deriv = pd.DataFrame()

    l0 = pq.read_table(nuevo / ".tacocat" / "level0.parquet").to_pandas()
    nube = pd.read_csv(T11 / f"cloud_{a.fam}.csv").set_index("id")
    antes = l0.copy()

    l0 = l0.set_index("id")
    for sid in l0.index:
        if sid in nube.index:
            l0.loc[sid, "target:cloud"] = float(nube.loc[sid, "target:cloud"])
            if "bg0:cloud" in nube.columns:
                l0.loc[sid, "bg:cloud0"] = float(nube.loc[sid, "bg0:cloud"])
            for k in (1, 2, 3):
                l0.loc[sid, f"bg:cloud{k}"] = float(nube.loc[sid, f"bg{k}:cloud"])
        for k in (1, 2, 3):
            if (str(sid), k) in cambio.index:
                r = cambio.loc[(str(sid), k)]
                l0.loc[sid, f"bg:sensor{k}"] = str(r["sensor"])
                l0.loc[sid, f"bg:date{k}"] = str(r["date"])[:10]
                l0.loc[sid, f"bg:cloud{k}"] = float(r["cloud"])
            if a.fam == "s2":
                l0.loc[sid, f"bg:sensor{k}"] = "S2"
        if a.fam == "s2":
            # bg0 conserva el satélite que registró MARS, sin guiones bajos sueltos
            l0.loc[sid, "bg:sensor0"] = str(l0.loc[sid, "bg:sensor0"]).strip("_")
        if sid in deriv.index:
            for k in (1, 2, 3):
                v = deriv.loc[sid, f"d{k}"]
                if np.isfinite(v):
                    l0.loc[sid, f"bg:difference{k}"] = float(v)
                v = deriv.loc[sid, f"m{k}"]
                if np.isfinite(v):
                    l0.loc[sid, f"bg:methane{k}"] = float(v)

    l0 = l0.reset_index()
    if "internal:source_file" not in l0.columns:
        ref = pq.read_table(T7 / a.fam / ".tacocat" / "level0.parquet").to_pandas()
        l0 = l0.merge(ref[["id", "internal:source_file"]], on="id", how="left")
    for c in antes.columns:
        try:
            l0[c] = l0[c].astype(antes[c].dtype)
        except Exception:  # noqa: BLE001
            pass
    # orden canónico de columnas (el de L89 y el que espera la auditoría)
    orden = list(antes.columns)
    if "internal:source_file" not in orden:
        orden.append("internal:source_file")
    l0 = l0[[c for c in orden if c in l0.columns]]
    if set(l0.columns) == set(ORDEN):
        l0 = l0[ORDEN]
    l0.to_parquet(nuevo / ".tacocat" / "level0.parquet", index=False)
    print("level0 colección:", l0.shape)
    nuevo_idx = l0.set_index("id")
    viejo_idx = antes.set_index("id").reindex(nuevo_idx.index)
    cambiadas = (nuevo_idx["bg:date1"] != viejo_idx["bg:date1"]).sum()
    print("muestras con la referencia 1 cambiada:", int(cambiadas))

    # identidad de la colección (la escribe t9 desde la tarjeta): se replica en cada zip
    import copy as _copy
    import json as _json
    col_base = _json.loads((nuevo / ".tacocat" / "COLLECTION.json").read_text())

    # misma tabla dentro de cada zip
    for z in sorted(nuevo.glob("*.tacozip")):
        with zipfile.ZipFile(z) as zf:
            nombres = zf.namelist()
            datos = {n: zf.read(n) for n in nombres if not n.startswith("METADATA/")}
            l1b = zf.read("METADATA/level1.parquet")
        ids = {n.split("/")[1] for n in nombres if n.startswith("DATA/") and not n.endswith("__meta__")}
        sub = l0[l0["id"].isin(ids)]
        b0 = parquet_bytes(sub)

        col = _copy.deepcopy(col_base)
        la = pd.to_numeric(sub["emission:lat"], errors="coerce").dropna()
        lo = pd.to_numeric(sub["emission:lon"], errors="coerce").dropna()
        tm = pd.to_datetime(sub["stac:time_start"], errors="coerce").dropna()
        if len(la) and len(lo) and len(tm):
            col["extent"] = {"spatial": [round(float(lo.min()), 3), round(float(la.min()), 3),
                                        round(float(lo.max()), 3), round(float(la.max()), 3)],
                             "temporal": [tm.min().isoformat(), tm.max().isoformat()]}
        col_json = _json.dumps(col, ensure_ascii=False, indent=1).encode("utf-8")
        orden = [n for n in nombres if not n.startswith("METADATA/") and n != "COLLECTION.json"]
        orden += ["METADATA/level0.parquet", "METADATA/level1.parquet", "COLLECTION.json"]
        pos, planes = 0, []
        for n in orden:
            if n == "METADATA/level0.parquet":
                b = b0
            elif n == "METADATA/level1.parquet":
                b = l1b
            elif n == "COLLECTION.json":
                b = col_json
            else:
                b = datos[n]
            planes.append((n, b))
            pos += 30 + len(n) + len(b)
        # recalcular posiciones de verdad
        pos = 0
        sitios = {}
        for n, b in planes:
            sitios[n] = offset_contenido(pos, n)
            pos += 30 + len(n) + len(b)
        cab = struct.pack("<I", 3) + struct.pack("<6q", sitios["METADATA/level0.parquet"], len(b0),
                                                 sitios["METADATA/level1.parquet"], len(l1b),
                                                 sitios["COLLECTION.json"], len(col_json))
        cab += b"\x00" * (CABECERA - len(cab))
        planes[0] = ("TACO_HEADER", cab)
        destino = z.with_suffix(".tmp")
        with zipfile.ZipFile(destino, "w", compression=zipfile.ZIP_STORED) as zf:
            for n, b in planes:
                info = zipfile.ZipInfo(n, date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_STORED
                info.external_attr = 0o644 << 16
                zf.writestr(info, b)
        destino.replace(z)
        print(f"  {z.name}: {len(sub)} muestras, level0 {len(b0)/1024:.0f} KB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
