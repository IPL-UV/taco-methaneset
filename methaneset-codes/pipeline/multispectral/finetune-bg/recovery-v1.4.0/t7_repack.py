"""T7 · repaquetado del finetune multispectral a v1.4.0.

Copia cada .tacozip publicado entrada por entrada (bytes identicos), añade las muestras
recuperadas y regenera el catalogo. No toca nunca el dataset publicado.

Lo delicado del formato:
  - internal:offset de una hoja = offset del registro local + 30 + len(nombre) + len(extra)
  - DATA/<uuid>/__meta__ es un parquet con una fila por hoja y sus offsets
  - por eso __meta__ se escribe al final de cada muestra, cuando ya se conocen los offsets

Uso:
  python v140/t7_repack.py --coleccion l89 --limit 3
  python v140/t7_repack.py --coleccion l89
  python v140/t7_repack.py --coleccion s2
"""
import argparse
import io
import pathlib
import shutil
import struct
import zipfile

import numpy as np
import pandas as pd
import rasterio
import tacotiff

PUB = pathlib.Path("/data/databases/METHANESET_TACOS")
OUT = PUB / "_v140/T7"
LEAVES = PUB / "_v140/T5/leaves"
HOJAS = PUB / "_v140/T5/hojas_64.csv"
CE = pathlib.Path("/data/databases/METHANE_DATASETS_TACO")
RAW = PUB / "_v140/T2/raw"
VERSION = "1.4.0"

MAPEO = {"emission": "target", "background": "bg0", "plume": "plume",
         "ch4": "ch4", "bg1": "bg1", "bg2": "bg2", "bg3": "bg3", "dem": "dem"}


def _local_offset(f, header_offset):
    """Offset real del contenido: cabecera local + nombre + campo extra."""
    f.seek(header_offset)
    cab = f.read(30)
    n, m = struct.unpack("<HH", cab[26:30])
    return header_offset + 30 + n + m


def _stats(ruta):
    with rasterio.open(ruta) as s:
        a = s.read().astype("float64")
    out = []
    for i in range(a.shape[0]):
        b = a[i][np.isfinite(a[i])]
        if b.size == 0:
            out.append([])
            continue
        q = np.percentile(b, [25, 50, 75, 95]).tolist()
        out.append([float(b.min()), float(b.max()), float(b.mean()), float(b.std()),
                    float(100 * b.size / a[i].size)] + q)
    return out


def _cabecera(ruta, referencia):
    """Cabecera TACOTIFF de la hoja. Si el TIFF no la declara, se copia una compatible."""
    try:
        return tacotiff.metadata_from_tiff(str(ruta))
    except Exception:
        if referencia is not None:
            return referencia
        return b""


def _escribir_hojas(zf, sid, destino, hojas_ref):
    """Escribe las 8 hojas de una muestra. Los offsets se calculan despues, del zip cerrado."""
    filas = []
    for hoja_src, hoja_dst in MAPEO.items():
        origen = destino / hoja_src
        datos = origen.read_bytes()
        zf.writestr(f"DATA/{sid}/{hoja_dst}", datos)
        filas.append({
            "id": hoja_dst, "type": "FILE",
            "geotiff:stats": _stats(origen),
            "taco:header": _cabecera(origen, hojas_ref.get(hoja_dst)),
            "internal:size": len(datos),
        })
    return filas


def _escanear(destino):
    """Offset real de cada hoja y de cada __meta__ en los zips ya cerrados."""
    pos, tam = {}, {}
    for z in destino.glob("*.tacozip"):
        with open(z, "rb") as f, zipfile.ZipFile(z) as zf:
            for info in zf.infolist():
                partes = info.filename.split("/")
                if len(partes) == 3 and partes[0] == "DATA":
                    off = _local_offset(f, info.header_offset)
                    if partes[2] == "__meta__":
                        tam[partes[1]] = (off, info.file_size)
                    else:
                        pos[(partes[1], partes[2])] = off
    return pos, tam


def _parquet(df):
    buf = io.BytesIO()
    df.to_parquet(buf, index=False)
    return buf.getvalue()


def _hojas_referencia(coleccion):
    """Cabecera de cada hoja tomada de una muestra publicada (fallback del TIFF)."""
    base = PUB / f"methaneset-{coleccion}-finetune"
    z = next(p for p in base.glob("*.tacozip"))
    ref = {}
    with zipfile.ZipFile(z) as zf:
        nombres = zf.namelist()
        extraidos = pathlib.Path("/tmp/opencode/refcab")
        extraidos.mkdir(parents=True, exist_ok=True)
        for hoja in MAPEO.values():
            hojas = [n for n in nombres if n.endswith(f"/{hoja}")]
            if not hojas:
                continue
            destino = extraidos / hoja
            destino.write_bytes(zf.read(hojas[0]))
            try:
                ref[hoja] = tacotiff.metadata_from_tiff(str(destino))
            except Exception:
                ref[hoja] = b""
    return ref


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coleccion", required=True, choices=["l89", "s2"])
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    publicado = PUB / f"methaneset-{args.coleccion}-finetune"
    destino = OUT / args.coleccion
    if destino.resolve() == publicado.resolve():
        raise SystemExit("la salida apunta al dataset publicado: abortado")

    nuevas = pd.read_csv(HOJAS)
    nuevas = nuevas[nuevas.coleccion == args.coleccion]
    if args.limit:
        nuevas = nuevas.head(args.limit)
    print(f"{args.coleccion}: {len(nuevas)} muestras nuevas")

    if destino.exists():
        shutil.rmtree(destino)
    destino.mkdir(parents=True)
    shutil.copytree(publicado / ".tacocat", destino / ".tacocat")

    l0_pub = pd.read_parquet(publicado / ".tacocat" / "level0.parquet")
    ref = _hojas_referencia(args.coleccion)

    # país de cada muestra nueva
    ce = pd.read_parquet(CE / args.coleccion / ".tacocat" / "level0.parquet").set_index("id")
    pais_de = {}
    for sid in nuevas.id:
        fila = ce.loc[sid]
        # site:country es el mismo nombre con el que estan repartidos los zips publicados
        pais = str(fila.get("site:country") or "Unknown").split(";")[0].strip()
        pais_de[sid] = pais.replace(" ", "_").replace("/", "_")

    filas_nuevas, meta_por_muestra = [], {}
    for zip_pub in sorted(publicado.glob("*.tacozip")):
        shutil.copy2(zip_pub, destino / zip_pub.name)

    for pais, grupo in pd.DataFrame({"id": nuevas.id, "pais": nuevas.id.map(pais_de)}).groupby("pais"):
        ruta = destino / f"methaneset-{args.coleccion}-finetune_{pais}.tacozip"
        if not ruta.exists():
            ruta.write_bytes(b"")
        with open(ruta, "r+b") as f, zipfile.ZipFile(f, "a", zipfile.ZIP_STORED) as zf:
            for sid in grupo.id:
                meta_por_muestra[sid] = _escribir_hojas(zf, sid, LEAVES / sid, ref)
        print(f"  {pais}: {len(grupo)} muestras -> {ruta.name}")

    # primera pasada cerrada: ahora se conocen los offsets de las hojas
    pos, tam_meta = _escanear(destino)
    for pais, grupo in pd.DataFrame({"id": nuevas.id, "pais": nuevas.id.map(pais_de)}).groupby("pais"):
        ruta = destino / f"methaneset-{args.coleccion}-finetune_{pais}.tacozip"
        with open(ruta, "r+b") as f, zipfile.ZipFile(f, "a", zipfile.ZIP_STORED) as zf:
            for sid in grupo.id:
                filas = meta_por_muestra[sid]
                for fila in filas:
                    fila["internal:offset"] = pos[(sid, fila["id"])]
                zf.writestr(f"DATA/{sid}/__meta__",
                            _parquet(pd.DataFrame(filas)[
                                ["id", "type", "geotiff:stats", "taco:header",
                                 "internal:offset", "internal:size"]]))
    pos, tam_meta = _escanear(destino)

    # level1: se reescribe el offset de TODAS las filas desde el zip nuevo
    l1 = pd.read_parquet(publicado / ".tacocat" / "level1.parquet")
    l1["muestra"] = l1["internal:relative_path"].str.split("/").str[0]
    l1["hoja"] = l1["internal:relative_path"].str.split("/").str[1]
    l1["internal:offset"] = [pos.get((m, h), o) for m, h, o in
                             zip(l1.muestra, l1.hoja, l1["internal:offset"])]

    nuevas_l1 = []
    for sid, meta in meta_por_muestra.items():
        idx = int(l0_pub["internal:current_id"].max()) + 1 + len(nuevas_l1) // 8 if "internal:current_id" in l0_pub else 0
        for r in meta:
            nuevas_l1.append({
                "id": r["id"], "type": "FILE",
                "geotiff:stats": r["geotiff:stats"], "taco:header": r["taco:header"],
                "internal:current_id": idx, "internal:parent_id": idx,
                "internal:offset": pos[(sid, r["id"])], "internal:size": r["internal:size"],
                "internal:relative_path": f"{sid}/{r['id']}",
                "internal:source_file": f"methaneset-{args.coleccion}-finetune_{pais_de[sid]}.tacozip",
            })
    l1_full = pd.concat([l1.drop(columns=["muestra", "hoja"]),
                         pd.DataFrame(nuevas_l1)], ignore_index=True)
    l1_full.to_parquet(destino / ".tacocat" / "level1.parquet", index=False)

    # level0: las publicadas tal cual + las nuevas con las columnas rellenadas de César
    base = l0_pub.iloc[0].to_dict()
    filas0 = []
    for sid in nuevas.id:
        fila = {k: base.get(k) for k in l0_pub.columns}
        c = ce.loc[sid]
        fila["id"] = sid
        fila["target:sensor"] = c.get("satellite:satellite")
        fila["target:tile"] = c.get("satellite:tile")
        fila["target:sza"] = c.get("satellite:sza")
        fila["target:vza"] = c.get("satellite:vza")
        fila["detection:ch4_fluxrate"] = c.get("methane:ch4_fluxrate")
        fila["detection:ch4_fluxrate_std"] = c.get("methane:ch4_fluxrate_std")
        fila["detection:sector"] = c.get("methane:sector")
        fila["detection:offshore"] = c.get("methane:offshore")
        fila["detection:wind_source"] = c.get("methane:wind_source")
        fila["meteo:wind_u"] = c.get("methane:wind_u")
        fila["meteo:wind_v"] = c.get("methane:wind_v")
        fila["site:country"] = c.get("site:country")
        fila["site:location_name"] = c.get("site:location_name")
        fila["split"] = c.get("split")
        fila["stac:crs"] = c.get("stac:crs")
        fila["stac:time_start"] = c.get("stac:time_start")
        fila["internal:current_id"] = int(l0_pub["internal:current_id"].max()) + 1 + len(filas0)
        fila["internal:source_file"] = f"methaneset-{args.coleccion}-finetune_{pais_de[sid]}.tacozip"
        fila["internal:offset"] = tam_meta.get(sid, (0, 0))[0]
        fila["internal:size"] = tam_meta.get(sid, (0, 0))[1]
        filas0.append(fila)
    l0_full = pd.concat([l0_pub, pd.DataFrame(filas0)], ignore_index=True)
    for col in l0_pub.columns:  # mismo dtype que lo publicado, o no se puede leer
        try:
            if str(l0_pub[col].dtype).startswith("datetime"):
                l0_full[col] = pd.to_datetime(l0_full[col], utc=True).astype(l0_pub[col].dtype)
            else:
                l0_full[col] = l0_full[col].astype(l0_pub[col].dtype)
        except Exception:
            pass
    l0_full.to_parquet(destino / ".tacocat" / "level0.parquet", index=False)

    print(f"level0: {len(l0_pub)} + {len(filas0)} = {len(l0_full)}")
    print(f"level1: {len(l1)} + {len(nuevas_l1)} = {len(l1_full)}")


if __name__ == "__main__":
    main()
