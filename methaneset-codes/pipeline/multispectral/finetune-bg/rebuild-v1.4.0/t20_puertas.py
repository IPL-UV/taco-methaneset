"""Puertas del rebuild: comprueba que solo cambió lo que debía.

  python t20_puertas.py --fam l89 --nuevo .../T16/l89 --viejo .../T7/l89
"""

import argparse
import hashlib
import json
import io
import pathlib
import struct
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import zipfile

sys.path.insert(0, str(pathlib.Path(__file__).parent))


def sha(b):
    return hashlib.sha256(b).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nuevo", required=True)
    ap.add_argument("--viejo", required=True)
    ap.add_argument("--cambios", default="/tmp/opencode/cambios.csv")
    ap.add_argument("--fam", required=True)
    a = ap.parse_args()
    nuevo, viejo = pathlib.Path(a.nuevo), pathlib.Path(a.viejo)
    fallos, avisos = [], []

    cam = pd.read_csv(a.cambios) if pathlib.Path(a.cambios).exists() else pd.DataFrame()
    cam = cam[cam["fam"] == a.fam] if len(cam) else cam
    reemplazadas = {(r["id"], f"bg{int(r['slot'])}") for _, r in cam.iterrows()}
    chips = {tuple(p.stem.split("__")) for p in
             pathlib.Path("/data/databases/METHANESET_TACOS/_v140/T14").glob("*.tif")}
    tocadas = reemplazadas | chips

    # --- 1. por zip: hashes, offsets, METADATA ---------------------------
    l0n = pq.read_table(nuevo / ".tacocat" / "level0.parquet").to_pandas()
    l1n = pq.read_table(nuevo / ".tacocat" / "level1.parquet").to_pandas()
    zips_n = sorted(nuevo.glob("*.tacozip"))
    zips_v = {z.name: z for z in viejo.glob("*.tacozip")}
    print(f"zips: nuevos {len(zips_n)} | viejos {len(zips_v)}")

    identicas = distintas_ok = distintas_mal = 0
    for zn in zips_n:
        zv = zips_v.get(zn.name)
        if zv is None:
            fallos.append(f"zip nuevo sin origen: {zn.name}")
            continue
        with zipfile.ZipFile(zv) as z0, zipfile.ZipFile(zn) as z1:
            n0 = {n for n in z0.namelist() if n.startswith("DATA/") and not n.endswith("__meta__")}
            n1 = {n for n in z1.namelist() if n.startswith("DATA/") and not n.endswith("__meta__")}
            if n0 != n1:
                fallos.append(f"{zn.name}: cambia el conjunto de hojas ({len(n0)} vs {len(n1)})")
            for n in sorted(n0 & n1):
                sid, hoja = n.split("/")[1], n.split("/")[2]
                h0, h1 = sha(z0.read(n)), sha(z1.read(n))
                if h0 == h1:
                    identicas += 1
                    if (sid, hoja) in tocadas:
                        avisos.append(f"{sid[:8]} {hoja}: marcada para cambiar y no cambió")
                else:
                    if (sid, hoja) in tocadas:
                        distintas_ok += 1
                    else:
                        distintas_mal += 1
                        fallos.append(f"{sid[:8]} {hoja}: cambió sin estar en la lista")
            # offsets de las hojas
            l1z = pq.read_table(io.BytesIO(z1.read("METADATA/level1.parquet"))).to_pandas()
            with open(zn, "rb") as f:
                malos = 0
                for off in l1z["internal:offset"]:
                    f.seek(int(off))
                    cab = f.read(2)
                    if cab not in (b"II", b"MM"):
                        malos += 1
            if malos:
                fallos.append(f"{zn.name}: {malos} offsets no apuntan a un TIFF")
            # TACO_HEADER
            h = z1.read("TACO_HEADER")
            v = struct.unpack("<6q", h[4:52])
            infos = {i.filename: i for i in z1.infolist()}
            for nombre, off, tam in (("METADATA/level0.parquet", v[0], v[1]),
                                     ("METADATA/level1.parquet", v[2], v[3]),
                                     ("COLLECTION.json", v[4], v[5])):
                i = infos.get(nombre)
                if i is None:
                    fallos.append(f"{zn.name}: falta {nombre}")
                    continue
                real = i.header_offset + 30 + len(nombre)
                if real != off or i.file_size != tam:
                    fallos.append(f"{zn.name}: TACO_HEADER miente en {nombre}")
            # METADATA interna == .tacocat del zip
            l0z = pq.read_table(io.BytesIO(z1.read("METADATA/level0.parquet"))).to_pandas()
            ids_z = set(l0z["id"])
            ids_d = {n.split("/")[1] for n in z1.namelist()
                     if n.startswith("DATA/") and not n.endswith("__meta__")}
            if ids_z != ids_d:
                fallos.append(f"{zn.name}: METADATA y DATA discrepan ({len(ids_z)} vs {len(ids_d)})")

    print(f"hojas idénticas: {identicas} | cambiadas esperadas: {distintas_ok} | "
          f"cambiadas NO esperadas: {distintas_mal}")

    # --- 2. level0 --------------------------------------------------------
    print(f"level0 {l0n.shape} | level1 {l1n.shape}")
    base0 = pq.read_table(viejo / ".tacocat" / "level0.parquet").to_pandas()
    from t13_auditoria import COLS as COLS_CANON  # noqa: PLC0415
    if set(l0n.columns) != set(base0.columns):
        fallos.append("columnas de level0 distintas")
    if list(l0n.columns) != list(COLS_CANON):
        fallos.append("orden de columnas de level0 no canónico")
    if len(l0n) != len(base0):
        fallos.append(f"level0 con {len(l0n)} filas y no {len(base0)}")
    if set(l0n["id"]) != set(base0["id"]):
        fallos.append("ids de level0 distintos")
    c0 = pd.to_numeric(l0n["bg:cloud0"], errors="coerce")
    if c0.isna().any() or c0.min() < 0 or c0.max() > 1:
        fallos.append(f"bg:cloud0 fuera de rango o NaN ({int(c0.isna().sum())} NaN)")
    ct0 = pd.to_numeric(l0n["target:cloud"], errors="coerce")
    if ct0.isna().any() or ct0.min() < 0 or ct0.max() > 1:
        fallos.append("target:cloud fuera de rango o NaN")
    for k in (1, 2, 3):
        c = pd.to_numeric(l0n[f"bg:cloud{k}"], errors="coerce")
        malas = int((c > 0.35).sum())
        print(f"  bg:cloud{k} > 0.35: {malas}")
        if malas:
            avisos.append(f"{malas} referencias siguen por encima del 35 %")
        if c.isna().any():
            fallos.append(f"bg:cloud{k} con NaN")
    for k in (1, 2, 3):
        vals = set(l0n[f"bg:sensor{k}"].astype(str).unique())
        if a.fam == "l89":
            raros = {v for v in vals if v not in ("LC08", "LC09")}
        else:
            raros = {v for v in vals if v not in ("S2", "S2A", "S2B", "S2C")}
        if raros:
            fallos.append(f"bg:sensor{k} con valores raros: {list(raros)[:4]}")
    if pd.to_numeric(l0n["bg:difference1"], errors="coerce").isna().any():
        avisos.append("bg:difference1 con NaN")

    # --- 2b. cobertura de las hojas y metadatos de colección ---------------
    pct = {}
    for _, r in l1n.iterrows():
        hoja = str(r["internal:relative_path"]).split("/")[-1]
        if hoja not in ("target", "bg0", "bg1", "bg2", "bg3"):
            continue
        try:
            pct.setdefault(hoja, []).append(float(np.asarray(r["geotiff:stats"])[0][4]))
        except Exception:  # noqa: BLE001
            pass
    for hoja, v in pct.items():
        bajas = [x for x in v if x < 50]
        if bajas:
            fallos.append(f"{hoja}: {len(bajas)} hojas con menos del 50 % de píxeles válidos")
        flojas = [x for x in v if x < 99]
        if flojas:
            avisos.append(f"{hoja}: {len(flojas)} hojas con menos del 99 % válidos (mín {min(flojas):.1f})")

    version = pq.read_table(nuevo / ".tacocat" / "level0.parquet").schema is not None  # noqa: F841
    col_c = json.loads((nuevo / ".tacocat" / "COLLECTION.json").read_text())
    for zn in zips_n:
        with zipfile.ZipFile(zn) as z1:
            c = json.loads(z1.read("COLLECTION.json"))
        if c.get("dataset_version") != col_c.get("dataset_version"):
            fallos.append(f"{zn.name}: dataset_version {c.get('dataset_version')} != "
                          f"{col_c.get('dataset_version')}")
        if c.get("keywords") != col_c.get("keywords"):
            fallos.append(f"{zn.name}: keywords distintas de la colección")
        if c.get("description") != col_c.get("description"):
            fallos.append(f"{zn.name}: descripción distinta de la colección")
        if not c.get("extent", {}).get("temporal"):
            fallos.append(f"{zn.name}: COLLECTION.json sin extensión temporal")

    # --- 3. tacoreader ----------------------------------------------------
    import tacoreader  # noqa: PLC0415
    tacoreader.use("pandas")
    ds = tacoreader.load(str(nuevo))
    m = len(ds.data)
    sub = ds.data.read(0)
    print(f"tacoreader: {m} muestras, {len(sub)} hojas en la primera")
    if len(sub) != 8:
        fallos.append(f"la primera muestra tiene {len(sub)} hojas")

    print("\nFALLOS:", len(fallos))
    for f in fallos[:20]:
        print("  -", f)
    print("AVISOS:", len(avisos))
    for f in avisos[:10]:
        print("  ·", f)
    return 1 if fallos else 0


if __name__ == "__main__":
    raise SystemExit(main())
