"""Rebuild del finetune multispectral: sustituye referencias, repara hojas vacías y
regenera los catálogos y los zips desde cero.

Se escribe cada zip nuevo (STORE, sin compresión), con:
  TACO_HEADER, DATA/<uuid>/{target,bg0..bg3,ch4,plume,dem,__meta__}, METADATA/*, COLLECTION.json

Las hojas que no se tocan se copian byte a byte desde el zip de origen; la puerta
posterior lo comprueba con sha256.

Uso:
  python t16_rebuild.py --fam l89 --solo-pais Argentina --salida /tmp/opencode/rb
  python t16_rebuild.py --fam l89 --salida /data/databases/METHANESET_TACOS/_v140/T16/l89
"""

import argparse
import concurrent.futures as cf
import hashlib
import io
import json
import pathlib
import shutil
import struct
import sys
import tempfile
import zipfile

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import rasterio
import tacotiff

SRC = pathlib.Path("/data/databases/METHANESET_TACOS/_v140/T7")
CHIPS = pathlib.Path("/data/databases/METHANESET_TACOS/_v140/T14")
CLOUD = pathlib.Path("/data/databases/METHANESET_TACOS/_v140/T11")
HOJAS = ["target", "bg0", "bg1", "bg2", "bg3", "ch4", "plume", "dem"]
CHIPS_5 = ["target", "bg0", "bg1", "bg2", "bg3"]
SIM = {"l89": (1, 2, 3, 5), "s2": (1, 2, 3, 11)}
SWIR = {"l89": (5, 6), "s2": (11, 12)}
CABECERA = 116
VERSION = "1.4.0"


# --------------------------------------------------------------------------- #
# utilidades
# --------------------------------------------------------------------------- #
def stats_arr(a: np.ndarray, nd: float | None = None) -> list:
    """Estadísticas por banda con máscara de nodata (la versión publicada contaba ceros)."""
    out = []
    for i in range(a.shape[0]):
        b = a[i].astype("float64")
        malo = ~np.isfinite(b)
        if nd is not None:
            malo |= (a[i] == nd)
        v = b[~malo]
        if v.size == 0:
            out.append(np.zeros(9, dtype="float32"))
            continue
        q = np.percentile(v, [25, 50, 75, 95]).tolist()
        out.append(np.asarray([float(v.min()), float(v.max()), float(v.mean()), float(v.std()),
                               float(100 * v.size / b.size)] + [float(x) for x in q],
                              dtype="float32"))
    contenedor = np.empty(len(out), dtype=object)
    for i, v in enumerate(out):
        contenedor[i] = v
    return contenedor


def _norm_stats(v) -> np.ndarray:
    """Normaliza las estadísticas a un array de arrays float32 (como los publicados)."""
    v = list(v)
    contenedor = np.empty(len(v), dtype=object)
    for i, x in enumerate(v):
        contenedor[i] = np.asarray(x, dtype="float32")
    return contenedor


def leer(path_or_bytes):
    if isinstance(path_or_bytes, bytes):
        with rasterio.open(io.BytesIO(path_or_bytes)) as s:
            return s.read()
    with rasterio.open(path_or_bytes) as s:
        return s.read()


def stats_de(datos: bytes) -> list:
    with rasterio.open(io.BytesIO(datos)) as s:
        return stats_arr(s.read(), s.nodata)


def diferencia(t: np.ndarray, b: np.ndarray, fam: str) -> float:
    """MARS: media_k de |t_k - g_k*b_k| con g_k = media(t_k)/media(b_k), en reflectancia."""
    difs = []
    for k in SIM[fam]:
        tk, bk = t[k].astype("float64"), b[k].astype("float64")
        ok = np.isfinite(tk) & np.isfinite(bk)
        if ok.sum() < 100:
            difs.append(np.nan)
            continue
        mt, mb = float(tk[ok].mean()), float(bk[ok].mean())
        if not (np.isfinite(mt) and np.isfinite(mb) and abs(mb) > 1e-12):
            difs.append(np.nan)
            continue
        g = mt / mb
        difs.append(float(np.abs(tk[ok] - g * bk[ok]).mean()))
    v = float(np.nanmean(difs))
    return v / 1e4 if fam == "s2" else v


def metano(b: np.ndarray, fam: str) -> float:
    """Fracción de píxeles del fondo con anomalía de ratio por debajo del 92 % de su mediana."""
    s1, s2 = SWIR[fam]
    b1, b2 = b[s1].astype("float64"), b[s2].astype("float64")
    ok = (b1 > 0) & (b2 > 0)
    if ok.sum() < 100:
        return np.nan
    r = b2[ok] / b1[ok]
    med = float(np.median(r))
    if not np.isfinite(med) or med <= 0:
        return np.nan
    return float(np.mean(r < 0.92 * med))


def cabecera(datos: bytes, ref: bytes) -> bytes:
    with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as f:
        f.write(datos)
        ruta = f.name
    try:
        return tacotiff.metadata_from_tiff(ruta)
    except Exception:  # noqa: BLE001
        return ref
    finally:
        pathlib.Path(ruta).unlink(missing_ok=True)


def parquet_bytes(df: pd.DataFrame, compresion="zstd") -> bytes:
    buf = io.BytesIO()
    df.to_parquet(buf, index=False, compression=compresion)
    return buf.getvalue()


def offset_contenido(pos: int, nombre: str) -> int:
    return pos + 30 + len(nombre)


# --------------------------------------------------------------------------- #
# rebuild de un zip
# --------------------------------------------------------------------------- #
def rebuild_zip(zip_src: pathlib.Path, destino: pathlib.Path, reemplazos: dict,
                chips: dict, nube: pd.DataFrame, fam: str, fuente: str,
                cambios: dict | None = None, satelites: dict | None = None,
                l1_glob: pd.DataFrame | None = None, l0_glob=None,
                col_base: bytes | None = None) -> dict:
    import copy as _copy
    cambios = cambios or {}
    satelites = satelites or {}
    with zipfile.ZipFile(zip_src) as z:
        nombres = z.namelist()
        cab_orig = z.read("TACO_HEADER") if "TACO_HEADER" in nombres else b"\x00" * CABECERA

        l0_src = l1_src = None
        if "METADATA/level0.parquet" in nombres:
            l0_src = pd.read_parquet(io.BytesIO(z.read("METADATA/level0.parquet")))
            l1_src = pd.read_parquet(io.BytesIO(z.read("METADATA/level1.parquet")))
        muestras = []
        vistos = set()
        for n in nombres:
            partes = n.split("/")
            if len(partes) == 3 and partes[0] == "DATA" and partes[2] != "__meta__":
                if partes[1] not in vistos:
                    vistos.add(partes[1])
                    muestras.append(partes[1])
        bytes_src = {n: z.read(n) for n in nombres if n.startswith("DATA/") and not n.endswith("__meta__")}

    # hay un zip (Poland) sin METADATA interna: se cae al catálogo de la colección
    if l0_src is None:
        if l0_glob is None or l1_glob is None:
            raise RuntimeError(f"{zip_src.name} sin METADATA y sin catálogo de respaldo")
        l1_src = (l1_glob.loc[[m for m in muestras if m in l1_glob.index.levels[0]]]
                  .reset_index().drop(columns=["muestra", "hoja"]))
        l1_src["internal:relative_path"] = l1_src["internal:relative_path"].astype(str)
        l1_src = l1_src[[c for c in l1_src.columns if c != "internal:source_file"]]
        l1_src = l1_src.set_index(l1_src["internal:relative_path"].str.split("/").str[0])
        l0_src = l0_glob.loc[[m for m in muestras if m in l0_glob.index]].reset_index()
        l0_src = l0_src[[c for c in l0_src.columns if c != "internal:source_file"]]

    COLS_L1 = list(l1_src.columns)
    l1_src["muestra"] = l1_src["internal:relative_path"].str.split("/").str[0]
    l1_src["hoja"] = l1_src["internal:relative_path"].str.split("/").str[1]
    l1_idx = l1_src.set_index(["muestra", "hoja"])
    # las 70 muestras recuperadas no están en el METADATA interno del zip (defecto D5):
    # para ellas se cae al catálogo de la colección.
    def dato(clave, col):
        if clave in l1_idx.index:
            v = l1_idx.loc[clave, col]
            if isinstance(v, pd.Series):
                v = v.iloc[0]
            return v
        if l1_glob is not None and clave in l1_glob.index:
            v = l1_glob.loc[clave, col]
            return v.iloc[0] if isinstance(v, pd.Series) else v
        raise KeyError(f"sin dato {clave} {col}")

    filas1, filas0, derivs = [], [], []
    pos = 0
    entradas = []  # (nombre, bytes)

    # hueco reservado para TACO_HEADER
    entradas.append(("TACO_HEADER", b"\x00" * CABECERA))
    pos += 30 + len("TACO_HEADER") + CABECERA

    for sid in muestras:
        offsets_muestra = {}
        arrays = {}
        for hoja in HOJAS:
            nombre = f"DATA/{sid}/{hoja}"
            clave = (sid, hoja)
            if clave in reemplazos:
                datos = pathlib.Path(reemplazos[clave]).read_bytes()
            elif clave in chips:
                datos = pathlib.Path(chips[clave]).read_bytes()
            else:
                datos = bytes_src.get(nombre)
                if datos is None:
                    raise RuntimeError(f"falta {nombre} en {zip_src.name}")
            ref = dato(clave, "taco:header")
            cab = ref if (clave not in reemplazos and clave not in chips) else cabecera(datos, ref)
            if hoja in CHIPS_5:
                arrays[hoja] = leer(datos)
                st = stats_arr(arrays[hoja])
            else:
                st = _norm_stats(dato(clave, "geotiff:stats"))
            off = offset_contenido(pos, nombre)
            offsets_muestra[hoja] = (off, len(datos))
            entradas.append((nombre, datos))
            pos += 30 + len(nombre) + len(datos)
            filas1.append({
                "id": hoja, "type": "FILE", "geotiff:stats": st, "taco:header": cab,
                "internal:current_id": int(dato(clave, "internal:current_id")),
                "internal:parent_id": int(dato(clave, "internal:parent_id")),
                "internal:offset": off, "internal:size": len(datos),
                "internal:relative_path": f"{sid}/{hoja}",
            })

        # __meta__ de la muestra, ya con los offsets de sus hojas
        meta = pd.DataFrame([
            {"id": h, "type": "FILE",
             "geotiff:stats": next(f["geotiff:stats"] for f in filas1 if f["internal:relative_path"] == f"{sid}/{h}"),
             "taco:header": next(f["taco:header"] for f in filas1 if f["internal:relative_path"] == f"{sid}/{h}"),
             "internal:offset": offsets_muestra[h][0], "internal:size": offsets_muestra[h][1]}
            for h in HOJAS])
        bmeta = parquet_bytes(meta)
        nombre_meta = f"DATA/{sid}/__meta__"
        off_meta = offset_contenido(pos, nombre_meta)
        entradas.append((nombre_meta, bmeta))
        pos += 30 + len(nombre_meta) + len(bmeta)

        coincide = l0_src[l0_src["id"] == sid]
        if len(coincide):
            fila0 = coincide.iloc[0].to_dict()
        elif l0_glob is not None and sid in l0_glob.index:
            fila0 = l0_glob.loc[sid].to_dict()
        else:
            raise KeyError(f"sin fila level0 para {sid}")
        fila0["id"] = sid
        fila0["internal:offset"] = off_meta
        fila0["internal:size"] = len(bmeta)

        # columnas derivadas sobre las hojas que se van a publicar
        deriv_fila = {"id": sid}
        t = arrays["target"]
        for k in (1, 2, 3):
            bg = arrays[f"bg{k}"]
            fila0[f"bg:difference{k}"] = diferencia(t, bg, fam)
            fila0[f"bg:methane{k}"] = metano(bg, fam)
            deriv_fila[f"d{k}"] = fila0[f"bg:difference{k}"]
            deriv_fila[f"m{k}"] = fila0[f"bg:methane{k}"]
            c = cambios.get(f"{sid}|bg{k}")
            if c is not None:
                fila0[f"bg:sensor{k}"], fila0[f"bg:date{k}"], fila0[f"bg:cloud{k}"] = c
            elif sid in nube.index:
                fila0[f"bg:cloud{k}"] = float(nube.loc[sid, f"bg{k}:cloud"])
            if fam == "s2" and (sid, k) in satelites:
                fila0[f"bg:sensor{k}"] = satelites[(sid, k)]
        if sid in nube.index:
            fila0["target:cloud"] = float(nube.loc[sid, "target:cloud"])
        filas0.append(fila0)
        derivs.append(deriv_fila)

    # COLLECTION.json por zip: misma identidad que la colección, extent real del zip
    col = None
    if col_base:
        col = _copy.deepcopy(json.loads(col_base.decode("utf-8")))
        lons = pd.to_numeric(l0_src["emission:lon"], errors="coerce").dropna()
        lats = pd.to_numeric(l0_src["emission:lat"], errors="coerce").dropna()
        tms = pd.to_datetime(l0_src["stac:time_start"], errors="coerce").dropna()
        if len(lons) and len(lats) and len(tms):
            col["extent"] = {
                "spatial": [round(float(lons.min()), 3), round(float(lats.min()), 3),
                            round(float(lons.max()), 3), round(float(lats.max()), 3)],
                "temporal": [tms.min().isoformat(), tms.max().isoformat()],
            }
    bj = json.dumps(col, ensure_ascii=False, indent=1).encode("utf-8") if col else b""

    l1_nuevo = pd.DataFrame(filas1)
    l0_nuevo = pd.DataFrame(filas0)

    # METADATA: mismas columnas que el original (61/9, sin internal:source_file)
    l0_zip = l0_nuevo[[c for c in l0_src.columns]]
    l1_zip = l1_nuevo[COLS_L1]
    b0 = parquet_bytes(l0_zip)
    b1 = parquet_bytes(l1_zip)
    off0 = offset_contenido(pos, "METADATA/level0.parquet")
    pos += 30 + len("METADATA/level0.parquet") + len(b0)
    off1 = offset_contenido(pos, "METADATA/level1.parquet")
    pos += 30 + len("METADATA/level1.parquet") + len(b1)
    offj = offset_contenido(pos, "COLLECTION.json") if bj else 0

    cab = struct.pack("<I", 3) + struct.pack("<6q", off0, len(b0), off1, len(b1), offj, len(bj))
    cab = cab + b"\x00" * (CABECERA - len(cab))

    entradas.append(("METADATA/level0.parquet", b0))
    entradas.append(("METADATA/level1.parquet", b1))
    entradas.append(("COLLECTION.json", bj))
    entradas[0] = ("TACO_HEADER", cab)

    destino.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destino, "w", compression=zipfile.ZIP_STORED) as z:
        for nombre, datos in entradas:
            info = zipfile.ZipInfo(nombre, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            info.external_attr = 0o644 << 16
            z.writestr(info, datos)

    l0_zip = l0_zip.assign(**{"internal:source_file": fuente})
    l1_zip = l1_zip.assign(**{"internal:source_file": fuente})
    return {"level0": l0_zip, "level1": l1_zip, "zip": destino,
            "deriv": pd.DataFrame(derivs)}


def _tarea(args):
    return rebuild_zip(*args)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fam", required=True, choices=["l89", "s2"])
    ap.add_argument("--salida", required=True)
    ap.add_argument("--cambios", default="/tmp/opencode/cambios.csv")
    ap.add_argument("--solo-pais", default=None)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()

    src = SRC / a.fam
    salida = pathlib.Path(a.salida)
    nube = pd.read_csv(CLOUD / f"cloud_{a.fam}.csv").set_index("id")

    cam = pd.DataFrame()
    p_cam = pathlib.Path(a.cambios)
    if p_cam.exists():
        cam = pd.read_csv(p_cam)
        cam = cam[cam["fam"] == a.fam] if "fam" in cam.columns else cam
    reemplazos = {(str(r["id"]), f"bg{int(r['slot'])}"): str(r["ruta"]) for _, r in cam.iterrows()}
    cambios = {f"{r['id']}|bg{int(r['slot'])}": (str(r["sensor"]), str(r["date"]), float(r["cloud"]))
               for _, r in cam.iterrows()}

    sat = pd.DataFrame()
    p_sat = pathlib.Path("/tmp/opencode/satelites_s2.csv")
    if a.fam == "s2" and p_sat.exists():
        sat = pd.read_csv(p_sat)
        mapa = {"sentinel-2a": "S2A", "sentinel-2b": "S2B", "sentinel-2c": "S2C"}
        sat["plataforma"] = sat["plataforma"].map(
            lambda x: mapa.get(str(x).lower(), x if str(x).startswith("S2") else ""))
        sat = sat[sat["plataforma"] != ""]
    satelites = {(r["id"], int(r["slot"])): r["plataforma"] for _, r in sat.iterrows()}

    chips = {}
    for p in CHIPS.glob("*.tif"):
        sid, hoja = p.stem.split("__")
        chips[(sid, hoja)] = str(p)
    print(f"{a.fam}: {len(reemplazos)} hojas sustituidas, {len(chips)} hojas reparadas, "
          f"{len(satelites)} satélites recuperados", flush=True)

    zips = sorted(src.glob("*.tacozip"))
    if a.solo_pais:
        zips = [z for z in zips if a.solo_pais in z.name]
    if salida.exists():
        shutil.rmtree(salida)
    salida.mkdir(parents=True)

    l0s, l1s, ds = [], [], []
    l1_base = pd.read_parquet(src / ".tacocat" / "level1.parquet")
    l1_base = l1_base.assign(
        muestra=l1_base["internal:relative_path"].str.split("/").str[0],
        hoja=l1_base["internal:relative_path"].str.split("/").str[1]).set_index(["muestra", "hoja"])
    l0_base = pd.read_parquet(src / ".tacocat" / "level0.parquet").set_index("id")

    # identidad de la colección: keywords correctas por familia y versión al día
    col_txt = (src / ".tacocat" / "COLLECTION.json").read_text()
    col = json.loads(col_txt)
    col["dataset_version"] = VERSION
    sat = "sentinel-2" if a.fam == "s2" else "landsat"
    col["keywords"] = [sat if k in ("landsat", "sentinel-2") else k for k in col["keywords"]]
    if a.fam == "s2" and "sentinel-2" not in col["keywords"]:
        col["keywords"].insert(1, "sentinel-2")
    col_base = json.dumps(col, ensure_ascii=False, indent=1).encode("utf-8")
    args = [(z, salida / z.name, reemplazos, chips, nube, a.fam, z.name, cambios, satelites,
             l1_base, l0_base, col_base) for z in zips]
    with cf.ProcessPoolExecutor(max_workers=a.workers) as ex:
        for n, res in enumerate(ex.map(_tarea, args), 1):
            l0s.append(res["level0"])
            l1s.append(res["level1"])
            ds.append(res["deriv"])
            print(f"  [{n}/{len(zips)}] {res['zip'].name}", flush=True)

    l0 = pd.concat(l0s, ignore_index=True)
    l1 = pd.concat(l1s, ignore_index=True)
    # el .tacocat del nivel de colección va en el orden publicado
    base0 = pd.read_parquet(src / ".tacocat" / "level0.parquet")
    falta = sorted(set(base0["id"]) - set(l0["id"]))
    if falta:
        print(f"  [aviso] {len(falta)} muestras del catálogo no salieron del rebuild: {falta[:3]}")
    l0 = l0.set_index("id").reindex(base0["id"]).reset_index()
    for c in base0.columns:
        try:
            l0[c] = l0[c].astype(base0[c].dtype)
        except Exception:  # noqa: BLE001
            pass
    l0 = l0[list(base0.columns)]
    base1 = pd.read_parquet(src / ".tacocat" / "level1.parquet")
    l1 = l1[list(base1.columns)]
    for c in base1.columns:
        try:
            l1[c] = l1[c].astype(base1[c].dtype)
        except Exception:  # noqa: BLE001
            pass
    (salida / ".tacocat").mkdir(exist_ok=True)
    l0.to_parquet(salida / ".tacocat" / "level0.parquet", index=False)
    l1.to_parquet(salida / ".tacocat" / "level1.parquet", index=False)
    (salida / ".tacocat" / "COLLECTION.json").write_bytes(col_base)
    for extra in ("README.md", "index.html"):
        p = src / ".tacocat" / extra
        if p.exists():
            shutil.copy2(p, salida / ".tacocat" / extra)
    for extra in ("README.md", "index.html"):
        p = src / extra
        if p.exists():
            shutil.copy2(p, salida / extra)
    pd.concat(ds, ignore_index=True).to_parquet("/tmp/opencode/derivadas.parquet", index=False)
    print(f"level0 {l0.shape} | level1 {l1.shape}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
