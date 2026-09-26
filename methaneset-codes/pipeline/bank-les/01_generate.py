"""Step 1 · One 49-band GeoTIFF and its metadata row per volume (physical plume and snapshot).

Usage
  python 01_generate.py --piloto                 a single volume, in /tmp, to check
  python 01_generate.py --salida RAW [-p N]      all volumes, N processes

Output: tifs/<uid>/<uid>.tif (49 x 90 x 120, float32, ppb per level at 3000 kg/h) and
filas/<uid>.parquet. If the parquet already exists, it is skipped: the run can be resumed.

The GeoTIFF is stored north up (like the projected bank), with the wind towards +x. The centre
of the source pixel is at (0, 0) m. No crop of the relaxation band and no threshold: raw.
"""
from __future__ import annotations
import argparse
import sys
import time
from multiprocessing import Pool
from pathlib import Path
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import Affine
from bankles import config as C
from bankles.metricas import fuentes, centro, extras, metricas
from bankles.esquema import COLUMNAS

FUENTES = fuentes()


def escribir(path: Path, vol: np.ndarray, fr: float, fc: float, level_heights) -> None:
    # the netCDF stores row 0 at the south; the GeoTIFF is stored north up, like the projected one.
    # CAREFUL: flip only the rows (axis 1); np.flipud on 3D would flip the level axis
    arr = vol[:, ::-1, :].astype("float32")
    nz, ny, nx = arr.shape
    tr = Affine(C.PX, 0, -(fc + 0.5) * C.PX, 0, -C.PX, (C.NY - fr - 0.5) * C.PX)
    with rasterio.open(path, "w", driver="GTiff", height=ny, width=nx, count=nz, dtype="float32",
                       compress="ZSTD", zstd_level=9, predictor=3, transform=tr) as dst:
        dst.write(arr)
        for k in range(nz):
            dst.set_band_description(k + 1, f"z={level_heights[k]:.1f} m")
        dst.update_tags(units=C.UNIDADES, q_ref_kg_h=C.Q_REF,
                        orientation="north-up (row 0 = north); wind towards +x (right); "
                                    "coordinates in m from the source; outer 5 px are a relaxation band")


def volumen(tipo: str, ws: float, p: int, snap: int, ext: dict, full: np.ndarray, sf: float,
            salida: Path) -> int:
    u = C.uid(tipo, p, ws, snap)
    fpar = salida / "filas" / f"{u}.parquet"
    if fpar.exists():
        return 0
    dtif = salida / "tifs" / u
    dtif.mkdir(parents=True, exist_ok=True)
    A = full.sum(0)
    fr, fc, fh, fw = centro(FUENTES[(tipo, p)])
    met = metricas(A, full, fr, fc, fw, fh, ext["level_heights"], ext["u10"], ext["v10"])
    escribir(dtif / f"{u}.tif", full, fr, fc, ext["level_heights"])
    fila = {
        "id": u, "path": f"tifs/{u}/{u}.tif",
        "methane:plume_uid": u, "methane:sim_type": tipo, "methane:emitter": C.emisor(tipo, p),
        "methane:plume_id": p, "methane:wind_speed": ws, "methane:snapshot_index": snap,
        "methane:snapshot_minute": snap / 2, "methane:time_utc": ext["time_utc"],
        "methane:source_file": f"{C.CARPETA_TIPO[tipo]}/{C.carpeta_viento(ws)}",
        "methane:q_ref_kg_h": C.Q_REF, "methane:units": C.UNIDADES, "methane:scale_factor": sf,
        **met}
    fpar.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([fila])[COLUMNAS].to_parquet(fpar, index=False)
    return 1


def pares_de(tipo, ws, proy):
    """(plume, snapshot): the 5 min of every emitter plus the exact snapshots of the projected bank."""
    base = [(p, s) for p in C.PLUMAS for s in C.SNAPS]
    extra = [(int(r.plume), int(r.snap)) for r in proy[(proy.tipo == tipo) & (proy.ws == ws)].itertuples()]
    return sorted(set(base + extra))


def grupo(args):
    tipo, ws, salida = args
    sys.path.insert(0, C.LOADER)
    from methanebank.loader import load_single_simulation
    import pandas as pd
    t0 = time.time()
    proy = pd.read_csv(C.PROYECTADO)
    pares = pares_de(tipo, ws, proy)
    snaps = sorted({s for _, s in pares})
    plumas = sorted({p for p, _ in pares})
    ext = extras(tipo, ws, snaps)
    n_tot, sf = load_single_simulation(C.LES, tipo, ws, plumas, snaps)
    n = 0
    for p, s in pares:
        full = np.asarray(n_tot[(p, s)], dtype="float64") / (C.N_TOTAL_AIR * C.PIXEL_AREA) * 1e9 * sf[p]
        n += volumen(tipo, ws, p, s, ext[s], full, float(sf[p]), Path(salida))
    print(f"{tipo} {ws:4.1f} m/s: {n} volumes, {time.time() - t0:.0f} s", flush=True)
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--salida", type=Path, default=C.RAW)
    ap.add_argument("-p", "--procesos", type=int, default=6)
    ap.add_argument("--piloto", action="store_true")
    a = ap.parse_args()
    if a.piloto:
        tipo, ws, p, s = "multi", 3.0, 5, 70
        ext = extras(tipo, ws, [s])
        sys.path.insert(0, C.LOADER)
        from methanebank.loader import load_single_simulation
        n_tot, sf = load_single_simulation(C.LES, tipo, ws, [p], [s])
        full = np.asarray(n_tot[(p, s)], dtype="float64") / (C.N_TOTAL_AIR * C.PIXEL_AREA) * 1e9 * sf[p]
        salida = Path("/tmp/opencode/bankles-piloto")
        n = volumen(tipo, ws, p, s, ext[s], full, float(sf[p]), salida)
        print("pilot:", n, "volume in", salida)
        return
    a.salida.mkdir(parents=True, exist_ok=True)
    trabajos = [(t, float(w), str(a.salida)) for t in C.TIPOS for w in C.WS]
    t0 = time.time()
    with Pool(a.procesos) as pool:
        total = sum(pool.imap_unordered(grupo, trabajos))
    print(f"done: {total} volumes in {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
