"""Step 5 · One GeoTIFF per physical plume and sun position, with its metadata row.

Usage
  python 05_generate_tifs.py --piloto DIR          a single plume (point source 5, 7 m/s, snap 93)
  python 05_generate_tifs.py --salida DIR [-p N]   every plume in trabajo/seleccion.csv, N processes

Each physical plume writes tifs/<plume_uid>/*.tif and filas/<plume_uid>.parquet. If the parquet
of a plume already exists, it is skipped: the run can be resumed.

GeoTIFF: float32, ZSTD, 20 m, north up (row 0 = north), wind to the right. The transform puts
the centre of the source pixel at (0, 0) m, so the coordinates are metres relative to the
source. No CRS: it is a local frame.
"""
from __future__ import annotations
import argparse
import math
import sys
import time
from multiprocessing import Pool
from pathlib import Path
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import Affine
from banco import config as C
from banco.geometria import fuentes, centro, rejilla
from banco.proyeccion import proyectar, recortar
from banco.esquema import COLUMNAS, emisor, uid as nombre_uid

NODOS = rejilla()
FUENTES = fuentes()


def medidas(A: np.ndarray, vol: np.ndarray, fc: float, fw: float) -> dict:
    """Measurements of the physical plume on the full column map (90 x 120), the same ones used by
    the criteria, and z_eff on the volume without the relaxation band."""
    px = A.max(axis=0)
    pico = float(A.max())
    W = C.VENTANA
    X = np.arange(W)
    Lg = np.log(np.clip(px[C.COL_BORDE + 1 - W:C.COL_BORDE + 1], 1e-3, None))
    slope = float(((X - X.mean()) * (Lg - Lg.mean())).sum() / ((X - X.mean()) ** 2).sum())
    lado = {"upwind": A[:, C.COL_ENTRADA].max(), "downwind": A[:, C.COL_BORDE].max(),
            # sides: only the stretch kept in the GeoTIFF (columns 5 to 114), as in 01_criteria
            "bottom": A[C.RELAX, C.RELAX:C.COL_BORDE + 1].max(),
            "top": A[C.NY - 1 - C.RELAX, C.RELAX:C.COL_BORDE + 1].max()}
    mmin = float(px[C.INTERIOR[0]:C.INTERIOR[1]].min())
    m_k = np.clip(vol.sum(axis=(1, 2)), 0, None)
    z_eff = float((np.arange(C.NZ) * C.DZ * m_k).sum() / m_k.sum())
    d = {"methane:peak_ppb": pico, "methane:mean_height_m": z_eff, "edge:downwind_slope": slope,
         "edge:downwind_ref_ppb": mmin, "edge:downwind_ref_frac": mmin / pico}
    for k, v in lado.items():
        d[f"edge:{k}_ppb"] = float(v)
        d[f"edge:{k}_frac"] = float(v) / pico
    # foreign methane: upwind strip (up to 100 m before the box), extended over the 110 columns
    D = A[C.RELAX:C.NY - C.RELAX, C.RELAX:C.NX - C.RELAX]
    anc = (D > C.RUIDO_REF).sum(axis=0) * C.PX          # visible width per column
    d["methane:width_max_m"] = float(anc.max())
    d["edge:downwind_width_m"] = float(anc[-1])
    c1 = int(np.floor(fc - fw / 2 - C.BARLOVENTO))
    U = A[C.RELAX:C.NY - C.RELAX, C.RELAX:c1 + 1]
    d["qa:foreign_mass_frac"] = float(U.sum() / U.shape[1] * D.shape[1] / D.sum())
    return d


def escribir(path: Path, arr_su: np.ndarray, src_row_su: float, src_col: float, etiquetas: dict):
    """Writes north up. arr_su has row 0 at the south (LES frame)."""
    arr = np.flipud(arr_su).astype("float32")
    h, w = arr.shape
    src_row = (h - 1) - src_row_su
    tr = Affine(C.PX, 0, -(src_col + 0.5) * C.PX, 0, -C.PX, (src_row + 0.5) * C.PX)
    with rasterio.open(path, "w", driver="GTiff", height=h, width=w, count=1, dtype="float32",
                       compress="ZSTD", zstd_level=9, predictor=3, transform=tr) as dst:
        dst.write(arr, 1)
        dst.update_tags(**etiquetas)
        dst.set_band_description(1, "XCH4 enhancement [ppb] at 3000 kg/h")
    return h, w, src_row


def pluma(tipo, ws, p, snap, full_ppb, sf, salida: Path) -> int:
    uid = nombre_uid(tipo, p, ws, snap)
    fparquet = salida / "filas" / f"{uid}.parquet"
    if fparquet.exists():
        return 0
    dtif = salida / "tifs" / uid
    dtif.mkdir(parents=True, exist_ok=True)
    A = full_ppb.sum(0)
    vol = full_ppb[:, C.RELAX:-C.RELAX, C.RELAX:-C.RELAX]
    tot = float(vol.sum())
    fr, fc, fh, fw = centro(FUENTES[(tipo, p)])
    base = {"methane:plume_uid": uid, "methane:sim_type": tipo, "methane:emitter": emisor(tipo, p),
            "methane:wind_speed": ws, "methane:snapshot_index": snap, "methane:snapshot_minute": snap / 2,
            "methane:scale_factor": sf, **medidas(A, vol, fc, fw),
            "methane:source_width_m": fw * C.PX, "methane:source_height_m": fh * C.PX}
    filas = []
    for nodo, (anillo, sza, raa) in enumerate(NODOS):
        mapa, f0, c0 = proyectar(vol, sza, raa)
        fuera = abs(tot - float(mapa.sum())) / tot
        arr, f, c, qa = recortar(mapa, f0, c0)
        src_row_su = (fr - C.RELAX) - f
        src_col = (fc - C.RELAX) - c
        nombre = f"{uid}_sza{sza:04.1f}_raa{raa:05.1f}"
        et = {"units": C.UNIDADES, "q_ref_kg_h": C.Q_REF, "plume_uid": uid, "sza": f"{sza:.4f}",
              "raa": f"{raa:.4f}", "convention": "north-up; wind to +x; RAA = sun azimuth, "
              "counterclockwise from the wind; coordinates in m from the source"}
        h, w, src_row = escribir(dtif / f"{nombre}.tif", arr, src_row_su, src_col, et)
        filas.append({"id": nombre, "path": f"tifs/{uid}/{nombre}.tif", **base,
                      "sun:sza": sza, "sun:raa": raa,
                      "array:width": w, "array:height": h,
                      "array:source_col": src_col, "array:source_row": src_row,
                      # pixels beyond (+) or short of (-) the domain without relaxation
                      # (columns 5 to 114, rows 5 to 84); (fr, fc) is the source in the LES, row 0 = south
                      "array:grow_upwind_px": round(src_col - (fc - C.RELAX)),
                      "array:grow_downwind_px": round((w - 1 - src_col) - (C.COL_BORDE - fc)),
                      "array:grow_top_px": round(src_row - (C.NY - 1 - C.RELAX - fr)),
                      "array:grow_bottom_px": round((h - 1 - src_row) - (fr - C.RELAX)),
                      **qa})
        assert fuera < 1e-9, f"{nombre}: mass outside the canvas {fuera:.1e}"
    fparquet.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(filas)[COLUMNAS].to_parquet(fparquet, index=False)
    return len(filas)


def grupo(args):
    """One wind speed and type run: loaded once with all its plumes and snapshots."""
    tipo, ws, pares, salida = args
    sys.path.insert(0, C.LOADER)
    from methanebank.loader import load_single_simulation
    t0 = time.time()
    plumas = sorted({p for p, _ in pares})
    snaps = sorted({s for _, s in pares})
    n_tot, sf = load_single_simulation(C.LES, tipo, ws, plumas, snaps)
    n = 0
    for p, s in pares:
        full = np.asarray(n_tot[(p, s)], dtype="float64") / (C.N_TOTAL_AIR * C.PIXEL_AREA) * 1e9 * sf[p]
        n += pluma(tipo, ws, p, s, full, float(sf[p]), Path(salida))
    print(f"{tipo} {ws:4.1f} m/s: {len(pares)} plumes, {n} files, {time.time() - t0:.0f} s", flush=True)
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--piloto", type=Path)
    ap.add_argument("--salida", type=Path)
    ap.add_argument("-p", "--procesos", type=int, default=8)
    a = ap.parse_args()
    sel = pd.read_csv(C.TRABAJO / "seleccion.csv")
    if a.piloto:
        sel = sel[(sel.tipo == "multi") & (sel.plume == 5) & (sel.ws == 7.0) & (sel.snap == 93)]
        salida = a.piloto
    else:
        salida = a.salida
    salida.mkdir(parents=True, exist_ok=True)
    trabajos = [(t, float(w), [(int(r.plume), int(r.snap)) for r in g.itertuples()], str(salida))
                for (t, w), g in sel.groupby(["tipo", "ws"])]
    t0 = time.time()
    if len(trabajos) == 1:
        total = grupo(trabajos[0])
    else:
        with Pool(a.procesos) as pool:
            total = sum(pool.imap_unordered(grupo, trabajos))
    print(f"done: {total} files in {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
