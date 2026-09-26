"""Measurements of the LES volume: real level heights, wind, mass and shape, edges, criteria and qa.

Everything is computed here, without depending on the CSVs of the projected bank, so that the les
bank is self-contained. The edge and criteria formulas are the same ones used by the projected bank
(01_criteria.py and 05_generate_tifs.py), so that the flags are comparable.
"""
from __future__ import annotations
import math
import re
import numpy as np
import netCDF4 as nc
from . import config as C

_PAT = re.compile(
    r"DO j = (\d+)\*\(jde \+ jds\)/90,\s*(\d+)\*\(jde \+ jds\)/90,\s*1\s*\n"
    r"\s*DO i = (\d+)\*\(ide \+ ids\)/120,\s*(\d+)\*\(ide \+ ids\)/120,\s*1\s*\n"
    r".*?tracer\(i, 1, j, P_plume(\d)\) = 1\.", re.S)


def fuentes() -> dict:
    """{(tipo, plume): (row0, row1, col0, col1)} inclusive, 0-based (Gorroño's solve_em.F)."""
    out = {}
    for tipo, sub in (("multi", "multisource"), ("area", "areasource")):
        txt = (C.CONF_WRF / sub / "solve_em.F").read_text(errors="replace")
        for j0, j1, i0, i1, p in _PAT.findall(txt):
            fj = lambda n: int(n) * (C.NY + 2) // 90
            fi = lambda n: int(n) * (C.NX + 2) // 120
            out[(tipo, int(p))] = (fj(j0) - 1, fj(j1) - 1, fi(i0) - 1, fi(i1) - 1)
    return out


def centro(caja):
    """(centre_row, centre_col, height_px, width_px)."""
    r0, r1, c0, c1 = caja
    return 0.5 * (r0 + r1), 0.5 * (c0 + c1), r1 - r0 + 1, c1 - c0 + 1


def _ficheros(tipo: str, ws: float):
    d = C.LES / C.CARPETA_TIPO[tipo] / C.carpeta_viento(ws)
    return sorted(p for p in d.iterdir() if p.name.startswith("auxhist24"))


def extras(tipo: str, ws: float, snaps) -> dict:
    """Per requested snapshot: real level heights (49), UTC time and domain means of U10/V10."""
    fs = _ficheros(tipo, ws)
    ds = [nc.Dataset(str(f)) for f in fs]
    try:
        PH = np.concatenate([d.variables["PH"][:] for d in ds], axis=0).astype("float32")
        PHB = np.concatenate([d.variables["PHB"][:] for d in ds], axis=0).astype("float32")
        H = (PH + PHB) / C.G
        del PH, PHB
        zab = H - H[:, 0][:, None, :, :]
        del H
        mid = 0.5 * (zab[:, :C.NZ] + zab[:, 1:C.NZ + 1])
        del zab
        altura = mid.mean(axis=(2, 3)).astype("float64")
        del mid
        U10 = np.concatenate([d.variables["U10"][:] for d in ds], axis=0).astype("float64")
        V10 = np.concatenate([d.variables["V10"][:] for d in ds], axis=0).astype("float64")
        u10 = U10.mean(axis=(1, 2))
        v10 = V10.mean(axis=(1, 2))
        del U10, V10
        times = ["".join(c.decode() for c in row) for d in ds for row in d.variables["Times"][:]]
    finally:
        for d in ds:
            d.close()
    return {s: {"level_heights": altura[s], "time_utc": times[s],
                "u10": float(u10[s]), "v10": float(v10[s])} for s in snaps}


def metricas(A, vol, fr, fc, fw, fh, level_heights, u10, v10) -> dict:
    """A: column map (ppb, 90 x 120). vol: volume (ppb, 49 x 90 x 120)."""
    pico = float(A.max())
    px = A.max(axis=0)
    W = C.VENTANA
    X = np.arange(W)
    Lg = np.log(np.clip(px[C.COL_BORDE + 1 - W:C.COL_BORDE + 1], 1e-3, None))
    slope = float(((X - X.mean()) * (Lg - Lg.mean())).sum() / ((X - X.mean()) ** 2).sum())
    mmin = float(px[C.INTERIOR[0]:C.INTERIOR[1]].min())
    bordes = {"upwind": A[:, C.COL_ENTRADA].max(), "downwind": A[:, C.COL_BORDE].max(),
              "bottom": A[C.RELAX, C.RELAX:C.COL_BORDE + 1].max(),
              "top": A[C.NY - 1 - C.RELAX, C.RELAX:C.COL_BORDE + 1].max()}
    D = A[C.RELAX:C.NY - C.RELAX, C.RELAX:C.NX - C.RELAX]
    anc = (D > C.RUIDO_REF).sum(axis=0) * C.PX
    c1 = int(np.floor(fc - fw / 2 - C.BARLOVENTO))
    U = A[C.RELAX:C.NY - C.RELAX, C.RELAX:c1 + 1]
    # if the source box leaves no upwind strip (a8, a9), there is no measurement: NaN, as in the projected bank
    if U.shape[1] == 0:
        fuera = float("nan")
    else:
        fuera = float(U.sum() / U.shape[1] * D.shape[1] / D.sum())

    # shape and mass on the interior volume (without the relaxation band), as in the projected bank
    vi = vol[:, C.RELAX:C.NY - C.RELAX, C.RELAX:C.NX - C.RELAX]
    xg = (np.arange(C.RELAX, C.NX - C.RELAX) - fc) * C.PX
    yg = (np.arange(C.RELAX, C.NY - C.RELAX) - fr) * C.PX
    zg = np.asarray(level_heights, dtype="float64")
    k, y, x = np.unravel_index(int(np.argmax(vi)), vi.shape)
    m = np.clip(vi, 0, None)
    tot = float(m.sum())
    mk = m.sum(axis=(1, 2))
    my = m.sum(axis=(0, 2))
    mx = m.sum(axis=(0, 1))
    cxd = float((mx * xg).sum() / tot)
    cxc = float((my * yg).sum() / tot)
    zef = float((mk * zg).sum() / tot)
    sxd = float(np.sqrt((mx * (xg - cxd) ** 2).sum() / tot))
    sxc = float(np.sqrt((my * (yg - cxc) ** 2).sum() / tot))
    szg = float(np.sqrt((mk * (zg - zef) ** 2).sum() / tot))
    pi = px[C.RELAX:C.NX - C.RELAX]
    cols = np.where(pi > C.RUIDO_REF)[0]
    length = float((cols.max() - cols.min()) * C.PX) if len(cols) else 0.0
    lev = np.where(vi.max(axis=(1, 2)) > C.RUIDO_REF)[0]
    dz = float(np.diff(zg).mean())
    height = float(zg[lev.max()] - zg[lev.min()] + dz) if len(lev) else 0.0
    ime = float(A[C.RELAX:C.NY - C.RELAX, C.RELAX:C.NX - C.RELAX].sum())

    ef = {kk: float(v) / pico for kk, v in bordes.items()}
    maxf = max(ef.values())
    apaga = bool(slope < 0 and bordes["downwind"] < mmin)
    ent_ok = bool(bordes["upwind"] < C.ENTRADA_FRAC * pico)
    lat_ok = bool(bordes["top"] < C.LATERAL_FRAC * pico and bordes["bottom"] < C.LATERAL_FRAC * pico)
    fuera_ok = bool(fuera < C.FUERA_MAX)
    ancho_ok = bool(anc[-1] <= C.ANCHO_BORDE_MAX)

    d = {
        "methane:peak_ppb": pico, "methane:peak_level": int(k),
        "methane:peak_downwind_m": float(xg[x]), "methane:peak_crosswind_m": float(yg[y]),
        "plume:mean_height_m": zef, "methane:height_std_m": szg,
        "plume:width_max_m": float(anc.max()),
        "plume:length_downwind_m": length, "plume:height_vertical_m": height,
        "methane:center_downwind_m": cxd, "methane:center_crosswind_m": cxc,
        "methane:sigma_downwind_m": sxd, "methane:sigma_crosswind_m": sxc,
        "methane:sigma_vertical_m": szg,
        "methane:column_peak_ppb": pico, "methane:ime_ppb_px": ime,
        "methane:mass_total_mol": tot / 1e9 * C.N_TOTAL_AIR * C.PIXEL_AREA,
        "methane:source_width_m": fw * C.PX, "methane:source_height_m": fh * C.PX,
        "edge:upwind_ppb": float(bordes["upwind"]), "edge:upwind_frac": ef["upwind"],
        "edge:downwind_ppb": float(bordes["downwind"]), "edge:downwind_frac": ef["downwind"],
        "edge:downwind_slope": slope, "edge:downwind_ref_ppb": mmin, "edge:downwind_ref_frac": mmin / pico,
        "edge:downwind_width_m": float(anc[-1]),
        "edge:top_ppb": float(bordes["top"]), "edge:top_frac": ef["top"],
        "edge:bottom_ppb": float(bordes["bottom"]), "edge:bottom_frac": ef["bottom"],
        "edge:foreign_mass_frac": fuera, "edge:max_frac": maxf,
        "edge:touches_edge": bool(maxf > C.TOCA_BORDE_FRAC),
        "criteria:fades_out": apaga, "criteria:ent_ok": ent_ok, "criteria:lat_ok": lat_ok,
        "criteria:fuera_ok": fuera_ok, "criteria:ancho_ok": ancho_ok,
        "criteria:pasa": bool(apaga and ent_ok and lat_ok and fuera_ok and ancho_ok),
        "array:nz": int(vol.shape[0]), "array:ny": int(vol.shape[1]), "array:nx": int(vol.shape[2]),
        "array:pixel_size_m": C.PX, "array:relax_px": C.RELAX,
        "array:z_top_m": float(zg[-1]), "array:dz_mean_m": dz,
        "grid:level_heights_m": [float(v) for v in zg],
        "wind:u10_mean": float(u10), "wind:v10_mean": float(v10),
        "wind:speed_10m": float(math.hypot(u10, v10)),
        "wind:dir_10m": float(math.degrees(math.atan2(u10, v10)) % 360.0),
        "qa:min_ppb": float(vol.min()), "qa:max_ppb": float(vol.max()),
        "qa:zeros_frac": float((vol == 0).mean()),
    }
    return d
