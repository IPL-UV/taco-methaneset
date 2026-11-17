"""MODULE 14b: wide-window matched filter (wwmf) for the EMIT granules.

Adds a fourth retrieval leaf, `wwmf.tif`, computed with the same methanex engine
and the same precision (float64) as mf/rmf/mag1c, but over a wider spectral window.

Window
------
Roger et al. (2024) define the wide-window matched filter as the same matched
filter run over 1000-2500 nm instead of the classical 2100-2450 nm; the broader
window constrains the background better and reduces retrieval artifacts. We
exclude the two water-vapour absorption windows (1340-1450 and 1800-1960 nm),
where the stored radiance is quantised more coarsely by the DISCARD_LSB pass of
MODULE 14, so including them only adds noise. Against the IMEO masks on the
published dataset this window raises the per-granule AUC from 0.714 to 0.738 and
cuts the granules below 0.6 from 45 to 10.

Output
------
One single-band float32 GeoTIFF per granule with the profile and nodata
convention of mf.tif (tile 128, zstd, sensor grid, no CRS), band description
`wwmf_ppm_m`. Written to `<granule>/wwmf.tif`, or to `$OUT/<granule>.tif` when
OUT is set, always through a `.part` file so an interrupted run never leaves a
truncated leaf behind.

Usage:
  python pipeline/emit/14b_generate_wwmf.py                     # in place, resumable
  OUT=/path/staging WORKERS=8 python pipeline/emit/14b_generate_wwmf.py
  LIMIT=20 WORKERS=8 python pipeline/emit/14b_generate_wwmf.py
"""
import multiprocessing as mp
import os
import pathlib
import sys

import numpy as np

sys.path.insert(0, os.environ.get("METHANEX_SRC", "/data/users/julio/methanset"))

TACO = pathlib.Path(os.environ.get(
    "METHANESET_EMIT", "/data/databases/METHANESET_TACOS/methaneset-emit"))
SAFETENSORS = pathlib.Path(os.environ.get("METHANEX_TEMPLATE",
                                          "/data/users/julio/methanset/ch4_emit.safetensors"))
OUT = pathlib.Path(os.environ["OUT"]) if os.environ.get("OUT") else None
LEAF = "wwmf.tif"
WIDE = (1000, 2500)
WV = ((1340, 1450), (1800, 1960))   # zonas con LSB descartados: intervalo semiabierto, como 10b
FILL = -9990.0                       # radiancia de relleno del producto
NODATA = -9999.0
LIMIT = int(os.environ.get("LIMIT", "0")) or None
WORKERS = int(os.environ.get("WORKERS", "8"))
BATCH = 64

_ENGINE = {}
_DEVICE = None


def _engine():
    """Motor por proceso, con la plantilla recortada a las bandas que se guardan sin perdida."""
    import torch
    from methanex import MethaneRetrieval
    from safetensors.torch import load_file

    if _DEVICE in _ENGINE:
        return _ENGINE[_DEVICE]
    t = load_file(str(SAFETENSORS))
    centers, full = t["centers"], t["template"]
    keep = ((centers >= WIDE[0]) & (centers < WIDE[1])).numpy()
    for lo, hi in WV:
        keep &= ~((centers >= lo) & (centers < hi)).numpy()
    idx = np.flatnonzero(keep)
    eng = MethaneRetrieval(device=_DEVICE, dtype=torch.float64,
                           wavelength_range=(float(centers[idx[0]]), float(centers[idx[-1]])))
    eng.template = full[keep].to(device=eng.device, dtype=torch.float64)
    eng.band_mask = torch.as_tensor(keep)
    _ENGINE[_DEVICE] = (keep, eng)
    return _ENGINE[_DEVICE]


def worker(granule):
    import rasterio
    import torch
    from methanex import MFConfig

    gran = pathlib.Path(granule)
    dst = (OUT / f"{gran.name}.tif") if OUT else (gran / LEAF)
    part = dst.with_suffix(".tif.part")
    if dst.exists() or part.exists():
        # un .part huerfano es de una corrida interrumpida: se rehace
        part.unlink(missing_ok=True)
        if dst.exists():
            return "skip"
    try:
        keep, eng = _engine()
        with rasterio.open(gran / "radiance.tif") as src:
            rad = src.read().astype("float32")
        bad = rad[rad.shape[0] // 2] <= FILL   # (filas, columnas): rasterio da (bandas, filas, cols)
        x = np.ascontiguousarray(np.transpose(rad, (1, 2, 0))[:, :, keep])
        del rad
        res = eng.retrieve(torch.from_numpy(x), MFConfig(batch_size=BATCH), display_pbar=False)
        ch4 = res.mf.cpu().numpy().astype("float64")
        del x, res
        torch.cuda.empty_cache()
        ch4[bad] = NODATA
        with rasterio.open(gran / "mf.tif") as ref:
            prof = ref.profile.copy()
        prof.update(count=1, dtype="float32", nodata=NODATA, compress="zstd",
                    tiled=True, blockxsize=128, blockysize=128, interleave="band")
        dst.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(part, "w", **prof) as out:
            out.write(np.nan_to_num(ch4, nan=NODATA, posinf=NODATA, neginf=NODATA).astype("float32"), 1)
            out.set_band_description(1, "wwmf_ppm_m")
        os.replace(part, dst)          # solo aparece con el nombre final si esta completo
        return "ok"
    except Exception as e:  # noqa: BLE001
        part.unlink(missing_ok=True)
        return f"error {type(e).__name__}: {e}"


def main():
    import tacoreader

    tacoreader.use("pandas")
    l0 = tacoreader.load(str(TACO)).flatten()
    granules = sorted({str(p).rsplit("/", 1)[0] for p in l0["gdal_vsi"].dropna()})
    if LIMIT:
        granules = granules[:LIMIT]
    devices = [f"cuda:{i % 2}" for i in range(WORKERS)]
    print(f"{len(granules)} granulos · {WORKERS} procesos · hoja {LEAF} · float64")

    counts = {"ok": 0, "skip": 0, "error": 0}
    with mp.get_context("fork").Pool(WORKERS) as pool:
        # cada worker recibe su device al arrancar: reasignamos por tandas de WORKERS
        jobs = [(g, devices[i % WORKERS]) for i, g in enumerate(granules)]
        for i, res in enumerate(pool.imap(_run, jobs, chunksize=1), 1):
            counts[res.split()[0]] = counts.get(res.split()[0], 0) + 1
            if res.startswith("error"):
                print(f"  {i}/{len(jobs)} {res}", flush=True)
            elif i % 50 == 0:
                print(f"  {i}/{len(jobs)} · {counts}", flush=True)
    print(f"\nhecho: {counts}")


def _run(job):
    gran, device = job
    global _DEVICE
    _DEVICE = device
    return worker(gran)


if __name__ == "__main__":
    main()
