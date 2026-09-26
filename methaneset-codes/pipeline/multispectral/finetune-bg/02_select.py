"""Step 2 · Picks the 3 best backgrounds per site among the raw candidates. In parallel.

Faithful to MARS:
  similarity = mean(|target - background * gain|), gain = mean_target/mean_background per band,
  over methane-insensitive bands (B02,B03,B04,B11; in Landsat B2,B3,B4,B6), with co-registration.
And also:
  - OmniCloudMask cloud mask (cloud+shadow fraction).
  - background methane through the SWIR2/SWIR1 ratio against the pool median.
It filters the flagged ones and keeps the top 3 most similar.

Usage:
  python 02_select.py --taco .../methaneset-l89-finetune --raw .../methaneset-l89-finetune-bg-raw \
                           --platform LC08 [--workers 8]
Output: <raw>/seleccion_<taco>_<platform>.csv
"""
from __future__ import annotations

import argparse
import csv
import pathlib
import re
import warnings
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import rasterio
import tacoreader
from scipy.ndimage import shift as nd_shift
from skimage.registration import phase_cross_correlation

COREG_REF_IDX = 3
SIM_BAND_INDEXES = {"landsat": [1, 2, 3, 5], "s2": [1, 2, 3, 11]}
SWIR_IDX = {"landsat": (5, 6), "s2": (11, 12)}
RGN_IDX = {"landsat": (3, 2, 4), "s2": (3, 2, 7)}
PLATFORM_FAMILY = {"LC08": "landsat", "LC09": "landsat", "S2": "s2"}

try:
    from omnicloudmask import predict_from_array as _ocm
    _HAS_OCM = True
except Exception:  # noqa: BLE001
    _HAS_OCM = False

_DATA = None
_CFG = {}


def _slug(t):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(t)).strip("_")


def coregister(target, bg, ref=COREG_REF_IDX):
    t, b = target[ref].astype(np.float64), bg[ref].astype(np.float64)
    if not np.any(t > 0) or not np.any(b > 0) or t.std() == 0 or b.std() == 0:
        return bg, np.nan
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            syx, err, _ = phase_cross_correlation(t, b, upsample_factor=10)
    except Exception:  # noqa: BLE001
        return bg, np.nan
    if not np.any(syx):
        return bg, float(err)
    return np.stack([nd_shift(bg[k], syx, order=1, mode="nearest") for k in range(bg.shape[0])]), float(err)


def diferencia_mars(target, bg, band_idx):
    """mean(|t - b*gain|) with per-band gain = mean(t)/mean(b). The gain cancels the
    scale (Cesar's target is in reflectance and the background in DN)."""
    t = target[band_idx].astype(np.float64)
    b = bg[band_idx].astype(np.float64)
    diffs = []
    for k in range(t.shape[0]):
        den = b[k].mean()
        g = t[k].mean() / den if den > 0 else 1.0
        diffs.append(np.abs(t[k] - g * b[k]).mean())
    return float(np.mean(diffs))


def cloud_frac(bg, rgn):
    if not _HAS_OCM:
        return np.nan
    arr = np.stack([bg[rgn[0]], bg[rgn[1]], bg[rgn[2]]]).astype(np.float32)
    try:
        pred = np.asarray(_ocm(arr)).squeeze()
    except Exception:  # noqa: BLE001
        return np.nan
    return float(np.mean(np.isin(pred, (1, 2, 3)))) if pred.size else np.nan


def methane_frac(bg, med, swir1, swir2, thr=0.92):
    b1, b2 = bg[swir1].astype(np.float64), bg[swir2].astype(np.float64)
    m1, m2 = med[swir1].astype(np.float64), med[swir2].astype(np.float64)
    v = (b1 > 0) & (b2 > 0) & (m1 > 0) & (m2 > 0)
    if not v.any():
        return np.nan
    r = np.divide(b2, b1, out=np.zeros_like(b2), where=b1 > 0) / np.divide(
        m2, m1, out=np.ones_like(m2), where=m1 > 0)
    return float(np.mean(r[v] < thr))


def _init(taco, raw, platform, top, cloud_max, methane_max):
    tacoreader.use("pandas")
    globals()["_DATA"] = tacoreader.load(taco).data
    _CFG.update(dict(taco=taco, raw=raw, platform=platform, top=top,
                     cloud_max=cloud_max, methane_max=methane_max,
                     fam=PLATFORM_FAMILY[platform]))


def _sitio(i):
    row = _DATA.iloc[i]
    if str(row.get("type", "FOLDER")) != "FOLDER":
        return [], []
    if PLATFORM_FAMILY.get(str(row.get("satellite:platform")), "s2") != _CFG["fam"]:
        return [], []
    tid = _slug(row.get("id"))
    site = pathlib.Path(_CFG["raw"]) / _CFG["platform"] / pathlib.Path(str(_CFG["taco"]).rstrip("/")).name / tid
    bgs = sorted(site.glob("bg*.tif")) if site.is_dir() else []
    if not bgs:
        return [], []
    try:
        with rasterio.open(_DATA.read(i).read(0)) as s:
            target = s.read()
    except Exception:  # noqa: BLE001
        return [], []
    fam = _CFG["fam"]
    band_idx, (s1, s2), rgn = SIM_BAND_INDEXES[fam], SWIR_IDX[fam], RGN_IDX[fam]
    cand, arrays = [], []
    for p in bgs:
        with rasterio.open(p) as s:
            bg = s.read()
        if bg.shape != target.shape:
            continue
        bg, _ = coregister(target, bg)
        cand.append(p)
        arrays.append(bg)
    if not cand:
        return [], []
    med = np.median(np.stack(arrays), axis=0)
    scored = []
    for p, bg in zip(cand, arrays):
        d = diferencia_mars(target, bg, band_idx)
        cb = cloud_frac(bg, rgn)
        m = methane_frac(bg, med, s1, s2)
        scored.append((p.name, d, cb, m))
    scored.sort(key=lambda x: (np.isnan(x[1]), x[1]))
    # clean = passes cloud (2%) and methane (3%). If there are not 3, it relaxes cloud to 35% (MARS 2nd pass).
    clean = [s for s in scored if not (s[2] > _CFG["cloud_max"]) and not (s[3] > _CFG["methane_max"])]
    pick = clean[:_CFG["top"]]
    if len(pick) < _CFG["top"]:
        relax = [s for s in scored if s not in pick and not (s[2] > _CFG["cloud_max"]) and not (s[3] > _CFG["methane_max"])]
        pick += relax[:_CFG["top"] - len(pick)]
    out = []
    for rank, (name, d, cb, m) in enumerate(pick, 1):
        date = re.search(r"_(\d{8})\.tif$", name).group(1)
        out.append([tid, _CFG["platform"], date, name, f"{d:.6f}", f"{cb:.4f}", f"{m:.4f}", rank])
    full = [[tid, _CFG["platform"], name, f"{d:.6f}", f"{cb:.4f}", f"{m:.4f}",
             int(not (cb > _CFG["cloud_max"])), int(not (m > _CFG["methane_max"]))]
            for (name, d, cb, m) in scored]
    return out, full


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--taco", required=True)
    ap.add_argument("--raw", required=True)
    ap.add_argument("--platform", required=True, choices=["LC08", "LC09", "S2"])
    ap.add_argument("--top", type=int, default=3)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--cloud-max", type=float, default=0.10)
    ap.add_argument("--methane-max", type=float, default=0.03)
    args = ap.parse_args()

    raiz = pathlib.Path(args.raw)
    taco_name = pathlib.Path(str(args.taco).rstrip("/")).name
    tacoreader.use("pandas")
    data = tacoreader.load(args.taco).data
    idx = [i for i in range(len(data)) if str(data.iloc[i].get("type", "FOLDER")) == "FOLDER"]
    if args.limit:
        idx = idx[:args.limit]
    print(f"sites to process: {len(idx)}  workers={args.workers}")

    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init,
                             initargs=(args.taco, args.raw, args.platform, args.top,
                                       args.cloud_max, args.methane_max)) as ex:
        resultados = list(ex.map(_sitio, idx, chunksize=4))

    rows = [r for out, _ in resultados for r in out]
    full = [r for _, fu in resultados for r in fu]
    outp = raiz / f"seleccion_{taco_name}_{args.platform}.csv"
    with open(outp, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["taco_id", "platform", "date", "file", "difference", "cloud", "methane", "rank"])
        w.writerows(rows)
    fullp = raiz / f"candidatos_{taco_name}_{args.platform}.csv"
    with open(fullp, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["taco_id", "platform", "file", "difference", "cloud", "methane", "pass_cloud", "pass_methane"])
        w.writerows(full)
    n3 = sum(1 for out, _ in resultados if len(out) >= 3)
    print(f"rows: {len(rows)} | sites with 3: {n3} of {len(idx)} -> {outp} (+ {fullp})")


if __name__ == "__main__":
    main()
