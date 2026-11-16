#!/usr/bin/env python
"""
SNR real vs control de mascara aleatoria — MethaneSET finetune.

Definicion exacta (Eq. snr del paper, misma logica que figure_tech_01.py):

    SNR_k = |mean(r_k | pluma) - mean(r_k | fondo)| / std(r_k | fondo)
    r_k   = target_k / bg0_k        (bg0 = fondo MARS; 'reference' en TACOv2)

Control: la mascara real de pluma se traslada a posiciones aleatorias
(uniforme sobre desplazamientos validos dentro del parche, descartando los
que salen). Misma area y forma. N_CTRL realizaciones; por muestra y banda
se guarda la mediana de esas realizaciones.

Reglas identicas al codigo de la figura:
  - pixeles validos: bg0 > 0 y ratio finito
  - se exige n_pluma >= 5, n_fondo >= 20 y std_fondo > 1e-9

Salidas (en --out):
  snr_samples_<tag>.csv   una fila por (muestra, banda) con real + control
  run_meta_<tag>.json     parametros y versiones

Uso:
  python compute_snr_control.py --sensor l89
  python compute_snr_control.py --sensor s2
  python compute_snr_control.py --sensor l89 --dataset-dir <folder>/methaneset-l89-finetune --leaf-bg reference --tag tacov2_l89
"""

import argparse
import json
import os
import pathlib
import time

import numpy as np
import pandas as pd
import rasterio
import tacoreader

SENSORS = {
    "s2": {
        "bands": ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08",
                  "B8A", "B09", "B10", "B11", "B12"],
        "wl": [443, 490, 560, 665, 705, 740, 783, 842, 865, 945, 1375, 1610, 2190],
        "path": os.path.join(os.environ.get("METHANESET_DATA", ""),
                             "methaneset-s2-finetune"),
    },
    "l89": {
        "bands": ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08",
                  "B09", "B10", "B11"],
        "wl": [443, 482, 562, 655, 865, 1610, 2200, 590, 1373, 10895, 12005],
        "path": os.path.join(os.environ.get("METHANESET_DATA", ""),
                             "methaneset-l89-finetune"),
    },
}


def load_pivot(dataset_dir):
    tacoreader.use("pandas")
    d = tacoreader.load(str(dataset_dir)).flatten()
    piv = d.pivot_table(index="l0:id", columns="l1:id",
                        values="gdal_vsi", aggfunc="first").reset_index()
    return piv


def band_stat(r, finite, mask, w):
    vp = finite & mask
    npix = int(vp.sum())
    if npix < 5:
        return None
    vb = finite & ~mask
    nb = int(vb.sum())
    if nb < 20:
        return None
    rp = r[vp]
    rb = r[vb]
    mu_p = float(rp.mean())
    mu_b = float(rb.mean())
    sig = float(rb.std())
    if sig < 1e-9:
        return None
    return {
        "snr": abs(mu_p - mu_b) / sig,
        "diff": mu_p - mu_b,
        "mu_p": mu_p,
        "mu_b": mu_b,
        "n_plume": npix,
        "n_bg": nb,
    }


def bbox_of(mask):
    rs, cs = np.nonzero(mask)
    if len(rs) == 0:
        return None
    return int(rs.min()), int(rs.max()), int(cs.min()), int(cs.max())


def random_placements(rng, H, W, bbox, n_ctrl, central_frac=None, max_tries=50000):
    """Desplazamientos enteros uniformes; se descarta si la mascara sale del parche.

    Si central_frac no es None, el centroide de la mascara desplazada debe
    quedar ademas dentro de una caja central de ese tamano (fraccion del lado).
    """
    r0, r1, c0, c1 = bbox
    rc = (r0 + r1) / 2.0
    cc = (c0 + c1) / 2.0
    out = []
    tries = 0
    while len(out) < n_ctrl and tries < max_tries:
        tries += 1
        dy = int(rng.integers(-(H - 1), H))
        dx = int(rng.integers(-(W - 1), W))
        if r0 + dy < 0 or r1 + dy > H - 1:
            continue
        if c0 + dx < 0 or c1 + dx > W - 1:
            continue
        if central_frac is not None:
            lim_r = central_frac * (H - 1) / 2.0
            lim_c = central_frac * (W - 1) / 2.0
            if abs(rc + dy - (H - 1) / 2.0) > lim_r:
                continue
            if abs(cc + dx - (W - 1) / 2.0) > lim_c:
                continue
        out.append((dy, dx))
    return out


def shifted_mask(plume, bbox, dy, dx):
    r0, r1, c0, c1 = bbox
    m = np.zeros_like(plume)
    m[r0 + dy:r1 + dy + 1, c0 + dx:c1 + dx + 1] = plume[r0:r1 + 1, c0:c1 + 1]
    return m


def process_sample(row, bands, leaf_bg, rng, n_ctrl, central_frac=None):
    with rasterio.open(row["target"]) as src:
        target = src.read().astype(np.float32)
    with rasterio.open(row[leaf_bg]) as src:
        bg = src.read().astype(np.float32)
    with rasterio.open(row["plume"]) as src:
        plume = src.read(1) == 1

    H, W = plume.shape
    bbox = bbox_of(plume)
    placements = (random_placements(rng, H, W, bbox, n_ctrl, central_frac)
                  if bbox else [])

    n_bands = min(target.shape[0], bg.shape[0], len(bands))
    records = []
    for b in range(n_bands):
        r = target[b] / np.where(bg[b] > 0, bg[b], np.nan)
        finite = np.isfinite(r)
        real = band_stat(r, finite, plume, W)
        if real is None:
            continue
        ctrl = []
        for dy, dx in placements:
            m = shifted_mask(plume, bbox, dy, dx)
            st = band_stat(r, finite, m, W)
            if st is not None:
                ctrl.append(st["snr"])
        ctrl = np.asarray(ctrl, dtype=float)
        rec = {
            "band": bands[b],
            "snr_real": real["snr"],
            "signed_diff": real["diff"],
            "mean_ratio_plume": real["mu_p"],
            "mean_ratio_bg": real["mu_b"],
            "n_plume": real["n_plume"],
            "n_bg": real["n_bg"],
            "mask_area": int(plume.sum()),
            "n_ctrl_valid": int(ctrl.size),
            "snr_ctrl_med": float(np.median(ctrl)) if ctrl.size else np.nan,
            "snr_ctrl_min": float(ctrl.min()) if ctrl.size else np.nan,
            "snr_ctrl_max": float(ctrl.max()) if ctrl.size else np.nan,
        }
        # temperatura/valor medio del target en pluma y fondo (diagnostico TIR)
        tp = target[b][finite & plume]
        tb = target[b][finite & ~plume]
        rec["target_mean_plume"] = float(tp.mean()) if tp.size else np.nan
        rec["target_mean_bg"] = float(tb.mean()) if tb.size else np.nan
        records.append(rec)
    return records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sensor", choices=["l89", "s2"], required=True)
    ap.add_argument("--dataset-dir", default=None)
    ap.add_argument("--leaf-bg", default="bg0")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--out", default=os.path.join(os.environ.get(
        "METHANESET_WORK", os.path.join(os.path.dirname(os.path.abspath(__file__)), "work")),
        "snr_control"))
    ap.add_argument("--n-ctrl", type=int, default=5)
    ap.add_argument("--ctrl-central-frac", type=float, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    cfg = SENSORS[args.sensor]
    dataset_dir = args.dataset_dir or cfg["path"]
    tag = args.tag or args.sensor
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print(f"[{tag}] dataset: {dataset_dir}")
    piv = load_pivot(dataset_dir)
    if args.limit:
        piv = piv.iloc[:args.limit]
    print(f"[{tag}] muestras: {len(piv)} | n_ctrl={args.n_ctrl} seed={args.seed}")

    rng = np.random.default_rng(args.seed)
    t0 = time.time()
    rows = []
    for i in range(len(piv)):
        if i % 100 == 0:
            el = time.time() - t0
            print(f"  {i}/{len(piv)}  ({el:.0f}s, {el / max(i, 1):.3f} s/muestra)")
        try:
            recs = process_sample(piv.iloc[i], cfg["bands"], args.leaf_bg,
                                  rng, args.n_ctrl, args.ctrl_central_frac)
        except Exception as exc:
            print(f"  [warn] muestra {piv.iloc[i].get('l0:id')}: {exc!r}")
            continue
        sid = piv.iloc[i]["l0:id"]
        for b, rec in enumerate(recs):
            rec.update({"sensor": args.sensor, "tag": tag, "sample_id": sid,
                        "band_index": b,
                        "wl": cfg["wl"][b] if b < len(cfg["wl"]) else np.nan})
            rows.append(rec)

    df = pd.DataFrame(rows)
    cols = ["sensor", "tag", "sample_id", "band_index", "band", "wl"] + \
           [c for c in df.columns if c not in
            ("sensor", "tag", "sample_id", "band_index", "band", "wl")]
    df = df[cols]
    out_csv = out / f"snr_samples_{tag}.csv"
    df.to_csv(out_csv, index=False)
    meta = {
        "script": "compute_snr_control.py",
        "sensor": args.sensor, "tag": tag,
        "dataset_dir": str(dataset_dir), "leaf_bg": args.leaf_bg,
        "n_ctrl": args.n_ctrl, "seed": args.seed, "limit": args.limit,
        "ctrl_central_frac": args.ctrl_central_frac,
        "n_samples_loaded": int(len(piv)),
        "n_rows": int(len(df)),
        "bands": cfg["bands"], "wl": cfg["wl"],
        "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_s": round(time.time() - t0, 1),
    }
    (out / f"run_meta_{tag}.json").write_text(json.dumps(meta, indent=1))
    print(f"[{tag}] OK -> {out_csv} ({len(df)} filas, {time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
