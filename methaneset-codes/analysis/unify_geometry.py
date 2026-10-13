"""MODULE 28 of the v2 pipeline: aligned windows that contain each annotated plume, by source (IMEO/CM/union) and by size (128, 256). Does not exclude by cloud.

level0:
  selection:{imeo,cm,plume}_plume_windows_128 / _256  (lists of plain structs)
  selection:n_{imeo,cm,plume}_windows_128 / _256
Audit: _plume_chips.parquet (id, source, bit, bbox, fits_128, fits_256, n128, n256)
"""
from __future__ import annotations
import pathlib, time
import numpy as np, pandas as pd, rasterio

ROOT = pathlib.Path("/data/databases/METHANESET_TACOS/methaneset-emit")
LEVEL0 = ROOT / "METADATA" / "level0.parquet"
AUDIT = pathlib.Path("/data/databases/METHANESET_TACOS/_plume_chips.parquet")
SIZES = (128, 256)
MARGIN = 10


def plume_boxes(mask):
    out = []
    for bit in range(1, 65):
        m = (mask & np.uint64(1 << (bit - 1))) != 0
        ys, xs = np.nonzero(m)
        if len(ys):
            out.append((bit, int(ys.min()), int(xs.min()), int(ys.max()), int(xs.max())))
    return out


def aligned_windows(r0, c0, r1, c1, h, w, S):
    rs = [r for r in range(0, h - S + 1, S) if r <= r0 - MARGIN and r + S - 1 >= r1 + MARGIN]
    cs = [c for c in range(0, w - S + 1, S) if c <= c0 - MARGIN and c + S - 1 >= c1 + MARGIN]
    return [{"col_off": c, "row_off": r, "width": S, "height": S} for r in rs for c in cs]


def dedup(L):
    return [dict(t) for t in {tuple(sorted(x.items())) for x in L}]


def main():
    df = pd.read_parquet(LEVEL0)
    cols = {}
    for src in ("imeo", "cm"):
        for S in SIZES:
            cols[f"selection:{src}_plume_windows_{S}"] = {}
            cols[f"selection:n_{src}_windows_{S}"] = {}
    for S in SIZES:
        cols[f"selection:plume_windows_{S}"] = {}
        cols[f"selection:n_plume_windows_{S}"] = {}
    rows = []
    t0 = time.time()
    for i, sid in enumerate(df["id"], 1):
        d = ROOT / "DATA" / sid
        with rasterio.open(d / "plume_imeo.tif") as a, rasterio.open(d / "plume_cm.tif") as b:
            mi = a.read(1); mc = b.read(1)
        h, w = mi.shape
        acc = {("imeo", S): [] for S in SIZES}
        acc.update({("cm", S): [] for S in SIZES})
        for src, mask in (("imeo", mi), ("cm", mc)):
            for bit, r0, c0, r1, c1 in plume_boxes(mask):
                n = {}
                for S in SIZES:
                    wins = aligned_windows(r0, c0, r1, c1, h, w, S)
                    acc[(src, S)].extend(wins); n[S] = len(wins)
                rows.append(dict(id=sid, source=src, bit=bit, r0=r0, c0=c0, r1=r1, c1=c1,
                                 fits_128=n[128] > 0, fits_256=n[256] > 0, n128=n[128], n256=n[256]))
        for S in SIZES:
            iw, cw = acc[("imeo", S)], acc[("cm", S)]
            both = dedup(iw + cw)
            cols[f"selection:imeo_plume_windows_{S}"][sid] = dedup(iw)
            cols[f"selection:cm_plume_windows_{S}"][sid] = dedup(cw)
            cols[f"selection:plume_windows_{S}"][sid] = both
            cols[f"selection:n_imeo_windows_{S}"][sid] = len(dedup(iw))
            cols[f"selection:n_cm_windows_{S}"][sid] = len(dedup(cw))
            cols[f"selection:n_plume_windows_{S}"][sid] = len(both)
        if i % 100 == 0 or i == len(df):
            print(f"[{i}/{len(df)}] {time.time()-t0:.0f}s", flush=True)
    viejas = [c for c in df.columns if c.startswith("selection:") and "plume_windows" in c]
    df = df.drop(columns=viejas, errors="ignore")
    for c, m in cols.items():
        if c.startswith("selection:n_"):
            df[c] = df["id"].map(m).fillna(0).astype(int)
        else:
            df[c] = df["id"].map(m).apply(lambda x: x if isinstance(x, list) else [])
    df.to_parquet(LEVEL0, index=False)
    pd.DataFrame(rows).to_parquet(AUDIT, index=False)
    print("level0:", df.shape, "| plumas auditadas:", len(rows))
    print(df[[c for c in df.columns if c.startswith("selection:n_")]].describe().round(1).to_string())


if __name__ == "__main__":
    main()
