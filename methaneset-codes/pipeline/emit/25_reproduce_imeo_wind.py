"""MODULE 25: can we REPRODUCE the wind IMEO publishes? (Julio, Aug 27)

Julio: "focus only on IMEO... if IMEO when extracting the wind value, how
it did it, and if you repeat the same thing does it match what they give, and why
would it not match".

IMEO declares ERA5-Land at 10 m. We use EXACTLY the same product.
If theirs and ours do not match 100%, the difference can only come
from HOW the number is extracted. There are two free decisions:

  TIME: ERA5-Land is HOURLY. An EMIT scene is taken, for example, at
    10:37. Is the previous hour used (10:00)? The next one (11:00)? The nearest
    one? Or is it interpolated between the two according to the minute? We
    interpolate to the minute (module 13); IMEO does not declare it.

  SPACE: ERA5-Land cells are ~9 km. Is the node nearest to the
    emission point taken, or is it interpolated between the neighbouring nodes?

This module downloads from Earth Engine, for a sample of single-source
plumes, the ERA5-Land values at the emission point at the previous and
the next hour, and compares EVERY variant against the wind IMEO publishes. The
closest variant is, with good probability, the one they use.

Outputs: console + assets/images/imeo-wind-repro.png
Usage:
  /data/users/julio/.conda/envs/deep/bin/python code/v2/25_reproduce_imeo_wind.py \
        > code/v2/25_reproduce_imeo_wind.log 2>&1
"""
import datetime
import json
import pathlib
import re
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

import ee
import numpy as np
import pandas as pd
import rasterio

_theme = pathlib.Path(os.environ.get("METHANESET_THEME", "/data/users/julio/Notes/01-Projects/methanset/assets/slides/theme"))
if _theme.exists():
    sys.path.insert(0, str(_theme))
import figstyle  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

ROOT = pathlib.Path(os.environ.get("METHANESET_ROOT", "/data/users/julio/Notes/01-Projects/methanset"))
IMG = ROOT / "assets" / "images"
PUB = ROOT / "assets" / "slides" / "public"
CROSS = ROOT / "assets" / "data" / "cross" / "2026-08-25"
TACO = pathlib.Path("/data/databases/METHANSET_TACOS/methaneset-emit")
DATA = TACO / "DATA"
TS = re.compile(r"_(\d{8}T\d{6})_")
COLL = "ECMWF/ERA5_LAND/HOURLY"
BANDS = ["u_component_of_wind_10m", "v_component_of_wind_10m"]
NMAX = 260          # sample plumes (one EE call per plume)
WORKERS = 6


def scene_dt(gid):
    t = TS.search(gid).group(1)
    return datetime.datetime.strptime(t, "%Y%m%dT%H%M%S")


def emission_lonlat(gid, code):
    """lon/lat of the emission point pixel (mask band 2)."""
    with rasterio.open(DATA / gid / "plume_imeo.tif") as s:
        b2 = s.read(2)
    ys, xs = np.nonzero(b2 == code)
    if not len(ys):
        return None
    with rasterio.open(DATA / gid / "latlon.tif") as s:
        lat, lon = s.read(1), s.read(2)
    return float(lon[ys[0], xs[0]]), float(lat[ys[0], xs[0]]), int(ys[0]), int(xs[0])


def era5_at(lon, lat, dt):
    """ERA5-Land u/v at the point's node, at the previous and next hour."""
    h0 = dt.replace(minute=0, second=0, microsecond=0)
    h1 = h0 + datetime.timedelta(hours=1)
    pt = ee.Geometry.Point([lon, lat])
    out = {}
    for tag, h in (("prev", h0), ("next", h1)):
        img = (ee.ImageCollection(COLL)
               .filterDate(h.isoformat(), (h + datetime.timedelta(hours=1)).isoformat())
               .first().select(BANDS))
        v = img.reduceRegion(ee.Reducer.first(), pt, 1000).getInfo()
        out[tag] = (v.get(BANDS[0]), v.get(BANDS[1]))
    return out, (dt - h0).total_seconds() / 3600.0


def one(job):
    gid, name, code, theirs = job
    try:
        got = emission_lonlat(gid, code)
        if got is None:
            return None
        lon, lat, y, x = got
        dt = scene_dt(gid)
        era, frac = era5_at(lon, lat, dt)
        if any(v is None for pair in era.values() for v in pair):
            return None
        up, vp = era["prev"]
        un, vn = era["next"]
        # our published layer, read at the same pixel
        with rasterio.open(DATA / gid / "wind.tif") as s:
            ours_u = float(s.read(1)[y, x])
            ours_v = float(s.read(2)[y, x])
        ui = up + (un - up) * frac
        vi = vp + (vn - vp) * frac
        near = ("prev", up, vp) if frac < 0.5 else ("next", un, vn)
        return {
            "granule": gid, "plume": name, "theirs": theirs, "frac_hour": frac,
            "sp_prev": float(np.hypot(up, vp)),
            "sp_next": float(np.hypot(un, vn)),
            "sp_interp": float(np.hypot(ui, vi)),
            "sp_nearest": float(np.hypot(near[1], near[2])),
            "sp_ours_layer": float(np.hypot(ours_u, ours_v)),
        }
    except Exception as e:
        print(f"  ERROR {gid}: {type(e).__name__}: {e}", flush=True)
        return None


def main():
    ee.Initialize(project="ee-contrerasnetk")
    md = pd.read_parquet(TACO / "METADATA" / "level0.parquet")
    md = md[(md["detection:n_imeo"] == 1) & (~md["selection:is_free"])]
    jobs = []
    for r in md.to_dict("records"):
        iw = json.loads(r["meteo:imeo_wind"] or "{}")
        cod = json.loads(r["detection:imeo_cod"] or "{}")
        for name, w in iw.items():
            if w and w.get("speed") and cod.get(name):
                jobs.append((r["id"], name, cod[name][0], float(w["speed"])))
                break
    jobs = jobs[:NMAX]
    print(f"single-source plumes to test: {len(jobs)}")

    res = []
    with ThreadPoolExecutor(WORKERS) as ex:
        futs = [ex.submit(one, j) for j in jobs]
        for i, f in enumerate(as_completed(futs), 1):
            r = f.result()
            if r:
                res.append(r)
            if i % 50 == 0:
                print(f"  {i}/{len(jobs)}", flush=True)
    d = pd.DataFrame(res)
    d.to_parquet(CROSS / "imeo_wind_repro.parquet", index=False)
    print(f"\nplumes resolved: {len(d)}\n")

    variants = [
        ("previous hour", "sp_prev"),
        ("next hour", "sp_next"),
        ("nearest hour", "sp_nearest"),
        ("interpolated to the minute", "sp_interp"),
        ("our published layer", "sp_ours_layer"),
    ]
    print(f"{'variant':26s} {'median |dif|':>14s} {'bias':>9s} "
          f"{'r':>6s} {'<=0.25 m/s':>11s} {'<=0.5':>7s}")
    stats_ = {}
    for label, col in variants:
        dif = d[col] - d.theirs
        r_ = np.corrcoef(d[col], d.theirs)[0, 1]
        stats_[label] = dif.abs().median()
        print(f"{label:26s} {dif.abs().median():14.3f} {dif.median():+9.3f} "
              f"{r_:6.3f} {(dif.abs()<=0.25).mean():10.0%} "
              f"{(dif.abs()<=0.5).mean():6.0%}")
    win = min(stats_, key=stats_.get)
    print(f"\n-> the variant closest to what IMEO publishes: **{win}** "
          f"({stats_[win]:.3f} m/s)")
    print(f"-> our published layer stands at "
          f"{stats_['our published layer']:.3f} m/s")
    dif_h = (d.sp_next - d.sp_prev).abs()
    print(f"\nhow much the wind changes from one hour to the next: median "
          f"{dif_h.median():.2f} m/s (p90 {dif_h.quantile(0.9):.2f})")
    print("that is the size of the effect that the hour choice can explain.")

    fig, ax = plt.subplots(figsize=(8.6, 4.8), constrained_layout=True)
    labels = [v[0] for v in variants]
    vals = [stats_[l] for l in labels]
    cols = [figstyle.PALETTE["teal"]] * len(labels)
    cols[int(np.argmin(vals))] = figstyle.PALETTE["orange"]
    ax.bar(range(len(labels)), vals, color=cols, width=0.62)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels([l.replace(" ", "\n", 1) for l in labels], fontsize=9)
    for x, v in enumerate(vals):
        ax.text(x, v, f"{v:.3f}", ha="center", va="bottom", fontsize=11,
                fontweight="bold")
    ax.set_ylabel("median |ours - IMEO published|  [m/s]")
    ax.set_title("Reproducing IMEO's published wind from ERA5-Land\n"
                 f"same product, {len(d)} single-source plumes: only the "
                 "extraction differs", fontsize=12)
    figstyle.save(fig, IMG / "imeo-wind-repro.png")
    shutil.copy2(IMG / "imeo-wind-repro.png", PUB / "imeo-wind-repro.png")
    print("\nfigure -> imeo-wind-repro.png")


if __name__ == "__main__":
    main()
