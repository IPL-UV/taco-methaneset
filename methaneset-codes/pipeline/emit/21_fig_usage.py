"""MODULE 21: the "how the dataset is used" figures (closing slides).

Julio's request (Aug 26): close the deck showing how the user can USE
the images, without yet entering chunks, training or injection:

  a) 1:1 orphans: a scene where each catalog saw something the other did not
  b) split balance: map per split + bars by year / region / sector
  c) wind over the plume: ERA5-Land field from wind.tif on top of the
     mask, with the PER-PLUME wind arrows from each institution

(The fourth one, "which flux to trust", lives in module 22 because it needs
the institutional wind from module 20.)

Outputs: assets/images/{orphans-11,splits-balance,wind-over-plume}.png
         (+ copy to assets/slides/public/)
Usage:
  /data/users/julio/.conda/envs/deep/bin/python code/v2/21_fig_usage.py \
        > code/v2/21_fig_usage.log 2>&1
"""
import json
import pathlib
import shutil
import sys

import numpy as np
import pandas as pd
import rasterio

_theme = pathlib.Path(os.environ.get("METHANESET_THEME", "/data/users/julio/Notes/01-Projects/methanset/assets/slides/theme"))
if _theme.exists():
    sys.path.insert(0, str(_theme))
import figstyle  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

ROOT = pathlib.Path(os.environ.get("METHANESET_ROOT", "/data/users/julio/Notes/01-Projects/methanset"))
IMG = ROOT / "assets" / "images"
PUB = ROOT / "assets" / "slides" / "public"
TACO = pathlib.Path("/data/databases/METHANSET_TACOS/methaneset-emit")
DATA = TACO / "DATA"
LEVEL0 = TACO / "METADATA" / "level0.parquet"


def layer(gid, name, band=1):
    with rasterio.open(DATA / gid / name) as s:
        a = s.read(band).astype("float32")
        nod = s.nodata
    if nod is not None and name != "plume_imeo.tif" and name != "plume_cm.tif":
        a[a == nod] = np.nan
    return a


def crop_around(mask, pad=90):
    ys, xs = np.nonzero(mask)
    y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad, mask.shape[0])
    x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad, mask.shape[1])
    return slice(y0, y1), slice(x0, x1)


# ---------------------------------------------------------------- a) orphans
# Scene diagnosed with Julio (Aug 27): an orphan does NOT always mean
# "the other catalog did not see it". Here CM registered the SAME plume twice
# (IoU 0.885 between them, sources 1,288 m apart); the Hungarian is 1:1, it matches
# the closest one and the twin is left as an orphan. The other orphan, however,
# is real: a site 40 km north that CM did not detect.
SCENE_11 = "EMIT_L1B_RAD_001_20240610T041715_2416203_007"


def _pt(srcb, code, lat, lon):
    ys, xs = np.nonzero(srcb == code)
    return (float(lon[ys[0], xs[0]]), float(lat[ys[0], xs[0]])) if len(ys) else None


def fig_orphans(md):
    gid = SCENE_11
    r = md[md["id"] == gid].iloc[0]
    icod, ccod = json.loads(r["detection:imeo_cod"]), json.loads(r["detection:cm_cod"])
    pairs, orph = json.loads(r["match:pairs"]), json.loads(r["match:orphans"])
    iflux, cflux = json.loads(r["detection:imeo_flux"]), json.loads(r["detection:cm_flux"])
    matched_cm = {p["cm"] for p in pairs}
    with rasterio.open(DATA / gid / "plume_imeo.tif") as s:
        mi, si = s.read(1), s.read(2)
    with rasterio.open(DATA / gid / "plume_cm.tif") as s:
        mc, sc = s.read(1), s.read(2)
    with rasterio.open(DATA / gid / "latlon.tif") as s:
        lat, lon = s.read(1), s.read(2)
    mag = layer(gid, "mag1c.tif")

    imeo_main = pairs[0]["imeo"]
    m_imeo = np.isin(mi, icod[imeo_main])
    cm_shapes = [(n, np.isin(mc, c), c[0], n in matched_cm) for n, c in ccod.items()]
    far = [n for n in orph["imeo"] if np.isin(mi, icod[n]).sum() > 500]
    m_far = np.isin(mi, icod[far[0]]) if far else np.zeros_like(m_imeo)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6.2),
                             gridspec_kw={"width_ratios": [1.25, 1]},
                             constrained_layout=True)

    # --- panel 1: the shared site, with CM's TWO identities ---
    ax = axes[0]
    sl = crop_around(m_imeo | np.logical_or.reduce([m for _, m, _, _ in cm_shapes]), 55)
    ax.imshow(np.clip(mag[sl], 0, np.nanpercentile(mag[sl], 99.6)), cmap="magma")
    ov = np.zeros(m_imeo[sl].shape + (4,), "float32")
    ov[m_imeo[sl]] = (0.18, 0.83, 0.75, 0.45)
    ax.imshow(ov)
    styles = [("-", "#fb923c"), ("--", "#a78bfa")]
    for k, (name, m, code, is_matched) in enumerate(cm_shapes):
        ls, col = styles[k % 2]
        ax.contour(m[sl].astype(float), levels=[0.5], colors=[col],
                   linewidths=2.0, linestyles=[ls])
        pt = _pt(sc, code, lat, lon)
        if pt:
            ys, xs = np.nonzero(sc[sl] == code)
            if len(ys):
                ax.plot(xs[0], ys[0], "o", ms=10, mfc=col, mec="#0f172a", mew=1.2)
    ys, xs = np.nonzero(si[sl] == icod[imeo_main][0])
    if len(ys):
        ax.plot(xs[0], ys[0], "*", ms=20, mfc="#2dd4bf", mec="#0f172a", mew=1.0)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("one site, three records: IMEO traced the whole plume,\n"
                 "Carbon Mapper filed it TWICE (IoU 0.885 between the two)",
                 fontsize=11)
    h = [Line2D([], [], marker="*", ls="none", ms=15, mfc="#2dd4bf", mec="#0f172a",
                label=f"IMEO {imeo_main} · {iflux[imeo_main]:.0f} kg/h · "
                      f"{int(m_imeo.sum()):,} px")]
    for k, (name, m, code, is_matched) in enumerate(cm_shapes):
        ls, col = styles[k % 2]
        tag = "MATCHED by Hungarian" if is_matched else "left as orphan"
        f = cflux.get(name)
        h.append(Line2D([], [], color=col, lw=2.4, ls=ls, marker="o", ms=8,
                        mfc=col, mec="#0f172a",
                        label=f"CM …{name[-13:]} · {f:.0f} kg/h · "
                              f"{int(m.sum()):,} px · {tag}"))
    ax.legend(handles=h, fontsize=8.4, loc="lower left", framealpha=0.9)

    # --- panel 2: the real orphan ---
    ax = axes[1]
    sl2 = crop_around(m_far, 90)
    ax.imshow(np.clip(mag[sl2], 0, np.nanpercentile(mag[sl2], 99.6)), cmap="magma")
    ov2 = np.zeros(m_far[sl2].shape + (4,), "float32")
    ov2[m_far[sl2]] = (0.65, 0.55, 0.98, 0.8)
    ax.imshow(ov2)
    ys, xs = np.nonzero(si[sl2] == icod[far[0]][0])
    if len(ys):
        ax.plot(xs[0], ys[0], "*", ms=18, mfc="#a78bfa", mec="#0f172a", mew=1.0)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"40 km north, a REAL orphan:\n"
                 f"IMEO {far[0]} · {iflux[far[0]]:.0f} kg/h, no CM record at all",
                 fontsize=11)
    fig.suptitle(f"{gid} · why an orphan is not always a disagreement",
                 fontsize=12.5)
    figstyle.save(fig, IMG / "orphans-11.png")
    shutil.copy2(IMG / "orphans-11.png", PUB / "orphans-11.png")
    print(f"  -> orphans-11.png ({len(pairs)} pair, {len(orph['cm'])} CM orphan "
          f"= twin, {len(orph['imeo'])} IMEO orphans)")


# ------------------------------------------------------------ b) split balance
def fig_splits(md):
    md = md.copy()
    md["year"] = md["emit:time_start"].astype(str).str[:4]
    lon = (md["spatial:bbox_west"] + md["spatial:bbox_east"]) / 2
    lat = (md["spatial:bbox_south"] + md["spatial:bbox_north"]) / 2
    COL = {"train": figstyle.PALETTE["teal"], "val": figstyle.PALETTE["orange"],
           "test": figstyle.PALETTE["slate"]}
    fig = plt.figure(figsize=(16, 6.6), constrained_layout=True)
    gs = fig.add_gridspec(2, 3, height_ratios=[1.35, 1])
    import cartopy.crs as ccrs
    import cartopy.feature as cfeat
    # one map PER SPLIT (Julio's request): this shows that each one covers the
    # whole world instead of keeping to one region
    for k, sp in enumerate(["train", "val", "test"]):
        axm = fig.add_subplot(gs[0, k], projection=ccrs.Robinson())
        axm.add_feature(cfeat.LAND, facecolor="#eef2f7", edgecolor="none")
        axm.add_feature(cfeat.OCEAN, facecolor="#f8fafc")
        axm.add_feature(cfeat.COASTLINE, lw=0.35, edgecolor="#94a3b8")
        axm.add_feature(cfeat.BORDERS, lw=0.2, edgecolor="#dbe2ea")
        axm.set_global()
        m = md["selection:split"] == sp
        # background grey: the other scenes, to read the mix
        axm.plot(lon[~m], lat[~m], "o", ms=2.2, alpha=0.35, mec="none",
                 color="#cbd5e1", ls="none", transform=ccrs.PlateCarree())
        axm.plot(lon[m], lat[m], "o", ms=4.2, alpha=0.9, mec="none",
                 color=COL[sp], ls="none", transform=ccrs.PlateCarree())
        cn = md.loc[m, "site:country"].nunique()
        axm.set_title(f"{sp} · {int(m.sum())} scenes · {cn} countries",
                      fontsize=11, color=COL[sp], fontweight="bold")

    for ax, key, title in ((fig.add_subplot(gs[1, 0]), "year", "by year"),
                           (fig.add_subplot(gs[1, 1]), "detection:sector",
                            "by dominant sector"),
                           (fig.add_subplot(gs[1, 2]), "selection:flux_bin",
                            "by flux bin")):
        col = md[key].replace("", "(no sector)")
        cnt = col.value_counts()
        tab = pd.crosstab(col, md["selection:split"], normalize="index") * 100
        tab = tab[[c for c in ["train", "val", "test"] if c in tab.columns]]
        tab.index = [f"{i}  n={cnt[i]}" for i in tab.index]  # small classes visible
        tab.plot(kind="barh", stacked=True, ax=ax, legend=False, width=0.78,
                 color=[COL[c] for c in tab.columns])
        for x in (70, 85):
            ax.axvline(x, color="#0f172a", ls=":", lw=1)
        ax.set_xlim(0, 100); ax.set_xlabel("% of scenes"); ax.set_ylabel("")
        ax.set_title(title, fontsize=10.5)
        ax.tick_params(labelsize=8.5)
    fig.suptitle("every axis reproduces the 70 / 15 / 15 target "
                 "(dotted lines), and no source crosses splits", fontsize=12.5)
    figstyle.save(fig, IMG / "splits-balance.png")
    shutil.copy2(IMG / "splits-balance.png", PUB / "splits-balance.png")
    print("  -> splits-balance.png")


# --------------------------------------------------------- c) wind over plume
def fig_wind(md):
    cand = md[(md["detection:n_imeo"] == 1) & (md["detection:n_cm"] >= 1)
              & (md["detection:imeo_flux_max"] > 2000)]
    row = cand.sort_values("detection:imeo_flux_max", ascending=False).iloc[0]
    gid = row["id"]
    print(f"[wind] {gid} · flux IMEO {row['detection:imeo_flux_max']:.0f} kg/h")
    mi = layer(gid, "plume_imeo.tif")
    mag = layer(gid, "mag1c.tif")
    with rasterio.open(DATA / gid / "wind.tif") as s:
        u, v = s.read(1), s.read(2)
    with rasterio.open(DATA / gid / "plume_imeo.tif") as s:
        src_band = s.read(2)
    sl = crop_around(mi > 0, pad=120)
    U, V = u[sl], v[sl]
    base = np.clip(mag[sl], 0, np.nanpercentile(mag[sl], 99.6))

    fig, ax = plt.subplots(figsize=(8.6, 6.4), constrained_layout=True)
    ax.imshow(base, cmap="magma")
    ov = np.zeros(mi[sl].shape + (4,), "float32")
    ov[mi[sl] > 0] = (0.18, 0.83, 0.75, 0.55)
    ax.imshow(ov)
    # the image is in SENSOR geometry (rotated with respect to north): the
    # u/v components are geographic, they must be projected onto the row and
    # column axes using the latlon gradient, or the arrows lie.
    lat = layer(gid, "latlon.tif", 1)[sl]
    lon = layer(gid, "latlon.tif", 2)[sl]
    coslat = np.cos(np.radians(np.nanmean(lat)))
    dx_e = np.nanmean(np.gradient(lon, axis=1)) * coslat   # east per column
    dx_n = np.nanmean(np.gradient(lat, axis=1))            # north per column
    dy_e = np.nanmean(np.gradient(lon, axis=0)) * coslat
    dy_n = np.nanmean(np.gradient(lat, axis=0))
    ecol = np.hypot(dx_e, dx_n); erow = np.hypot(dy_e, dy_n)
    ux, uy = dx_e / ecol, dx_n / ecol      # unit vector of +column in (E, N)
    vx, vy = dy_e / erow, dy_n / erow      # unit vector of +row in (E, N)
    Ucol = U * ux + V * uy                 # wind projected onto columns
    Urow = U * vx + V * vy                 # and onto rows
    bearing = (np.degrees(np.arctan2(ux, uy))) % 360
    print(f"  +column axis points to {bearing:.0f} deg from north")
    step = max(base.shape[0] // 14, 1)
    yy, xx = np.mgrid[0:base.shape[0]:step, 0:base.shape[1]:step]
    # quiver with angles='uv' (default) ignores that the y axis is inverted:
    # +row is DOWN on screen, so the component is negated.
    ax.quiver(xx, yy, Ucol[::step, ::step], -Urow[::step, ::step],
              color="#e2e8f0", scale=70, width=0.0035, alpha=0.9)
    ys, xs = np.nonzero(src_band[sl] > 0)
    if len(ys):
        ax.plot(xs, ys, "o", ms=9, mfc="#fde047", mec="#a16207", mew=1.6,
                ls="none")
    ax.set_xticks([]); ax.set_yticks([])
    sp = float(np.hypot(U, V).mean())
    ax.set_title(f"{gid}\nplume mask (teal), emission point (yellow) and the "
                 f"ERA5-Land wind of wind.tif, rotated into sensor axes · scene mean {sp:.1f} m/s",
                 fontsize=11)
    figstyle.save(fig, IMG / "wind-over-plume.png")
    shutil.copy2(IMG / "wind-over-plume.png", PUB / "wind-over-plume.png")
    print("  -> wind-over-plume.png")


def wind_vs_plume(md, nmax=400):
    """Wind validation: plume bearing from its source vs bearing towards
    which the wind of wind.tif blows. If the layer is correct they must match."""
    md = md[(md["detection:n_imeo"] == 1) & (~md["selection:is_free"])].head(nmax)
    diffs = []
    for gid in md["id"]:
        try:
            with rasterio.open(DATA / gid / "plume_imeo.tif") as s:
                m, sp = s.read(1), s.read(2)
            ys, xs = np.nonzero(sp > 0)
            py, px = np.nonzero(m > 0)
            if not len(ys) or len(py) < 30:
                continue
            with rasterio.open(DATA / gid / "latlon.tif") as s:
                lat, lon = s.read(1), s.read(2)
            with rasterio.open(DATA / gid / "wind.tif") as s:
                u, v = s.read(1), s.read(2)
            sy, sx = ys[0], xs[0]
            de = (lon[py, px].mean() - lon[sy, sx]) * np.cos(np.radians(lat[sy, sx]))
            dn = lat[py, px].mean() - lat[sy, sx]
            if np.hypot(de, dn) < 1e-4:
                continue
            b_pl = np.degrees(np.arctan2(de, dn)) % 360
            b_wd = np.degrees(np.arctan2(u[sy, sx], v[sy, sx])) % 360
            diffs.append(abs((b_pl - b_wd + 180) % 360 - 180))
        except Exception:
            continue
    d = np.array(diffs)
    print(f"[wind validation] {len(d)} single-source plumes · "
          f"median difference {np.median(d):.0f} deg · "
          f"<=45 deg in {(d<=45).mean():.0%} · <=90 in {(d<=90).mean():.0%}")
    return d


def main():
    md = pd.read_parquet(LEVEL0)
    md = md[~md["selection:is_free"]]
    fig_orphans(md)
    fig_splits(pd.read_parquet(LEVEL0))
    fig_wind(md)
    wind_vs_plume(md)


if __name__ == "__main__":
    main()
