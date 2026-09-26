"""MODULE 15 of the v2 pipeline: generate masks (plume_imeo.tif + plume_cm.tif).

The last double layer of the sample, with Julio's FULL nomenclature (faithful
port of v1's 04.7-04.10 + 04.14, verified back then):

  band 1 (mask):    uint8 instance codes. A matched IMEO-CM pair shares THE
                     SAME code in both files; orphans get their own code;
                     partial overlaps get a zone code (> n_base). Large ones
                     first, small ones on top.
  band 2 (sources): the emission point of each source at its nearest pixel
                     (cap 0.05 degrees), with the base code.

v2 sources (recorded decisions):
  IMEO: polygons from the PORTAL geojson (dated snapshot, the same as the
        cross-match); canonical emission point from unep_sources.
  CM:   alpha band (==255) of each plume_tif (module 15a), dissolved by
        (granule, source) with the association from module 15b; canonical
        point of the CM source.
  Frees (25): empty masks (all zero), 2 equal bands.

Matching (Julio's thresholds, sensitivity sweep pending):
  local UTM per scene; cost (1 - max_containment) + dist_sources/2000 m;
  1:1 Hungarian assignment; accept if inter >= 3500 m2 and (IoU >= 0.1 or
  containment >= 0.5 or sources at <= 1000 m); intra-overlaps >= 3500 m2;
  containment >= 0.9 -> the inner code enters the outer one.

Outputs:
  METHANSET_TACOS/emit/<granule>/plume_imeo.tif and plume_cm.tif
  cross/<date>/mask_codes.parquet  (codes and matching per scene, for the
                                     metadata module 16)

Hard assert: no code goes above 255 (uint8).
Usage:
  nohup /data/users/julio/.conda/envs/deep/bin/python code/v2/15_generate_masks.py \
        > code/v2/15_generate_masks.log 2>&1 &
"""
import json
import pathlib
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import combinations
from math import asin, cos, radians, sin, sqrt

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import shapely
from pyproj import CRS
from rasterio.features import shapes as rio_shapes
from scipy.optimize import linear_sum_assignment
from shapely import make_valid
from shapely.geometry import shape
from shapely.ops import unary_union

DATA = pathlib.Path(__file__).resolve().parent.parent.parent / "assets" / "data"
OUT_ROOT = pathlib.Path("/data/databases/METHANSET_TACOS/emit")
CM_DIR = pathlib.Path("/data/databases/METHANSET_TACOS/cm_plume_tifs")
TS_G = re.compile(r"EMIT_L1B_RAD_001_(\d{8}T\d{6})")
WORKERS = 8
PT_MAX_DEG = 0.05
IOU_MIN, IO_PRIMARY = 0.1, 0.5
MIN_INTER, SRC_NEAR = 3500.0, 1000.0
ALPHA_W, D_REF = 1.0, 2000.0
CONTAIN_TH = 0.9
MIN_INTRA = 3500.0
BASE = dict(driver="GTiff", count=2, dtype="uint8", nodata=0, tiled=True,
            blockxsize=128, blockysize=128, compress="zstd", interleave="band")

# globals that the fork inherits
G_IMEO = {}      # ts -> list of dicts (id, geom, src, pt)
G_CM = {}        # ts -> list of dicts (source_name, geom, pt, plume_ids)
G_SCENES = {}    # ts -> (scene_id, is_free)


def utm_crs(lon, lat):
    zone = int((lon + 180) // 6) + 1
    return CRS.from_epsg((32600 if lat >= 0 else 32700) + zone)


def haversine_m(lon1, lat1, lon2, lat2):
    R = 6_371_000.0
    dlon, dlat = radians(lon2 - lon1), radians(lat2 - lat1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    return 2 * R * asin(sqrt(a))


def vectorize_cm_tif(path):
    try:
        with rasterio.open(path) as src:
            if src.count < 4:
                return None
            alpha = src.read(4)
            mask = (alpha == 255).astype(np.uint8)
            if mask.sum() == 0:
                return None
            polys = [shape(g) for g, v in rio_shapes(
                mask, mask=mask.astype(bool), transform=src.transform) if v == 1]
            if not polys:
                return None
            geom = unary_union(polys)
            if not geom.is_valid:
                geom = geom.buffer(0)
            g = gpd.GeoSeries([geom], crs=src.crs).to_crs(4326).iloc[0]
            return make_valid(g)
    except Exception:
        return None


# ---------- matching + codes (port of 04.7-04.10 / 40) ----------

def match_and_codes(imeo, cm):
    n, m = len(imeo), len(cm)
    all_geoms = [x["geom"] for x in imeo + cm]
    c = unary_union(all_geoms).centroid
    utm = utm_crs(c.x, c.y)
    gi = gpd.GeoSeries([x["geom"] for x in imeo], crs=4326).to_crs(utm)
    gj = gpd.GeoSeries([x["geom"] for x in cm], crs=4326).to_crs(utm)

    cost = np.ones((n, m)); metr = {}
    for i in range(n):
        for j in range(m):
            inter = gi[i].intersection(gj[j]).area
            if inter == 0:
                continue
            union = gi[i].area + gj[j].area - inter
            iou = inter / union if union > 0 else 0
            io_i = inter / gi[i].area if gi[i].area else 0
            io_j = inter / gj[j].area if gj[j].area else 0
            pi, pj = imeo[i]["pt"], cm[j]["pt"]
            d = (haversine_m(pi[0], pi[1], pj[0], pj[1])
                 if all(np.isfinite([*pi, *pj])) else np.inf)
            metr[(i, j)] = dict(iou=iou, inter=inter, io_i=io_i, io_j=io_j, dist=d)
            overlap = max(io_i, io_j)
            d_norm = d / D_REF if np.isfinite(d) else 1e3
            cost[i, j] = (1 - overlap) + ALPHA_W * d_norm

    def accepts(i, j):
        mm = metr.get((i, j))
        if mm is None or mm["inter"] < MIN_INTER:
            return False
        return (mm["iou"] >= IOU_MIN or max(mm["io_i"], mm["io_j"]) >= IO_PRIMARY
                or mm["dist"] <= SRC_NEAR)

    accept = []
    if n and m:
        ri, cj = linear_sum_assignment(cost)
        accept = [(int(i), int(j)) for i, j in zip(ri, cj) if accepts(int(i), int(j))]
    matched_i = {i for i, _ in accept}
    matched_j = {j for _, j in accept}
    orph_i = [i for i in range(n) if i not in matched_i]
    orph_j = [j for j in range(m) if j not in matched_j]

    def intra(gs, k):
        out = []
        for a, b in combinations(range(k), 2):
            if not gs[a].intersects(gs[b]):
                continue
            it = gs[a].intersection(gs[b]).area
            if it < MIN_INTRA:
                continue
            out.append((a, b, it / gs[a].area if gs[a].area else 0,
                        it / gs[b].area if gs[b].area else 0))
        return out
    intra_i = intra(gi, n) if n >= 2 else []
    intra_j = intra(gj, m) if m >= 2 else []

    def dirty(pairs_intra):
        d = set()
        for a, b, ioa, iob in pairs_intra:
            if max(ioa, iob) >= CONTAIN_TH:
                d.add(b if ioa >= iob else a)
            else:
                d.add(a); d.add(b)
        return d
    dirty_i, dirty_j = dirty(intra_i), dirty(intra_j)

    perfect = [(i, j) for i, j in accept if i not in dirty_i and j not in dirty_j]
    impure = [(i, j) for i, j in accept if i in dirty_i or j in dirty_j]

    cc = 1
    P_i, P_j = {}, {}
    for i, j in perfect + impure:
        P_i[i] = cc; P_j[j] = cc; cc += 1
    for i in orph_i:
        P_i[i] = cc; cc += 1
    for j in orph_j:
        P_j[j] = cc; cc += 1
    n_base = cc - 1

    L_i = {k: [v] for k, v in P_i.items()}
    L_j = {k: [v] for k, v in P_j.items()}

    def add_zones(pairs_intra, P, L, c):
        for a, b, ioa, iob in pairs_intra:
            if a not in P or b not in P:
                continue
            if max(ioa, iob) >= CONTAIN_TH:
                inner = a if ioa >= iob else b
                outer = b if inner == a else a
                z = P[inner]
                if z not in L[outer]:
                    L[outer].append(z)
            else:
                L[a].append(c); L[b].append(c); c += 1
        return c
    cc = add_zones(intra_i, P_i, L_i, cc)
    cc = add_zones(intra_j, P_j, L_j, cc)
    return L_i, L_j, n_base, cc - 1, accept, orph_i, orph_j, len(perfect)


# ---------- rasterization (port of 04.14) ----------

def stamp(raster, lon, lat, geom, value):
    minx, miny, maxx, maxy = geom.bounds
    cand = (lon >= minx) & (lon <= maxx) & (lat >= miny) & (lat <= maxy)
    idx = np.flatnonzero(cand)
    if idx.size == 0:
        return
    inside = shapely.contains_xy(geom, lon[idx], lat[idx])
    raster[idx[inside]] = value


def stamp_point(raster, lon, lat, lon0, lat0, value):
    if not (np.isfinite(lon0) and np.isfinite(lat0)):
        return
    d2 = (lon - lon0) ** 2 + (lat - lat0) ** 2
    k = int(np.argmin(d2))
    if d2[k] <= PT_MAX_DEG ** 2:
        raster[k] = value


def paint_side(lon, lat, shape2d, entries, codes, n_base):
    b1 = np.zeros(lon.size, dtype=np.uint8)
    b2 = np.zeros(lon.size, dtype=np.uint8)
    items = [(k, entries[k]["geom"], codes[k][0]) for k in codes]
    items.sort(key=lambda t: t[1].area, reverse=True)   # large ones first
    for _, g, base in items:
        stamp(b1, lon, lat, g, base)
    zone_ids = {}
    for k, cs in codes.items():
        for z in cs[1:]:
            if z > n_base:
                zone_ids.setdefault(z, []).append(k)
    for z, ks in zone_ids.items():
        if len(ks) != 2:
            continue
        a, b = ks
        inter = entries[a]["geom"].intersection(entries[b]["geom"])
        if not inter.is_empty:
            stamp(b1, lon, lat, inter, z)
    for k, cs in codes.items():
        lo0, la0 = entries[k]["pt"]
        stamp_point(b2, lon, lat, lo0, la0, cs[0])
    return b1.reshape(shape2d), b2.reshape(shape2d)


def write_mask(path, b1, b2):
    tmp = path.with_suffix(".tif.part")
    with rasterio.open(tmp, "w", width=b1.shape[1], height=b1.shape[0],
                       **BASE) as dst:
        dst.write(b1, 1); dst.write(b2, 2)
        dst.set_band_description(1, "mask")
        dst.set_band_description(2, "sources")
    tmp.rename(path)


def generate_one(ts):
    scene, is_free = G_SCENES[ts]
    out_dir = OUT_ROOT / scene
    f_im, f_cm = out_dir / "plume_imeo.tif", out_dir / "plume_cm.tif"
    # if the masks already exist, ONLY the code metadata is recomputed
    # (cheap) so that mask_codes.parquet always comes out complete
    solo_meta = f_im.exists() and f_cm.exists()
    try:
        with rasterio.open(out_dir / "latlon.tif") as s:
            lat2 = s.read(1); lon2 = s.read(2)
        shape2d = lat2.shape
        lon, lat = lon2.ravel().astype(np.float64), lat2.ravel().astype(np.float64)

        if is_free:
            if not solo_meta:
                z = np.zeros(shape2d, dtype=np.uint8)
                write_mask(f_im, z, z)
                write_mask(f_cm, z, z)
            return ts, "", {"scene": scene, "is_free": True, "imeo_cod": "{}",
                            "cm_cod": "{}", "n_base": 0, "max_code": 0,
                            "n_pairs": 0, "n_orph_imeo": 0, "n_orph_cm": 0}

        imeo = G_IMEO.get(ts, [])
        cm = []
        for e in G_CM.get(ts, []):
            geoms = [vectorize_cm_tif(CM_DIR / f"{p}.tif") for p in e["plume_ids"]]
            geoms = [g for g in geoms if g is not None]
            if not geoms:
                continue
            u = unary_union(geoms)
            if not u.is_valid:
                u = u.buffer(0)
            cm.append({**e, "geom": u})

        L_i, L_j, n_base, max_code, accept, orph_i, orph_j, n_perfect = \
            match_and_codes(imeo, cm) if (imeo or cm) else \
            ({}, {}, 0, 0, [], [], [], 0)
        assert max_code <= 255, f"{scene}: {max_code} codes, does not fit in uint8"

        if not solo_meta:
            b1i, b2i = paint_side(lon, lat, shape2d,
                                  {k: imeo[k] for k in range(len(imeo))}, L_i, n_base)
            b1c, b2c = paint_side(lon, lat, shape2d,
                                  {k: cm[k] for k in range(len(cm))}, L_j, n_base)
            write_mask(f_im, b1i, b2i)
            write_mask(f_cm, b1c, b2c)

        imeo_cod = {imeo[k]["id"]: v for k, v in L_i.items()}
        cm_cod = {cm[k]["source_name"]: v for k, v in L_j.items()}
        pairs = [{"imeo": imeo[i]["id"], "cm": cm[j]["source_name"]}
                 for i, j in accept]
        orphans = {"imeo": [imeo[i]["id"] for i in orph_i],
                   "cm": [cm[j]["source_name"] for j in orph_j]}
        return ts, "", {"scene": scene, "is_free": False,
                        "imeo_cod": json.dumps(imeo_cod),
                        "cm_cod": json.dumps(cm_cod),
                        "pairs": json.dumps(pairs),
                        "orphans": json.dumps(orphans),
                        "n_perfect": n_perfect,
                        "n_base": n_base, "max_code": max_code,
                        "n_pairs": len(accept),
                        "n_orph_imeo": len(orph_i), "n_orph_cm": len(orph_j)}
    except Exception as e:
        return ts, f"{type(e).__name__}: {e}", None


def main():
    global G_IMEO, G_CM, G_SCENES
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
    cross_dir = sorted((DATA / "cross").iterdir())[-1]
    splits = pd.read_parquet(cross_dir / "splits.parquet")

    scene_dirs = {TS_G.search(p.name).group(1).lower(): p.name
                  for p in OUT_ROOT.iterdir() if TS_G.search(p.name)}
    G_SCENES = {r.granule_ts: (scene_dirs[r.granule_ts], bool(r.is_free))
                for r in splits.itertuples() if r.granule_ts in scene_dirs}

    imeo_dir = sorted((DATA / "imeo").iterdir())[-1]
    pl = gpd.read_file(imeo_dir / "unep_methanedata_detected_plumes.geojson")
    pl = pl[pl.satellite.str.contains("EMIT", na=False)].copy()
    pl["granule_ts"] = pl.tile.str.extract(r"EMIT_L1B_RAD_001_(\d{8}T\d{6})")[0].str.lower()
    pl = pl[pl.granule_ts.isin(set(G_SCENES))]
    src = gpd.read_file(imeo_dir / "unep_methanedata_detected_sources.geojson")
    spt = {r.source_name: (r.geometry.x, r.geometry.y) for r in src.itertuples()}
    # IMEO is keyed by EMITTER (Julio's decision, like v1): in practice
    # 1 emitter = 1 plume per scene; the 2 double pairs merge into one geometry
    for (ts_, sn), grp in pl.groupby(["granule_ts", "source_name"]):
        geom = make_valid(unary_union([make_valid(g) for g in grp.geometry]))
        pt = spt.get(sn, (float(grp.lon.iloc[0]), float(grp.lat.iloc[0])))
        G_IMEO.setdefault(ts_, []).append({"id": sn, "geom": geom, "pt": pt})

    man = pd.read_parquet(CM_DIR / "manifest.parquet")
    cs = pd.read_parquet(CM_DIR / "sources.parquet")
    man = man.merge(cs, on="plume_id", how="left")
    man = man[man.granule_ts.isin(set(G_SCENES))]
    man["source_name"] = man.source_name.fillna(man.plume_id)
    for (ts, sn), grp in man.groupby(["granule_ts", "source_name"]):
        lo = grp.src_lon.dropna()
        la = grp.src_lat.dropna()
        pt = (float(lo.iloc[0]), float(la.iloc[0])) if len(lo) else \
             (float(grp.lon.iloc[0] or np.nan), float(grp.lat.iloc[0] or np.nan))
        G_CM.setdefault(ts, []).append(
            {"source_name": sn, "pt": pt, "plume_ids": list(grp.plume_id)})

    todo = list(G_SCENES)
    if limit:
        todo = todo[:limit]
    print(f"scenes: {len(todo)} · IMEO plumes: {sum(len(v) for v in G_IMEO.values())} · "
          f"CM sources: {sum(len(v) for v in G_CM.values())}", flush=True)

    rows, t0, errs = [], time.time(), 0
    with ProcessPoolExecutor(max_workers=WORKERS) as ex:
        futs = [ex.submit(generate_one, t) for t in todo]
        for i, fut in enumerate(as_completed(futs), 1):
            ts, err, meta = fut.result()
            if err and err != "already existed":
                errs += 1
                print(f"  ERROR {ts}: {err}", flush=True)
            if meta:
                rows.append(meta)
            if i % 50 == 0 or i == len(futs):
                print(f"  {i}/{len(todo)}  ({i/(time.time()-t0):.2f} scenes/s)",
                      flush=True)

    df = pd.DataFrame(rows)
    df.to_parquet(cross_dir / "mask_codes.parquet", index=False)
    ok = df[~df.is_free]
    print(f"\nscenes with codes: {len(ok)} · total pairs: {int(ok.n_pairs.sum())} · "
          f"global max code: {int(ok.max_code.max()) if len(ok) else 0} · errors {errs}")
    print(f"codes -> {cross_dir / 'mask_codes.parquet'}")


if __name__ == "__main__":
    main()
