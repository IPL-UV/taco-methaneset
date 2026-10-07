"""MODULE 17b of pipeline v2: the final metadata table (scene level).

REVISION 2 after Julio's feedback (Aug 26), changes applied:
  - IMEO is identified by EMITTER (source_name), as in v1: imeo_ids,
    imeo_flux and imeo_cod are keyed by emitter (masks re-keyed in 15).
  - detection:coverage_imeo/cm/union: % of plume pixels per scene,
    measured from the masks themselves.
  - imeo_flux / cm_flux as DICTS emitter -> flux (null if the plume
    has no fluxrate; detection:n_noflux counts them).
  - detection:sectors per plume with the v1 structure
    {sector: {"imeo": [...], "cm": [...]}} + dominant detection:sector.
  - sensor: MEANS only and coherent renaming: sza/vza (solar and view
    zenith) + saa/vaa (solar and view azimuth). Min/max out.
  - meteo: next to the angles (mean of our own wind.tif, covers frees).
  - spatial:imeo_points/cm_points in WKT MULTIPOINT (as in v1).
  - selection at the END: split, is_free, flux_bin. consensus is removed
    (redundant: in v2 consensus == not is_free by construction).
  - out: match:max_code and n_base (internal diagnostic, stays in
    mask_codes.parquet); in: match:pairs, orphans, n_perfect (v1).
  - standard rounding: angles/wind/amf/elevation/coverage 3 decimals,
    flux 2, radiance 4, bbox 6, path_length 1, earth_sun 5.

Output: cross/<date>/metadata_level0.parquet
Usage:
  /data/users/julio/.conda/envs/deep/bin/python code/v2/17b_build_metadata.py \
        > code/v2/17b_build_metadata.log 2>&1
"""
import json
import pathlib
import re
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed

import h5py
import numpy as np
import pandas as pd
import rasterio
from shapely.geometry import MultiPoint

DATA = pathlib.Path(__file__).resolve().parent.parent.parent / "assets" / "data"
# the emit/ workshop no longer exists: samples live in the TACO (hardlinks)
ROOT = pathlib.Path("/data/databases/METHANSET_TACOS/methaneset-emit/DATA")
# CM manifest/sources: dated backup in the vault (cm_plume_tifs deleted)
CM_DIR = pathlib.Path("/data/users/julio/Notes/01-Projects/methanset/assets/data/carbonmapper/2026-08-25")
RAD_ROOTS = [pathlib.Path("/data/databases/MARS-Hyperspectral/EMIT_FULL"),
             pathlib.Path("/data/databases/MARS-Hyperspectral_complement/EMIT_FULL")]
OBS_ROOTS = [pathlib.Path("/data/databases/MARS-Hyperspectral_complement/EMIT_OBS"),
             pathlib.Path("/data/databases/METHANE_DATASETS_EMIT_NEW_TACO/OBS")]
V1_L0 = pathlib.Path("/data/databases/METHANE_DATASETS_TACOv2/methaneset-emit/"
                     "methaneset-emit/METADATA/level0.parquet")
TS_G = re.compile(r"EMIT_L1B_RAD_001_(\d{8}T\d{6})")
TS_ANY = re.compile(r"(\d{8}T\d{6})")
WEAK = 1000.0
WORKERS = 8

COLUMN_ORDER = [
    "id", "type",
    "detection:n_imeo", "detection:n_cm",
    "detection:coverage_imeo", "detection:coverage_cm", "detection:coverage_union",
    "detection:imeo_ids", "detection:cm_ids",
    "detection:imeo_flux", "detection:cm_flux",
    "detection:sectors", "detection:sector_list", "detection:sector",
    "detection:n_imeo_weak", "detection:n_imeo_strong", "detection:n_cm_noflux",
    "detection:n_cm_dup_pairs", "detection:cm_dup_groups",
    "detection:imeo_flux_total", "detection:imeo_flux_max",
    "detection:cm_flux_total", "detection:cm_flux_max",
    "detection:imeo_cod", "detection:cm_cod",
    "match:pairs", "match:orphans", "match:n_pairs", "match:n_perfect",
    "match:n_orph_imeo", "match:n_orph_cm",
    "sensor:shape_rows", "sensor:shape_cols",
    "sensor:sza_mean", "sensor:vza_mean", "sensor:saa_mean", "sensor:vaa_mean",
    "sensor:amf_mean", "sensor:phase_mean",
    "sensor:path_length_mean", "sensor:earth_sun_distance",
    "meteo:wind_u", "meteo:wind_v", "meteo:wind_speed",
    "meteo:imeo_wind", "meteo:cm_wind",
    "emit:flight_line", "emit:time_start", "emit:time_end",
    "radiance:min", "radiance:max", "radiance:elev_min_m", "radiance:elev_max_m",
    "site:country",
    "spatial:bbox_west", "spatial:bbox_south", "spatial:bbox_east",
    "spatial:bbox_north", "spatial:imeo_points", "spatial:cm_points",
    "selection:split", "selection:is_free", "selection:flux_bin",
]

ROUND = {3: ["sensor:sza_mean", "sensor:vza_mean", "sensor:saa_mean",
             "sensor:vaa_mean", "sensor:amf_mean", "sensor:phase_mean",
             "meteo:wind_u",
             "meteo:wind_v", "meteo:wind_speed", "radiance:elev_min_m",
             "radiance:elev_max_m", "detection:coverage_imeo",
             "detection:coverage_cm", "detection:coverage_union"],
         2: ["detection:imeo_flux_total", "detection:imeo_flux_max",
             "detection:cm_flux_total", "detection:cm_flux_max"],
         4: ["radiance:min", "radiance:max"],
         6: ["spatial:bbox_west", "spatial:bbox_south", "spatial:bbox_east",
             "spatial:bbox_north"],
         1: ["sensor:path_length_mean"],
         5: ["sensor:earth_sun_distance"]}

G = {}

# CM sectors come as IPCC codes; they are mapped to the IMEO vocabulary
def cm_sector_name(code):
    """CM IPCC map -> IMEO vocabulary, prefix-robust (1B1a, etc.)."""
    c = (code or "").strip()
    for pref, name in (("1B1", "Coal"), ("1B2", "Oil and Gas"),
                       ("1A", "Energy"), ("6", "Waste"), ("4B", "Livestock")):
        if c.startswith(pref):
            return name
    return "Other" if c else "Unknown"


def _index(roots, pattern):
    idx = {}
    for root in roots:
        for f in root.glob(pattern):
            m = TS_ANY.search(f.name)
            if m:
                idx.setdefault(m.group(1).lower(), str(f))
    return idx


def obs_stats(path):
    with h5py.File(path, "r") as h:
        names = [b.decode() if isinstance(b, bytes) else str(b)
                 for b in h["sensor_band_parameters/observation_bands"][:]]
        obs = h["obs"]
        def band(key):
            i = next(k for k, n in enumerate(names) if key in n.lower())
            a = obs[:, :, i]
            return a[a > -9000]
        out = {"sensor:sza_mean": float(band("to-sun zenith").mean()),
               "sensor:vza_mean": float(band("to-sensor zenith").mean()),
               "sensor:saa_mean": float(band("to-sun azimuth").mean()),
               "sensor:vaa_mean": float(band("to-sensor azimuth").mean()),
               "sensor:phase_mean": float(band("solar phase").mean()),
               "sensor:path_length_mean": float(band("path length").mean()),
               "sensor:earth_sun_distance": float(band("earth-sun").mean())}
        out["sensor:amf_mean"] = float(
            1 / np.cos(np.radians(out["sensor:sza_mean"]))
            + 1 / np.cos(np.radians(out["sensor:vza_mean"])))
        return out


def build_row(ts):
    g = G[ts]
    d = ROOT / g["scene"]
    r = {"id": g["scene"], "type": "FOLDER"}
    try:
        with rasterio.open(d / "radiance.tif") as s:
            r["sensor:shape_rows"], r["sensor:shape_cols"] = s.height, s.width
        cached = G[ts].get("minmax")
        if cached:
            r["radiance:min"], r["radiance:max"] = cached
        else:
            # TRUE min/max over the full cube, in band blocks
            rmin, rmax = np.inf, -np.inf
            with rasterio.open(d / "radiance.tif") as s:
                for b0 in range(1, s.count + 1, 32):
                    blk = s.read(list(range(b0, min(b0 + 32, s.count + 1))))
                    v = blk[blk > -9000]
                    if v.size:
                        rmin = min(rmin, float(v.min()))
                        rmax = max(rmax, float(v.max()))
            r["radiance:min"] = rmin if np.isfinite(rmin) else np.nan
            r["radiance:max"] = rmax if np.isfinite(rmax) else np.nan
        with rasterio.open(d / "elevation.tif") as s:
            e = s.read(1)
            r["radiance:elev_min_m"] = float(np.nanmin(e))
            r["radiance:elev_max_m"] = float(np.nanmax(e))
        with rasterio.open(d / "latlon.tif") as s:
            la, lo = s.read(1), s.read(2)
            r["spatial:bbox_west"], r["spatial:bbox_east"] = float(lo.min()), float(lo.max())
            r["spatial:bbox_south"], r["spatial:bbox_north"] = float(la.min()), float(la.max())
        with rasterio.open(d / "wind.tif") as s:
            u, v_ = s.read(1), s.read(2)
            r["meteo:wind_u"] = float(u.mean()); r["meteo:wind_v"] = float(v_.mean())
            r["meteo:wind_speed"] = float(np.hypot(u, v_).mean())
        with rasterio.open(d / "plume_imeo.tif") as s:
            mi = s.read(1) > 0
        with rasterio.open(d / "plume_cm.tif") as s:
            mc = s.read(1) > 0
        r["detection:coverage_imeo"] = float(mi.mean() * 100)
        r["detection:coverage_cm"] = float(mc.mean() * 100)
        r["detection:coverage_union"] = float((mi | mc).mean() * 100)
        r.update(obs_stats(g["obs"]))
        with h5py.File(g["rad"], "r") as h:
            for k, col in (("flight_line", "emit:flight_line"),
                           ("time_coverage_start", "emit:time_start"),
                           ("time_coverage_end", "emit:time_end")):
                val = h.attrs.get(k, b"")
                r[col] = val.decode() if isinstance(val, bytes) else str(val)
        r.update(g["det"])
        for nd, cols in ROUND.items():
            for c in cols:
                if c in r and r[c] is not None and np.isfinite(r[c]):
                    r[c] = round(r[c], nd)
        return r, ""
    except Exception as e:
        return r, f"{type(e).__name__}: {e}"


def wkt_points(pairs):
    pts = [(a, b) for a, b in pairs if np.isfinite(a) and np.isfinite(b)]
    return MultiPoint(pts).wkt if pts else "MULTIPOINT EMPTY"


def main():
    cross_dir = sorted((DATA / "cross").iterdir())[-1]
    splits = pd.read_parquet(cross_dir / "splits.parquet")
    codes = pd.read_parquet(cross_dir / "mask_codes.parquet")
    codes["ts"] = codes.scene.str.extract(TS_G)[0].str.lower()
    codes = codes.set_index("ts")
    scene_of = {TS_G.search(p.name).group(1).lower(): p.name
                for p in ROOT.iterdir() if TS_G.search(p.name)}
    rad_idx = _index(RAD_ROOTS, "*_RAD_*.nc")
    obs_idx = _index(OBS_ROOTS, "*_OBS_*.nc")

    # latest snapshot that DOES contain the plume csv (there are dated folders
    # for other IMEO products, e.g. the response-rate)
    imeo_dir = sorted(d for d in (DATA / "imeo").iterdir()
                      if (d / "unep_methanedata_detected_plumes.csv").exists())[-1]
    im = pd.read_csv(imeo_dir / "unep_methanedata_detected_plumes.csv")
    im = im[im.satellite.str.contains("EMIT", na=False)].copy()
    im["ts"] = im.tile.str.extract(r"EMIT_L1B_RAD_001_(\d{8}T\d{6})")[0].str.lower()
    src = pd.read_csv(imeo_dir / "unep_methanedata_detected_sources.csv")
    src_pt = {r.source_name: (float(r.lon), float(r.lat)) for r in src.itertuples()}

    man = pd.read_parquet(CM_DIR / "manifest.parquet").merge(
        pd.read_parquet(CM_DIR / "sources.parquet"), on="plume_id", how="left")
    man["source_name"] = man.source_name.fillna(man.plume_id)

    # PER-PLUME wind published by each institution (module 20). This is the one
    # THEY used to invert the flux: different from meteo:wind_* (our ERA5-Land,
    # scene mean).
    wind_pq = cross_dir / "plume_wind.parquet"
    wind_by_scene = {}
    if wind_pq.exists():
        pw = pd.read_parquet(wind_pq)
        for (gran, cat), grp in pw.groupby(["granule", "catalog"]):
            d = {}
            for key, g in grp.groupby("plume_key"):
                u, v = g.wind_u.mean(), g.wind_v.mean()
                if pd.isna(u) or pd.isna(v):
                    d[key] = None
                    continue
                e = {"u": round(float(u), 3), "v": round(float(v), 3),
                     "speed": round(float(np.hypot(u, v)), 3)}
                if cat == "cm":  # CM publishes direction, IMEO components
                    e["dir"] = round(float((np.degrees(np.arctan2(-u, -v))) % 360), 3)
                d[key] = e
            wind_by_scene[(gran, cat)] = d
        print(f"institutional wind: {len(wind_by_scene)} (scene, catalog)")
    else:
        print("WARNING: plume_wind.parquet missing (run 20); wind columns empty")

    v1 = pd.read_parquet(V1_L0, columns=["id", "site:country"]).set_index("id")
    sel = pd.read_parquet(cross_dir / "selection.parquet").set_index("granule_ts")

    # radiance min/max cache from the previous table (the expensive step)
    mm_cache = {}
    prev = cross_dir / "metadata_level0.parquet"
    if prev.exists():
        p = pd.read_parquet(prev, columns=["id", "radiance:min", "radiance:max"])
        mm_cache = {r.id: (r._2, r._3) for r in p.itertuples()
                    if pd.notna(r._2)}
        print(f"cache min/max: {len(mm_cache)} scenes")

    global G
    for row in splits.itertuples():
        ts = row.granule_ts
        if ts not in scene_of or ts not in rad_idx or ts not in obs_idx:
            print(f"WARNING {ts}: missing scene/RAD/OBS, skipping")
            continue
        scene = scene_of[ts]
        pi = im[im.ts == ts]
        pc = man[man.granule_ts == ts]

        # IMEO by emitter (flux summed if an emitter brings 2 plumes)
        iflux = {}
        for sn, grp in pi.groupby("source_name"):
            f = grp.ch4_fluxrate.dropna()
            iflux[sn] = round(float(f.sum()), 2) if len(f) else None
        cflux = {}
        for sn, grp in pc.groupby("source_name"):
            f = grp.emission_auto.dropna()
            cflux[sn] = round(float(f.sum()), 2) if len(f) else None

        sectors = {}
        for sn, grp in pi.groupby("source_name"):
            sec = grp.sector.dropna()
            sec = sec.iloc[0] if len(sec) else "Unknown"
            sectors.setdefault(sec, {"imeo": [], "cm": []})["imeo"].append(sn)
        for sn, grp in pc.groupby("source_name"):
            sec = grp.sector.dropna()
            sec = sec.iloc[0] if len(sec) else None
            sec = cm_sector_name(sec)
            sectors.setdefault(sec, {"imeo": [], "cm": []})["cm"].append(sn)
        sec_counts = Counter()
        for sec, d_ in sectors.items():
            sec_counts[sec] += len(d_["imeo"]) + len(d_["cm"])
        dominant = sec_counts.most_common(1)[0][0] if sec_counts else ""

        iwind = wind_by_scene.get((scene, "imeo"), {})
        cwind = wind_by_scene.get((scene, "cm"), {})

        det = {
            "meteo:imeo_wind": json.dumps(iwind),
            "meteo:cm_wind": json.dumps(cwind),
            "detection:n_imeo": int(pi.source_name.nunique()),
            "detection:n_cm": int(pc.source_name.nunique()),
            "detection:imeo_ids": json.dumps(sorted(pi.source_name.dropna().unique())),
            "detection:cm_ids": json.dumps(sorted(pc.source_name.dropna().unique())),
            "detection:imeo_flux": json.dumps(iflux),
            "detection:cm_flux": json.dumps(cflux),
            "detection:sectors": json.dumps(sectors),
            "detection:sector_list": json.dumps(sorted(sectors)),
            "detection:sector": dominant,
            # weak/strong ALWAYS on the IMEO flux (the design baseline);
            # the completeness filter guarantees 0 < flux for every IMEO plume.
            "detection:n_imeo_weak": int(((pi.ch4_fluxrate > 0)
                                          & (pi.ch4_fluxrate < WEAK)).sum()),
            "detection:n_imeo_strong": int((pi.ch4_fluxrate >= WEAK).sum()),
            # only CM can come unquantified (structural in its catalog)
            "detection:n_cm_noflux": int(sum(1 for v in cflux.values()
                                             if v is None)),
            "detection:imeo_flux_total": float(pi.ch4_fluxrate.sum()),
            "detection:imeo_flux_max": (float(pi.ch4_fluxrate.max())
                                        if pi.ch4_fluxrate.notna().any() else np.nan),
            "detection:cm_flux_total": float(pc.emission_auto.sum()),
            "detection:cm_flux_max": (float(pc.emission_auto.max())
                                      if pc.emission_auto.notna().any() else np.nan),
            "spatial:imeo_points": wkt_points(
                [src_pt.get(sn, (np.nan, np.nan))
                 for sn in sorted(pi.source_name.dropna().unique())]),
            "spatial:cm_points": wkt_points(
                [tuple(x) for x in pc.groupby("source_name")[["src_lon", "src_lat"]]
                 .first().dropna().itertuples(index=False)]),
            "selection:split": row.split,
            "selection:is_free": bool(row.is_free),
            "selection:flux_bin": ("free" if row.is_free else
                                   ("has_weak" if ((pi.ch4_fluxrate > 0)
                                                   & (pi.ch4_fluxrate < WEAK)).any()
                                    else "strong_only")),
            "site:country": (sel.loc[ts, "country"] if ts in sel.index
                             else v1["site:country"].get(scene, "")),
        }
        if ts in codes.index:
            c = codes.loc[ts]
            def _i(key):
                v = c.get(key)
                return int(v) if pd.notna(v) else 0
            def _s(key, default):
                v = c.get(key)
                return v if isinstance(v, str) else default
            det.update({"detection:imeo_cod": _s("imeo_cod", "{}"),
                        "detection:cm_cod": _s("cm_cod", "{}"),
                        "match:pairs": _s("pairs", "[]"),
                        "match:orphans": _s("orphans", '{"imeo": [], "cm": []}'),
                        "match:n_pairs": _i("n_pairs"),
                        "match:n_perfect": _i("n_perfect"),
                        "match:n_orph_imeo": _i("n_orph_imeo"),
                        "match:n_orph_cm": _i("n_orph_cm")})
        G[ts] = {"scene": scene, "rad": rad_idx[ts], "obs": obs_idx[ts],
                 "det": det, "minmax": mm_cache.get(scene)}

    print(f"scenes with full context: {len(G)}", flush=True)
    rows, errs, t0 = [], 0, time.time()
    with ProcessPoolExecutor(max_workers=WORKERS) as ex:
        futs = [ex.submit(build_row, ts) for ts in G]
        for i, fut in enumerate(as_completed(futs), 1):
            r, err = fut.result()
            if err:
                errs += 1
                print(f"  ERROR {r['id']}: {err}", flush=True)
            else:
                rows.append(r)
            if i % 100 == 0 or i == len(futs):
                print(f"  {i}/{len(G)}  ({i/(time.time()-t0):.2f}/s)", flush=True)

    df = pd.DataFrame(rows).sort_values("id")

    # warning from the CM overlap bug (module 19e): CM plume pairs from
    # DIFFERENT sources with IoU >= 0.5 in the scene. If > 0, n_cm and the CM
    # source partition are an upper bound (see slide 'One plume, two identities').
    dup_pq = cross_dir / "cm_dup_scenes.parquet"
    if dup_pq.exists():
        # the GROUP of twin sources is published (not loose pairs): a single
        # plume can be archived under up to 6 CM sources.
        dup = pd.read_parquet(dup_pq)[["id", "n_cm_dup_pairs", "cm_dup_groups"]]
        dup = dup.rename(columns={"n_cm_dup_pairs": "detection:n_cm_dup_pairs",
                                  "cm_dup_groups": "detection:cm_dup_groups"})
        df = df.merge(dup, on="id", how="left")
        df["detection:n_cm_dup_pairs"] = (
            df["detection:n_cm_dup_pairs"].fillna(0).astype(int))
        df["detection:cm_dup_groups"] = df["detection:cm_dup_groups"].fillna("[]")
        print(f"n_cm_dup_pairs: {int((df['detection:n_cm_dup_pairs']>0).sum())} "
              "scenes with CM duplicates")
    else:
        print("WARNING: cm_dup_scenes.parquet missing (run 19e); column omitted")

    cols = [c for c in COLUMN_ORDER if c in df.columns]
    extra = [c for c in df.columns if c not in cols]
    df = df[cols + extra]
    out = cross_dir / "metadata_level0.parquet"
    df.to_parquet(out, index=False)
    print(f"\nrows {len(df)} x columns {len(df.columns)} · errors {errs}")
    print("order:", list(df.columns)[:6], "...", list(df.columns)[-3:])
    print(f"-> {out}")


if __name__ == "__main__":
    main()
