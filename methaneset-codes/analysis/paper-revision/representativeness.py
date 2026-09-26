"""Representativeness of the EMIT subset of MethaneSET (700 + 21).

Compares the universe of original IMEO MARS and Carbon Mapper detections
(snapshot 2026-08-25, the same one used to build the paper cross) against the
plumes that fall inside the 700 selected scenes.

Populations:
  IMEO_all   EMIT plumes from the IMEO portal (all)
  IMEO_pool  EMIT plumes inside the 1,497 clean candidates (consensus+no
             nodata+complete IMEO flux): the selector's immediate "before"
  IMEO_sel   EMIT plumes inside the 700 selected scenes
  CM_all     EMIT CH4 plumes from the CM catalog (all)
  CM_sel     EMIT CH4 plumes inside the 700 selected scenes
Also, at scene level: IMEO granules (4,091), consensus (1,779), pool (1,497)
and selection (700), with each scene's flux_max.

Does not touch the vault: reads and writes only in /tmp/opencode/rep_emit/.
"""
import json
import pathlib
import re

import geopandas as gpd
import numpy as np
import pandas as pd

DATA = pathlib.Path("/data/users/julio/Notes/01-Projects/methanset/assets/data")
SNAP = "2026-08-25"
CROSS = DATA / "cross" / SNAP
TACO = pathlib.Path(
    "/data/databases/METHANESET_TACOS/methaneset-emit/METADATA/level0.parquet")
NE = pathlib.Path(
    "/data/users/julio/.local/share/cartopy/shapefiles/natural_earth/cultural"
    "/ne_110m_admin_0_countries.shp")
OUT = pathlib.Path("/tmp/opencode/rep_emit/resultados.md")

TS_IMEO = re.compile(r"EMIT_L1B_RAD_001_(\d{8}T\d{6})")
TS_CM = re.compile(r"^emi(\d{8}t\d{6})")
BINS = [0, 100, 300, 1000, 3000, np.inf]
LABELS = ["<100", "100-300", "300-1000", "1000-3000", ">3000"]

# Same region map as code/v2/05_balance_analysis.py (source of the pool).
REGION = {
    "Turkmenistan": "Central Asia", "Kazakhstan": "Central Asia",
    "Uzbekistan": "Central Asia", "Tajikistan": "Central Asia",
    "Kyrgyzstan": "Central Asia", "Afghanistan": "Central Asia",
    "Iran": "Middle East", "Iraq": "Middle East", "Saudi Arabia": "Middle East",
    "Kuwait": "Middle East", "Oman": "Middle East", "Yemen": "Middle East",
    "Syria": "Middle East", "United Arab Emirates": "Middle East",
    "Qatar": "Middle East", "Israel": "Middle East", "Jordan": "Middle East",
    "Bahrain": "Middle East",
    "Algeria": "North Africa", "Libya": "North Africa", "Egypt": "North Africa",
    "Tunisia": "North Africa", "Morocco": "North Africa", "Sudan": "North Africa",
    "Nigeria": "Sub-Saharan Africa", "South Africa": "Sub-Saharan Africa",
    "Angola": "Sub-Saharan Africa", "Mozambique": "Sub-Saharan Africa",
    "United States of America": "North America", "United States": "North America",
    "Mexico": "North America", "Canada": "North America",
    "Argentina": "Latin America", "Venezuela": "Latin America",
    "Colombia": "Latin America", "Brazil": "Latin America",
    "Peru": "Latin America", "Bolivia": "Latin America", "Chile": "Latin America",
    "India": "South/East Asia", "Pakistan": "South/East Asia",
    "China": "South/East Asia", "Bangladesh": "South/East Asia",
    "Indonesia": "South/East Asia", "Australia": "South/East Asia",
}
# Natural Earth -> IMEO portal names.
NE2PORTAL = {
    "Russia": "Russian Federation", "Iran": "Iran (Islamic Republic of)",
    "Syria": "Syrian Arab Republic", "Venezuela": "Venezuela",
    "Bolivia": "Bolivia (Plurinational State of)", "Türkiye": "Türkiye",
    "Turkey": "Türkiye", "Vietnam": "Viet Nam",
    "United Republic of Tanzania": "Tanzania",
    "Republic of the Congo": "Congo",
    "Democratic Republic of the Congo": "Democratic Republic of the Congo",
}
# From regional vocabulary to REGION keys.
REGION_ALIAS = {
    "Iran (Islamic Republic of)": "Iran",
    "Syrian Arab Republic": "Syria",
    "Venezuela (Bolivarian Republic of)": "Venezuela",
    "Bolivia (Plurinational State of)": "Bolivia",
    "Russian Federation": "Russian Federation",
}


def region_of(country):
    if not isinstance(country, str):
        return "Other"
    return REGION.get(REGION_ALIAS.get(country, country), "Other")


def pct_table(s, order=LABELS):
    s = pd.Series(s).dropna()
    s = s[s > 0]
    cat = pd.cut(s, BINS, labels=LABELS, right=False)
    sh = cat.value_counts(normalize=True).reindex(order).fillna(0)
    q = np.percentile(s, [10, 25, 50, 75, 90]) if len(s) else [np.nan] * 5
    return {
        "n": len(s),
        "weak<1000": f"{100 * (s < 1000).mean():.0f}",
        **{k: f"{100 * sh[k]:.0f}" for k in order},
        "p10": f"{q[0]:.0f}", "p25": f"{q[1]:.0f}", "p50": f"{q[2]:.0f}",
        "p75": f"{q[3]:.0f}", "p90": f"{q[4]:.0f}",
    }


def md_table(rows, index_name="population"):
    df = pd.DataFrame(rows).T
    df.index.name = index_name
    head = "| " + index_name + " | " + " | ".join(df.columns) + " |"
    sep = "|" + "---|" * (len(df.columns) + 1)
    body = "\n".join(
        f"| {i} | " + " | ".join(str(v) for v in r) + " |"
        for i, r in df.iterrows())
    return "\n".join([head, sep, body])


def topn_table(series_by_pop, n=10):
    """Top-n countries of the first criterion; rest grouped as Other."""
    keys = series_by_pop[list(series_by_pop)[0]].value_counts().head(n).index
    out = {}
    for name, s in series_by_pop.items():
        s = s.dropna()
        tot = len(s)
        out[name] = {}
        for k in keys:
            out[name][k] = f"{100 * (s == k).mean():.1f} ({int((s == k).sum())})"
        out[name]["Other"] = (f"{100 * (~s.isin(keys)).mean():.1f} "
                              f"({int((~s.isin(keys)).sum())})")
        out[name]["n"] = tot
    return out


def sect_imeo(s):
    m = {"Oil and Gas": "Oil and Gas", "Coal": "Coal", "Waste": "Waste"}
    return s.map(lambda x: m.get(x, "Other"))


def sect_cm(code):
    c = "" if pd.isna(code) else str(code).strip()
    for pref, name in (("1B1", "Coal"), ("1B2", "Oil and Gas"),
                       ("1A", "Other"), ("6", "Waste"), ("4B", "Other")):
        if c.startswith(pref):
            return name
    return "Other" if c else "Unknown"


def main():
    sel = pd.read_parquet(CROSS / "selection.parquet")
    cand = pd.read_parquet(CROSS / "candidates.parquet")
    gran = pd.read_parquet(CROSS / "granules.parquet")
    l0 = pd.read_parquet(TACO)
    sel_ts = set(sel.granule_ts)
    pool_ts = set(cand.granule_ts)

    # ---- IMEO ----
    im = pd.read_csv(DATA / "imeo" / SNAP / "unep_methanedata_detected_plumes.csv")
    im = im[im.satellite.str.contains("EMIT", na=False)].copy()
    im["granule_ts"] = im.tile.str.extract(TS_IMEO)[0].str.lower()
    im["year"] = im.tile_date.str[:4]
    im = im[im.granule_ts.notna()]
    im_all = im
    im_pool = im[im.granule_ts.isin(pool_ts)]
    im_sel = im[im.granule_ts.isin(sel_ts)]
    fl = {k: v.ch4_fluxrate for k, v in
          [("IMEO_all", im_all), ("IMEO_pool", im_pool), ("IMEO_sel", im_sel)]}

    # ---- CM ----
    cm = pd.read_parquet(DATA / "carbonmapper" / SNAP / "plumes.parquet")
    cm = cm[(cm.gas == "CH4") & (cm.instrument == "emi")].copy()
    cm["granule_ts"] = cm.plume_id.str.extract(TS_CM)[0]
    cm["year"] = pd.to_datetime(cm.scene_timestamp, format="mixed",
                                utc=True).dt.year.astype(str)
    cm_all = cm[cm.granule_ts.notna()]
    cm_sel = cm_all[cm_all.granule_ts.isin(sel_ts)]
    cfl = {"CM_all": cm_all.emission_auto, "CM_sel": cm_sel.emission_auto}

    # ---- CM geocoding and IMEO verification ----
    world = gpd.read_file(NE)[["ADMIN", "geometry"]]
    for name, d in (("CM_all", cm_all), ("CM_sel", cm_sel),
                    ("IMEO_check", im_all)):
        g = gpd.GeoDataFrame(
            d, geometry=gpd.points_from_xy(d.lon, d.lat), crs="EPSG:4326")
        j = gpd.sjoin(g, world, predicate="within", how="left")
        j = j[~j.index.duplicated(keep="first")]
        miss = j.ADMIN.isna()
        if miss.any():
            jn = gpd.sjoin_nearest(
                g[miss.values].to_crs(3857), world.to_crs(3857), how="left")
            jn = jn[~jn.index.duplicated(keep="first")]
            j.loc[miss, "ADMIN"] = jn.ADMIN.values
        j["nat"] = j.ADMIN.map(lambda x: NE2PORTAL.get(x, x))
        if name == "CM_all":
            cm_all = cm_all.assign(country_ne=j.nat.values)
        elif name == "CM_sel":
            cm_sel = cm_sel.assign(country_ne=j.nat.values)
        else:
            im_all = im_all.assign(country_ne=j.nat.values)
    agree = (im_all.country_ne == im_all.country).mean()

    # ---- flux tables ----
    t1_imeo = {k: pct_table(v) for k, v in fl.items()}
    t1_cm = {k: pct_table(v) for k, v in cfl.items()}
    taco_imeo, taco_cm = [], []
    for _, r in l0[~l0["selection:is_free"]].iterrows():
        taco_imeo += list(json.loads(r["detection:imeo_flux"] or "{}").values())
        taco_cm += list(json.loads(r["detection:cm_flux"] or "{}").values())
    t1_taco = {"TACO_IMEO_sel": pct_table(pd.Series(taco_imeo)),
               "TACO_CM_sel": pct_table(pd.Series(taco_cm))}

    # scene level (IMEO flux_max per granule)
    g_all = gran.copy()
    g_cons = gran[gran.consensus]
    g_pool = cand[["granule_ts", "flux_max"]]
    g_sel = sel[["granule_ts", "flux_max"]]
    t2 = {
        "IMEO_gran_all": pct_table(g_all.imeo_flux_max),
        "IMEO_gran_consensus": pct_table(g_cons.imeo_flux_max),
        "IMEO_gran_pool": pct_table(g_pool.flux_max),
        "IMEO_gran_sel": pct_table(g_sel.flux_max),
    }
    # CM at scene level (max emission_auto per granule)
    cm_g = cm_all.groupby("granule_ts").emission_auto.max()
    t2_cm = {
        "CM_gran_all": pct_table(cm_g),
        "CM_gran_sel": pct_table(
            cm_all[cm_all.granule_ts.isin(sel_ts)].groupby("granule_ts")
            .emission_auto.max()),
    }

    # ---- geography ----
    geos = {
        "IMEO_all": im_all.country,
        "IMEO_pool": im_pool.country,
        "IMEO_sel": im_sel.country,
        "CM_all": cm_all.country_ne,
        "CM_sel": cm_sel.country_ne,
    }
    country_tbl = topn_table(geos)
    regs = {k: v.map(region_of) for k, v in geos.items()}
    reg_keys = ["Central Asia", "Middle East", "North Africa", "North America",
                "Latin America", "South/East Asia", "Sub-Saharan Africa"]
    region_tbl = {}
    for name, s in regs.items():
        region_tbl[name] = {
            k: f"{100 * (s == k).mean():.1f} ({int((s == k).sum())})"
            for k in reg_keys}
        region_tbl[name]["Other"] = (
            f"{100 * (~s.isin(reg_keys)).mean():.1f} "
            f"({int((~s.isin(reg_keys)).sum())})")
        region_tbl[name]["n"] = len(s)

    # ---- sectors ----
    sectors = {
        "IMEO_all": sect_imeo(im_all.sector),
        "IMEO_pool": sect_imeo(im_pool.sector),
        "IMEO_sel": sect_imeo(im_sel.sector),
        "CM_all": cm_all.sector.map(sect_cm),
        "CM_sel": cm_sel.sector.map(sect_cm),
    }
    sec_keys = ["Oil and Gas", "Coal", "Waste", "Other", "Unknown"]
    sector_tbl = {}
    for name, s in sectors.items():
        s = s.fillna("Unknown")
        sector_tbl[name] = {
            k: f"{100 * (s == k).mean():.1f} ({int((s == k).sum())})"
            for k in sec_keys}
    # granule: dominant sector of the selection scenes vs pool
    sector_tbl["SEL_gran_scene"] = {
        k: f"{(sel.sector == k).mean() * 100:.1f} "
           f"({int((sel.sector == k).sum())})"
        for k in ["Oil and Gas", "Coal", "Waste", "Other"]}
    sector_tbl["POOL_gran_scene"] = {
        k: f"{(cand.sector == k).mean() * 100:.1f} "
           f"({int((cand.sector == k).sum())})"
        for k in ["Oil and Gas", "Coal", "Waste", "Other"]}

    # ---- time ----
    years = {
        "IMEO_all": im_all.year,
        "IMEO_pool": im_pool.year,
        "IMEO_sel": im_sel.year,
        "CM_all": cm_all.year,
        "CM_sel": cm_sel.year,
    }
    ykeys = sorted(set().union(*[set(v.dropna()) for v in years.values()]))
    year_tbl = {}
    for name, s in years.items():
        s = s.dropna()
        year_tbl[name] = {
            k: f"{100 * (s == k).mean():.1f} ({int((s == k).sum())})"
            for k in ykeys}
        year_tbl[name]["n"] = len(s)

    # ---- markdown assembly ----
    L = []
    A = L.append
    A("# Representativeness of the EMIT subset (MethaneSET)\n")
    A(f"- Snapshots: IMEO and CM from `{SNAP}` (the same as in the paper cross).")
    A("- 'After' = plumes that fall inside the 700 scenes of "
      "`selection.parquet`; the 21 free granules contribute no plumes.")
    A("- Flux classes in kg/h. Percentiles in kg/h. "
      "NaN and zeros are excluded (n is reported).")
    A(f"- CM geocoding with Natural Earth 110m (sjoin + nearest "
      f"neighbor); portal-vs-NE country agreement in IMEO EMIT: {100 * agree:.1f}%.")
    A("- The snapshot CM catalog ends on 2025-11-12; the IMEO universe "
      "reaches 2026-07-24 (1,339 plumes), so the consensus cannot cover "
      "2026.\n")

    A("## 0. Populations and funnel\n")
    A(f"| level | n |")
    A(f"|---|---|")
    A(f"| IMEO EMIT granules (plumes) | {im_all.granule_ts.nunique():,} "
      f"({len(im_all):,} plumes) |")
    A(f"| IMEO∩CM consensus | {int(gran.consensus.sum()):,} granules |")
    A(f"| clean pool without nodata + complete IMEO flux | "
      f"{len(cand):,} granules ({len(im_pool):,} plumes) |")
    A(f"| final selection | {len(sel):,} granules ({len(im_sel):,} plumes) "
      f"+ 21 free |")
    A(f"| CM EMIT CH4 granules (plumes) | "
      f"{cm_all.granule_ts.nunique():,} ({len(cm_all):,} plumes) |")
    A(f"| CM inside the selection | {cm_sel.granule_ts.nunique():,} "
      f"granules ({len(cm_sel):,} plumes) |")
    A(f"| IMEO with NaN/0 flux | {int(im_all.ch4_fluxrate.isna().sum())} NaN "
      f"+ {int((im_all.ch4_fluxrate == 0).sum())} zeros |")
    A(f"| CM without quantified flux | "
      f"{int(cm_all.emission_auto.isna().sum())} NaN + "
      f"{int((cm_all.emission_auto == 0).sum())} zeros |\n")

    A("## 1a. Flux per PLUME (kg/h)\n")
    A("**IMEO**\n")
    A(md_table(t1_imeo))
    A("\n**Carbon Mapper** (emission_auto; NaN = not quantified)\n")
    A(md_table(t1_cm))
    A("\n**Check against the TACO metadata (the 700, plumes per emitter)**\n")
    A(md_table(t1_taco))
    A("")

    A("## 1b. Maximum flux per SCENE (kg/h)\n")
    A("**IMEO** (scene max)\n")
    A(md_table(t2))
    A("\n**CM** (scene max)\n")
    A(md_table(t2_cm))
    A("")

    A("## 2. Countries (top 10 by IMEO_all + Other)\n")
    A(md_table(country_tbl))
    A("\n## Regions (map of `05_balance_analysis.py`)\n")
    A(md_table(region_tbl))
    A("")

    A("## 3. Sectors (% of plumes)\n")
    A(md_table(sector_tbl))
    A("\n*IMEO: grouped portal sector. CM: IPCC code mapped "
      "(1B1→Coal, 1B2→Oil and Gas, 6→Waste, rest→Other). "
      "The last two rows are the dominant sector of the SCENE "
      "(what the selector balanced).*\n")

    A("## 4. Observation year (% of plumes)\n")
    A(md_table(year_tbl))

    OUT.write_text("\n".join(L))
    print("\n".join(L))
    print(f"\n[saved to {OUT}]")


if __name__ == "__main__":
    main()
