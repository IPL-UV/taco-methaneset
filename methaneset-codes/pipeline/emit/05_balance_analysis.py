"""MODULE 05 of the v2 pipeline: balance-axis analysis over the clean universe.

Input: the consensus granules with 0% nodata (modules 00 + 04).
Julio's question: which combination of axes (flux, year, region, sector,
sources, plumes, country...) gives the best stratification for an
ultra-balanced set? This module does NOT select: it measures the occupancy of
each combination of strata in order to choose with numbers.

Output: console (distributions per axis + occupancy per combination)
        and cross/<date>/candidates.parquet (enriched per-granule table,
        input of selector 06).

Usage:
  /data/users/julio/.conda/envs/deep/bin/python code/v2/05_balance_analysis.py \
        > code/v2/05_balance_analysis.log 2>&1
"""
import itertools
import pathlib
import re

import numpy as np
import pandas as pd

DATA = pathlib.Path(__file__).resolve().parent.parent.parent / "assets" / "data"
TS = re.compile(r"EMIT_L1B_RAD_001_(\d{8}T\d{6})")
WEAK = 1000.0

REGION = {
    # Central Asia
    "Turkmenistan": "Central Asia", "Kazakhstan": "Central Asia",
    "Uzbekistan": "Central Asia", "Tajikistan": "Central Asia",
    "Kyrgyzstan": "Central Asia", "Afghanistan": "Central Asia",
    # Middle East
    "Iran": "Middle East", "Iraq": "Middle East", "Saudi Arabia": "Middle East",
    "Kuwait": "Middle East", "Oman": "Middle East", "Yemen": "Middle East",
    "Syria": "Middle East", "United Arab Emirates": "Middle East",
    "Qatar": "Middle East", "Israel": "Middle East", "Jordan": "Middle East",
    "Bahrain": "Middle East",
    # North Africa
    "Algeria": "North Africa", "Libya": "North Africa", "Egypt": "North Africa",
    "Tunisia": "North Africa", "Morocco": "North Africa", "Sudan": "North Africa",
    # Rest of Africa
    "Nigeria": "Sub-Saharan Africa", "South Africa": "Sub-Saharan Africa",
    "Angola": "Sub-Saharan Africa", "Mozambique": "Sub-Saharan Africa",
    # North America
    "United States of America": "North America", "United States": "North America",
    "Mexico": "North America", "Canada": "North America",
    # Latin America
    "Argentina": "Latin America", "Venezuela": "Latin America",
    "Colombia": "Latin America", "Brazil": "Latin America",
    "Peru": "Latin America", "Bolivia": "Latin America", "Chile": "Latin America",
    # South and East Asia
    "India": "South/East Asia", "Pakistan": "South/East Asia",
    "China": "South/East Asia", "Bangladesh": "South/East Asia",
    "Indonesia": "South/East Asia", "Australia": "South/East Asia",
}


def mode_or_mixed(s):
    m = s.mode()
    return m.iloc[0] if len(m) else "unknown"


def occupancy(df, axes, targets=(500, 600, 700)):
    g = df.groupby(axes, observed=True).size()
    total = len(g)
    print(f"\n-- {' x '.join(axes)} --")
    print(f"   non-empty strata: {total} · median scenes/stratum: {g.median():.0f} "
          f"· min: {g.min()} · max: {g.max()}")
    for n in targets:
        q = n / total
        fillable = (g >= q).sum()
        cap = int(np.minimum(g, np.ceil(q)).sum())
        print(f"   target {n}: quota/stratum ~{q:.1f} · strata that fill it: "
              f"{fillable}/{total} · N reachable with a hard quota: {cap}")
    return g


def main():
    cross_dir = sorted((DATA / "cross").iterdir())[-1]
    nod = pd.read_parquet(cross_dir / "nodata.parquet")
    clean = set(nod[nod.nodata_pct == 0].granule_ts)
    print(f"clean universe (consensus + 0% nodata): {len(clean):,}")

    imeo_dir = sorted((DATA / "imeo").iterdir())[-1]
    im = pd.read_csv(imeo_dir / "unep_methanedata_detected_plumes.csv")
    im = im[im.satellite.str.contains("EMIT", na=False)].copy()
    im["granule_ts"] = im.tile.str.extract(TS)[0].str.lower()
    im = im[im.granule_ts.isin(clean)]

    # IMEO flux-completeness FILTER (Julio's decision, Aug 26):
    # drop any scene with an IMEO plume without flux (NaN) or with flux==0
    # (a flux of 0 makes no physical sense for a detected plume).
    # The CM side does NOT filter: its non-quantification is structural to the
    # catalog (25% of scenes) and filtering it would kill 35% of the weak
    # resource; it is declared.
    bad = set(im[im.ch4_fluxrate.isna() | (im.ch4_fluxrate == 0)].granule_ts)
    print(f"IMEO flux filter: dropped {len(bad)} scenes with a NaN/0 plume")
    im = im[~im.granule_ts.isin(bad)]

    agg = im.groupby("granule_ts").agg(
        n_plumes=("id_plume", "count"),
        n_weak=("ch4_fluxrate", lambda s: int(((s > 0) & (s < WEAK)).sum())),
        n_strong=("ch4_fluxrate", lambda s: int((s >= WEAK).sum())),
        n_sources=("source_name", "nunique"),
        sector=("sector", mode_or_mixed),
        country=("country", mode_or_mixed),
        flux_max=("ch4_fluxrate", "max"),
        lat=("lat", "mean"), lon=("lon", "mean"),
    ).reset_index()
    agg["year"] = agg.granule_ts.str[:4]
    agg["flux_bin"] = np.where(agg.n_weak > 0, "has_weak", "strong_only")
    agg["region"] = agg.country.map(REGION).fillna("Other")

    out = cross_dir / "candidates.parquet"
    agg.to_parquet(out, index=False)
    print(f"enriched table -> {out}  ({len(agg):,} granules)\n")

    print("== distributions per axis ==")
    for col in ["flux_bin", "year", "region", "sector"]:
        vc = agg[col].value_counts()
        print(f"\n{col}:")
        for k, v in vc.items():
            print(f"   {k:22s} {v:5d}  ({v/len(agg):.0%})")
    print(f"\nplumes per scene: median {agg.n_plumes.median():.0f} · "
          f"max {agg.n_plumes.max()} · scenes with >=2 sources: "
          f"{(agg.n_sources >= 2).sum()}")
    print(f"distinct countries: {agg.country.nunique()} "
          f"(top: {dict(agg.country.value_counts().head(6))})")

    print("\n== occupancy per combination of strata ==")
    occupancy(agg, ["flux_bin", "year"])
    occupancy(agg, ["flux_bin", "region"])
    occupancy(agg, ["flux_bin", "year", "region"])
    occupancy(agg, ["flux_bin", "year", "sector"])
    occupancy(agg, ["flux_bin", "year", "region", "sector"])


if __name__ == "__main__":
    main()
