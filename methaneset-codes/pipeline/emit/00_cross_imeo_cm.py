"""MODULE 00 of the official MethaneSET v2 pipeline: the IMEO x CM cross-match.

What it does, in order:
  1. Downloads the day's fresh catalogs (IMEO portal + Carbon Mapper API)
     to assets/data/<source>/<YYYY-MM-DD>/. If today's snapshot already
     exists, it reuses it (idempotent).
  2. Builds the EMIT granule key on both sides:
       IMEO:  tile  = EMIT_L1B_RAD_001_<YYYYMMDDTHHMMSS>_...  -> timestamp
       CM:    plume_id = emi<yyyymmddthhmmss>p...             -> timestamp
  3. Reports the cross-match TOTALS (only granules with a plume):
     plumes and granules per catalog, consensus (granule in both), exclusives.
  4. Saves the per-granule table for the consensus and the exclusives to
     assets/data/cross/<YYYY-MM-DD>/granules.parquet, which is the input of
     the following modules (nodata, balance, splits).

Usage (with twin log, pipeline convention):
  cd 01-Projects/methanset
  nohup /data/users/julio/.conda/envs/deep/bin/python code/v2/00_cross_imeo_cm.py \
        > code/v2/00_cross_imeo_cm.log 2>&1 &
"""
import datetime
import pathlib
import re
import subprocess
import sys

import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent          # code/v2
CODE = HERE.parent                                       # code (fetchers live there)
DATA = CODE.parent / "assets" / "data"
PY = sys.executable

IMEO_PLUMES = "unep_methanedata_detected_plumes.csv"
TS_IMEO = re.compile(r"EMIT_L1B_RAD_001_(\d{8}T\d{6})")
TS_CM = re.compile(r"^emi(\d{8}t\d{6})")


def ensure_snapshot(source, script, check_file):
    day = datetime.date.today().isoformat()
    dest = DATA / source / day
    if (dest / check_file).exists():
        print(f"[{source}] today's snapshot already exists: {dest}")
        return dest
    print(f"[{source}] downloading snapshot {day}...")
    subprocess.run([PY, str(CODE / script)], check=True)
    if not (dest / check_file).exists():
        raise RuntimeError(f"download of {source} did not leave {check_file} in {dest}")
    return dest


def main():
    imeo_dir = ensure_snapshot("imeo", "fetch_imeo.py", IMEO_PLUMES)
    cm_dir = ensure_snapshot("carbonmapper", "fetch_carbonmapper.py", "plumes.parquet")

    im = pd.read_csv(imeo_dir / IMEO_PLUMES)
    im = im[im.satellite.str.contains("EMIT", na=False)].copy()
    im["granule_ts"] = im.tile.str.extract(TS_IMEO)[0].str.lower()
    im = im[im.granule_ts.notna()]

    cm = pd.read_parquet(cm_dir / "plumes.parquet")
    cm = cm[(cm.gas == "CH4") & (cm.instrument == "emi")].copy()
    cm["granule_ts"] = cm.plume_id.str.extract(TS_CM)[0]
    cm = cm[cm.granule_ts.notna()]

    g_im, g_cm = set(im.granule_ts), set(cm.granule_ts)
    both = g_im & g_cm

    print("\n== IMEO x CM CROSS-MATCH over EMIT (only granules with a plume) ==")
    print(f"IMEO:  {len(im):,} plumes in {len(g_im):,} granules")
    print(f"CM:    {len(cm):,} CH4 plumes in {len(g_cm):,} granules")
    print(f"consensus (granule in BOTH): {len(both):,} granules")
    print(f"IMEO only: {len(g_im - g_cm):,}   CM only: {len(g_cm - g_im):,}")
    n_im_both = im.granule_ts.isin(both).sum()
    n_cm_both = cm.granule_ts.isin(both).sum()
    print(f"plumes inside the consensus: IMEO {n_im_both:,} + CM {n_cm_both:,}")

    agg_im = im.groupby("granule_ts").agg(
        n_imeo=("id_plume", "count"), imeo_flux_max=("ch4_fluxrate", "max"),
        imeo_flux_med=("ch4_fluxrate", "median"),
        first_country=("country", "first"), first_source=("source_name", "first"))
    agg_cm = cm.groupby("granule_ts").agg(
        n_cm=("plume_id", "count"), cm_flux_max=("emission_auto", "max"),
        cm_flux_med=("emission_auto", "median"))
    g = agg_im.join(agg_cm, how="outer")
    g["n_imeo"] = g.n_imeo.fillna(0).astype(int)
    g["n_cm"] = g.n_cm.fillna(0).astype(int)
    g["consensus"] = (g.n_imeo > 0) & (g.n_cm > 0)
    g["year"] = g.index.str[:4].astype(int)

    out = DATA / "cross" / datetime.date.today().isoformat()
    out.mkdir(parents=True, exist_ok=True)
    g.reset_index().to_parquet(out / "granules.parquet", index=False)
    print(f"\nper-granule table -> {out / 'granules.parquet'}  ({len(g):,} rows)")
    print("\nconsensus by year:")
    print(g[g.consensus].groupby("year").size().to_string())


if __name__ == "__main__":
    main()
