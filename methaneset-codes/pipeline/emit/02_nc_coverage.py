"""MODULE 02 of the v2 pipeline: nc coverage for the cross-match universe.

Question: of the CONSENSUS granules from module 00 (plume in IMEO and in CM),
how many RAD and OBS are already on disk and how many would have to be
downloaded? This defines the real cost of the nodata analysis (which needs
the RAD).

Roots searched (by timestamp in the file name):
  RAD: MARS-Hyperspectral/EMIT_FULL (Cesar) + MARS-Hyperspectral_complement/EMIT_FULL
  OBS: MARS-Hyperspectral_complement/EMIT_OBS + METHANE_DATASETS_EMIT_NEW_TACO/OBS

Output:
  console (the summary) and assets/data/cross/<date>/nc_coverage.parquet with,
  per consensus granule: path of the RAD and of the OBS if they exist, or
  empty if missing.

Usage:
  /data/users/julio/.conda/envs/deep/bin/python code/v2/02_nc_coverage.py \
        > code/v2/02_nc_coverage.log 2>&1
"""
import pathlib
import re

import pandas as pd

DATA = pathlib.Path(__file__).resolve().parent.parent.parent / "assets" / "data"
RAD_ROOTS = [
    pathlib.Path("/data/databases/MARS-Hyperspectral/EMIT_FULL"),
    pathlib.Path("/data/databases/MARS-Hyperspectral_complement/EMIT_FULL"),
]
OBS_ROOTS = [
    pathlib.Path("/data/databases/MARS-Hyperspectral_complement/EMIT_OBS"),
    pathlib.Path("/data/databases/METHANE_DATASETS_EMIT_NEW_TACO/OBS"),
]
TS = re.compile(r"(\d{8}T\d{6})")


def index_root(roots, pattern):
    """timestamp (lowercase) -> path. If a ts appears in several roots, the first one wins."""
    idx = {}
    for root in roots:
        if not root.exists():
            continue
        for f in root.glob(pattern):
            m = TS.search(f.name)
            if m:
                idx.setdefault(m.group(1).lower(), str(f))
    return idx


def main():
    cross_dir = sorted((DATA / "cross").iterdir())[-1]
    g = pd.read_parquet(cross_dir / "granules.parquet")
    cons = g[g.consensus].copy()
    print(f"cross-match used: {cross_dir.name} · consensus granules: {len(cons):,}")

    rad = index_root(RAD_ROOTS, "*_RAD_*.nc")
    obs = index_root(OBS_ROOTS, "*_OBS_*.nc")
    print(f"RAD indexed on disk: {len(rad):,} · OBS indexed: {len(obs):,}")

    cons["rad_path"] = cons.granule_ts.map(rad).fillna("")
    cons["obs_path"] = cons.granule_ts.map(obs).fillna("")
    have_rad = (cons.rad_path != "").sum()
    have_obs = (cons.obs_path != "").sum()
    have_both = ((cons.rad_path != "") & (cons.obs_path != "")).sum()

    print("\n== consensus coverage ==")
    print(f"with RAD on disk:  {have_rad:,} / {len(cons):,}  (missing {len(cons)-have_rad:,})")
    print(f"with OBS on disk:  {have_obs:,} / {len(cons):,}  (missing {len(cons)-have_obs:,})")
    print(f"with BOTH:         {have_both:,} / {len(cons):,}")

    falta_rad = cons[cons.rad_path == ""]
    gb_rad = len(falta_rad) * 0.99
    print(f"\nestimated cost of downloading the missing RADs: ~{gb_rad:,.0f} GB "
          f"(at ~0.99 GB per RAD; OBS ~0.11 GB extra each)")
    print("missing by year:")
    print(falta_rad.groupby("year").size().to_string())

    out = cross_dir / "nc_coverage.parquet"
    cons.to_parquet(out, index=False)
    print(f"\ntable -> {out}")


if __name__ == "__main__":
    main()
