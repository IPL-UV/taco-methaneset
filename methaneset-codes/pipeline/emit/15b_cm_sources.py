"""MODULE 15b of the v2 pipeline: associate each CM plume with its source.

The CM API does not include the source in the plume listing; it is queried per
plume at /catalog/source/plume/name/{plume_id}, which returns source_name and
the canonical point of the source. It is needed for:
  - the per-source dissolve (several CM records of one same cloud)
  - band 2 of plume_cm.tif (emission point)
  - the IMEO x CM matching (distance between sources)

Input: cm_plume_tifs/manifest.parquet (from 15a).
Output: cm_plume_tifs/sources.parquet (plume_id, source_name, src_lon, src_lat)

8 threads with retries; resumable (re-reads what is already resolved if it
exists).
Usage:
  nohup /data/users/julio/.conda/envs/deep/bin/python code/v2/15b_cm_sources.py \
        > code/v2/15b_cm_sources.log 2>&1 &
"""
import pathlib
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests

OUT_DIR = pathlib.Path("/data/databases/METHANSET_TACOS/cm_plume_tifs")
API = "https://api.carbonmapper.org/api/v1/catalog/source/plume/name/{}"
WORKERS = 8


def fetch(pid, ses):
    for attempt in range(3):
        try:
            r = ses.get(API.format(pid), timeout=60)
            if r.status_code == 404:
                return pid, None, None, None, ""
            r.raise_for_status()
            x = r.json()
            pt = (x.get("point") or {}).get("coordinates", [None, None])
            return pid, x.get("source_name"), pt[0], pt[1], ""
        except Exception as e:
            time.sleep(3 * (attempt + 1))
            err = f"{type(e).__name__}: {e}"
    return pid, None, None, None, err


def main():
    man = pd.read_parquet(OUT_DIR / "manifest.parquet")
    done = {}
    prev = OUT_DIR / "sources.parquet"
    if prev.exists():
        d = pd.read_parquet(prev)
        done = {r.plume_id for r in d[d.source_name.notna()].itertuples()}
        rows = d[d.source_name.notna()].to_dict("records")
    else:
        rows = []
    todo = [p for p in man.plume_id.unique() if p not in done]
    print(f"plumes: {len(man.plume_id.unique())} · previously resolved: {len(done)} · "
          f"to query: {len(todo)}", flush=True)

    ses = requests.Session()
    t0, errs = time.time(), 0
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = [ex.submit(fetch, p, ses) for p in todo]
        for i, fut in enumerate(as_completed(futs), 1):
            pid, name, lo, la, err = fut.result()
            if err:
                errs += 1
                print(f"  ERROR {pid}: {err}", flush=True)
            rows.append({"plume_id": pid, "source_name": name,
                         "src_lon": lo, "src_lat": la})
            if i % 250 == 0 or i == len(futs):
                print(f"  {i}/{len(todo)}  ({i/(time.time()-t0):.1f}/s)", flush=True)

    df = pd.DataFrame(rows)
    df.to_parquet(prev, index=False)
    sin = df.source_name.isna().sum()
    print(f"\nresolved {df.source_name.notna().sum():,} · without source {sin} · "
          f"errors {errs} -> {prev}")


if __name__ == "__main__":
    main()
