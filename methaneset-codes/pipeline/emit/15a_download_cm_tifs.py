"""MODULE 15a of the v2 pipeline: download the plume_tif files from Carbon Mapper.

The CM API does not deliver polygons (geometry_json is a point): the mask
lives in the alpha band of each plume_tif. This module downloads the tifs of
the CH4/EMIT plumes whose granules are in our 725 split.

Key detail: the plume_tif URLs come PRE-SIGNED and expire, so each tif is
downloaded as soon as its API page delivers it (the URL list cannot be saved
for later).

Output:
  /data/databases/METHANSET_TACOS/cm_plume_tifs/<plume_id>.tif
  /data/databases/METHANSET_TACOS/cm_plume_tifs/manifest.parquet
    (plume_id, granule_ts, gas, sector, emission_auto, lon, lat, scene_id)

Resumable (skips tifs already downloaded). Usage:
  nohup /data/users/julio/.conda/envs/deep/bin/python code/v2/15a_download_cm_tifs.py \
        > code/v2/15a_download_cm_tifs.log 2>&1 &
"""
import pathlib
import re
import time

import pandas as pd
import requests

DATA = pathlib.Path(__file__).resolve().parent.parent.parent / "assets" / "data"
OUT = pathlib.Path("/data/databases/METHANSET_TACOS/cm_plume_tifs")
API = "https://api.carbonmapper.org/api/v1/catalog/plumes/annotated"
TS_CM = re.compile(r"^emi(\d{8}t\d{6})")
LIMIT = 500


def main():
    cross_dir = sorted((DATA / "cross").iterdir())[-1]
    wanted = set(pd.read_parquet(cross_dir / "splits.parquet").granule_ts)
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"target granules: {len(wanted)}", flush=True)

    rows, offset, dl, skip, t0 = [], 0, 0, 0, time.time()
    ses = requests.Session()
    while True:
        r = ses.get(API, params={"limit": LIMIT, "offset": offset,
                                 "instrument": "emi"}, timeout=180)
        r.raise_for_status()
        items = r.json()["items"]
        if not items:
            break
        for x in items:
            m = TS_CM.match(x["plume_id"] or "")
            if not m or m.group(1) not in wanted or x.get("gas") != "CH4":
                continue
            geo = x.get("geometry_json") or {}
            lonlat = geo.get("coordinates", [None, None])
            rows.append({"plume_id": x["plume_id"], "granule_ts": m.group(1),
                         "gas": x["gas"], "sector": x.get("sector"),
                         "emission_auto": x.get("emission_auto"),
                         "lon": lonlat[0], "lat": lonlat[1],
                         "scene_id": x.get("scene_id")})
            url = x.get("plume_tif")
            if not url:
                continue
            fp = OUT / f"{x['plume_id']}.tif"
            if fp.exists() and fp.stat().st_size > 0:
                skip += 1
                continue
            try:
                with ses.get(url, stream=True, timeout=120) as rr:
                    rr.raise_for_status()
                    tmp = fp.with_suffix(".tif.part")
                    with open(tmp, "wb") as f:
                        for chunk in rr.iter_content(1 << 18):
                            f.write(chunk)
                    tmp.rename(fp)
                dl += 1
            except Exception as e:
                print(f"  ERROR {x['plume_id']}: {type(e).__name__}: {e}",
                      flush=True)
        offset += len(items)
        print(f"  offset {offset} · in target {len(rows)} · downloaded {dl} · "
              f"already there {skip} · {offset/(time.time()-t0):.0f} items/s",
              flush=True)
        if len(items) < LIMIT:
            break

    pd.DataFrame(rows).to_parquet(OUT / "manifest.parquet", index=False)
    print(f"\nmanifest: {len(rows)} plumes · downloaded {dl} · already there {skip}")
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
