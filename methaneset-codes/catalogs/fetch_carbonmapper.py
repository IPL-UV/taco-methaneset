"""Downloads the Carbon Mapper public plume catalog to a dated parquet.

Important notes found while using it (22 Aug 2026):
  - The endpoint answers WITHOUT a token. Sending an expired token returns 401,
    sending none returns 200. Do not use Authorization.
  - The `gas` parameter IS IGNORED: asking for gas=CH4 returns the whole catalog.
    You have to split by the `gas` field of each record.
  - `instrument` does filter. Valid values: GAO, ang, av3, emi, tan, ssc.

Usage:  python fetch_carbonmapper.py
Output: assets/data/carbonmapper/<YYYY-MM-DD>/plumes.parquet
"""
import datetime
import pathlib
import time

import pandas as pd
import requests

API = "https://api.carbonmapper.org/api/v1/catalog/plumes/annotated"
LIMIT = 1000
KEEP = ["plume_id", "gas", "instrument", "platform", "scene_timestamp",
        "emission_auto", "emission_uncertainty_auto", "sector", "status",
        "plume_bounds", "published_at",
        # PER-PLUME wind published by CM: it is the one THEY used to
        # invert emission_auto (speed + direction, not components)
        "wind_speed_avg_auto", "wind_direction_avg_auto"]
OUT = pathlib.Path(__file__).resolve().parent.parent / "assets" / "data" / "carbonmapper"


def centroid(bounds):
    """plume_bounds comes as [west, south, east, north]."""
    if not bounds or len(bounds) != 4:
        return None, None
    return (bounds[0] + bounds[2]) / 2, (bounds[1] + bounds[3]) / 2


def main():
    rows, offset, t0 = [], 0, time.time()
    while True:
        r = requests.get(API, params={"limit": LIMIT, "offset": offset}, timeout=180)
        r.raise_for_status()
        items = r.json()["items"]
        if not items:
            break
        for x in items:
            rec = {k: x.get(k) for k in KEEP}
            rec["lon"], rec["lat"] = centroid(x.get("plume_bounds"))
            rows.append(rec)
        offset += len(items)
        if len(items) < LIMIT:
            break
    df = pd.DataFrame(rows).drop(columns=["plume_bounds"])
    day = datetime.date.today().isoformat()
    dest = OUT / day
    dest.mkdir(parents=True, exist_ok=True)
    df.to_parquet(dest / "plumes.parquet", index=False)
    print(f"{len(df)} plumes in {time.time()-t0:.0f}s")
    print(df["gas"].value_counts().to_dict())
    print("saved to", dest / "plumes.parquet")


if __name__ == "__main__":
    main()
