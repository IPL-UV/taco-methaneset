"""Build the browser-side EMIT point index from the published TACO metadata.

Reads METADATA/level0.parquet of methaneset-emit (FOLDER format) on Hugging Face
and writes a small GeoJSON of plume locations. Run from the repo root:

    python scripts/build_emit_index.py
"""

from __future__ import annotations

import json
import pathlib
import urllib.request

import pandas as pd

URL = (
    "https://huggingface.co/datasets/tacofoundation/methaneset/"
    "resolve/main/methaneset-emit/METADATA/level0.parquet"
)
OUT = pathlib.Path("public/data/emit.geojson")


def num(value):
    if value is None or pd.isna(value):
        return None
    return float(value)


def main() -> None:
    cache = pathlib.Path("/tmp/methaneset_emit_level0.parquet")
    if not cache.exists():
        urllib.request.urlretrieve(URL, cache)
    df = pd.read_parquet(cache)

    features = []
    for _, row in df.iterrows():
        west, east = num(row["spatial:bbox_west"]), num(row["spatial:bbox_east"])
        south, north = num(row["spatial:bbox_south"]), num(row["spatial:bbox_north"])
        if None in (west, east, south, north):
            continue
        flux = [num(row.get("detection:imeo_flux_max")), num(row.get("detection:cm_flux_max"))]
        flux = [f for f in flux if f is not None]
        features.append(
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [(west + east) / 2, (south + north) / 2]},
                "properties": {
                    "id": row["id"],
                    "country": row.get("site:country"),
                    "n_imeo": int(row["detection:n_imeo"]),
                    "n_cm": int(row["detection:n_cm"]),
                    "flux_max": max(flux) if flux else None,
                    "sector": row.get("detection:sector"),
                    "split": row.get("selection:split"),
                    "date": str(row.get("emit:time_start", ""))[:10],
                },
            }
        )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"type": "FeatureCollection", "features": features}))
    print(f"{len(features)} granules -> {OUT}")


if __name__ == "__main__":
    main()
