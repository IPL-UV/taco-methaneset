"""Build the browser-side plume index from the published TACO metadata.

Reads the level0 Parquet indexes of the three spatially referenced datasets
(EMIT and the two multispectral finetune sets) from Hugging Face and writes a
single GeoJSON with one point per observed plume. The plume bank and bank-les
have no geographic reference, so they are intentionally excluded.

    python scripts/build_plume_index.py
"""

from __future__ import annotations

import json
import pathlib
import urllib.request

import pandas as pd

HF = "https://huggingface.co/datasets/tacofoundation/methaneset/resolve/main"
CACHE = pathlib.Path("/tmp/methaneset_index")
OUT = pathlib.Path("public/data/plumes.geojson")

SOURCES = [
    # dataset, sensor label, metadata path inside the dataset
    ("methaneset-emit", "EMIT", "methaneset-emit/METADATA/level0.parquet"),
    ("methaneset-s2-finetune", "Sentinel-2", "methaneset-s2-finetune/.tacocat/level0.parquet"),
    ("methaneset-l89-finetune", "Landsat 8/9", "methaneset-l89-finetune/.tacocat/level0.parquet"),
]


def num(value):
    if value is None or pd.isna(value):
        return None
    return float(value)


def fetch(path: str) -> pd.DataFrame:
    CACHE.mkdir(parents=True, exist_ok=True)
    local = CACHE / path.replace("/", "__")
    if not local.exists():
        urllib.request.urlretrieve(f"{HF}/{path}", local)
    return pd.read_parquet(local)


def emit_points(df: pd.DataFrame, sensor: str) -> list[dict]:
    features = []
    for _, row in df.iterrows():
        west, east = num(row["spatial:bbox_west"]), num(row["spatial:bbox_east"])
        south, north = num(row["spatial:bbox_south"]), num(row["spatial:bbox_north"])
        if None in (west, east, south, north):
            continue
        fluxes = [num(row.get("detection:imeo_flux_max")), num(row.get("detection:cm_flux_max"))]
        fluxes = [f for f in fluxes if f is not None]
        features.append(
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [(west + east) / 2, (south + north) / 2]},
                "properties": {
                    "dataset": "methaneset-emit",
                    "sensor": sensor,
                    "country": row.get("site:country"),
                    "date": str(row.get("emit:time_start", ""))[:10],
                    "flux": max(fluxes) if fluxes else None,
                    "flux_kind": "max",
                    "n_imeo": int(row["detection:n_imeo"]),
                    "n_cm": int(row["detection:n_cm"]),
                    "sector": row.get("detection:sector"),
                },
            }
        )
    return features


def multispectral_points(df: pd.DataFrame, sensor: str, dataset: str) -> list[dict]:
    features = []
    for _, row in df.iterrows():
        lat, lon = num(row.get("emission:lat")), num(row.get("emission:lon"))
        if lat is None or lon is None:
            continue
        country = row.get("site:country") or ""
        features.append(
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [lon, lat]},
                "properties": {
                    "dataset": dataset,
                    "sensor": sensor,
                    "country": country,
                    "date": str(row.get("stac:time_start", ""))[:10],
                    "flux": num(row.get("detection:ch4_fluxrate")),
                    "flux_kind": "ch4",
                    "id": str(row.get("id", "")),
                    "file": f"{dataset}_{country.replace(' ', '_')}.tacozip",
                    "viz": "multispectral",
                },
            }
        )
    return features


def main() -> None:
    features: list[dict] = []
    for dataset, sensor, path in SOURCES:
        df = fetch(path)
        if dataset == "methaneset-emit":
            features += emit_points(df, sensor)
        else:
            features += multispectral_points(df, sensor, dataset)
        print(f"{dataset}: {len(df)} rows")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"type": "FeatureCollection", "features": features}))
    print(f"total: {len(features)} points -> {OUT}")


if __name__ == "__main__":
    main()
