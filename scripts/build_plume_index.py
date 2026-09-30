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
import re
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


POINT_RE = re.compile(r"-?\d+\.?\d*(?:[eE][-+]?\d+)?")


def wkt_points(value) -> list[tuple[float, float]]:
    if not isinstance(value, str):
        return []
    nums = [float(x) for x in POINT_RE.findall(value)]
    return list(zip(nums[0::2], nums[1::2]))


def as_list(value) -> list:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return []
    return list(value) if isinstance(value, (list, tuple)) else []


def as_dict(value) -> dict:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return {}
    return dict(value) if isinstance(value, dict) else {}


def fetch(path: str) -> pd.DataFrame:
    CACHE.mkdir(parents=True, exist_ok=True)
    local = CACHE / path.replace("/", "__")
    if not local.exists():
        urllib.request.urlretrieve(f"{HF}/{path}", local)
    return pd.read_parquet(local)


def emit_points(df: pd.DataFrame, sensor: str) -> list[dict]:
    features = []
    for _, row in df.iterrows():
        matches = [
            (
                "IMEO",
                as_list(row.get("detection:imeo_ids")),
                wkt_points(row.get("spatial:imeo_points")),
                as_dict(row.get("detection:imeo_flux")),
            ),
            (
                "Carbon Mapper",
                as_list(row.get("detection:cm_ids")),
                wkt_points(row.get("spatial:cm_points")),
                as_dict(row.get("detection:cm_flux")),
            ),
        ]
        for system, ids, points, fluxes in matches:
            seen = set()
            for i, (lon, lat) in enumerate(points):
                source = ids[i] if i < len(ids) else ""
                key = source or (lon, lat)
                if key in seen:
                    continue
                seen.add(key)
                features.append(
                    {
                        "type": "Feature",
                        "geometry": {"type": "Point", "coordinates": [lon, lat]},
                        "properties": {
                            "dataset": "methaneset-emit",
                            "sensor": sensor,
                            "system": system,
                            "country": row.get("site:country"),
                            "date": str(row.get("emit:time_start", ""))[:10],
                            "flux": num(fluxes.get(source)),
                            "flux_kind": "max",
                            "id": f"{row['id']}:{source}" if source else str(row["id"]),
                            "source": source,
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
