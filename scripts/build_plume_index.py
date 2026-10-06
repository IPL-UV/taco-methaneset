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
from collections import Counter

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
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return []
    return list(value) if isinstance(value, (list, tuple)) else []


def as_dict(value) -> dict:
    if hasattr(value, "tolist"):
        value = value.tolist()
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


def bit_map(row: pd.Series, column: str) -> dict[str, int]:
    bits = as_dict(row.get(column))
    out = {}
    for bit, plume in bits.items():
        try:
            out[str(plume)] = int(bit)
        except (TypeError, ValueError):
            raise SystemExit(f"bad bit {bit!r} in {column} of {row.get('id')}")
    return out


def emit_points(df: pd.DataFrame, sensor: str) -> list[dict]:
    features = []
    for _, row in df.iterrows():
        bit_maps = {
            "IMEO": bit_map(row, "detection:imeo_bits"),
            "Carbon Mapper": bit_map(row, "detection:cm_bits"),
        }
        pair_map = {}
        for pair in as_list(row.get("match:pairs")):
            if not isinstance(pair, dict):
                continue
            imeo = str(pair.get("imeo") or "")
            cm = str(pair.get("cm") or "")
            if not imeo or not cm:
                continue
            for key in (("IMEO", imeo), ("Carbon Mapper", cm)):
                if key in pair_map:
                    raise SystemExit(f"duplicate pair entry for {key} in {row.get('id')}")
                pair_map[key] = ("Carbon Mapper", cm) if key[0] == "IMEO" else ("IMEO", imeo)
        orphans = as_dict(row.get("match:orphans"))
        orphan_imeo = {str(x) for x in as_list(orphans.get("imeo"))}
        orphan_cm = {str(x) for x in as_list(orphans.get("cm"))}
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
                value = ids[i] if i < len(ids) else ""
                source = "" if value is None or pd.isna(value) else str(value)
                key = source or (lon, lat)
                if key in seen:
                    continue
                seen.add(key)
                mate = pair_map.get((system, source))
                if mate:
                    match, pair_source, pair_id = "pair", mate[0], mate[1]
                elif source in (orphan_imeo if system == "IMEO" else orphan_cm):
                    match, pair_source, pair_id = "orphan", None, None
                else:
                    match, pair_source, pair_id = None, None, None
                bit = bit_maps[system].get(source)
                if source and bit is None:
                    raise SystemExit(f"plume {source} has no bit in {row.get('id')}")
                pair_bit = bit_maps[mate[0]].get(mate[1]) if mate else None
                if mate and pair_bit is None:
                    raise SystemExit(f"pair {mate[1]} has no bit in {row.get('id')}")
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
                            "match": match,
                            "pair_source": pair_source,
                            "pair_id": pair_id,
                            "bit": bit,
                            "pair_bit": pair_bit,
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
            for column in ("match:pairs", "match:orphans", "detection:imeo_bits", "detection:cm_bits"):
                if column not in df.columns:
                    raise SystemExit(
                        f"{path} has no {column}: it is a stale cache, remove "
                        f"{CACHE}/methaneset-emit__METADATA__level0.parquet and rerun"
                    )
            points = emit_points(df, sensor)
            matches = Counter(p["properties"].get("match") for p in points)
            if not matches["pair"] and not matches["orphan"]:
                raise SystemExit(f"{path}: no pair or orphan features, match metadata not read")
            print(f"{dataset} match: {dict(matches)}")
            features += points
        else:
            features += multispectral_points(df, sensor, dataset)
        print(f"{dataset}: {len(df)} rows")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"type": "FeatureCollection", "features": features}))
    print(f"total: {len(features)} points -> {OUT}")


if __name__ == "__main__":
    main()
