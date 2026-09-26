"""Rebuilds the pretraining TACO: re-zips the extracted rasters with the fixed table.

Leaves per sample: target, reference, dem (the tifs are already COG TILE ZSTD).
Usage: python 05_build_taco.py --ds methaneset-s2-pretraining --salida DIR [--limit N]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pydantic
import tacotoolbox
from tacotoolbox.datamodel import Sample, Taco, Tortilla
from tacotoolbox.sample.datamodel import SampleExtension
from tacotoolbox.sample.extensions.geotiff_stats import GeotiffStats
from tacotoolbox.sample.extensions.tacotiff import Header

WORK = Path("/data/databases/_pre_work_julio")
HOJAS = [("target", "target"), ("reference", "reference"), ("dem", "dem")]

ORDEN = [
    "id", "target:sensor", "site:country", "site:location_name", "split",
    "detection:case_study", "detection:sector", "detection:isplume",
    "detection:ch4_fluxrate", "detection:ch4_fluxrate_std", "detection:wind_source",
    "detection:offshore",
    "target:tile", "target:sza", "target:vza",
    "stac:crs", "stac:geotransform", "stac:tensor_shape", "stac:time_start",
    "meteo:wind_u", "meteo:wind_v",
    "geoenrich:admin_countries", "geoenrich:admin_states", "geoenrich:admin_districts",
    "geoenrich:elevation", "geoenrich:population", "geoenrich:temperature",
    "quality:percentage_clear", "quality:observability", "quality:notified",
    "quality:last_update", "majortom:code", "bg:tile0",
]


class Fila(SampleExtension):
    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)
    datos: dict[str, Any]
    esquema: Any

    def get_schema(self) -> pa.Schema:
        return self.esquema

    def get_field_descriptions(self) -> dict[str, str]:
        return {c: c for c in self.datos}

    def _compute(self, sample) -> pa.Table:
        return pa.Table.from_pylist([self.datos], schema=self.esquema)


def ordenar_tacocat(salida: Path) -> None:
    import pyarrow.parquet as pq

    f = Path(salida).parent / ".tacocat" / "level0.parquet"
    if not f.exists():
        return
    tab = pq.read_table(f)
    internos = [c for c in tab.column_names if c.startswith("internal:")]
    orden = (
        [c for c in ORDEN if c in tab.column_names]
        + [c for c in tab.column_names if c not in ORDEN and not c.startswith("internal:")]
        + internos
    )
    pq.write_table(tab.select(orden), f, compression="zstd")
    print("tacocat reordered")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", required=True)
    ap.add_argument("--salida", required=True)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()

    sys.path.insert(0, f"/tmp/opencode/pre/{a.ds}")
    from dataset.config import (
        COLLECTION_DESCRIPTION,
        COLLECTION_KEYWORDS,
        COLLECTION_PROVIDERS,
        COLLECTION_TASKS,
        COLLECTION_TITLE,
    )

    raw = WORK / a.ds / "raw"
    t = pd.read_csv(raw / "tabla_propuesta.csv")
    t = t.where(pd.notna(t), None)
    paths = pd.read_csv(raw / "paths_cesar.csv")
    if a.limit:
        t = t.head(a.limit)
        paths = paths[paths["id"].isin(set(t["id"]))]
    pmap = paths.set_index("id")

    extras = [c for c in t.columns if c != "id"]
    esquema = pa.Schema.from_pandas(t[extras], preserve_index=False)

    muestras, fallos = [], 0
    for idx in range(len(t)):
        if idx and idx % 5000 == 0:
            print(f"  {idx}/{len(t)} samples", flush=True)
        row = t.iloc[idx]
        try:
            pathrow = pmap.loc[row["id"]]
            leaves = []
            for hoja, col in HOJAS:
                src = pathrow[col]
                if not isinstance(src, str) or not src:
                    continue
                s = Sample(id=hoja, path=src)
                s.extend_with(Header())
                s.extend_with(GeotiffStats())
                leaves.append(s)
            child = Tortilla(samples=leaves, strict_schema=True)
            samp = Sample(id=row["id"], path=child)
            import numpy as np

            datos = {}
            for k in extras:
                v = row[k]
                if isinstance(v, float) and pd.isna(v):
                    v = None
                if isinstance(v, np.ndarray):
                    v = v.tolist()
                datos[k] = v
            samp.extend_with(Fila(datos=datos, esquema=esquema))
            muestras.append(samp)
        except Exception as ex:  # noqa: BLE001
            fallos += 1
            print(f"[fail] {row['id']}: {type(ex).__name__} {ex}")
    print(f"samples: {len(muestras)} | failed: {fallos}")

    root = Tortilla(samples=muestras, strict_schema=True)
    taco = Taco(
        tortilla=root,
        id=a.ds,
        title=COLLECTION_TITLE,
        dataset_version="1.0.0",
        licenses=["CC-BY-4.0"],
        description=COLLECTION_DESCRIPTION,
        tasks=COLLECTION_TASKS,
        keywords=COLLECTION_KEYWORDS,
        providers=[{"name": p.name, "roles": list(p.roles)} for p in COLLECTION_PROVIDERS],
    )
    out = Path(a.salida) / f"{a.ds}.tacozip"
    print("writing TACO to", a.salida)
    zips = tacotoolbox.create(
        taco, out, output_format="zip", split_size="4GB",
        group_by="site:country", consolidate=True,
    )
    print("created:", *zips, sep="\n  ")
    ordenar_tacocat(out)

    coll = Path(a.salida) / ".tacocat" / "COLLECTION.json"
    try:
        tacotoolbox.generate_markdown(
            input=coll, output=Path(a.salida) / "README.md",
            dataset_example_path=str(Path(a.salida) / ".tacocat"),
        )
        tacotoolbox.generate_html(
            input=coll, output=Path(a.salida) / "index.html",
            dataset_example_path=str(Path(a.salida) / ".tacocat"), catalogue_url=None,
        )
        print("README and index generated")
    except FileNotFoundError:
        print("no .tacocat on disk: skipping README/index")


if __name__ == "__main__":
    main()
