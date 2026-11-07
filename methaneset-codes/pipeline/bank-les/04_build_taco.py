"""Step 4 · Packages the cubes into a TACO and generates its README.md (in English).

Runs with tacotoolbox 0.26.x (majortom environment), the same API as the projected bank.

Usage
  python 04_build_taco.py RAW --prueba 300 --split 3MB --salida DIR   a few rows, to check
  python 04_build_taco.py RAW                                          the whole bank, in core.config.FINAL

Each row of tabla.parquet is a Sample: its 49-band GeoTIFF and its columns as metadata, with the
English description of each one (core/schema.py). The description of the collection is
DATASET_README.md without its column table.
"""
from __future__ import annotations
import argparse
import json
import math
import re
import tempfile
import zipfile
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
import pyarrow as pa
import pydantic
import tacotoolbox
from tacotoolbox.datamodel import Sample, Tortilla, Taco
from tacotoolbox.sample.datamodel import SampleExtension
from core import config as C
from core.schema import COLUMNS, DESCRIPTIONS

AQUI = Path(__file__).parent


class Row(SampleExtension):
    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)
    data: dict[str, Any]
    schema: Any

    def get_schema(self) -> pa.Schema:
        return self.schema

    def get_field_descriptions(self) -> dict[str, str]:
        return DESCRIPTIONS

    def _compute(self, sample) -> pa.Table:
        return pa.Table.from_pylist([self.data], schema=self.schema)


def description() -> str:
    """DATASET_README.md without the title or the columns section (the generated README renders it)."""
    txt = (AQUI / "DATASET_README.md").read_text(encoding="utf-8")
    txt = re.sub(r"\A# .*\n+", "", txt)
    txt = re.sub(r"## Columns\n.*?(?=\n## )", "", txt, flags=re.S)
    return txt.strip() + "\n\n"


def clean(v):
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.floating):
        return None if math.isnan(v) else float(v)
    if isinstance(v, np.bool_):
        return bool(v)
    if isinstance(v, np.ndarray):
        return v.tolist()
    return v


def order_tacocat(salida: Path) -> None:
    import pyarrow.parquet as pq
    f = salida.parent / ".tacocat" / "level0.parquet"
    if not f.exists():
        return
    tab = pq.read_table(f)
    orden = ["id", "type"] + COLUMNS[2:]
    orden += [c for c in tab.column_names if c not in orden]
    assert set(orden) == set(tab.column_names)
    pq.write_table(tab.select(orden), f, compression="zstd")


def readme(salida: Path, zips: list[Path]) -> Path:
    cat = salida.parent / ".tacocat" / "COLLECTION.json"
    if cat.exists():
        coll = cat
    else:
        with zipfile.ZipFile(zips[0]) as z:
            nombre = next(n for n in z.namelist() if n.endswith("COLLECTION.json"))
            coll = Path(tempfile.mkdtemp()) / "COLLECTION.json"
            coll.write_bytes(z.read(nombre))
    faltan = [c for c in COLUMNS[2:] if c not in json.dumps(json.loads(coll.read_text()))]
    assert not faltan, f"columns missing from COLLECTION.json: {faltan}"
    ejemplo = str(salida.parent / ".tacocat") if cat.exists() else str(zips[0])
    destino = salida.parent / "README.md"
    tacotoolbox.generate_markdown(coll, destino, dataset_example_path=ejemplo)
    tacotoolbox.generate_html(coll, salida.parent / "index.html", dataset_example_path=ejemplo,
                              catalogue_url=None)
    return destino


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("raw", type=Path)
    ap.add_argument("--salida", type=Path, default=C.FINAL / "methaneset-bank-les.tacozip")
    ap.add_argument("--prueba", type=int, default=0)
    ap.add_argument("--split", default="4GB")
    a = ap.parse_args()
    t = pd.read_parquet(a.raw / "tabla.parquet")
    assert list(t.columns) == COLUMNS, "the table does not follow core/schema.py"
    assert set(DESCRIPTIONS) == set(COLUMNS[2:]), "descriptions missing or extra"
    if a.prueba:
        t = t.sample(a.prueba, random_state=0)
    extras = COLUMNS[2:]
    schema = pa.Schema.from_pandas(t[extras], preserve_index=False)
    muestras = []
    for vals in t.itertuples(index=False, name=None):
        d = dict(zip(t.columns, vals))
        s = Sample(id=d["id"], path=(a.raw / d["path"]).resolve())
        s.extend_with(Row(data={c: clean(d[c]) for c in extras}, schema=schema))
        muestras.append(s)
    taco = Taco(
        tortilla=Tortilla(muestras),
        id="methaneset_bank_les",
        title="MethaneSET bank LES",
        dataset_version=C.VERSION,
        description=description(),
        licenses=["CC-BY-4.0"],
        providers=[{"name": "Image and Signal Processing Group (ISP-UV)", "roles": ["producer"]},
                   {"name": "LARS-UPV, Gorroño et al. (WRF-LES simulations)", "roles": ["licensor"]}],
        tasks=["other"],
        keywords=["methane", "plume", "LES", "WRF", "XCH4", "cube", "3D", "synthetic"],
    )
    a.salida.parent.mkdir(parents=True, exist_ok=True)
    zips = tacotoolbox.create(taco, a.salida, output_format="zip", split_size=a.split, consolidate=True)
    print("written:", *zips, sep="\n  ")
    # tacotoolbox only consolidates when there are several zips; with just one we create it here, so
    # that tacoreader.load(".../methaneset-bank-les/") works the same as with the projected bank
    if not (a.salida.parent / ".tacocat").exists():
        from tacotoolbox.tacocat import create_tacocat
        create_tacocat(inputs=list(zips), output=a.salida.parent, validate_schema=True)
        print("consolidated in", a.salida.parent / ".tacocat")
    order_tacocat(a.salida)
    print("README:", readme(a.salida, zips))


if __name__ == "__main__":
    main()
