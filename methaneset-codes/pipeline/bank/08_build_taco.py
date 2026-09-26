"""Step 8 · Packages the bank into a TACO and generates its README.md. Only after Julio approves the table.

Runs with tacotoolbox 0.26.x (Sample / Tortilla / Taco API), the one that writes a header that
tacoreader 2.x reads; the tacotoolbox 0.5.0-beta of the deep environment writes one that the current
reader rejects. For now: the Python of César's majortom environment.

Usage
  python 08_build_taco.py RAW --prueba 300 --split 3MB --salida DIR   a few rows, to check
  python 08_build_taco.py RAW                                          the whole bank, in config.FINAL

Each row of tabla.parquet is a Sample: its GeoTIFF and its columns as metadata, with the
English description of each one (core/schema.py). The description of the collection is
DATASET_README.md without its column table: that table is rendered by the generated README from the
descriptions. The constant values (3000 kg/h, ppb, 20 m, nadir view weight) go in that text.
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
from banco import config as C
from banco.esquema import COLUMNAS, DESCRIPCIONES

AQUI = Path(__file__).parent


class Fila(SampleExtension):
    """The metadata of one table row, with its schema and its descriptions."""
    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)
    datos: dict[str, Any]
    esquema: Any

    def get_schema(self) -> pa.Schema:
        return self.esquema

    def get_field_descriptions(self) -> dict[str, str]:
        return DESCRIPCIONES

    def _compute(self, sample) -> pa.Table:
        return pa.Table.from_pylist([self.datos], schema=self.esquema)


def descripcion() -> str:
    """DATASET_README.md without the title (the generated README sets it) and without the columns section."""
    txt = (AQUI / "DATASET_README.md").read_text(encoding="utf-8")
    txt = re.sub(r"\A# .*\n+", "", txt)
    txt = re.sub(r"## Columns\n.*?(?=\n## )", "", txt, flags=re.S)
    return txt.strip() + "\n\n"


def limpio(v):
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.floating):
        return None if math.isnan(v) else float(v)
    return v


def ordenar_tacocat(salida: Path) -> None:
    """tacotoolbox sorts the columns alphabetically when consolidating .tacocat/ (_column_utils.py);
    inside each zip and in COLLECTION.json the schema order is preserved. The consolidated parquet is
    rewritten with id, type, the schema columns and the internal ones at the end."""
    import pyarrow.parquet as pq
    f = salida.parent / ".tacocat" / "level0.parquet"
    if not f.exists():
        return
    tab = pq.read_table(f)
    orden = ["id", "type"] + COLUMNAS[2:]
    orden += [c for c in tab.column_names if c not in orden]
    assert set(orden) == set(tab.column_names)
    pq.write_table(tab.select(orden), f, compression="zstd")


def readme(salida: Path, zips: list[Path]) -> Path:
    """README.md next to the TACO, from its COLLECTION.json."""
    cat = salida.parent / ".tacocat" / "COLLECTION.json"
    if cat.exists():
        coll = cat
    else:
        with zipfile.ZipFile(zips[0]) as z:
            nombre = next(n for n in z.namelist() if n.endswith("COLLECTION.json"))
            coll = Path(tempfile.mkdtemp()) / "COLLECTION.json"
            coll.write_bytes(z.read(nombre))
    faltan = [c for c in COLUMNAS[2:] if c not in json.dumps(json.loads(coll.read_text()))]
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
    ap.add_argument("--salida", type=Path, default=C.FINAL / "methaneset-bank.tacozip")
    ap.add_argument("--prueba", type=int, default=0)
    ap.add_argument("--split", default="4GB")
    a = ap.parse_args()
    t = pd.read_parquet(a.raw / "tabla.parquet")
    assert list(t.columns) == COLUMNAS, "the table does not follow core/schema.py"
    assert set(DESCRIPCIONES) == set(COLUMNAS[2:]), "descriptions missing or extra"
    if a.prueba:
        t = t.sample(a.prueba, random_state=0)
    extras = COLUMNAS[2:]
    esquema = pa.Schema.from_pandas(t[extras], preserve_index=False)
    muestras = []
    for vals in t.itertuples(index=False, name=None):
        d = dict(zip(t.columns, vals))
        s = Sample(id=d["id"], path=(a.raw / d["path"]).resolve())
        s.extend_with(Fila(datos={c: limpio(d[c]) for c in extras}, esquema=esquema))
        muestras.append(s)
    taco = Taco(
        tortilla=Tortilla(muestras),
        id="methaneset_bank",
        title="MethaneSET bank",
        dataset_version="1.0.0",
        description=descripcion(),
        licenses=["CC-BY-4.0"],
        providers=[{"name": "Image and Signal Processing Group (ISP-UV)", "roles": ["producer"]},
                   {"name": "LARS-UPV, Gorroño et al. (WRF-LES simulations)", "roles": ["licensor"]}],
        tasks=["other"],
        keywords=["methane", "plume", "LES", "WRF", "XCH4", "hyperspectral", "EMIT", "synthetic"],
    )
    a.salida.parent.mkdir(parents=True, exist_ok=True)
    zips = tacotoolbox.create(taco, a.salida, output_format="zip", split_size=a.split, consolidate=True)
    print("written:", *zips, sep="\n  ")
    ordenar_tacocat(a.salida)
    print("README:", readme(a.salida, zips))


if __name__ == "__main__":
    main()
