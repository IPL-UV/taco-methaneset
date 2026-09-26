"""Step 5 · Builds the finetune TACO: Cesar's leaves (target, bg0, ch4, plume, dem) + bg1/2/3 + table.

Reuses Cesar's rasters as-is (extracts them from the tacozip) and adds our bg1/2/3.
Leaves per sample (fixed ids): target, bg0, bg1, bg2, bg3, ch4, plume, dem.
The table is tabla_propuesta.csv. Runs with the majortom environment.

Usage: python 05_build_taco.py --salida DIR [--limit N]
"""
from __future__ import annotations
import argparse
import shutil
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pydantic
from osgeo import gdal
import tacotoolbox
from tacotoolbox.datamodel import Sample, Tortilla, Taco
from tacotoolbox.sample.datamodel import SampleExtension
from tacotoolbox.sample.extensions.tacotiff import Header
from tacotoolbox.sample.extensions.geotiff_stats import GeotiffStats
from taco_descriptions import DESCRIPCIONES, DESCRIPCION

gdal.UseExceptions()
COG = ["COMPRESS=ZSTD", "PREDICTOR=2", "BLOCKSIZE=224", "BIGTIFF=IF_SAFER", "OVERVIEWS=NONE",
       "INTERLEAVE=TILE"]

RAW = Path("/data/databases/METHANESET_TACOS/methaneset-l89-finetune-bg-raw")
TABLA = RAW / "tabla_propuesta.csv"
PATHS = RAW / "paths_cesar.csv"
STAGE = RAW / "stage"
HOJAS = [("target", "cesar:target"), ("bg0", "cesar:reference"), ("bg1", "bg1"),
         ("bg2", "bg2"), ("bg3", "bg3"), ("ch4", "cesar:ch4"), ("plume", "cesar:plume"),
         ("dem", "cesar:dem")]


class Fila(SampleExtension):
    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)
    datos: dict[str, Any]
    esquema: Any

    def get_schema(self) -> pa.Schema:
        return self.esquema

    def get_field_descriptions(self) -> dict[str, str]:
        return {c: DESCRIPCIONES.get(c, c) for c in self.datos}

    def _compute(self, sample) -> pa.Table:
        return pa.Table.from_pylist([self.datos], schema=self.esquema)


ORDEN = ["id","type","target:sensor","site:country","site:location_name","split",
 "detection:case_study","detection:sector","detection:ch4_fluxrate","detection:ch4_fluxrate_std","detection:wind_source","detection:offshore",
 "plume:geometry","emission:lat","emission:lon",
 "target:tile","target:sza","target:vza","target:cloud","stac:crs","stac:geotransform","stac:tensor_shape","stac:time_start",
 "meteo:wind_u","meteo:wind_v",
 "geoenrich:admin_countries","geoenrich:admin_states","geoenrich:admin_districts","geoenrich:elevation","geoenrich:population","geoenrich:temperature",
 "quality:percentage_clear","quality:observability","quality:notified","quality:last_update","majortom:code",
 "bg:sensor0","bg:tile0","bg:date0","bg:cloud0",
 "bg:sensor1","bg:date1","bg:cloud1","bg:difference1","bg:methane1",
 "bg:sensor2","bg:date2","bg:cloud2","bg:difference2","bg:methane2",
 "bg:sensor3","bg:date3","bg:cloud3","bg:difference3","bg:methane3"]


def ordenar_tacocat(salida: Path) -> None:
    import pyarrow.parquet as pq
    f = Path(salida).parent / ".tacocat" / "level0.parquet"
    if not f.exists():
        return
    tab = pq.read_table(f)
    internos = [c for c in tab.column_names if c.startswith("internal:")]
    orden = [c for c in ORDEN if c in tab.column_names] + internos
    pq.write_table(tab.select(orden), f, compression="zstd")
    print("tacocat reordered")


def a_cog(src: str, dst: Path) -> None:
    if dst.exists():
        return
    gdal.Translate(str(dst), src, format="COG", creationOptions=COG)


def build_sample(row, paths, esquema) -> Sample:
    d = STAGE / row.id
    d.mkdir(parents=True, exist_ok=True)
    leaves = []
    for hoja, col in HOJAS:
        src = getattr(paths, col) if col in paths._fields else None
        if src is None or (isinstance(src, float) and pd.isna(src)):
            continue
        f = d / f"{hoja}.tif"
        a_cog(str(src), f)
        s = Sample(id=hoja, path=str(f))
        s.extend_with(Header()); s.extend_with(GeotiffStats())
        leaves.append(s)
    child = Tortilla(samples=leaves, strict_schema=True)
    sample = Sample(id=row.id, path=child)
    datos = {k: (None if (isinstance(v, float) and pd.isna(v)) else v) for k, v in row._asdict().items() if k != "Index"}
    import numpy as np
    datos = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in datos.items()}
    sample.extend_with(Fila(datos=datos, esquema=esquema))
    return sample


def main():
    global RAW, TABLA, PATHS, STAGE
    ap = argparse.ArgumentParser()
    ap.add_argument("--salida", required=True)
    ap.add_argument("--raw", default=str(RAW))
    ap.add_argument("--id", default="methaneset-l89-finetune")
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()

    RAW = Path(a.raw); TABLA = RAW / "tabla_propuesta.csv"; PATHS = RAW / "paths_cesar.csv"; STAGE = RAW / "stage"

    t = pd.read_csv(TABLA)
    t = t.where(pd.notna(t), None)
    # the TACO requires a fixed structure: only sites with all 3 backgrounds
    n0 = len(t)
    t = t[t["bg:date1"].notna() & t["bg:date2"].notna() & t["bg:date3"].notna()].reset_index(drop=True)
    print(f"sites with all 3 backgrounds: {len(t)} of {n0}")
    paths = pd.read_csv(PATHS)
    if a.limit:
        t = t.head(a.limit); paths = paths.head(a.limit)
    pmap = paths.set_index("id")
    extras = [c for c in t.columns if c != "id"]
    esquema = pa.Schema.from_pandas(t[extras], preserve_index=False)

    muestras, fallos = [], 0
    for idx in range(len(t)):
        row = t.iloc[idx]
        try:
            pathrow = pmap.loc[row["id"]]
            d = STAGE / row["id"]; d.mkdir(parents=True, exist_ok=True)
            leaves = []
            for hoja, col in HOJAS:
                src = pathrow[col]
                if not isinstance(src, str):
                    continue
                f = d / f"{hoja}.tif"
                a_cog(src, f)
                s = Sample(id=hoja, path=str(f)); s.extend_with(Header()); s.extend_with(GeotiffStats())
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
    taco = Taco(tortilla=root, id=a.id, title=a.id,
                dataset_version="1.0.0", licenses=["CC-BY-4.0"],
                description=DESCRIPCION,
                tasks=["detection"], keywords=["methane", "landsat", "background", "plume", "taco"],
                providers=[{"name": "Image and Signal Processing Group (ISP-UV)", "roles": ["producer"]},
                           {"name": "UNEP IMEO, MARS (background pairs)", "roles": ["licensor"]}])
    print("writing TACO to", a.salida)
    out = Path(a.salida) / f"{a.id}.tacozip"
    zips = tacotoolbox.create(taco, out, output_format="zip", split_size="4GB",
                              group_by="site:country", consolidate=True)
    print("created:", *zips, sep="\n  ")
    ordenar_tacocat(out)
    import tacotoolbox as tt
    coll = Path(a.salida) / ".tacocat" / "COLLECTION.json"
    try:
        tt.generate_markdown(input=coll, output=Path(a.salida) / "README.md",
                             dataset_example_path=str(Path(a.salida) / ".tacocat"))
        tt.generate_html(input=coll, output=Path(a.salida) / "index.html",
                         dataset_example_path=str(Path(a.salida) / ".tacocat"), catalogue_url=None)
        print("README and index generated")
    except FileNotFoundError:
        print("no .tacocat on disk (new package version): skipping README/index")


if __name__ == "__main__":
    main()
