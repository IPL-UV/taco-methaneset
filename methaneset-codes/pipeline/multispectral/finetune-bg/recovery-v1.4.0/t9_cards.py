"""T9 · tarjetas de la v1.4.0 (README de la coleccion e index.html).

El README publicado es la tarjeta que se ve en Hugging Face. Se parte de la version 1.3.0,
se corrigen la prosa y las cifras, y se reescribe COLLECTION.json para que el `.tacocat`
diga lo mismo que la tarjeta. El index.html se regenera solo en la descripcion y las cuentas.

Uso:
  python v140/t9_cards.py --coleccion l89
  python v140/t9_cards.py --coleccion s2
"""
import argparse
import json
import pathlib
import re

import pandas as pd

AQUI = pathlib.Path(__file__).resolve().parent
CARDS = AQUI / "cards"
T7 = pathlib.Path("/data/databases/METHANESET_TACOS/_v140/T7")

REFERENCIAS = {
    "l89": """## References

`bg0` is the single reference that comes with the MARS-S2L pair. `bg1` to `bg3` are three extra
references that we curate for every target with the MARS-S2L rule: query Landsat 8 and 9 together
within ±120 days (before or after) of the target, drop candidates with more than 50 % cloud, cloud
shadow or missing pixels, drop the target's own pass, and keep the three most similar by the MARS
metric on methane-insensitive bands (B2, B3, B4, B6). When ±120 days does not yield 40 candidates
the window widens to ±240 days and, if still thin, to the same day-of-year ±45 days over earlier
years, and adjacent WRS-2 paths (±1) are admitted when the site sits on a scene edge. Of the 4,059
reference leaves that carry a date, 119 lie beyond ±120 days (2.9 %); 659 of the 4,059 references
added before v1.4.0 come from an adjacent path.

All four references are stored as top-of-atmosphere reflectance (Landsat Collection 2 Level-1
`T1_TOA`), in the same scale and dtype as `target`.

`bg:methane1`--`bg:methane3` is the fraction of reference pixels that carry a methane enhancement of
their own. References are ranked to minimise it; it is below 0.03 in 4,012 of the 4,110 reference
leaves (97.6 %), and the column lets you drop the remaining ones if your application needs a
stricter bound.

`bg:difference1`--`bg:difference3` is the MARS score of each reference against the target,
computed on the published TOA leaves. It is not a ranking: many candidates are near-ties (the
median gap between the first and the second reference is 0.0005), so the order of `bg1`--`bg3`
carries no meaning.
""",
    "s2": """## References

`bg0` is the single reference that comes with the MARS-S2L pair. `bg1` to `bg3` are three extra
references that we curate for every target with the MARS-S2L rule: query Sentinel-2 acquisitions
within ±120 days (before or after) of the target, drop candidates with more than 50 % cloud, cloud
shadow or missing pixels, drop the target's own pass, and keep the three most similar by the MARS
metric on methane-insensitive bands (B02, B03, B04, B11). When ±120 days does not yield 40
candidates the window widens to ±240 days and, if still thin, to the same day-of-year ±45 days over
earlier years. The 159 references added in v1.4.0 all lie within ±120 days of their target.

All four references are stored in the same BOA reflectance scale and dtype as `target`.

`bg:methane1`--`bg:methane3` is the fraction of reference pixels that carry a methane enhancement of
their own. References are ranked to minimise it, and it is below 0.03 in all 10,815 reference
leaves.

`bg:difference1`--`bg:difference3` is the MARS score of each reference against the target,
computed on the published leaves. It is not a ranking: many candidates are near-ties, so the order
of `bg1`--`bg3` carries no meaning.
""",
}


def _mascara(desc, coleccion, cifras):
    """Cambia el bloque de References, la version, las particiones y la cobertura."""
    inicio = desc.index("## References")
    fin = desc.index("## Columns")
    desc = desc[:inicio] + REFERENCIAS[coleccion] + "\n" + desc[fin:]
    desc = re.sub(r"\*\*Version\*\*: [0-9.]+", "**Version**: 1.4.0", desc)
    if "**Version**" not in desc:
        desc = desc.replace("## Dataset Information\n", "## Dataset Information\n\n**Version**: 1.4.0\n", 1)
    desc = re.sub(r"\*\*Partitions\*\*: \d+ files",
                  f"**Partitions**: {cifras['ficheros']} files", desc)
    desc = re.sub(r"\*\*Spatial coverage\*\*: \[[^\]]*\] \(WGS84\)",
                  f"**Spatial coverage**: [{cifras['bbox']}] (WGS84)", desc)
    desc = re.sub(r"\*\*Root\*\*: FOLDER \([\d,]+ samples\)",
                  f"**Root**: FOLDER ({cifras['muestras']:,} samples)", desc)
    desc = re.sub(r"- Level 1: ([^(]+) \([\d,]+ samples\)",
                  lambda m: f"- Level 1: {m.group(1)} ({cifras['hojas']:,} samples)", desc)
    return desc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coleccion", required=True, choices=["l89", "s2"])
    a = ap.parse_args()

    carpeta = T7 / a.coleccion
    l0 = pd.read_parquet(carpeta / ".tacocat" / "level0.parquet")
    l1 = pd.read_parquet(carpeta / ".tacocat" / "level1.parquet")

    lat = pd.to_numeric(l0["emission:lat"], errors="coerce")
    lon = pd.to_numeric(l0["emission:lon"], errors="coerce")
    bbox = ", ".join(f"{v:.3f}" for v in (lon.min(), lat.min(), lon.max(), lat.max()))
    ficheros = sorted(l0["internal:source_file"].unique())
    por_fichero = l0.groupby("internal:source_file").size()
    cifras = {
        "muestras": len(l0),
        "hojas": len(l1),
        "ficheros": len(ficheros),
        "bbox": bbox,
    }

    base = (CARDS / f"{a.coleccion}_README_v130.md").read_text()
    desc = _mascara(base, a.coleccion, cifras)
    cuerpo = desc.split("## Dataset Information")[0].strip()
    cuerpo = re.sub(r"^#\s+methaneset-[a-z0-9]+-finetune\s*", "", cuerpo).strip() + "\n"

    (CARDS / f"{a.coleccion}_README.md").write_text(desc)
    (carpeta / "README.md").write_text(desc)
    print(f"README: {len(desc.splitlines())} lineas | {cifras}")

    # COLLECTION.json: lo que la tarjeta dice tiene que estar tambien en el catalogo
    ruta = carpeta / ".tacocat" / "COLLECTION.json"
    coll = json.loads(ruta.read_text())
    coll["description"] = "\n" + cuerpo
    coll["dataset_version"] = "1.4.0"
    coll["keywords"] = [k.replace("background", "reference") for k in coll["keywords"]]
    for prov in coll["providers"]:
        prov["name"] = prov["name"].replace("(background pairs)", "(reference pairs)")
    coll["taco:pit_schema"]["root"]["n"] = len(l0)
    coll["taco:pit_schema"]["shape"] = [int(por_fichero.max()), 8]
    extents = []
    for f in ficheros:
        sub = l0[l0["internal:source_file"] == f]
        la = pd.to_numeric(sub["emission:lat"], errors="coerce")
        lo = pd.to_numeric(sub["emission:lon"], errors="coerce")
        extents.append({"file": f, "id": coll["id"],
                        "spatial": [round(lo.min(), 3), round(la.min(), 3),
                                    round(lo.max(), 3), round(la.max(), 3)]})
    coll["taco:sources"] = {
        "count": len(ficheros),
        "ids": [coll["id"]] * len(ficheros),
        "files": ficheros,
        "extents": extents,
    }
    ruta.write_text(json.dumps(coll, indent=2, ensure_ascii=False))

    # index.html: descripcion y recuentos
    html = carpeta / "index.html"
    if html.exists():
        import markdown
        t = html.read_text()
        nuevo = markdown.markdown(cuerpo, extensions=["tables"])
        t = re.sub(r'(<div class="description-content">).*?(</div>)',
                   lambda m: m.group(1) + nuevo + m.group(2),
                   t, count=1, flags=re.S)
        t = t.replace("<td>1,353</td>", f"<td>{len(l0):,}</td>")
        t = t.replace("<td>10,824</td>", f"<td>{len(l1):,}</td>")
        t = t.replace("<td>3,552</td>", f"<td>{len(l0):,}</td>")
        t = t.replace("<td>28,416</td>", f"<td>{len(l1):,}</td>")
        html.write_text(t)
        print("index.html: descripcion y cuentas actualizadas")


if __name__ == "__main__":
    main()
