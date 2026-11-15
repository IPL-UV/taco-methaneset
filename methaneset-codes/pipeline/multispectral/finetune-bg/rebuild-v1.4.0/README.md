# Rebuild of the multispectral finetune (v1.4.0, 14 nov 2026)

The first v1.4.0 release passed the smoke tests but not a full audit: empty chips, cloud measured
at the wrong resolution, references outside the ±120-day window and stale metadata inside the
zips. This folder holds the scripts that audited both collections, repaired the leaves and
rewrote the `.tacozip` files from scratch without touching a single pixel that did not need it.

The rule is the MARS-S2L one: references within ±120 days of the target, cloud screened in two
rounds (5 % strict, 35 % when the strict round leaves fewer than three candidates), ranking by
the MARS score on methane-insensitive bands. Methane is **not** a selection criterion, because
MARS does not use it (`isplume` is disabled in `location_image.py`); `bg:methane` is published as
a diagnostic.

| script | what it does |
|---|---|
| `t11_cloud.py` | CloudSEN12 at native resolution on every published leaf |
| `t13_auditoria.py` | full audit: structure, column coherence, raster units, recompute of cloud, difference and methane |
| `t14_chips.py` | re-downloads empty leaves from Earth Engine, picking the granule that really covers the chip |
| `t17_satelites.py` | tries to recover the Sentinel-2 spacecraft from the STAC catalogue |
| `t19_reemplazos.py` | turns the selection into the replacement table (`cambios.csv`) |
| `t23_refs_l89.py` | replaces Landsat references whose scene barely overlaps the chip |
| `t24_nube_chips.py` | measures cloud on the repaired chips (the old values were computed on zeros) |
| `t16_rebuild.py` | writes each zip from scratch: `TACO_HEADER`, offsets and `__meta__` recalculated |
| `t21_level0.py` | rewrites `level0` and the per-zip `COLLECTION.json` inside and outside the zips |
| `t20_puertas.py` | gates: sha256 per leaf, offsets against the real TIFFs, header against the zip, `METADATA` against `DATA`, canonical columns, cloud ≤35 %, `tacoreader` |
| `t22_subir.py` | uploads to a branch, compares the manifest and tests a download pinned to the commit before touching `main` |

`plan-reparacion.md` has the defect table and the decisions; `CHANGELOG.md` has the final numbers
and the sha256 of every published `.tacozip`.

Order of a full rebuild:

```bash
python t11_cloud.py --coleccion l89 && python t11_cloud.py --coleccion s2
python t19_reemplazos.py
python t16_rebuild.py --fam l89 --salida .../T16/l89
python t21_level0.py --fam l89 --nuevo .../T16/l89
python t20_puertas.py --fam l89 --nuevo .../T16/l89 --viejo .../T7/l89
python t13_auditoria.py --taco .../T16/l89 --fam l89
```
