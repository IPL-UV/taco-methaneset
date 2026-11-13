"""Descriptions (English) of the table and of the collection of the L89 finetune with the 4 references.

OJO al reconstruir: `generate_markdown` sobrescribe el README.md de la coleccion con este texto.
DESCRIPCION es la de L89 y DESCRIPCION_S2 la de Sentinel-2; no se mezclan (la card de S2 llego a
describir escenas de Landsat por reutilizar la de L89). `tacotoolbox` 0.5.2 ya no trae
`generate_markdown`, asi que estas descripciones se aplican con `recovery-v1.4.0/t9_cards.py`.
"""

DESCRIPCION = """
Landsat 8/9 plume scenes for supervised methane detection, each with the original target scene
(MARS-S2L) and four references of the same site.

## Samples

Every sample is one plume id, a FOLDER with eight leaves:

| leaf | what it is |
|---|---|
| `target` | the Landsat scene with the plume (from MARS-S2L) |
| `bg0` | the reference that MARS itself selected for that target (the original `reference` scene) |
| `bg1`, `bg2`, `bg3` | three extra references, curated here |
| `ch4` | the MBMP enhancement of the target (from MARS) |
| `plume` | binary plume mask of the target |
| `dem` | Copernicus GLO-30 elevation |

## References

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

## Columns

All columns describe the target unless prefixed `bg:`, which describe the four references
(`bg:sensor0` to `bg:sensor3`, and so on).

The `target:cloud` and `bg:cloud0`--`bg:cloud3` columns are the cloud and shadow fraction of each chip, computed with OmniCloudMask.

Some columns are constant in this dataset and are kept for compatibility: `detection:offshore` is
always False (onshore plumes only) and `quality:observability` is `clear` for all but three samples.
`detection:isplume` was always True and is not shipped as a column.
"""

DESCRIPCION_S2 = """
# methaneset-s2-finetune


Sentinel-2 plume scenes for supervised methane detection, each with the original target scene
(MARS-S2L) and four references of the same site.

## Samples

Every sample is one plume id, a FOLDER with eight leaves:

| leaf | what it is |
|---|---|
| `target` | the Sentinel-2 scene with the plume (from MARS-S2L) |
| `bg0` | the reference that MARS itself selected for that target (the original `reference` scene) |
| `bg1`, `bg2`, `bg3` | three extra references, curated here |
| `ch4` | the MBMP enhancement of the target (from MARS) |
| `plume` | binary plume mask of the target |
| `dem` | Copernicus GLO-30 elevation |

## References

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

## Columns

All columns describe the target unless prefixed `bg:`, which describe the four references
(`bg:sensor0` to `bg:sensor3`, and so on).

The `target:cloud` and `bg:cloud0`--`bg:cloud3` columns are the cloud and shadow fraction of each chip, computed with OmniCloudMask.

Some columns are constant in this dataset and are kept for compatibility: `detection:offshore` is
always False (onshore plumes only) and `quality:observability` is `clear` for all but three samples.
`detection:isplume` was always True and is not shipped as a column.
"""

DESCRIPCION_POR_COLECCION = {"l89": DESCRIPCION, "s2": DESCRIPCION_S2}

DESCRIPCIONES = {
    "target:sensor": "Satellite of the target scene: LC08 (Landsat 8) or LC09 (Landsat 9)",
    "site:country": "Country of the emission source",
    "site:location_name": "Site name of the emission source",
    "split": "train / validation / test split inherited from MARS-S2L, grouped by emitter",
    "detection:case_study": "Case study region (e.g. Turkmenistan, Algeria)",
    "detection:sector": "Economic sector of the source (Oil and Gas, Coal)",
    "detection:ch4_fluxrate": "Emission rate of the plume estimated by MARS [kg/h]",
    "detection:ch4_fluxrate_std": "Uncertainty of detection:ch4_fluxrate [kg/h]",
    "detection:wind_source": "Reanalysis used for the wind: ERA5-Land onshore, GEOS-FP offshore",
    "detection:offshore": "Whether the source is offshore. Always False in this dataset",
    "plume:geometry": "Plume polygon of the target in WKT/WKB, EPSG:4326",
    "emission:lat": "Latitude of the emission point [deg]",
    "emission:lon": "Longitude of the emission point [deg]",
    "target:tile": "Landsat granule id of the target scene",
    "target:sza": "Solar zenith angle of the target [deg]",
    "target:vza": "Viewing zenith angle of the target [deg]",
    "target:cloud": "Cloud and shadow fraction of the target chip (OmniCloudMask)",
    "stac:crs": "CRS of the chip (local UTM)",
    "stac:geotransform": "GeoTransform of the chip: 10 m pixels, 200 x 200",
    "stac:tensor_shape": "Shape of the chip: 11 bands, 200 rows, 200 columns",
    "stac:time_start": "Acquisition time of the target scene",
    "meteo:wind_u": "Eastward wind component at the emission point [m/s]",
    "meteo:wind_v": "Northward wind component at the emission point [m/s]",
    "geoenrich:admin_countries": "Administrative country (may repeat site:country)",
    "geoenrich:admin_states": "Administrative state or province",
    "geoenrich:admin_districts": "Administrative district",
    "geoenrich:elevation": "Elevation of the emission point, Copernicus GLO-30 [m]",
    "geoenrich:population": "Mean population density around the emission point (HRSL)",
    "geoenrich:temperature": "Mean climate temperature at the site [K]",
    "quality:percentage_clear": "MARS cloud-clear fraction of the target scene [%]",
    "quality:observability": "MARS observability tag: clear or bad_retrieval",
    "quality:notified": "Whether the emitter was notified upstream",
    "quality:last_update": "Last update of the MARS record",
    "majortom:code": "Internal identifier of the MARS scene",
    "bg:sensor0": "Satellite of bg0 (the MARS reference): LC08 or LC09",
    "bg:tile0": "Landsat granule id of bg0 (the MARS reference)",
    "bg:date0": "Acquisition date of bg0",
    "bg:cloud0": "Cloud and shadow fraction of bg0 (OmniCloudMask)",
    "bg:sensor1": "Satellite of bg1: LC08 or LC09",
    "bg:date1": "Acquisition date of bg1",
    "bg:cloud1": "Cloud and shadow fraction of bg1 (OmniCloudMask)",
    "bg:difference1": "MARS similarity of bg1 to the target (lower is more similar)",
    "bg:methane1": "Fraction of bg1 pixels with a methane enhancement of its own",
    "bg:sensor2": "Satellite of bg2: LC08 or LC09",
    "bg:date2": "Acquisition date of bg2",
    "bg:cloud2": "Cloud and shadow fraction of bg2 (OmniCloudMask)",
    "bg:difference2": "MARS similarity of bg2 to the target (lower is more similar)",
    "bg:methane2": "Fraction of bg2 pixels with a methane enhancement of its own",
    "bg:sensor3": "Satellite of bg3: LC08 or LC09",
    "bg:date3": "Acquisition date of bg3",
    "bg:cloud3": "Cloud and shadow fraction of bg3 (OmniCloudMask)",
    "bg:difference3": "MARS similarity of bg3 to the target (lower is more similar)",
    "bg:methane3": "Fraction of bg3 pixels with a methane enhancement of its own",
}
