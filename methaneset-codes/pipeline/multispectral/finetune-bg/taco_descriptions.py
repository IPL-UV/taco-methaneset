"""Descriptions (English) of the table and of the collection of the L89 finetune with the 4 backgrounds."""

DESCRIPCION = """
Landsat 8/9 plume scenes for supervised methane detection, each with the original target scene
(MARS-S2L) and four clean backgrounds of the same site.

## Samples

Every sample is one plume id, a FOLDER with eight leaves:

| leaf | what it is |
|---|---|
| `target` | the Landsat scene with the plume (from MARS-S2L) |
| `bg0` | the background that MARS itself selected for that target (the original `reference`) |
| `bg1`, `bg2`, `bg3` | three extra clean backgrounds, curated here |
| `ch4` | the MBMP enhancement of the target (from MARS) |
| `plume` | binary plume mask of the target |
| `dem` | Copernicus GLO-30 elevation |

## Backgrounds

`bg0` is the single background that comes with the MARS-S2L pair. `bg1` to `bg3` are three extra
backgrounds that we curate for every target, with the same MARS method: query Landsat 8 and 9
together within ±120 days (before or after), filter by cloud and constellation, discard the target's
own pass, and keep the three most similar by the MARS metric on methane-insensitive bands (B2, B3,
B4, B6). When ±120 days does not yield 40 candidates, the pool is filled with the same day-of-year
±45 days up to two years back. All four backgrounds are clean of methane.

## Columns

All columns describe the target unless prefixed `bg:`, which describe the four backgrounds
(`bg:sensor0` to `bg:sensor3`, and so on).

Some columns are constant in this dataset and are kept for compatibility: `detection:offshore` is
always False (onshore plumes only) and `quality:observability` is `clear` for all but three samples.
`detection:isplume` was always True and is not shipped as a column.
"""


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
