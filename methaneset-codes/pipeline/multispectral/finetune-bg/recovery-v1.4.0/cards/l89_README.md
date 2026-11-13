
# methaneset-l89-finetune


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

## Dataset Information

**Version**: 1.4.0


**License**: CC-BY-NC-SA-4.0

**Keywords**: methane, landsat, reference, plume, taco

**Tasks**: detection

## Dataset Overview

**Partitions**: 19 files
**Spatial coverage**: [-103.982, -50.746, 151.104, 43.553] (WGS84)

## Dataset Structure (Root-Sibling Uniform Tree)

**Root**: FOLDER (1,370 samples)

**Hierarchy**:

- Level 1: FILE → FILE → FILE → FILE → FILE → FILE → FILE → FILE (10,960 samples)

## Metadata Fields

### LEVEL0

| Field | Type | Description |
|-------|------|-------------|
| `id` | `string` | Unique sample identifier within parent scope. Must be unique among siblings. |
| `type` | `string` | Sample type discriminator (FILE or FOLDER). |
| `target:sensor` | `string` | Satellite of the target scene: LC08 (Landsat 8) or LC09 (Landsat 9) |
| `detection:case_study` | `string` | Case study region (e.g. Turkmenistan, Algeria) |
| `detection:ch4_fluxrate` | `double` | Emission rate of the plume estimated by MARS [kg/h] |
| `detection:ch4_fluxrate_std` | `double` | Uncertainty of detection:ch4_fluxrate [kg/h] |
| `detection:offshore` | `bool` | Whether the source is offshore. Always False in this dataset |
| `detection:sector` | `string` | Economic sector of the source (Oil and Gas, Coal) |
| `detection:wind_source` | `string` | Reanalysis used for the wind: ERA5-Land onshore, GEOS-FP offshore |
| `geoenrich:admin_countries` | `string` | Administrative country (may repeat site:country) |
| `geoenrich:admin_districts` | `string` | Administrative district |
| `geoenrich:admin_states` | `string` | Administrative state or province |
| `geoenrich:elevation` | `double` | Elevation of the emission point, Copernicus GLO-30 [m] |
| `geoenrich:population` | `double` | Mean population density around the emission point (HRSL) |
| `geoenrich:temperature` | `double` | Mean climate temperature at the site [K] |
| `majortom:code` | `string` | Internal identifier of the MARS scene |
| `meteo:wind_u` | `double` | Eastward wind component at the emission point [m/s] |
| `meteo:wind_v` | `double` | Northward wind component at the emission point [m/s] |
| `plume:geometry` | `string` | Plume polygon of the target in WKT/WKB, EPSG:4326 |
| `quality:last_update` | `string` | Last update of the MARS record |
| `quality:notified` | `bool` | Whether the emitter was notified upstream |
| `quality:observability` | `string` | MARS observability tag: clear or bad_retrieval |
| `quality:percentage_clear` | `double` | MARS cloud-clear fraction of the target scene [%] |
| `target:sza` | `double` | Solar zenith angle of the target [deg] |
| `target:saa` | `double` | Solar azimuth angle of the target [deg] |
| `target:raa` | `double` | Sun azimuth relative to the wind, `(wind_dir - SAA) mod 360` [deg] |
| `target:tile` | `string` | Landsat granule id of the target scene |
| `target:vza` | `double` | Viewing zenith angle of the target [deg] |
| `site:country` | `string` | Country of the emission source |
| `site:location_name` | `string` | Site name of the emission source |
| `split` | `string` | train / validation / test split inherited from MARS-S2L, grouped by emitter |
| `stac:crs` | `string` | CRS of the chip (local UTM) |
| `stac:geotransform` | `string` | GeoTransform of the chip: 10 m pixels, 200 x 200 |
| `stac:tensor_shape` | `string` | Shape of the chip: 11 bands, 200 rows, 200 columns |
| `stac:time_start` | `string` | Acquisition time of the target scene |
| `bg:sensor0` | `string` | Satellite of bg0 (the MARS reference): LC08 or LC09 |
| `bg:tile0` | `string` | Landsat granule id of bg0 (the MARS reference) |
| `bg:date0` | `string` | Acquisition date of bg0 |
| `bg:cloud0` | `double` | Cloud and shadow fraction of bg0 |
| `bg:sensor1` | `string` | Satellite of bg1: LC08 or LC09 |
| `bg:date1` | `string` | Acquisition date of bg1 |
| `bg:cloud1` | `double` | Cloud and shadow fraction of bg1 |
| `bg:difference1` | `double` | MARS similarity of bg1 to the target (lower is more similar) |
| `bg:methane1` | `double` | Fraction of bg1 pixels with a methane enhancement of its own |
| `bg:sensor2` | `string` | Satellite of bg2: LC08 or LC09 |
| `bg:date2` | `string` | Acquisition date of bg2 |
| `bg:cloud2` | `double` | Cloud and shadow fraction of bg2 |
| `bg:difference2` | `double` | MARS similarity of bg2 to the target (lower is more similar) |
| `bg:methane2` | `double` | Fraction of bg2 pixels with a methane enhancement of its own |
| `bg:sensor3` | `string` | Satellite of bg3: LC08 or LC09 |
| `bg:date3` | `string` | Acquisition date of bg3 |
| `bg:cloud3` | `double` | Cloud and shadow fraction of bg3 |
| `bg:difference3` | `double` | MARS similarity of bg3 to the target (lower is more similar) |
| `bg:methane3` | `double` | Fraction of bg3 pixels with a methane enhancement of its own |
| `target:cloud` | `double` | Cloud and shadow fraction of the target chip |
| `emission:lat` | `double` | Latitude of the emission point [deg] |
| `emission:lon` | `double` | Longitude of the emission point [deg] |
| `internal:current_id` | `int64` | Current sample position at this level (0-indexed). Enables O(1) random access and relational JOINs (ZIP, FOLDER, TACOCAT). |
| `internal:parent_id` | `int64` | Foreign key referencing parent sample position in previous level (ZIP, FOLDER, TACOCAT). |

### LEVEL1

| Field | Type | Description |
|-------|------|-------------|
| `id` | `string` | Unique sample identifier within parent scope. Must be unique among siblings. |
| `type` | `string` | Sample type discriminator (FILE or FOLDER). |
| `geotiff:stats` | `list<item: list<item: float>>` | Per-band statistics (List[List[Float32]]): categorical mode returns class probabilities, continuous mode returns [min, max, mean, std, valid%, p25, p50, p75, p95] |
| `taco:header` | `binary` | Binary TACOTIFF header (35 bytes + tile counts) for fast reading without IFD parsing |
| `internal:current_id` | `int64` | Current sample position at this level (0-indexed). Enables O(1) random access and relational JOINs (ZIP, FOLDER, TACOCAT). |
| `internal:parent_id` | `int64` | Foreign key referencing parent sample position in previous level (ZIP, FOLDER, TACOCAT). |
| `internal:relative_path` | `string` | Relative path from DATA/ directory. Format: {parent_path}/{id} or {id} for level0 (ZIP, FOLDER, TACOCAT). |


## Usage

### Python

```python
# pip install tacoreader
import tacoreader

ds = tacoreader.load("https://huggingface.co/datasets/tacofoundation/methaneset/resolve/main/methaneset-l89-finetune/.tacocat/")
print(f"ID: {ds.id}")
print(f"Version: {ds.version}")
print(f"Samples: {len(ds.data)}")
```

### R

```r
# Coming soon: R support is planned but not yet available
# install.packages("tacoreader")
library(tacoreader)

ds <- load_taco("https://huggingface.co/datasets/tacofoundation/methaneset/resolve/main/methaneset-l89-finetune/.tacocat/")
cat(sprintf("ID: %s\n", ds$id))
cat(sprintf("Version: %s\n", ds$version))
cat(sprintf("Samples: %d\n", nrow(ds$data)))
```

### Julia

```julia
# Coming soon: Julia support is planned but not yet available
# using Pkg; Pkg.add("TacoReader")
using TacoReader

ds = load_taco("https://huggingface.co/datasets/tacofoundation/methaneset/resolve/main/methaneset-l89-finetune/.tacocat/")
println("ID: ", ds.id)
println("Version: ", ds.version)
println("Samples: ", size(ds.data, 1))
```

## Data Providers

**Image and Signal Processing Group (ISP-UV)** — *producer*

**UNEP IMEO, MARS (reference pairs)** — *licensor*




---

Generated with ❤️ using [TacoToolbox](https://github.com/tacotoolbox/tacotoolbox) v0.26.9

## Schema changelog (v1.2.0)

`target:raa` was corrected to match the `sun:raa` convention of `methaneset-bank`: sun azimuth
relative to the wind, counterclockwise from the wind direction, i.e. `(wind_dir - SAA) mod 360`.
The previous value used the opposite sign, so querying the bank with it mirrored the plume's solar
parallax. No column was renamed. Null when the wind is calm (<0.1 m/s) or SAA is missing.

## Data fix (2026-11-02)

`bg1`--`bg3` were previously stored as raw Level-1 DN (`C02/T1`, uint16) while `target` and `bg0`
were TOA reflectance (`C02/T1_TOA`, float32). All four references are now downloaded from
`C02/T1_TOA`, so every leaf in a sample shares the same radiometric scale. All leaves were
re-encoded as COG with 256-pixel blocks; the other leaves keep their values.
`bg:difference1`--`bg:difference3` is now computed on the TOA leaves. The previous values compared
DN references with TOA targets. Three references were replaced because their scenes did not cover the chip:
`83f82646` bg1 was 100 % nodata (now `LC09_181040_20240321`), and `a204333a` bg3 (4 % valid) and
`3c682e7c` bg3 (15 % valid) were re-selected as `LC09_169037_20230907` and `LC08_157033_20240609`.
Their metadata columns were updated.

## Reference identity fix (2026-11-03)

Every `bg0`--`bg3` leaf was matched against the Landsat Collection 2 `T1_TOA` catalog by its
pixels, not by trusting the recorded date: each chip was compared with every candidate scene over
the site (WRS-2 path ±2, ±240 days) on two independent 600 m windows, and the matching scene was
taken as the leaf's identity. `bg:sensor0` is now derived from `bg:tile0` (it had been copied from
`target:sensor`, which was wrong in 523 rows), and 115 `bg1`--`bg3` rows whose `bg:sensor`/`bg:date`
pair did not correspond to any scene over the site were corrected to the scene actually stored.
References that repeated a sibling image were re-selected with the documented MARS rule and
rejected if their pixel hash matched another reference of the same sample. `bg:difference` and
`bg:methane` were recomputed on the final leaves. After the fix no sample repeats a scene among
`target` and `bg0`--`bg3`, and no two leaves of a sample are pixel-identical. The GEE asset id of
every leaf is listed in `data/gee_ids_l89_finetune.csv` in the
[code repository](https://github.com/IPL-UV/taco-methaneset).

## Known limitation: thermal no-data

`B10` and `B11` are brightness temperature from `T1_TOA`. Outside the TIRS swath (100 km) the
product has no data, and those pixels are stored as `-inf` (the TOA conversion of the fill value).
This affects 71 of the 6,765 reflectance leaves in 42 of the 1,353 samples (3 %), `target` and
references alike. In 49 of those leaves the `-inf` is confined to `B10`/`B11`; in 22 it also
reaches `B1`--`B9` because the chip straddles the edge of the scene, and in two references
(`468c9372` `bg1` and `a025cd0f` `bg3`, both `LC08_165038_20250205`) only 2.4 % of the pixels are
valid. Mask with `np.isfinite` before normalising, and check per-leaf valid fraction if you
subsample.

## Known issue: references that coincide with another target

The Landsat archive over a site is small, and 1,875 of the 5,412 references (516 `bg0`, 1,359
`bg1`--`bg3`, in 857 samples) are pixel-identical to the `target` of another sample of the same
site; they therefore contain that sample's plume. `data/cross_sample_backgrounds.csv` in the
[code repository](https://github.com/IPL-UV/taco-methaneset) lists every one of them with the
source sample and the number of plume pixels, so you can mask those pixels or exclude the leaves.
The curated `bg1`--`bg3` were deduplicated against the target and against each other, but not
against the targets of other samples.

## Known issue: unverified reference identities

38 `bg1`--`bg3` leaves (in 10 samples) could not be matched to a `T1_TOA` scene because the chip
is displaced relative to the scene grid; their `bg:sensor`/`bg:date` keep the identity recorded by
the selection pipeline.
