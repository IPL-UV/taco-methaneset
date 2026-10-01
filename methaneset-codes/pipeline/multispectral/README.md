# Multispectral (Sentinel-2 and Landsat 8/9)

Two independent builds share the same conventions:

- `finetune-bg/` — scenes with expert-verified plume masks, plus the matched background.
- `pretraining/` — confirmed plume-free scenes with their paired reference.

The pairs, annotations and temporal split come from MARS-S2L. The imagery is re-retrieved at
native resolution instead of redistributing the original six common bands: 13 bands for
Sentinel-2 and 9 for Landsat OLI, resampled to 10 m and cut into 200x200 chips around the emitter.

Each chip is a Cloud-Optimized GeoTIFF with Zstandard compression and `INTERLEAVE=TILE`, so
reading the two SWIR bands does not decompress the other eleven.

| Folder | Modules |
|---|---|
| `finetune-bg/` | `01_download.py`, `02_select.py`, `03_proposal_table.py`, `04_sensors.py`, `05_build_taco.py`, `06_recompute_ch4.py` |
| `pretraining/` | `01_prepare_table.py`, `05_build_taco.py`, `run_full.sh`, `scan_pre_ids.py`, `scan_ref_nulls.py` |

Shared helpers live next to the modules: `config_sensors.py`, `s2_prep.py`, `taco_descriptions.py`.

## Recomputing the `ch4` enhancement

`06_recompute_ch4.py` rebuilds the `ch4` leaf of both finetune sets with the MARS retrieval
(`marss2l`) after the January 2026 change that keeps negative ΔXCH₄ values. It recomputes from
`target` + `bg0` + `plume`, replaces only the `ch4` entry of every tacozip, and updates the
offsets, sizes and statistics of the internal tables. Needs the public
[marss2l](https://github.com/UNEP-IMEO-MARS/marss2l) code (LGPL-3.0) and
`integrated_transmittances.json`.
