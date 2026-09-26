# EMIT (hyperspectral)

Granules enter the dataset only when **both** IMEO MARS and Carbon Mapper verified a plume in the
same EMIT granule. The consensus is the point of this product.

| Module | What it does |
|---|---|
| `00_cross_imeo_cm.py` | Fetches fresh catalogs and crosses them by granule id |
| `01_consolidate_nc.py` | Standardizes the local radiance store into ordered folders |
| `02_nc_coverage.py` | Checks which RAD/OBS files of the consensus are already on disk |
| `03_download_missing_nc.py` | Downloads the missing ones from LP DAAC (`EMITL1BRAD` v001) |
| `04_nodata_scan.py` | Percent of nodata per candidate |
| `05_balance_analysis.py` | Balance axes over the clean granules |
| `06_select_balanced.py` | Final selection, one row per granule |
| `06b`, `06c`, `06d` | Figures: before and after, flux versus pixels, selection map |
| `07_propose_splits.py` | Proposes the train, validation and test split |
| `08_chunking_bench.py` | Chunking benchmark for the radiance store |
| `09_wind_intrascene.py` | Wind variability inside a scene |
| `10_generate_radiance.py` | Writes the radiance Cloud-Optimized GeoTIFF |
| `10b_apply_lsb.py` | SNR-adaptive mantissa compression (GDAL `DISCARD_LSB`) |
| `11_generate_latlon_glt.py` | Per-pixel latitude and longitude, plus the GLT layers |
| `12_generate_elevation.py` | Copernicus DEM elevation |
| `13_generate_wind.py` | Wind field per scene |
| `14_generate_retrievals.py` | Matched-filter products: `mf`, `rmf` and `mag1c` |
| `15_generate_masks.py` | Plume masks from IMEO and Carbon Mapper |
| `15a`, `15b` | Download the Carbon Mapper rasters and sources |
| `16_verify_samples.py` | Verifies each sample before packaging |
| `17a`, `17b` | Download the observation products and build the metadata table |
| `18_build_taco.py` | Builds the TACO collection |
| `19*`, `21`, `22`, `23`, `24`, `25` | Analyses and figures kept for provenance |
| `20_subir_radiance.py` | Uploads the radiance store |
| `26_upload_hf.py` | Uploads the collection to Hugging Face |

Rules that are not negotiable:

- **No nodata**: a granule with nodata does not enter. It is a selection criterion, not a column.
- **Emission point always from the catalog**, never the centroid of the mask.
- Flow bins: `weak < 1000 kg/h <= strong` (STARCOP; EMIT reaches 90% POD around 1060 kg/h).

Two details that matter: radiance is stored in **sensor coordinates** (only the GLT is
georeferenced), and compression is SNR-adaptive, keeping the whole SWIR window lossless.
