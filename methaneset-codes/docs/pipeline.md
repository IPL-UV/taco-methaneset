# Pipeline

End-to-end description of how the seven collections are built. The reasoning behind each decision
is included, because most of it was learned by hitting the problem first.

## 1. Catalogs

Two independent monitoring systems provide the annotations:

- **IMEO MARS** (UNEP) publishes verified detections on the Eye on Methane platform. For the
  multispectral sensors, MARS-S2L monitors known emitter sites and analysts delineate plumes by
  hand. For EMIT, annotations come from manual inspection of matched-filter maps.
- **Carbon Mapper** processes EMIT with a column-wise matched filter and runs a structured quality
  control before releasing a plume.

`catalogs/` snapshots both into dated folders. Snapshots are never overwritten: if the portal
changes tomorrow, yesterday's numbers still reproduce.

## 2. EMIT

The product keeps only granules where **both** systems verified a plume, so every scene carries
two independent labels.

1. Cross the catalogs by EMIT granule id (`00`).
2. Check which radiance files are already on disk and download the missing ones from LP DAAC
   (`01` to `03`).
3. Discard granules with nodata. This is a selection criterion, not a column: a scene with gaps
   does not enter.
4. Rank by plume extent and keep a balanced subset (`04` to `06`).

Assets per granule: the 285-band L1B radiance hypercube, the matched-filter products
(`mf`, `rmf`, `mag1c`), the two masks, elevation, the GLT and per-pixel lat/lon.

Two details that matter:

- Radiance is stored in **sensor coordinates**. Only the GLT is georeferenced, so drawing it on a
  map requires orthorectification through the GLT.
- Compression is SNR-adaptive: the SWIR region is stored losslessly, VNIR discards bits well below
  the instrument noise floor. That keeps the weak 1665 nm absorption and the strong 2300 nm band
  intact for both standard and combined matched filters.

## 3. Multispectral

The pairs, annotations and temporal split come from MARS-S2L. We do not redistribute the original
data: the imagery is re-retrieved at native resolution, which recovers the red-edge, water-vapor
and cirrus bands that the original six-band release drops.

- All bands resampled to 10 m and cut into 200x200 chips around the emitter.
- Cloud-Optimized GeoTIFF with Zstandard and `INTERLEAVE=TILE`: a reader can seek to the two SWIR
  tiles without decompressing the other eleven bands.
- Confirmed plume-free scenes form the **pretraining** subset; scenes with verified masks form the
  **finetune** subset.

The temporal split follows MARS-S2L: training through November 2023, validation in 2021, testing
from December 2023 to June 2024.

## 4. Plume bank

The bank comes from WRF-LES 3D tracer simulations at 20 m resolution. The 3D volume is projected
into 2D column enhancements following the satellite parallax geometry: each layer shifts by
`h * tan(theta)`, and the solar and viewing paths are projected separately and combined with
air-mass weights.

The projected bank spans a grid of wind speeds, solar zenith angles, solar azimuths, wind
directions and source configurations, all at a reference emission rate of 3000 kg/h. Every map is
a real 3D simulation passed through the geometry, so the bank can be used for physically
consistent augmentation. The raw simulation cubes are published separately as `methaneset-bank-les`.

Two traps:

- `wind_dir` is not meteorological. 0 degrees is East, counter-clockwise.
- The LES domain is nested in a periodic parent, so a plume can re-enter the domain. The input
  filter removes that second copy.

## 5. TACO build

`pipeline/taco/` wraps the assets in the TACO contract:

1. An inventory of scenes with their files and metadata (`contexts.py`).
2. The metadata namespaces as pydantic models (`extensions.py`).
3. Files as leaves, scenes as nested samples, everything in the root tortilla (`levels.py`).
4. Identity and license (`collection.py`).
5. Write and document (`build.py`).

The builder computes nothing: the assets already exist and it only organizes them. Before
writing, the schema is validated, so a mismatch fails early.

## Where the numbers come from

Sizes and sample counts quoted in the README come from the published tree on Hugging Face, not
from the manuscript. The datasets have been re-uploaded a few times, so counts move; when in
doubt, measure again.
