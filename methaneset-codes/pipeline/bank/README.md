# Plume bank (projected)

The projected bank: 2D column enhancements from WRF-LES 3D tracer simulations, spanning a grid of
wind speeds, solar zenith angles, solar azimuths, wind directions and source configurations.

| Module | What it does |
|---|---|
| `01_criteria.py` | Selection criteria for the snapshots |
| `02_selection.py` | Snapshot-by-snapshot selection |
| `03_grid.py` | Output grid |
| `04_projection_test.py` | Projection test |
| `05_generate_tifs.py` | Writes the enhancement GeoTIFFs |
| `06_table.py` | Index table |
| `07_inspection.py`, `07b_inspection_figure.py` | Visual inspection |
| `08_build_taco.py` | Builds the TACO collection |
| `09_verify_taco.py` | Verifies the result |
| `core/` | Shared package: `config`, `schema`, `geometry`, `projection` |

Two traps worth remembering:

- `wind_dir` here is not meteorological: 0 degrees is East, counter-clockwise.
- The LES domain is nested in a periodic parent, so a plume can re-enter the domain. The input
  filter removes that second copy.
