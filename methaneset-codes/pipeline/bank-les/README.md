# Plume bank (raw simulation cubes)

The source of the plume bank: the 3D tracer volumes from WRF-LES, published without projection and
without cropping, so the bank can be reprocessed under new geometries.

| Module | What it does |
|---|---|
| `01_generate.py` | Writes the raw volumes |
| `02_table.py` | Index table |
| `03_inspection.py` | Visual inspection |
| `04_build_taco.py` | Builds the TACO collection |
| `05_verify.py` | Verifies the result |
| `06_upload_hf.py` | Uploads the collection to Hugging Face |
| `core/` | Shared package: `config`, `schema`, `metrics` |
