"""Configuration of the methaneset-emit (FOLDER) build.

Step 0: everything editable lives here. No paths or identity scattered
across the modules.

PREPARED, NOT RUN YET: check INVENTARIO and ASSETS_DIR before the first run.
"""

import sys
from pathlib import Path

for _p in Path(__file__).resolve().parents:
    if (_p / "config" / "paths.py").exists():
        sys.path.insert(0, str(_p / "config"))
        break
import paths as P  # noqa: E402


# ── Inputs ────────────────────────────────────────────────────────────────
# Scene inventory: one row per granule with precomputed metadata
# (produced by the data pipeline, not by this builder).
INVENTARIO = P.root("METHANESET_INVENTORY", "PENDIENTE/methaneset_master.parquet")

# Folder with the per-scene assets: <granule_id>/{radiance,mf,...}.tif
ASSETS_DIR = P.root("METHANESET_ASSETS_DIR", "PENDIENTE/RETRIEVALS")

# The 9 files that make up each sample, in the dataset's canonical order.
ASSETS = [
    "radiance.tif", "mf.tif", "rmf.tif", "mag1c.tif",
    "plume_imeo.tif", "plume_cm.tif",
    "elevation.tif", "glt.tif", "latlon.tif",
]

# ── Output ────────────────────────────────────────────────────────────────
OUTPUT = P.root("METHANESET_EMIT_OUT", "PENDIENTE/methaneset-emit")   # destination folder
BUILD = {
    "output_format": "folder",   # the zip comes later: "zip" + split_size
    "consolidate": True,
    "clean_previous_outputs": True,
    "validate_schema": True,
    "sample_limit": None,        # e.g. 3 for a test build
    "workers": 8,
}

# ── COLLECTION.json identity ──────────────────────────────────────────────
COLLECTION_ID = "methaneset-emit"
COLLECTION_VERSION = "1.0.0"
COLLECTION_LICENSES = ["CC-BY-4.0"]
