"""Path resolution for the MethaneSET pipelines.

Every root is read from the environment with a default for the development
server, so the code runs unchanged on a laptop, a cluster or a notebook.
Set the variables in your shell or in a `.env` file; nothing is hardcoded in
the modules.
"""
from __future__ import annotations

import os
from pathlib import Path


def root(var: str, default: str | None = None) -> Path:
    """Environment-driven path. Raises if the variable is missing and has no default."""
    value = os.environ.get(var) or default
    if value is None:
        raise RuntimeError(f"Set {var} in your environment")
    return Path(value).expanduser()


# Repository root: the folder that contains pipeline/, catalogs/, config/.
REPO = Path(__file__).resolve().parents[1]

# Working roots.
DATA = root("METHANESET_DATA", "/data/databases/METHANESET_TACOS")
ASSETS = root("METHANESET_ASSETS", str(REPO / "assets"))
CACHE = root("METHANESET_CACHE", str(REPO / ".cache"))

# Upstream sources used by the plume bank.
LES = root("METHANESET_LES", "/data/databases/zenodo_methane/home/wrfshared/WRF_gorrono/benchmark_sim")
WRF_CONF = root("METHANESET_WRF_CONF", "/data/databases/Julio/gorrono-les-configuracion/WRF_configuration_files")
LOADER = os.environ.get("METHANESET_LOADER", "/data/users/ceayca/methanebank")

# Derived bank paths.
BANK_RAW = root("METHANESET_BANK_RAW", str(DATA / "methaneset-bank-vza0-raw"))
BANK_FINAL = root("METHANESET_BANK_FINAL", str(DATA / "methaneset-bank-vza0"))
BANK_LES_RAW = root("METHANESET_BANK_LES_RAW", str(DATA / "methaneset-bank-les-raw"))

# Assets used to build the plume bank (profiles, side borders, widths, outside masks).
INJECTION_ASSETS = root(
    "METHANESET_INJECTION_ASSETS",
    "/data/users/julio/Notes/01-Projects/methanset/01-injection-notes/assets/code",
)
BANK_LES_FINAL = root("METHANESET_BANK_LES_FINAL", str(DATA / "methaneset-bank-les"))
