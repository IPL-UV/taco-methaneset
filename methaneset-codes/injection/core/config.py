"""
config.py

Physical constants, grid definitions, and default paths for the
methane plume injection pipeline.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Default paths
# ---------------------------------------------------------------------------
DIR_EMIT_SCENES = "https://huggingface.co/datasets/tacofoundation/methaneset/resolve/main/methaneset-emit"
DIR_PLUME_BANK  = "https://huggingface.co/datasets/tacofoundation/methaneset/resolve/main/methaneset-bank"

# Resolved relative to this file (core/), not the current working directory.
# The EMIT .npy files live next to this module in core/.
_PROJECT_DIR = Path(__file__).resolve().parent
EMIT_WAVELENGTHS_FILE = _PROJECT_DIR / "emit_wavelengths.npy"
EMIT_FWHM_FILE        = _PROJECT_DIR / "emit_fwhm.npy"

# ---------------------------------------------------------------------------
# Patch & plume settings
# ---------------------------------------------------------------------------
PATCH_H = 128                 # patch height [pixels]
PATCH_W = 128                 # patch width [pixels]
N_PLUMES = None               # number of plumes per patch (None = random 1-10, see run_injection randint(1, 11))
SOURCE = "multi"              # "area" or "multi" (None = random per plume)
MIN_SEPARATION = 15           # min pixels between sources
MARGIN = 10                   # min pixels from patch edge for sources

Q_MIN_KGH = 100               # minimum emission rate [kg/h]
Q_MAX_KGH = 10000.0           # maximum emission rate [kg/h]
Q_REF_KGH = 3000.0            # plume bank reference emission rate [kg/h]
Q_TARGET_KGH = None           # emission rate per plume (None = random Q_MIN–Q_MAX)

# ---------------------------------------------------------------------------
# Wind
# ---------------------------------------------------------------------------
USE_WIND_FIELD = False        # use wind.tif field (True) or fixed wind (False)
WIND_SPEED = None             # m/s (None = random from bank)
WIND_DIR = None               # deg (None = random 0–359)

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------
MR_BACKGROUND_PPB = 1900.0     # atmospheric methane concentration [ppb]
ENHANCEMENT_PIXEL_RES = 20.0   # bank pixel resolution [m]
EMIT_NODATA = -9999.0

# ---------------------------------------------------------------------------
# Methane absorption windows (nm)
# ---------------------------------------------------------------------------
CH4_WINDOWS = [(1000, 2500)]