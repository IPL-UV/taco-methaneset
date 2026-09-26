"""methaneset-bank-les v1 · single configuration.

The 3D cubes of Gorroño's LES, just as they come out of the model, published as TACO. One row =
one volume (49 levels x 90 x 120) of a physical plume in a snapshot. Not projected, without
cropping the relaxation band and without any zeroing: that is the user's choice, the README explains it.

LES conventions (the same as in the netCDF):
- x = column = wind direction; y = row = crosswind, from south to north.
- The netCDF stores row 0 at the south. The GeoTIFF is stored north up and the wind towards +x.
- Height of level k: the centre between the staggered layers k and k+1 of (PH + PHB) / g.
"""
from __future__ import annotations
from pathlib import Path


import sys
from pathlib import Path

for _p in Path(__file__).resolve().parents:
    if (_p / "config" / "paths.py").exists():
        sys.path.insert(0, str(_p / "config"))
        break
import paths as P  # noqa: E402

# paths, resolved from the environment (see config/paths.py)
LES = P.LES
LOADER = P.LOADER
CONF_WRF = P.WRF_CONF
CODE = Path(__file__).resolve().parents[1]
RAW = P.BANK_LES_RAW
FINAL = P.BANK_LES_FINAL

# LES grid
PX = 20.0
NY, NX, NZ = 90, 120, 49
NY_STAG = NY + 2 * 0
RELAX = 5
G = 9.81
N_TOTAL_AIR = 3.6055e5      # mol/m2 of air in the column, the same constant as the projected bank
PIXEL_AREA = PX * PX
Q_REF = 3000.0              # kg/h

# bank universe: every emitter and every wind speed, the 7 snapshots every 5 min of the
# valid window (minutes 30 to 60) PLUS the exact snapshots used by the projected bank, so that
# everything that originated the projected bank is also here without projecting. The turbulence
# memory is 1 to 2.5 min (measured), so the 5 min are already independent samples.
WS = [2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0, 7.0, 8.0, 10.0]
SNAPS = [60, 70, 80, 90, 100, 110, 119]
PROYECTADO = Path(__file__).resolve().parents[1] / "snapshots-proyectado.csv"
TIPOS = ("multi", "area")
PLUMAS = list(range(1, 10))
CARPETA_TIPO = {"multi": "multisource_simulations", "area": "areasource_simulations"}

# criteria of the projected bank, to flag them here too (the same numbers)
COL_ENTRADA = RELAX
COL_BORDE = NX - 1 - RELAX
VENTANA = 30
INTERIOR = (40, COL_BORDE + 1 - VENTANA)
ENTRADA_FRAC = 0.02
LATERAL_FRAC = 0.02
BARLOVENTO = 5
FUERA_MAX = 0.002
ANCHO_BORDE_MAX = 60.0
RUIDO_REF = 50.0
TOCA_BORDE_FRAC = 0.05

UNIDADES = "ppb"
VERSION = "1.0.0"


def carpeta_viento(ws: float) -> str:
    return f"ugeo_{int(ws)}ms" + ("5" if ws % 1 else "")


def emisor(tipo: str, p: int) -> str:
    """a1 to a9 the area sources, p1 to p9 the point sources."""
    return ("a" if tipo == "area" else "p") + str(p)


def uid(tipo: str, p: int, ws: float, snap: int) -> str:
    return f"{emisor(tipo, p)}_w{ws:04.1f}_s{snap:03d}"
