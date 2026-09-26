"""methaneset-bank v3 · single configuration.

Everything that is a bank decision lives here; the numbered scripts import it.

Conventions
-----------
- 0-based indices over the LES mass grid: 120 columns (x, downwind) and
  90 rows (y, from south to north). The LES wind always blows towards +x.
- Height of level k: z_k = k * DZ, as in Gorroño's and César's code.
- RAA is the SOLAR azimuth relative to the wind, counterclockwise from +x. With EMIT angles
  (from north, clockwise): RAA = (wind heading - SAA) mod 360.
- The footprint of the solar path shifts towards RAA + 180, the side opposite the sun.
- The rasters are stored north up: row 0 of the GeoTIFF = north; the wind goes to the right.
"""
from __future__ import annotations
import math


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
PERFILES = P.INJECTION_ASSETS / "perfiles-todos.npz"
LATERALES = P.INJECTION_ASSETS / "bordes-laterales-todos.csv"
ANCHOS = P.INJECTION_ASSETS / "ancho-todos.csv"
FUERA = P.INJECTION_ASSETS / "fuera-todos.csv"
CODE = Path(__file__).resolve().parents[1]
TRABAJO = CODE / "trabajo"
RAW = P.BANK_RAW
FINAL = P.BANK_FINAL

# LES
PX = 20.0                      # m, LES pixel
DZ = 20.0                      # m, level thickness
NY, NX, NZ = 90, 120, 49
RELAX = 5                      # relaxation band pixels per side
N_TOTAL_AIR = 3.6055e5         # mol/m2 of air in the column
PIXEL_AREA = PX * PX
Q_REF = 3000.0                 # kg/h
WS = [2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0, 7.0, 8.0, 10.0]
SNAPS = list(range(60, 120))   # minutes 30 to 59.5
TIPOS = ("multi", "area")
CARPETA_TIPO = {"multi": "multisource_simulations", "area": "areasource_simulations"}


def carpeta_viento(ws: float) -> str:
    return f"ugeo_{int(ws)}ms" + ("5" if ws % 1 else "")


# selection criteria
COL_ENTRADA = RELAX                     # 5
COL_BORDE = NX - 1 - RELAX              # 114
VENTANA = 30                            # slope columns, 600 m
INTERIOR = (40, COL_BORDE + 1 - VENTANA)  # columns 40 to 84
ENTRADA_FRAC = 0.02                     # upwind edge below 2% of the peak
LATERAL_FRAC = 0.02                     # north (row 84) and south (row 5) edges below 2% of the peak
BARLOVENTO = 5                          # px: the foreign methane strip ends 100 m before the source box
FUERA_MAX = 0.002                       # estimated foreign methane below 0.2% of the mass (twice the zero budget)
RUIDO_REF = 50.0                        # ppb, EMIT standard noise: sets the visible width of the plume
ANCHO_BORDE_MAX = 60.0                  # m: the plume leaves the domain narrower than one EMIT pixel
SEP = 16                                # 8 minutes between snapshots
OBJETIVOS = (75, 90, 105)               # minutes 37.5, 45 and 52.5

# geometry
Z_REF = 330.0                  # m, grid reference height: above the z_eff of every plume (max. 330)
TOL = 60.0                     # m, one EMIT pixel
SZA_TOPE = 70.5                # degrees; with TOL and Z_REF it gives 17 rings, the last one at 71.0
VZA_REF = 0.0                  # degrees: nadir (24 Sep reprocess; 8.4 before, median of the 721 EMIT scenes)
A_V = 1.0 / math.cos(math.radians(VZA_REF))

# saving
UMBRAL_CEROS = 1e-4            # maximum threshold, fraction of the maximum of each array
PRESUPUESTO_MASA = 1e-3        # the zeros cannot take more than 0.1% of the mass
UNIDADES = "ppb"
