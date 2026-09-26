"""Step 2 · Choice of up to 3 snapshots per run.

Among those passing the four criteria, the trio closest to minutes 37.5, 45 and 52.5 with 8
minutes or more between consecutive ones; if there is no trio, the best pair; if there is none,
the closest to the centre. Every emitter with something that passes is included.
Output: trabajo/seleccion.csv, one row per physical plume.
"""
import itertools
import numpy as np
import pandas as pd
from banco import config as C
from banco.esquema import uid as nombre_uid

c = pd.read_csv(C.TRABAJO / "criterios.csv")
O = C.OBJETIVOS


def elegir(ok):
    best = None
    for a, b, d in itertools.combinations(ok, 3):
        if b - a >= C.SEP and d - b >= C.SEP:
            cost = abs(a - O[0]) + abs(b - O[1]) + abs(d - O[2])
            if best is None or cost < best[0]:
                best = (cost, [a, b, d])
    if best:
        return best[1]
    for a, b in itertools.combinations(ok, 2):
        if b - a >= C.SEP:
            cost = abs(a - O[0]) + abs(b - O[2])
            if best is None or cost < best[0]:
                best = (cost, [a, b])
    if best:
        return best[1]
    return [min(ok, key=lambda s: abs(s - O[1]))] if len(ok) else []


filas = []
for (t, w, p), g in c.groupby(["tipo", "ws", "plume"]):
    for s in elegir(np.sort(g[g.pasa].snap.values)):
        filas.append(dict(tipo=t, ws=w, plume=int(p), snap=int(s), minuto=s / 2,
                          plume_uid=nombre_uid(t, int(p), w, int(s))))
d = pd.DataFrame(filas)
d.to_csv(C.TRABAJO / "seleccion.csv", index=False)
print(f"{len(d)} physical plumes")
print(d.groupby(["tipo", "plume"]).size().to_string())
