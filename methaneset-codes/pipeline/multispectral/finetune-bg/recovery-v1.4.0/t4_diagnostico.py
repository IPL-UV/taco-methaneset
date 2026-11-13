"""T4 · por que no llega cada muestra perdida a tres referencias (v1.4.0).

Lee las candidatas puntuadas por `02_seleccionar.py` y escribe, muestra a muestra, cuantas
referencias hay y cual es la limitante que la deja fuera. Nada de "no se pudo": cada fila lleva
el motivo y la cifra que lo sostiene.

Limites que se aplican, en este orden:
  1. ventana        ±120 dias alrededor de la pluma (la de MARS-S2L)
  2. cobertura      >=90 % de pixeles validos (filtro de la descarga)
  3. nube estricta  <=2 %  (primera pasada de MARS-S2L)
  4. metano         <=3 % de pixeles con realce propio
  5. nube relajada  <=35 % (segunda pasada de MARS-S2L)

Uso:
  python v140/t4_diagnostico.py
"""
import pathlib

import pandas as pd

RAW = pathlib.Path("/data/databases/METHANESET_TACOS/_v140/T2/raw")
OUT = pathlib.Path("/data/databases/METHANESET_TACOS/_v140/T4")
CLOUD_STRICT, CLOUD_RELAX, METHANE_MAX = 0.02, 0.50, 0.03


def motivo(g):
    n = len(g)
    if n == 0:
        return "sin candidatas en la ventana de ±120 días"
    estricta = ((g.cloud <= CLOUD_STRICT) & (g.methane <= METHANE_MAX)).sum()
    relajada = ((g.cloud <= CLOUD_RELAX) & (g.methane <= METHANE_MAX)).sum()
    if estricta >= 3:
        return f"ok con 2 % de nube ({estricta} válidas)"
    if relajada >= 3:
        return f"ok solo relajando nube al 50 % (regla de MARS) ({relajada} válidas, {estricta} al 2 %)"
    if relajada == 0:
        if (g.methane <= METHANE_MAX).sum() == 0:
            return f"todas con metano > 3 % ({n} candidatas)"
        return f"todas con nube > 50 % ({n} candidatas)"
    return f"solo {relajada} de las 3 necesarias ({n} candidatas, {estricta} al 2 % de nube)"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    tablas = []
    for f in sorted(RAW.glob("candidatos_*.csv")):
        d = pd.read_csv(f)
        if d.empty:
            continue
        d["coleccion"] = f.stem.split("_")[1]
        d["sensor"] = f.stem.split("_")[2]
        tablas.append(d)
    if not tablas:
        raise SystemExit("no hay candidatos: lanza t3_select primero")
    d = pd.concat(tablas, ignore_index=True)

    filas = []
    for (col, sen, tid), g in d.groupby(["coleccion", "sensor", "taco_id"]):
        estricta = ((g.cloud <= CLOUD_STRICT) & (g.methane <= METHANE_MAX)).sum()
        relajada = ((g.cloud <= CLOUD_RELAX) & (g.methane <= METHANE_MAX)).sum()
        filas.append({
            "coleccion": col, "sensor": sen, "id": tid,
            "candidatas": len(g),
            "ok_nube_2": int(estricta),
            "ok_nube_35": int(relajada),
            "nube_mediana": round(float(g.cloud.median()), 3),
            "metano_mediano": round(float(g.methane.median()), 3),
            "motivo": motivo(g),
        })
    out = pd.DataFrame(filas).sort_values(["coleccion", "sensor", "id"])
    out.to_csv(OUT / "diagnostico_255.csv", index=False)

    resumen = (out.assign(recuperable=out.ok_nube_35 >= 3)
                  .groupby(["coleccion", "sensor", "recuperable"]).size()
                  .unstack(fill_value=0))
    print(resumen.to_string())
    print(f"\nfilas: {len(out)} · con 3 referencias: {int((out.ok_nube_35 >= 3).sum())}")
    print("\nmotivos mas frecuentes:")
    print(out.loc[out.ok_nube_35 < 3, "motivo"].str.replace(r"\d+", "N", regex=True)
             .value_counts().head(6).to_string())


if __name__ == "__main__":
    main()
