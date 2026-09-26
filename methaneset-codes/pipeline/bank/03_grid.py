"""Step 3 · Grid of sun positions. Output: trabajo/rejilla.csv."""
import pandas as pd
from banco import config as C
from banco.geometria import rejilla

d = pd.DataFrame(rejilla(), columns=["anillo", "sza", "raa"])
d.insert(0, "nodo", range(len(d)))
C.TRABAJO.mkdir(exist_ok=True)
d.to_csv(C.TRABAJO / "rejilla.csv", index=False)
print(f"{len(d)} sun positions in {d.anillo.nunique()} rings")
print(d.groupby("anillo").agg(sza=("sza", "first"), azimuts=("raa", "size")).round(1).to_string())
