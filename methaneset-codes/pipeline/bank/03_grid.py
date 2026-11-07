"""Step 3 · Grid of sun positions. Output: trabajo/nodes.csv."""
import pandas as pd
from core import config as C
from core.geometry import nodes

d = pd.DataFrame(nodes(), columns=["anillo", "sza", "raa"])
d.insert(0, "nodo", range(len(d)))
C.WORK.mkdir(exist_ok=True)
d.to_csv(C.WORK / "nodes.csv", index=False)
print(f"{len(d)} sun positions in {d.anillo.nunique()} rings")
print(d.groupby("anillo").agg(sza=("sza", "first"), azimuts=("raa", "size")).round(1).to_string())
