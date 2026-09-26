"""Step 6 · Joins the rows of every plume into one table.

Usage: python 06_table.py DIR   (DIR is the output of step 5)
Output: DIR/tabla.parquet and DIR/tabla.csv
"""
import sys
from pathlib import Path
import pandas as pd

d = Path(sys.argv[1])
partes = sorted((d / "filas").glob("*.parquet"))
t = pd.concat([pd.read_parquet(p) for p in partes], ignore_index=True)
t = t.sort_values(["methane:emitter", "methane:wind_speed",
                   "methane:snapshot_index", "sun:sza", "sun:raa"]).reset_index(drop=True)
t.to_parquet(d / "tabla.parquet", index=False)
t.to_csv(d / "tabla.csv", index=False)
print(f"{len(partes)} physical plumes · {len(t)} rows · {t.shape[1]} columns")
