"""Step 2 · Joins the rows of every plume into the raw table.

Input: raw/filas/*.parquet (one per volume). Output: raw/tabla.parquet and raw/tabla.csv.
"""
import pandas as pd
from core import config as C
from core.schema import COLUMNS


def main():
    partes = sorted((C.RAW / "filas").glob("*.parquet"))
    if not partes:
        raise SystemExit(f"no rows in {C.RAW / 'filas'}")
    t = pd.concat([pd.read_parquet(f) for f in partes], ignore_index=True)[COLUMNS]
    t.to_parquet(C.RAW / "tabla.parquet", index=False)
    t.to_csv(C.RAW / "tabla.csv", index=False)
    print(f"{len(t)} rows and {len(t.columns)} columns from {t['id'].nunique()} volumes -> {C.RAW}")


if __name__ == "__main__":
    main()
