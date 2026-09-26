"""Alignment of the four background references relative to the target date.

bg0..bg3 come from different Sentinel-2/Landsat-8-9 dates. This script reads
the TACO level0 metadata of the previously selected samples and reports
|date(target) - date(bg_k)| in days for k = 0..3.

Run: /data/users/julio/.conda/envs/deep/bin/python 05_ref_dates.py
"""

import glob
import io
import os
import re
import zipfile

import numpy as np
import pandas as pd

BASES = {
    "s2": "/data/databases/METHANESET_TACOS/methaneset-s2-finetune",
    "l89": "/data/databases/METHANESET_TACOS/methaneset-l89-finetune",
}
OUT = "/tmp/opencode/ridge_multiref"


def parse_date(v):
    s = str(v)
    m = re.search(r"(\d{4})-?(\d{2})-?(\d{2})", s)
    if not m:
        m = re.search(r"(\d{4})(\d{2})(\d{2})", s)
    return pd.Timestamp(f"{m.group(1)}-{m.group(2)}-{m.group(3)}") if m else pd.NaT


def main():
    for sensor, base in BASES.items():
        sel = pd.read_csv(os.path.join(OUT, f"samples_multiref_{sensor}.csv"))
        rows = []
        for path in sorted(glob.glob(os.path.join(base, "*.tacozip"))):
            z = zipfile.ZipFile(path)
            l0 = pd.read_parquet(io.BytesIO(z.read("METADATA/level0.parquet")))
            z.close()
            keep = l0[l0["id"].isin(set(sel["uuid"]))].copy()
            if keep.empty:
                continue
            keep["t_date"] = keep["target:tile"].map(parse_date)
            for k in range(4):
                col = f"bg:date{k}" if f"bg:date{k}" in keep else f"bg:sensor{k}"
                keep[f"bg{k}_date"] = keep[col].map(parse_date)
            rows.append(keep[["id", "t_date"] + [f"bg{k}_date" for k in range(4)]])
        df = pd.concat(rows).rename(columns={"id": "uuid"})
        df = sel[["uuid", "split"]].merge(df, on="uuid", how="left")
        print(f"\n=== {sensor}: n = {len(df)} selected scenes")
        for k in range(4):
            d = (df["t_date"] - df[f"bg{k}_date"]).dt.days.abs()
            print(
                f"  bg{k}: median {d.median():.0f} d, IQR {d.quantile(0.25):.0f}-{d.quantile(0.75):.0f} d, "
                f"missing {int(d.isna().sum())}"
            )
        df.to_csv(os.path.join(OUT, f"ref_date_offsets_{sensor}.csv"), index=False)


if __name__ == "__main__":
    main()
