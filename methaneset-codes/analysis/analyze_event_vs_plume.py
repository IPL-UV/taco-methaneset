"""Quantifies the disconnect between EVENT and PLUME in the IMEO bank.

The `isplume` field in info.json describes the SITE (whether the logged
location was a plume, a confuser or a rejected detection), not the IMAGE. The
useful label is plumemask.tif. This script puts numbers on that difference
using the batch of 200 complete granules.

Usage:  python analyze_event_vs_plume.py
"""
import pathlib

import pandas as pd

FULLTILES = pathlib.Path("/data/users/julio/methanset/data/fulltiles_test_v4c.csv")


def main():
    ft = pd.read_csv(FULLTILES)
    eventos = int(ft["num_events"].sum())
    plumas = int(ft["num_plumes"].sum())
    vacios = int(((ft.num_events > 0) & (ft.num_plumes == 0)).sum())

    print(f"tiles in the batch      : {len(ft)}")
    print(f"declared events         : {eventos}")
    print(f"real plumes             : {plumas}")
    print(f"events without plume    : {eventos - plumas}  ({100*(eventos-plumas)/eventos:.0f}%)")
    print(f"tiles with events and 0 plumes: {vacios}  ({100*vacios/len(ft):.0f}% of the batch)")
    print()
    print("Conclusion: filtering positives by event or by isplume lets in more than")
    print("half of the images without gas. Plume pixels must be counted in plumemask.tif.")


if __name__ == "__main__":
    main()
