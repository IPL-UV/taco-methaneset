"""Analyzes the naming mess and the revisits in the IMEO HF repo.

Distinguishes two things that look the same:
  - same location on several dates  -> normal, it is a time series, but it
    forces splits to be cut by location and not by tile
  - same plume in several folders    -> that is a duplicate (see the geo IoU
    analysis in code_to_evaluate_more_emit/16_dedup_pool_iou.py)

Usage:  python analyze_hf_naming.py
"""
import collections
import re

from inventory_hf_manifest import lfs_paths

PATRON = re.compile(r"^(EMIT_L1B_RAD_001_(\d{8})T\d{6}_\d+_\d+)_(.+)$")


def familia(loc):
    if loc.startswith("EMIT_CH4_PlumeComplex"):
        return "NASA JPL catalog"
    if re.match(r"^EMIT_\d+$", loc):
        return "IMEO numbering"
    if re.match(r"^[A-Z]{3}_", loc):
        return "ISO3 country code"
    return "hand-made, no pattern"


def main():
    tiles = sorted({p.split("/")[1] for p in lfs_paths()
                    if p.startswith("EMIT/") and p.count("/") > 1})
    por_loc = collections.defaultdict(list)
    granules = set()
    for t in tiles:
        m = PATRON.match(t)
        if not m:
            continue
        granules.add(m.group(1))
        por_loc[m.group(3)].append(m.group(2))

    print(f"tiles         : {len(tiles)}")
    print(f"granules      : {len(granules)}")
    print(f"locations     : {len(por_loc)}")

    repetidas = {k: v for k, v in por_loc.items() if len(v) > 1}
    print(f"\nlocations in several granules: {len(repetidas)} "
          f"({100*len(repetidas)/len(por_loc):.0f}%)")
    print("the most repeated (they are different dates, not duplicates):")
    for k, v in sorted(repetidas.items(), key=lambda x: -len(x[1]))[:5]:
        print(f"   {k:30s} {len(v)} tiles, {len(set(v))} unique dates, "
              f"{min(v)}..{max(v)}")

    print("\nname families:")
    for k, v in collections.Counter(familia(l) for l in por_loc).most_common():
        print(f"   {k:22s} {v}")


if __name__ == "__main__":
    main()
