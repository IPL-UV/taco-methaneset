"""MODULE 19e: the CM overlap bug, PER SCENE -> a warning column for users.

Julio's request (Aug 26): "a column warning in which scenes not to trust
the CM criterion too much" (the overlap: one plume registered under
two different sources, slide 'One plume, two identities').

Method: for each published scene with >=2 CM sources, download the
plume-outline.geojson files of its plumes from the CM STAC (local cache in WKT,
resumable: the workshop tifs no longer exist) and count the pairs of
plumes from DIFFERENT SOURCES with IoU >= 0.5 (same 'same plume' threshold as
the matching). That count is `detection:n_cm_dup_pairs` in level0: if > 0,
n_cm and the CM source partition of that scene are an upper bound.

It also stores WHICH sources are twins (`detection:cm_dup_list`, Julio's
request Aug 27): knowing that the scene has overlaps is not enough, one must be
able to ask "does the CM plume I match with my IMEO plume have a twin?".

Outputs: assets/data/carbonmapper/2026-08-26-outlines/outlines.parquet (cache)
         assets/data/cross/2026-08-25/cm_dup_scenes.parquet
Usage:
  nohup /data/users/julio/.conda/envs/deep/bin/python code/v2/19e_cm_dup_scenes.py \
        > code/v2/19e_cm_dup_scenes.log 2>&1 &
"""
import itertools
import json
import pathlib
import re

import numpy as np
import pandas as pd
import requests
from shapely import make_valid, wkt as shp_wkt
from shapely.geometry import shape
from shapely.ops import unary_union

ROOT = pathlib.Path(os.environ.get("METHANESET_ROOT", "/data/users/julio/Notes/01-Projects/methanset"))
CM_DIR = ROOT / "assets" / "data" / "carbonmapper" / "2026-08-25"
CROSS = ROOT / "assets" / "data" / "cross" / "2026-08-25"
CACHE_DIR = ROOT / "assets" / "data" / "carbonmapper" / "2026-08-26-outlines"
CACHE = CACHE_DIR / "outlines.parquet"
LEVEL0 = pathlib.Path("/data/databases/METHANSET_TACOS/methaneset-emit/METADATA/level0.parquet")
STAC = "https://api.carbonmapper.org/api/v1/stac/search"
IOU_DUP = 0.5  # 'same plume' threshold, consistent with the matching


def outline_wkt(pid, ses):
    r = ses.post(STAC, json={"ids": [pid], "limit": 10}, timeout=60)
    r.raise_for_status()
    feats = sorted(r.json().get("features", []),
                   key=lambda it: it.get("collection", ""), reverse=True)
    for it in feats:
        assets = it.get("assets", {})
        a = assets.get("plume-outline.geojson") or assets.get("plume.geojson")
        if a:
            gj = ses.get(a["href"], timeout=60).json()
            g = unary_union([make_valid(shape(f["geometry"]))
                             for f in gj.get("features", [gj])])
            return g.wkt
    return None


def main():
    md = pd.read_parquet(LEVEL0)
    ts_of = {re.search(r"_(\d{8}T\d{6})_", g).group(1).lower().replace("t", "t"): g
             for g in md["id"]}
    ts_of = {k.replace("t", "t"): v for k, v in ts_of.items()}
    # manifest granule_ts: '20231024t070157'
    ts_pub = {re.search(r"_(\d{8})T(\d{6})_", g).group(1)
              + "t" + re.search(r"_(\d{8})T(\d{6})_", g).group(2): g
              for g in md["id"]}

    man = pd.read_parquet(CM_DIR / "manifest.parquet")
    src = pd.read_parquet(CM_DIR / "sources.parquet")
    man = man.merge(src[["plume_id", "source_name"]], on="plume_id", how="left")
    man["source_name"] = man.source_name.fillna(man.plume_id)
    man = man[man.granule_ts.isin(ts_pub)]
    multi = man.groupby("granule_ts").source_name.nunique()
    scenes = multi[multi >= 2].index
    todo = man[man.granule_ts.isin(scenes)]
    print(f"published scenes with >=2 CM sources: {len(scenes)} · "
          f"plumes to resolve: {len(todo)}")

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = (pd.read_parquet(CACHE) if CACHE.exists()
             else pd.DataFrame(columns=["plume_id", "wkt"]))
    have = set(cache.plume_id)
    ses = requests.Session()
    rows = []
    missing = [p for p in todo.plume_id if p not in have]
    print(f"in cache: {len(have & set(todo.plume_id))} · to download: {len(missing)}")
    for i, pid in enumerate(missing, 1):
        try:
            w = outline_wkt(pid, ses)
        except Exception as e:  # network: retried in another run
            print(f"  ERROR {pid}: {e}")
            w = None
        rows.append({"plume_id": pid, "wkt": w})
        if i % 50 == 0:
            print(f"  {i}/{len(missing)}")
            pd.concat([cache, pd.DataFrame(rows)]).to_parquet(CACHE)
    cache = pd.concat([cache, pd.DataFrame(rows)]) if rows else cache
    cache.to_parquet(CACHE)
    ok = cache.dropna(subset=["wkt"])
    print(f"outlines resolved: {len(ok)}/{len(cache)}")

    geom = {r.plume_id: shp_wkt.loads(r.wkt) for r in ok.itertuples()}
    out = []
    for ts, grp in todo.groupby("granule_ts"):
        n_touch = n_dup = 0
        max_iou = 0.0
        dup_pairs = []   # WHICH sources are twins, not just how many
        for a, b in itertools.combinations(grp.itertuples(), 2):
            if a.source_name == b.source_name:
                continue
            ga, gb = geom.get(a.plume_id), geom.get(b.plume_id)
            if ga is None or gb is None or not ga.intersects(gb):
                continue
            iou = ga.intersection(gb).area / ga.union(gb).area
            n_touch += 1
            max_iou = max(max_iou, iou)
            if iou >= IOU_DUP:
                n_dup += 1
                dup_pairs.append([a.source_name, b.source_name, round(iou, 3)])
        # overlaps are NOT only pairs: there are triplets and even groups of 6
        # sources describing a single plume (Julio, Aug 27). They are grouped by
        # connected components so the user reads GROUPS, not loose pairs.
        parent = {}

        def find(x):
            parent.setdefault(x, x)
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for a, b, _ in dup_pairs:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb
        groups = {}
        for n in list(parent):
            groups.setdefault(find(n), []).append(n)
        dup_groups = sorted((sorted(g) for g in groups.values()),
                            key=lambda g: (-len(g), g[0]))
        out.append({"id": ts_pub[ts], "n_cm_dup_pairs": n_dup,
                    "n_cm_touch_pairs": n_touch, "cm_dup_max_iou": max_iou,
                    "cm_dup_list": json.dumps(dup_pairs),
                    "n_cm_dup_groups": len(dup_groups),
                    "cm_dup_max_group": max((len(g) for g in dup_groups), default=0),
                    "cm_dup_groups": json.dumps(dup_groups)})
    dd = pd.DataFrame(out)
    dd.to_parquet(CROSS / "cm_dup_scenes.parquet")
    print(f"\nscenes evaluated: {len(dd)} · with dup (IoU>={IOU_DUP}): "
          f"{(dd.n_cm_dup_pairs>0).sum()} · total dup pairs: {dd.n_cm_dup_pairs.sum()}")
    g = dd[dd.n_cm_dup_groups > 0]
    print(f"twin groups: {int(g.n_cm_dup_groups.sum())} · "
          f"largest group: {int(dd.cm_dup_max_group.max())} sources for one plume · "
          f"scenes with a 3+ group: {int((dd.cm_dup_max_group>=3).sum())}")
    print(f"-> {CROSS/'cm_dup_scenes.parquet'}")


if __name__ == "__main__":
    main()
