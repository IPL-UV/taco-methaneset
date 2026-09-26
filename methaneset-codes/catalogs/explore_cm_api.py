"""Map of the Carbon Mapper public API: endpoints and STAC collections.

Clean version of cm_api_map.py from the second batch (deep). Useful to
regenerate the inventory of what CM publishes and to spot new endpoints.

Findings this script confirms (22 Aug 2026):
  - The openapi lives at /api/v1/openapi.json (the root /openapi.json gives 404).
  - Read access without a token for catalog/* and stac/*.
  - Besides the plume catalog there are: scenes with coverage
    (catalog/scenes/coverage), sources, bulk CSV/GeoJSON, map tiles,
    and a full STAC with products per level (l2b CMF ... l4a plumes).
  - Access levels (verified 22 Aug 2026): plumes, bulk and STAC are
    public; scenes and scenes/coverage return 401 without an account; tasking and
    publish_plumes are restricted. An expired token gives 401 everywhere.

Usage:  python explore_cm_api.py
Output: assets/data/carbonmapper/<date>/api_paths.txt and stac_collections.txt
"""
import datetime
import pathlib

import requests

BASE = "https://api.carbonmapper.org"
OUT = pathlib.Path(__file__).resolve().parent.parent / "assets" / "data" / "carbonmapper"


def main():
    dest = OUT / datetime.date.today().isoformat()
    dest.mkdir(parents=True, exist_ok=True)

    spec = requests.get(f"{BASE}/api/v1/openapi.json", timeout=60).json()
    paths = sorted(spec.get("paths", {}))
    (dest / "api_paths.txt").write_text("\n".join(paths) + "\n")
    publicos = [p for p in paths if p.startswith(("/api/v1/catalog", "/stac"))]
    print(f"{len(paths)} endpoints in the openapi, {len(publicos)} from catalog/STAC")
    for p in publicos:
        print("  ", p)

    cols = requests.get(f"{BASE}/api/v1/stac/collections", timeout=60).json()
    lineas = [f"{c['id']:28s} {c.get('title', '')}" for c in cols.get("collections", [])]
    (dest / "stac_collections.txt").write_text("\n".join(lineas) + "\n")
    print(f"\n{len(lineas)} STAC collections:")
    for l in lineas:
        print("  ", l)


if __name__ == "__main__":
    main()
