"""download_backgrounds.py — downloads the per-site background POOL of a methane TACO.

Works for BOTH TACOs (Landsat 8/9 and Sentinel-2): the sensor is auto-detected per
row from satellite:platform, so you run the SAME command pointing at either TACO.

Core idea
---------
For each site (FOLDER row = scene with a plume) it downloads a pool of PAST images
of the SAME site without a plume ("background"), as a reference of how the site looks
without methane. The pool already comes filtered by clouds/nodata on the GEE server;
the final fine filter (methane enhancement, similarity ranking) runs AFTER, on the
downloaded pixels.

Key decisions
-------------
1. EXACT TACO GEOTRANSFORM. The RasterTransform is rebuilt as-is from the TACO
   (stac:geotransform + stac:crs + stac:tensor_shape), it is not re-centered. That way
   backgrounds stay pixel-aligned with the plume and the stack is piled without
   resampling.

2. SAME SENSOR AS THE PLUME. LC08→LANDSAT/LC08/C02/T1, LC09→.../LC09/..., S2→
   COPERNICUS/S2_HARMONIZED (a single collection, no merge → no GEE errors).

3. ±45 DAY SEASONAL WINDOW. It walks back year by year within
   ±SEASON_DAYS around the plume's day-of-year; a raw pool is gathered and it
   stops once it reaches RAW_CAP candidates (the closest in season win).

4. CLOUD + NODATA FILTER ON THE SERVER (add_metrics).
   - coverage_pct: % of valid pixels (no nodata) in the ROI. Avoids split scenes.
   - cloud score: % of "clear" pixels in the ROI. CloudScore+ (cs_cdf) for S2,
     QA_PIXEL bits for Landsat. Cloudy scenes are discarded BEFORE downloading.
   It dedups by date (best scene per day), sorts nearest-season-first and caps
   at N. Note: LANDSAT is not "temporal" according to getAsset, so temporal
   discovery is done here by hand (filterBounds+filterDate), not via discover_images.

Usage
-----
    python scripts/download_backgrounds.py --taco /path/taco_l89 --out bg_l89 --limit 5 --dry-run
    python scripts/download_backgrounds.py --taco /path/taco_s2  --out bg_s2

Requires an environment authenticated with Earth Engine and cubexpress installed.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import datetime as dt
import pathlib
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor

import ee
import tacoreader

import cubexpress
from cubexpress.catalog.adaptive import is_rate_limit_error
from cubexpress.geo.geometry import rt_to_geometry
from cubexpress.geo.transform import RasterTransform
from cubexpress.request.row import RequestRow
from cubexpress.request.table import RequestTable


def _with_backoff(fn, *, tries: int = 6, base: float = 1.5):
    """Runs fn(); if EE answers rate-limit ('Too Many Requests'), retries with
    exponential backoff + jitter. Other errors propagate as-is."""
    for k in range(tries):
        try:
            return fn()
        except Exception as ex:  # noqa: BLE001
            if is_rate_limit_error(ex) and k < tries - 1:
                time.sleep(base * (2 ** k) + random.random())
                continue
            raise

# --------------------------------------------------------------------------- #
# Per-sensor presets. The platform of each site (satellite:platform) picks the
# preset. Each site discovers within a SINGLE collection (same sensor as the plume).
# --------------------------------------------------------------------------- #
# Bands to DOWNLOAD, in TACO tensor order (Landsat 11, S2 13). QA bands are
# not downloaded; QA_PIXEL is still used for the cloud score (reads the full asset).
LANDSAT_BANDS = ("B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B9", "B10", "B11")
S2_BANDS = ("B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B10", "B11", "B12")

SENSOR_PRESETS = {
    "LC08": {"collection": "LANDSAT/LC08/C02/T1", "earliest": dt.date(2013, 3, 18), "label": "LC08", "family": "landsat", "bands": LANDSAT_BANDS},
    "LC09": {"collection": "LANDSAT/LC09/C02/T1", "earliest": dt.date(2021, 10, 31), "label": "LC09", "family": "landsat", "bands": LANDSAT_BANDS},
    "S2A":  {"collection": "COPERNICUS/S2_HARMONIZED", "earliest": dt.date(2015, 6, 27), "label": "S2", "family": "s2", "bands": S2_BANDS},
    "S2B":  {"collection": "COPERNICUS/S2_HARMONIZED", "earliest": dt.date(2015, 6, 27), "label": "S2", "family": "s2", "bands": S2_BANDS},
    "S2C":  {"collection": "COPERNICUS/S2_HARMONIZED", "earliest": dt.date(2015, 6, 27), "label": "S2", "family": "s2", "bands": S2_BANDS},
}


def preset_for_platform(platform: str):
    """Sensor preset. S2A/S2B/S2C (and any future S2*) → same S2_HARMONIZED.
    Landsat stays explicit because LC08 and LC09 are different collections."""
    p = SENSOR_PRESETS.get(platform)
    if p is not None:
        return p
    if platform.startswith("S2"):  # S2D, etc.
        return SENSOR_PRESETS["S2A"]
    return None

# --------------------------------------------------------------------------- #
# Default parameters
# --------------------------------------------------------------------------- #
TACO_DEFAULT = "/data/databases/METHANE_DATASETS_TACOv2/methaneset-l89-pretraining"
OUT_DEFAULT = "/data/databases/METHANESET_TACOS"  # downloads root (raw)
BANDS: tuple[str, ...] | None = None  # None = all native bands of the asset
N_BACKGROUNDS = 15        # candidates to download per site (later filtered down to 3)
SEASON_DAYS = 45          # ±days around the plume's day-of-year
MAX_YEARS_BACK = 2        # years back for the seasonal fill
PLUME_BUFFER_DAYS = 5     # margin before the plume date (avoids the scene itself)
RAW_CAP = 40              # raw candidates to gather before scoring
CLEAR_MIN = 70.0          # minimum % of clear pixels in the ROI (cloud score)
COVERAGE_MIN = 90.0       # minimum % of valid pixels (no nodata) in the ROI
NWORKERS = 20             # download pool workers (one bulk download per batch)
DISCOVER_WORKERS = 8      # concurrent threads for discovery (PHASE 1)
CHUNK = 500               # sites per batch: they are discovered, then downloaded all together
EE_PROJECT = "ee-contrerasnetk"
MARS_DAYS = 120           # symmetric MARS window (±days)


# --------------------------------------------------------------------------- #
# Per-sensor cloud scores (passed to cubexpress.add_metrics)
#   They return % of CLEAR pixels over the ROI (0..100). add_metrics detects the
#   `scale` parameter and passes an adaptive scale (it does not read at native resolution).
# --------------------------------------------------------------------------- #
def s2_clear_score(image, geometry, source_ids=None, *, scale=None):
    """% clear with CloudScore+ (cs_cdf >= 0.6 = clear).

    Works on a single scene (matches by system:index) AND on a mosaic: if
    `source_ids` arrives (the merged granules), it rebuilds CloudScore+ from those
    granules and mosaics it the same way as the image, so the score is honest.
    """
    csp = ee.ImageCollection("GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED")
    if source_ids is not None:  # mosaic: rebuild cs_cdf from the source granules
        cs = csp.filter(ee.Filter.inList("system:index", ee.List(source_ids))).select("cs_cdf").mosaic()
    else:                       # single scene: by its own system:index
        cs = csp.filter(ee.Filter.eq("system:index", image.get("system:index"))).first()
        cs = ee.Image(ee.Algorithms.If(cs, cs, ee.Image.constant(0).rename("cs_cdf"))).select("cs_cdf")
    frac = cs.gte(0.6).reduceRegion(
        reducer=ee.Reducer.mean(), geometry=geometry, scale=scale or 20,
        maxPixels=int(1e9), bestEffort=True,
    ).get("cs_cdf")
    return ee.Number(ee.Algorithms.If(frac, frac, 0)).multiply(100)


def landsat_clear_score(image, geometry, *, scale=None):
    """% clear with Collection-2 QA_PIXEL (bits: 1 dilated, 2 cirrus, 3 cloud, 4 shadow)."""
    qa = image.select("QA_PIXEL")
    bad = (
        qa.bitwiseAnd(1 << 1).neq(0)
        .Or(qa.bitwiseAnd(1 << 2).neq(0))
        .Or(qa.bitwiseAnd(1 << 3).neq(0))
        .Or(qa.bitwiseAnd(1 << 4).neq(0))
    )
    clear = bad.Not().rename("clear")
    frac = clear.reduceRegion(
        reducer=ee.Reducer.mean(), geometry=geometry, scale=scale or 30,
        maxPixels=int(1e9), bestEffort=True,
    ).get("clear")
    return ee.Number(ee.Algorithms.If(frac, frac, 0)).multiply(100)


SCORE_FUNCS = {"s2": s2_clear_score, "landsat": landsat_clear_score}


# --------------------------------------------------------------------------- #
# TACO → sites
# --------------------------------------------------------------------------- #
def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(text)).strip("_")


def rt_from_taco_row(row) -> RasterTransform:
    """EXACT RasterTransform from the TACO's STAC columns.

    stac:geotransform = GDAL order [tx, sx, shear_x, ty, shear_y, sy];
    stac:tensor_shape = [n_bands, height, width].
    """
    gt = list(row["stac:geotransform"])
    tx, sx, shear_x, ty, shear_y, sy = gt[0], gt[1], gt[2], gt[3], gt[4], gt[5]
    _, height, width = (int(v) for v in row["stac:tensor_shape"])
    return RasterTransform(
        crs=str(row["stac:crs"]),
        translate_x=float(tx),
        translate_y=float(ty),
        scale_x=float(sx),
        scale_y=float(sy),
        width=int(width),
        height=int(height),
        shear_x=float(shear_x),
        shear_y=float(shear_y),
    )


def plume_date_from_row(row) -> dt.date | None:
    """Plume date from stac:time_start (can come as NaT)."""
    ts = row["stac:time_start"]
    try:
        if ts is None or str(ts) == "NaT":
            return None
        return dt.date(int(ts.year), int(ts.month), int(ts.day))
    except Exception:
        return None


def load_sites(taco_path: str, limit: int | None = None, platform_filter: str | None = None,
               only_ids: set | None = None) -> list[dict]:
    """List of per-site dicts: {id, rt, plume_date, platform, collection, earliest, family}.

    platform_filter: if given (e.g. "LC08", "LC09", "S2"), it only returns the
    sites with that sensor label. Useful to run L8/L9/S2 separately.
    """
    tacoreader.use("pandas")
    taco = tacoreader.load(taco_path)
    table = taco.data

    # taco.data is a TacoDataFramePandas (not a regular DataFrame): it has no
    # iterrows/columns, but it does have len and iloc. We iterate by position.
    sites: list[dict] = []
    for idx in range(len(table)):
        row = table.iloc[idx]
        if str(row.get("type", "FOLDER")) != "FOLDER":
            continue

        platform = str(row.get("satellite:platform") or "").strip()
        preset = preset_for_platform(platform)
        if preset is None:
            print(f"  [skip] idx={idx}: unknown platform {platform!r}")
            continue
        if platform_filter is not None and preset["label"] != platform_filter:
            continue

        plume_date = plume_date_from_row(row)
        if plume_date is None:
            print(f"  [skip] idx={idx}: no plume date (stac:time_start NaT)")
            continue

        try:
            rt = rt_from_taco_row(row)
        except Exception as ex:
            print(f"  [skip] idx={idx}: invalid geotransform ({ex})")
            continue

        # taco_id: unique UUID per sample → folder (location_name is NOT unique).
        taco_id = _slug(row.get("id") or f"row{idx}")
        if only_ids is not None and taco_id not in only_ids:
            continue
        name = row.get("site:location_name") or row.get("majortom:code") or taco_id
        sites.append(
            {
                "taco_id": taco_id,
                "id": _slug(name),  # human-readable name for the files
                "rt": rt,
                "plume_date": plume_date,
                "platform": preset["label"],
                "collection": preset["collection"],
                "earliest": preset["earliest"],
                "family": preset["family"],
                "bands": preset["bands"],
            }
        )
        if limit is not None and len(sites) >= limit:
            break
    return sites


# --------------------------------------------------------------------------- #
# Seasonal window
# --------------------------------------------------------------------------- #
def _anchor_for_year(plume_date: dt.date, year: int) -> dt.date:
    """Same plume date in another year (Feb 29 → Feb 28 if not a leap year)."""
    try:
        return dt.date(year, plume_date.month, plume_date.day)
    except ValueError:
        return dt.date(year, plume_date.month, 28)


def season_window(plume_date: dt.date, year_off: int, earliest: dt.date):
    """Window [start, end] of ±SEASON_DAYS around the plume's doy, `year_off`
    years back. In the plume's year (year_off=0) the end is cut before the plume.
    Returns None if it is empty or starts before `earliest`.
    """
    year = plume_date.year - year_off
    anchor = _anchor_for_year(plume_date, year)
    start = anchor - dt.timedelta(days=SEASON_DAYS)
    end = anchor + dt.timedelta(days=SEASON_DAYS)
    if year_off == 0:
        end = plume_date - dt.timedelta(days=PLUME_BUFFER_DAYS)
    if start < earliest:
        start = earliest
    if end <= start:
        return None
    return start, end


# --------------------------------------------------------------------------- #
# OWN temporal discovery (bypasses the Landsat not-temporal bug)
# --------------------------------------------------------------------------- #
_BANDS_CACHE: dict[str, tuple] = {}


def collection_bands(collection: str) -> tuple:
    if collection not in _BANDS_CACHE:
        info = cubexpress.inspect_asset(collection, with_bands=True)
        _BANDS_CACHE[collection] = tuple(info.bands or ())
    return _BANDS_CACHE[collection]


def discover_dated(collection: str, rt: RasterTransform, start: str, end: str) -> list[RequestRow]:
    """Enumerates granules of an ImageCollection over the ROI in [start, end).

    Runs filterBounds+filterDate and fetches (system:index, system:time_start) in ONE
    call. Builds RequestRows with the EXACT rt and metadata['date']. It does not use
    discover_images because that one flags Landsat/C02/T1 as non-temporal.
    """
    geom = rt_to_geometry(rt)
    col = ee.ImageCollection(collection).filterBounds(geom).filterDate(start, end)

    def _feat(img):
        return ee.Feature(None, {"g": img.get("system:index"), "t": img.get("system:time_start")})

    feats = _with_backoff(lambda: col.map(_feat).getInfo())["features"]
    bands = collection_bands(collection)
    short = collection.rstrip("/").split("/")[-1]

    rows = []
    for f in feats:
        p = f["properties"]
        gran, t = p.get("g"), p.get("t")
        if gran is None or t is None:
            continue
        d = dt.datetime.fromtimestamp(t / 1000.0, tz=dt.timezone.utc).date()
        rows.append(
            RequestRow(
                id=f"{short}_{gran}",
                raster_transform=rt,
                image=f"{collection}/{gran}",
                bands=bands,
                metadata={"date": d.strftime("%Y%m%d")},
            )
        )
    return rows


# --------------------------------------------------------------------------- #
# Per-site selection
# --------------------------------------------------------------------------- #
def _row_date(r) -> dt.date | None:
    d = (r.metadata or {}).get("date")
    if not d or len(str(d)) != 8:
        return None
    s = str(d)
    return dt.date(int(s[:4]), int(s[4:6]), int(s[6:]))


def _key(r) -> tuple:
    return (str((r.metadata or {}).get("date")), str(r.image).rsplit("/", 1)[-1])


def _score(r):
    return (r.metadata or {}).get("score")


def _coverage(r):
    return (r.metadata or {}).get("coverage_pct")


def _passes(r, clear_min: float, coverage_min: float) -> bool:
    sc, cov = _score(r), _coverage(r)
    return sc is not None and sc >= clear_min and cov is not None and cov >= coverage_min


def _dedup_per_date(rows: list) -> list:
    """One scene per date: the best (score, coverage)."""
    best: dict[dt.date, object] = {}
    for r in rows:
        d = _row_date(r)
        if d is None:
            continue
        cur = best.get(d)
        rank = ((_score(r) or 0), (_coverage(r) or 0))
        if cur is None or rank > ((_score(cur) or 0), (_coverage(cur) or 0)):
            best[d] = r
    return list(best.values())


def gather_raw(site: dict) -> list[RequestRow]:
    """Raw pool: first the MARS window (tile_date ±120, before or after); if it does not
    reach RAW_CAP, it fills with the same day-of-year ±SEASON_DAYS up to MAX_YEARS_BACK
    years back (past only). Landsat 8 and 9 are searched together."""
    rt, plume_date = site["rt"], site["plume_date"]
    platform = site["platform"]
    if platform in ("LC08", "LC09"):
        collections = ("LANDSAT/LC08/C02/T1", "LANDSAT/LC09/C02/T1")
    else:
        collections = (site["collection"],)
    cutoff = plume_date - dt.timedelta(days=PLUME_BUFFER_DAYS)

    raw: dict[tuple, RequestRow] = {}

    # 1) symmetric MARS window
    s = (plume_date - dt.timedelta(days=MARS_DAYS)).isoformat()
    e = (plume_date + dt.timedelta(days=MARS_DAYS)).isoformat()
    for col in collections:
        try:
            rows = discover_dated(col, rt, s, e)
        except Exception as ex:
            print(f"      [warn] discover {col} {s}..{e}: {type(ex).__name__} {ex}")
            rows = []
        for r in rows:
            d = _row_date(r)
            if d is None or abs((d - plume_date).days) < PLUME_BUFFER_DAYS:
                continue
            raw[_key(r)] = r

    # 2) seasonal fill (past only)
    if len(raw) < RAW_CAP:
        for year_off in range(0, MAX_YEARS_BACK + 1):
            win = season_window(plume_date, year_off, site["earliest"])
            if win is None:
                continue
            ws, we = win
            for col in collections:
                try:
                    rows = discover_dated(col, rt, ws.isoformat(), we.isoformat())
                except Exception:
                    rows = []
                for r in rows:
                    d = _row_date(r)
                    if d is None or d >= cutoff:
                        continue
                    raw[_key(r)] = r
            if len(raw) >= RAW_CAP:
                break
    return list(raw.values())


def select_backgrounds(site: dict, cloud_filter: bool, clear_min: float, coverage_min: float) -> list:
    """Returns up to N (RequestRow, date), filtered by clouds/coverage and sorted
    nearest-season-first."""
    plume_date = site["plume_date"]
    raw = gather_raw(site)
    if not raw:
        return []

    # No mosaic: each granule is scored and _dedup_per_date keeps the best
    # scene per date. If a date is split across tiles, the partial scene
    # is dropped by coverage_min. (Mosaicking clashed with the package ids.)
    if cloud_filter:
        scored = list(cubexpress.add_metrics(RequestTable(tuple(raw)), score_fn=SCORE_FUNCS[site["family"]]))
        good = [r for r in scored if _passes(r, clear_min, coverage_min)]
    else:
        scored = good = list(raw)

    def order_key(r):
        d = _row_date(r)
        return (plume_date.year - d.year, -d.toordinal())

    chosen = sorted(_dedup_per_date(good), key=order_key)[:N_BACKGROUNDS]

    # Fill: if the filter left fewer than N, complete with the clearest available
    # (one per new date), to avoid going back to EE. The later fine filter cleans them up.
    if cloud_filter and len(chosen) < N_BACKGROUNDS:
        used_dates = {_row_date(r) for r in chosen}
        rest = [r for r in _dedup_per_date(scored) if _row_date(r) not in used_dates]
        rest.sort(key=lambda r: -((_score(r) or 0) + (_coverage(r) or 0)))
        for r in rest:
            chosen.append(r)
            if len(chosen) >= N_BACKGROUNDS:
                break
        chosen.sort(key=order_key)

    return [(r, _row_date(r)) for r in chosen]


def build_site_rows(site: dict, taco_name: str, chosen: list) -> tuple[list, list]:
    """Builds the RequestRows of one site for the MASSIVE RequestSet.

    Each row id is its RELATIVE PATH to the output root:
        <sensor>/<taco>/<UUID>/bg{i}__{sensor}_{date}
    so a single express() call writes every file into its site folder.
    Only sensor bands are kept (Landsat 11 / S2 13).

    Returns (rows, rel_ids).
    """
    bands = tuple(BANDS or site["bands"])
    rows, rel_ids = [], []
    for i, (r, d) in enumerate(chosen):
        rid = f"{site['platform']}/{taco_name}/{site['taco_id']}/bg{i:02d}__{site['platform']}_{d.strftime('%Y%m%d')}"
        rows.append(dataclasses.replace(r, id=rid, bands=bands))
        rel_ids.append(rid)
    return rows, rel_ids


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    global N_BACKGROUNDS, SEASON_DAYS

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--taco", default=TACO_DEFAULT, help="path to the TACO")
    ap.add_argument("--out", default=OUT_DEFAULT, help="download root folder")
    ap.add_argument("--platform", default=None, choices=["LC08", "LC09", "S2"],
                    help="process only one sensor (to run L8/L9/S2 separately)")
    ap.add_argument("--limit", type=int, default=None, help="process only N sites (test)")
    ap.add_argument("--n", type=int, default=N_BACKGROUNDS, help="max backgrounds per site")
    ap.add_argument("--nworkers", type=int, default=NWORKERS, help="download pool workers")
    ap.add_argument("--discover-workers", type=int, default=DISCOVER_WORKERS,
                    help="concurrent threads for discovery (PHASE 1). Lower it if EE rate-limits")
    ap.add_argument("--chunk", type=int, default=CHUNK, help="sites per batch (discover → download together)")
    ap.add_argument("--clear-min", type=float, default=CLEAR_MIN, help="minimum %% clear in the ROI")
    ap.add_argument("--coverage-min", type=float, default=COVERAGE_MIN, help="minimum %% valid in the ROI")
    ap.add_argument("--no-cloud-filter", action="store_true", help="do NOT filter clouds/coverage")
    ap.add_argument("--overwrite", action="store_true", help="redo already downloaded sites (.done)")
    ap.add_argument("--season-days", type=int, default=SEASON_DAYS, help="±days of the seasonal window")
    ap.add_argument("--only-ids", default=None, help="txt with taco_ids (one per line): re-download ONLY those")
    ap.add_argument("--dry-run", action="store_true", help="discover and select, do NOT download")
    args = ap.parse_args()

    N_BACKGROUNDS = args.n
    SEASON_DAYS = args.season_days
    cloud_filter = not args.no_cloud_filter

    only_ids = None
    if args.only_ids:
        only_ids = {ln.strip() for ln in pathlib.Path(args.only_ids).read_text().splitlines() if ln.strip()}
        print(f"only-ids: {len(only_ids)} sites to re-download")

    ee.Initialize(
        project=EE_PROJECT,
        opt_url="https://earthengine-highvolume.googleapis.com",
    )

    # Output layout: <out>/<sensor>/<taco_id>/<site>/*.tif
    out_root = pathlib.Path(args.out)
    taco_name = pathlib.Path(str(args.taco).rstrip("/")).name
    out_root.mkdir(parents=True, exist_ok=True)
    log_path = out_root / f"selection_log_{taco_name}{('_' + args.platform) if args.platform else ''}.csv"
    log_rows = []

    sites = load_sites(args.taco, limit=args.limit, platform_filter=args.platform, only_ids=only_ids)
    by_platform: dict[str, int] = {}
    for s in sites:
        by_platform[s["platform"]] = by_platform.get(s["platform"], 0) + 1
    print(f"TACO: {taco_name}   sites: {len(sites)}  {by_platform}  cloud_filter={cloud_filter}")

    def site_dir_of(s):
        return out_root / s["platform"] / taco_name / s["taco_id"]

    # Resume: drop already finished sites (.done) from the start, unless overwrite.
    if args.overwrite:
        pending = list(sites)
    else:
        pending = [s for s in sites if not (site_dir_of(s) / ".done").exists()]
    print(f"pending: {len(pending)} / {len(sites)}   workers={args.nworkers}  batch={args.chunk}\n")

    t0 = time.time()
    n_chunks = (len(pending) + args.chunk - 1) // max(args.chunk, 1)
    for ci, start in enumerate(range(0, len(pending), args.chunk), 1):
        chunk = pending[start:start + args.chunk]

        # --- PHASE 1: discover + select the WHOLE batch, CONCURRENTLY ---
        # Each site is independent (discover + score over its own ROI), so they
        # run in parallel with a thread pool. EE getInfo is I/O, it scales well;
        # discover_dated already retries with backoff on rate-limit.
        def _select(site):
            try:
                return site, select_backgrounds(site, cloud_filter, args.clear_min, args.coverage_min), None
            except Exception as ex:  # noqa: BLE001
                return site, None, ex

        with ThreadPoolExecutor(max_workers=args.discover_workers) as pool:
            selected = list(pool.map(_select, chunk))

        all_rows = []
        chunk_sites = []  # (site, site_dir, rel_ids)
        for site, chosen, err in selected:
            if err is not None:
                print(f"  [error] {site['taco_id'][:8]} selection: {type(err).__name__} {err}")
                log_rows.append([site["taco_id"], site["id"], site["platform"], site["plume_date"], 0, "", "", ""])
                continue
            if not chosen:
                log_rows.append([site["taco_id"], site["id"], site["platform"], site["plume_date"], 0, "", "", ""])
                continue

            log_rows.append(
                [site["taco_id"], site["id"], site["platform"], site["plume_date"], len(chosen),
                 "|".join(d.strftime("%Y%m%d") for _, d in chosen),
                 "|".join(f"{(_score(r) or 0):.0f}" for r, _ in chosen),
                 "|".join(str(r.image) for r, _ in chosen)]  # GEE id of each background
            )
            if args.dry_run:
                continue

            site_dir = site_dir_of(site)
            site_dir.mkdir(parents=True, exist_ok=True)  # pre-create for the bulk express
            rows, rel_ids = build_site_rows(site, taco_name, chosen)
            all_rows.extend(rows)
            chunk_sites.append((site, site_dir, rel_ids))

        # --- PHASE 2: ONE single bulk download of the whole batch (saturated pool) ---
        if not args.dry_run and all_rows:
            table = RequestTable(tuple(all_rows))
            result = cubexpress.express(table, out_root, nworkers=args.nworkers,
                                        overwrite=args.overwrite, verbose=True)
            # .done per site whose files ALL came out
            done_now = 0
            for site, site_dir, rel_ids in chunk_sites:
                if all(rid in result.paths for rid in rel_ids):
                    (site_dir / ".done").write_text("ok\n")
                    done_now += 1
            print(f"  batch {ci}/{n_chunks}: {len(all_rows)} imgs, {result.n_succeeded} ok, "
                  f"{result.n_failed} failed, {done_now}/{len(chunk_sites)} sites .done")

        # --- progress / ETA by processed sites ---
        processed = min(start + args.chunk, len(pending))
        rate = (time.time() - t0) / max(processed, 1)
        eta_h = rate * (len(pending) - processed) / 3600.0
        print(f"  ··· {processed}/{len(pending)} sites  {rate:.1f}s/site  ETA ~{eta_h:.1f} h\n")

    with open(log_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["taco_id", "site_name", "platform", "plume_date", "n_backgrounds", "background_dates", "clear_scores", "background_gee_ids"])
        w.writerows(log_rows)
    print(f"\nSelection log: {log_path}")


if __name__ == "__main__":
    main()
