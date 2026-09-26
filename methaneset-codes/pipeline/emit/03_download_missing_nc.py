"""MODULE 03 of the v2 pipeline: download the missing consensus nc files.

Reads the module 02 table (nc_coverage.parquet) and downloads from LP DAAC,
via earthaccess (~/.netrc of Earthdata), whatever is missing:

  missing RAD -> MARS-Hyperspectral_complement/EMIT_FULL/
  OBS of those same granules -> MARS-Hyperspectral_complement/EMIT_OBS/

Notes:
  - The table only carries the granule timestamp; the full id is resolved
    by searching per day in CMR and filtering by native-id containing the
    timestamp (same trick as downloader 23 of the second batch).
  - Atomic (downloads to .part and renames), resumable (skips what is already
    downloaded), 3 retries per granule.

Usage (long, with nohup and twin log):
  cd 01-Projects/methanset
  nohup /data/users/julio/.conda/envs/deep/bin/python code/v2/03_download_missing_nc.py \
        > code/v2/03_download_missing_nc.log 2>&1 &
"""
import pathlib
import time

import earthaccess
import pandas as pd

DATA = pathlib.Path(__file__).resolve().parent.parent.parent / "assets" / "data"
COMP = pathlib.Path("/data/databases/MARS-Hyperspectral_complement")
DEST = {"RAD": COMP / "EMIT_FULL", "OBS": COMP / "EMIT_OBS"}
SHORT_NAME = "EMITL1BRAD"
RETRIES = 3


def stream(session, url, dest_dir, tag):
    name = url.split("/")[-1]
    fp = dest_dir / name
    if fp.exists() and fp.stat().st_size > 0:
        print(f"  {tag} already exists  {name}", flush=True)
        return True
    tmp = fp.with_suffix(fp.suffix + ".part")
    t0 = time.time()
    with session.get(url, stream=True, timeout=600) as r:
        r.raise_for_status()
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
    tmp.rename(fp)
    gb = fp.stat().st_size / 1e9
    print(f"  {tag} ok  {name}  {gb:.2f} GB in {time.time()-t0:.0f}s", flush=True)
    return True


def main():
    cross_dir = sorted((DATA / "cross").iterdir())[-1]
    cov = pd.read_parquet(cross_dir / "nc_coverage.parquet")
    falta = cov[cov.rad_path == ""].granule_ts.tolist()
    print(f"coverage used: {cross_dir.name} · missing RAD: {len(falta)}", flush=True)
    if not falta:
        print("nothing to download")
        return

    earthaccess.login(strategy="netrc")
    session = earthaccess.get_requests_https_session()

    ok, failed = 0, []
    for i, ts in enumerate(sorted(falta), 1):
        tag = f"[{i}/{len(falta)}]"
        day = f"{ts[:4]}-{ts[4:6]}-{ts[6:8]}"
        stub = ts.upper()
        got = False
        for attempt in range(1, RETRIES + 1):
            try:
                results = earthaccess.search_data(
                    short_name=SHORT_NAME, temporal=(day, day), count=2000)
                match = [r for r in results if stub in r["meta"]["native-id"]]
                if not match:
                    print(f"  {tag} NOT FOUND in CMR: {ts}", flush=True)
                    break
                urls = []
                for r in match:
                    urls += [u for u in r.data_links() if u.endswith(".nc")]
                for kind in ("RAD", "OBS"):
                    # filter by file NAME: the whole URL contains the
                    # RAD granule folder and would match the OBS as well
                    for u in [u for u in urls if f"_{kind}_" in u.split("/")[-1]]:
                        stream(session, u, DEST[kind], f"{tag} {kind}")
                got = True
                break
            except Exception as e:
                wait = 10 * attempt
                print(f"  {tag} attempt {attempt} failed ({e}); retrying in {wait}s",
                      flush=True)
                time.sleep(wait)
        if got:
            ok += 1
        else:
            failed.append(ts)

    print(f"\ncompleted {ok}/{len(falta)} granules · failed: {len(failed)}")
    for f in failed:
        print("  failed:", f)
    print("rerunning this script retries only what is missing (resumable)")


if __name__ == "__main__":
    main()
