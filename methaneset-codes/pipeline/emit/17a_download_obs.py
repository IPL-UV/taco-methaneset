"""MODULE 17a of pipeline v2: download the OBS files missing from the split.

The OBS files (per-pixel angles, path length, etc.) feed the sensor:
columns of the metadata table (17b). Same mechanism verified in module 03:
CMR search by day + timestamp, earthaccess with ~/.netrc.

Destination: /data/databases/MARS-Hyperspectral_complement/EMIT_OBS/
Resumable. Usage:
  nohup /data/users/julio/.conda/envs/deep/bin/python code/v2/17a_download_obs.py \
        > code/v2/17a_download_obs.log 2>&1 &
"""
import pathlib
import re
import time

import earthaccess
import pandas as pd

DATA = pathlib.Path(__file__).resolve().parent.parent.parent / "assets" / "data"
OBS_ROOTS = [
    pathlib.Path("/data/databases/MARS-Hyperspectral_complement/EMIT_OBS"),
    pathlib.Path("/data/databases/METHANE_DATASETS_EMIT_NEW_TACO/OBS"),
]
DEST = OBS_ROOTS[0]
TS = re.compile(r"(\d{8}T\d{6})")
RETRIES = 3


def stream(session, url, tag):
    name = url.split("/")[-1]
    fp = DEST / name
    if fp.exists() and fp.stat().st_size > 0:
        return
    tmp = fp.with_suffix(fp.suffix + ".part")
    with session.get(url, stream=True, timeout=600) as r:
        r.raise_for_status()
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
    tmp.rename(fp)
    print(f"  {tag} ok {name} {fp.stat().st_size/1e6:.0f} MB", flush=True)


def main():
    cross_dir = sorted((DATA / "cross").iterdir())[-1]
    splits = pd.read_parquet(cross_dir / "splits.parquet")
    have = set()
    for root in OBS_ROOTS:
        for f in root.glob("*_OBS_*.nc"):
            m = TS.search(f.name)
            if m:
                have.add(m.group(1).lower())
    falta = sorted(ts for ts in splits.granule_ts if ts not in have)
    print(f"split: {len(splits)} · OBS already on disk: {len(splits)-len(falta)} · "
          f"to download: {len(falta)}", flush=True)
    if not falta:
        return

    earthaccess.login(strategy="netrc")
    session = earthaccess.get_requests_https_session()
    ok, failed = 0, []
    for i, ts in enumerate(falta, 1):
        tag = f"[{i}/{len(falta)}]"
        day = f"{ts[:4]}-{ts[4:6]}-{ts[6:8]}"
        stub = ts.upper()
        got = False
        for attempt in range(1, RETRIES + 1):
            try:
                res = earthaccess.search_data(short_name="EMITL1BRAD",
                                              temporal=(day, day), count=2000)
                urls = []
                for r in res:
                    if stub in r["meta"]["native-id"]:
                        urls += [u for u in r.data_links()
                                 if "_OBS_" in u.split("/")[-1] and u.endswith(".nc")]
                if not urls:
                    print(f"  {tag} NO OBS in CMR: {ts}", flush=True)
                    break
                for u in urls:
                    stream(session, u, tag)
                got = True
                break
            except Exception as e:
                time.sleep(10 * attempt)
                err = f"{type(e).__name__}: {e}"
        if got:
            ok += 1
        else:
            failed.append(ts)
        if i % 50 == 0:
            print(f"  {i}/{len(falta)}", flush=True)
    print(f"\ncompleted {ok}/{len(falta)} · failed {len(failed)}: {failed[:5]}")


if __name__ == "__main__":
    main()
