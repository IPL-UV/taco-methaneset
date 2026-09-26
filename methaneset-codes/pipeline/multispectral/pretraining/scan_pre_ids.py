"""Pretraining rescan, saving the ids whose target is all-null."""
import shutil, tempfile, time, zipfile
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np, rasterio

BASE = Path("/data/databases/METHANE_DATASETS_TACOv2")
LOG = Path("/tmp/opencode/scan_pre_ids.log")


def log(m):
    line = f"[{time.strftime('%H:%M:%S')}] {m}"
    print(line, flush=True)
    LOG.open("a").write(line + "\n")


def scan_zip(args):
    ds, zpath = args
    zpath = Path(zpath)
    ids = []
    tmp = Path(tempfile.mkdtemp(prefix="sbn_"))
    try:
        with zipfile.ZipFile(zpath) as z:
            for n in z.namelist():
                if not n.endswith("/target"):
                    continue
                sid = n.split("/")[-2]
                (tmp / "x.tif").write_bytes(z.read(n))
                try:
                    with rasterio.open(tmp / "x.tif") as r:
                        a = r.read()
                        nod = r.nodata
                        if nod is not None and (a == nod).all():
                            ids.append(sid)
                except Exception as e:
                    ids.append(f"ERR:{sid}:{e}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return zpath.name, ids


def main():
    for ds in ("methaneset-s2-pretraining", "methaneset-l89-pretraining"):
        zips = sorted((BASE / ds).glob("*.tacozip"))
        log(f"{ds}: {len(zips)} zips")
        total = 0
        all_ids = []
        with ProcessPoolExecutor(max_workers=24) as ex:
            futs = {ex.submit(scan_zip, (ds, str(z))): z.name for z in zips}
            for f in as_completed(futs):
                name, ids = f.result()
                all_ids.extend(ids)
                total += 1
                log(f"{ds} [{total}/{len(zips)}] {name}: {len(ids)} nulls")
        out = Path(f"/tmp/opencode/excluir_pre_{ds}.txt")
        out.write_text("\n".join(all_ids) + "\n")
        log(f"{ds}: {len(all_ids)} all-null targets -> {out}")
    log("DONE")


if __name__ == "__main__":
    main()
