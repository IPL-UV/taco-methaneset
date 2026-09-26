"""MODULE 20 of pipeline v2: upload the EMIT radiance to HF as it becomes ready.

Module 10b leaves each radiance as COG+128+LSB (verified there); this
script uploads it as soon as it is ready, without waiting for the whole rebuild.

It uploads in BATCHES of 8 files per commit (HF limits to 128 commits/hour) and
with 4 batches in parallel. If HF replies 429, it waits and retries the same batch.

Resumable: what is uploaded stays in /tmp/opencode/emit_radiance_uploaded.txt

Usage:
  nohup /data/users/julio/.conda/envs/deep/bin/python code/v2/20_subir_radiance.py \
        > code/v2/20_subir_radiance.out 2>&1 &
"""
import pathlib
import time
from concurrent.futures import ThreadPoolExecutor

from huggingface_hub import CommitOperationAdd, HfApi
from osgeo import gdal

gdal.UseExceptions()

ROOT = pathlib.Path("/data/databases/METHANESET_TACOS/methaneset-emit/DATA")
ESTADO = pathlib.Path("/tmp/opencode/emit_radiance_uploaded.txt")
REPO = "tacofoundation/methaneset"
MSG = "Rebuild EMIT radiance: COG + Zstandard + SNR-adaptive bit discarding"
BATCH = 8
CONCURRENT = 4
LOG = pathlib.Path(__file__).with_suffix(".log")


def log(m):
    line = f"[{time.strftime('%H:%M:%S')}] {m}"
    print(line, flush=True)
    LOG.open("a").write(line + "\n")


def subidas():
    if ESTADO.exists():
        return set(ESTADO.read_text().split())
    return set()


def listo(p):
    ds = gdal.Open(str(p))
    if ds is None:
        return False
    md = ds.GetMetadata("IMAGE_STRUCTURE")
    block = ds.GetRasterBand(1).GetBlockSize()
    ds = None
    return md.get("LAYOUT") == "COG" and block == [128, 128]


def subir_lote(paths):
    api = HfApi()
    ops = [CommitOperationAdd(
        path_in_repo=f"methaneset-emit/DATA/{p.parent.name}/radiance.tif",
        path_or_fileobj=str(p)) for p in paths]
    for intento in range(1, 9):
        try:
            api.create_commit(repo_id=REPO, repo_type="dataset",
                              operations=ops, commit_message=MSG)
            with ESTADO.open("a") as fh:
                for p in paths:
                    fh.write(p.parent.name + "\n")
            return len(paths), "ok"
        except Exception as e:  # noqa: BLE001
            if "429" in str(e):
                log(f"429 on batch of {len(paths)}, waiting 300 s (attempt {intento})")
                time.sleep(300)
            else:
                log(f"error on batch of {len(paths)}: {type(e).__name__} {str(e)[:100]}")
                time.sleep(30)
    return len(paths), "fail"


def main():
    escenas = sorted(p / "radiance.tif" for p in ROOT.iterdir()
                     if (p / "radiance.tif").exists())
    log(f"{len(escenas)} scenes | already uploaded: {len(subidas())} | batches of {BATCH}")
    t0 = time.time()
    while time.time() - t0 < 16 * 3600:
        hechas = subidas()
        if len(hechas) >= len(escenas):
            log("all uploaded")
            break
        listas = [p for p in escenas if p.parent.name not in hechas and listo(p)]
        if not listas:
            time.sleep(120)
            continue
        lotes = [listas[i:i + BATCH] for i in range(0, len(listas), BATCH)]
        with ThreadPoolExecutor(CONCURRENT) as ex:
            res = list(ex.map(subir_lote, lotes))
        ok = sum(n for n, e in res if e == "ok")
        fallos = sum(1 for _, e in res if e != "ok")
        log(f"{len(lotes)} batches: {ok} uploads, {fallos} failed batches | total {len(subidas())}/{len(escenas)}")
    log("FIN")


if __name__ == "__main__":
    main()
