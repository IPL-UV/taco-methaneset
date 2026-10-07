"""Step 9 · Rebuilds the multispectral pretraining sets with the new columns.

Changes (metadata ONLY, no leaf is touched and sza/vza are not recomputed):
  - removes `detection:isplume`
  - adds `target:saa` and `target:raa` (from `gee_angles/saa_<sensor>.csv`)

Since the tacozip stores the entries STORED and the metadata goes LAST, rewriting
level0/level1/COLLECTION does not move the leaf offsets -> they are preserved as is
(verified at the end of each file).

Rebuilds the tacozip in the official folder, updates `.tacocat/level0.parquet`,
`.tacocat/COLLECTION.json` and `index.html`.

Usage:
  python 09_rebuild_pretraining.py --sensor s2 [--dry-run]
"""

from __future__ import annotations

import argparse
import io
import json
import pathlib
import zipfile

import pandas as pd

BASE = pathlib.Path(__file__).parent
ROOT = pathlib.Path("/data/databases/METHANESET_TACOS")
DATASETS = {("l89","pretraining"): "methaneset-l89-pretraining", ("s2","pretraining"): "methaneset-s2-pretraining",
            ("l89","finetune"): "methaneset-l89-finetune", ("s2","finetune"): "methaneset-s2-finetune"}

DESC = {
    "target:saa": "Solar azimuth angle of the target acquisition [deg, clockwise from north].",
    "target:raa": "Sun azimuth relative to the wind direction [deg], counterclockwise from the wind direction (matches the bank sun:raa).",
}


def patch_collection(coll: dict) -> dict:
    fs = coll.get("taco:field_schema", {})
    l0 = [f for f in fs.get("level0", []) if f[0] != "detection:isplume"]
    names = [f[0] for f in l0]
    if "target:saa" not in names:
        i = names.index("target:sza")
        l0.insert(i + 1, ["target:saa", "double", DESC["target:saa"]])
        l0.insert(i + 2, ["target:raa", "double", DESC["target:raa"]])
    fs["level0"] = l0
    coll["taco:field_schema"] = fs
    return coll


def verify_layers(zpath: pathlib.Path, infos_orig: list, l0: pd.DataFrame) -> bool:
    """The DATA entries must stay byte for byte at the SAME position in the zip.

    The TACO `internal:offset/size` point to the leaves; since we do not touch any
    leaf, its position (header_offset), size and CRC must be identical.
    """
    ok = True
    with zipfile.ZipFile(zpath) as z2:
        orig = {i.filename: i for i in infos_orig}
        for i2 in z2.infolist():
            if not i2.filename.startswith("DATA/"):
                continue
            o = orig.get(i2.filename)
            if o is None or i2.header_offset != o.header_offset or i2.file_size != o.file_size or i2.CRC != o.CRC:
                print(f"    altered leaf: {i2.filename} (off {o.header_offset if o else None}->{i2.header_offset})")
                ok = False
                break
    return ok


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sensor", required=True, choices=["l89", "s2"])
    ap.add_argument("--split", default="pretraining", choices=["pretraining", "finetune"])
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--only", default="", help="process only the files that contain this text")
    args = ap.parse_args()

    ds = DATASETS[(args.sensor, args.split)]
    folder = ROOT / ds
    cache = pd.read_csv(BASE / "gee_angles" / f"saa_{args.sensor}_{args.split}.csv")
    print(f"{ds}: {len(cache)} rows of saa/raa")

    files = sorted(folder.glob("*.tacozip"))
    if args.only:
        files = [f for f in files if args.only in f.name]
    all_files = sorted(folder.glob("*.tacozip"))
    pos = sum(len(pd.read_parquet(io.BytesIO(zipfile.ZipFile(f).read("METADATA/level0.parquet")), columns=["target:tile"])) for f in all_files[:all_files.index(files[0])]) if args.only else 0
    new_l0 = []
    for f in files:
        with zipfile.ZipFile(f) as z:
            l0 = pd.read_parquet(io.BytesIO(z.read("METADATA/level0.parquet")))
            chk = cache.iloc[pos:pos + len(l0)]
            if not (chk["target:tile"].astype(str).values == l0["target:tile"].astype(str).values).all():
                raise SystemExit(f"misaligned at {f.name} (pos {pos})")
            l0 = l0.drop(columns=[c for c in ["detection:isplume"] if c in l0.columns])
            l0["target:saa"] = chk["saa"].values
            l0["target:raa"] = chk["raa"].values
            pos += len(l0)

            if args.dry_run:
                new_l0.append(l0)
                continue

            l1 = pd.read_parquet(io.BytesIO(z.read("METADATA/level1.parquet")))
            buf = io.BytesIO()
            l0.to_parquet(buf, index=False)
            coll = patch_collection(json.loads(z.read("COLLECTION.json")))
            # NOTE: write in PHYSICAL order (header_offset), not the central directory order
            infos = sorted(z.infolist(), key=lambda i: i.header_offset)
            tmp = f.with_suffix(".tacozip.new")
            with zipfile.ZipFile(tmp, "w", zipfile.ZIP_STORED) as w:
                for info in infos:
                    name = info.filename
                    if name == "METADATA/level0.parquet":
                        data = buf.getvalue()
                    elif name == "COLLECTION.json":
                        data = json.dumps(coll).encode()
                    else:
                        data = z.read(name)
                    zi = zipfile.ZipInfo(name, date_time=info.date_time)
                    zi.compress_type = zipfile.ZIP_STORED
                    zi.external_attr = info.external_attr
                    w.writestr(zi, data)
            if not verify_layers(tmp, infos, l0):
                raise SystemExit(f"leaves moved in {tmp}")
            f.unlink()
            tmp.rename(f)
            new_l0.append(l0)
        print(f"  {f.name}: rows {len(l0)} ok")

    if args.dry_run:
        print("DRY-RUN: nothing was written")
        return

    # dataset .tacocat: level0 concatenated from ALL the files (already patched)
    tac = folder / ".tacocat"
    all_l0 = [pd.read_parquet(io.BytesIO(zipfile.ZipFile(f).read("METADATA/level0.parquet")))
              for f in all_files]
    pd.concat(all_l0, ignore_index=True).to_parquet(tac / "level0.parquet", index=False)
    cj = json.loads((tac / "COLLECTION.json").read_text())
    (tac / "COLLECTION.json").write_text(json.dumps(patch_collection(cj), indent=2, ensure_ascii=False))
    # index.html: remove the isplume row and add saa/raa after the sza row
    import re
    idx = folder / "index.html"
    t = idx.read_text()
    row_isplume = re.compile(r"\s*<tr>\s*<td class=\"field-name\">detection:isplume</td>.*?</tr>", re.S)
    t, n = row_isplume.subn("", t)
    row_sza = re.compile(r"(<tr>\s*<td class=\"field-name\">target:sza</td>.*?</tr>)", re.S)
    def make_name(name):
        return ("\n                        <tr>\n"
                f"                            <td class=\"field-name\">{name}</td>\n"
                "                            <td class=\"field-type\">double</td>\n"
                f"                            <td class=\"field-description\">{DESC[name]}</td>\n"
                "                        </tr>")
    if 'class="field-name">target:saa<' in t:
        n2 = 0
    else:
        t, n2 = row_sza.subn(lambda m: m.group(1) + make_name("target:saa") + make_name("target:raa"), t, count=1)
    idx.write_text(t)
    print(f"index.html: isplume removed ({n}), saa+raa added ({n2})")
    print(f".tacocat + index.html updated in {folder}")


if __name__ == "__main__":
    main()
