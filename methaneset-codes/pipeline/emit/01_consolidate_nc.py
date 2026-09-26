"""MODULE 01 of the v2 pipeline: ordering the EMIT nc files.

HISTORY: the original plan was to move julio's nc files into Cesar's folders
(MARS-Hyperspectral/EMIT_FULL and EMIT_OBS), but it is IMPOSSIBLE: they are
ceayca:ceayca with 775 and julio cannot write there. Julio's decision (Aug 25):
instead, julio's store is renamed and standardized with the SAME structure as
Cesar's, so that any code can resolve a granule by looking at two twin roots:

  /data/databases/MARS-Hyperspectral/            (Cesar's, read-only)
      EMIT_FULL/  *_RAD_*.nc
      EMIT_OBS/   (no nc today)
  /data/databases/MARS-Hyperspectral_complement/ (julio's, writable)
      EMIT_FULL/  *_RAD_*.nc
      EMIT_OBS/   *_OBS_*.nc

What this script does (idempotent, never overwrites):
  1. Renames METHANE_DATASETS_EMIT_200 -> MARS-Hyperspectral_complement
     (if already renamed, it continues).
  2. Creates EMIT_FULL/ and EMIT_OBS/ inside and moves its *_RAD_*.nc and
     *_OBS_*.nc there respectively.
  3. Prints the final inventory of both roots.

Usage:
  /data/users/julio/.conda/envs/deep/bin/python code/v2/01_consolidate_nc.py \
        > code/v2/01_consolidate_nc.log 2>&1
"""
import pathlib
import shutil

OLD = pathlib.Path("/data/databases/METHANE_DATASETS_EMIT_200")
COMP = pathlib.Path("/data/databases/MARS-Hyperspectral_complement")
CESAR = pathlib.Path("/data/databases/MARS-Hyperspectral")


def move_all(src_dir, pattern, dest):
    moved = skipped = 0
    for f in sorted(src_dir.glob(pattern)):
        target = dest / f.name
        if target.exists():
            print(f"  ALREADY EXISTS, not moving: {f.name}")
            skipped += 1
            continue
        shutil.move(str(f), str(target))
        moved += 1
    print(f"{src_dir} {pattern} -> {dest.name}/: moved {moved}, skipped {skipped}")


def main():
    print("NOTE: moving into MARS-Hyperspectral/EMIT_FULL|EMIT_OBS is impossible")
    print("(ceayca:ceayca 775, julio has no write access). Julio's complement")
    print("is standardized with the same structure.\n")

    if OLD.exists() and not COMP.exists():
        OLD.rename(COMP)
        print(f"renamed: {OLD.name} -> {COMP.name}")
    elif COMP.exists():
        print(f"{COMP.name} already exists, continuing")
    else:
        raise RuntimeError("I can find neither the old folder nor the new one")

    (COMP / "EMIT_FULL").mkdir(exist_ok=True)
    (COMP / "EMIT_OBS").mkdir(exist_ok=True)
    move_all(COMP, "*_RAD_*.nc", COMP / "EMIT_FULL")
    move_all(COMP, "*_OBS_*.nc", COMP / "EMIT_OBS")

    print("\nFinal inventory:")
    for root in (CESAR, COMP):
        rad = len(list((root / "EMIT_FULL").glob("*_RAD_*.nc")))
        obs = len(list((root / "EMIT_OBS").glob("*_OBS_*.nc")))
        print(f"  {root}: EMIT_FULL {rad} RAD nc · EMIT_OBS {obs} OBS nc")
    resto = [p.name for p in COMP.iterdir() if p.name not in ("EMIT_FULL", "EMIT_OBS")]
    print(f"  other entries in {COMP.name}: {len(resto)}")
    for n in resto[:10]:
        print("   ", n)


if __name__ == "__main__":
    main()
