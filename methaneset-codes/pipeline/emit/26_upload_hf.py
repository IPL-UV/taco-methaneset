"""MODULE 26: upload the TACO to Hugging Face (tacofoundation/methaneset).

Context (Julio, Aug 27): the previous version had ended up nested as
`methaneset-emit/methaneset-emit/`. That whole folder was deleted (commit
"remove methaneset-emit", the other 5 datasets in the repo intact) and now ours
is uploaded at the root, as `methaneset-emit/`.

Path trick: `upload_large_folder` does not accept `path_in_repo`, it uploads the
content of the folder you give it. Since /data/databases/METHANSET_TACOS/ already
contains ONLY methaneset-emit/, pointing there reproduces exactly the
desired structure and does NOT nest again.

It is resumable: if it is cut off, relaunch it and it continues where it was (it
uses a cache in .cache/huggingface inside the folder).

Usage:
  nohup /data/users/julio/.conda/envs/deep/bin/python code/v2/26_upload_hf.py \
        > code/v2/26_upload_hf.log 2>&1 &
"""
import pathlib
import sys

from huggingface_hub import HfApi, upload_large_folder

REPO = "tacofoundation/methaneset"
PARENT = pathlib.Path("/data/databases/METHANSET_TACOS")
TACO = PARENT / "methaneset-emit"
WORKERS = 12


def main():
    api = HfApi()
    who = api.whoami()
    print(f"user: {who.get('name')}", flush=True)

    # the parent folder must contain ONLY the taco, or garbage would be uploaded
    hijos = [p.name for p in PARENT.iterdir() if not p.name.startswith(".")]
    if hijos != ["methaneset-emit"]:
        sys.exit(f"ABORT: {PARENT} contains {hijos}, expected only "
                 "['methaneset-emit']")
    for req in ("COLLECTION.json", "METADATA", "DATA", "README.md"):
        if not (TACO / req).exists():
            sys.exit(f"ABORT: missing {req} in {TACO}")
    n = sum(1 for p in TACO.rglob("*") if p.is_file())
    print(f"to upload: {n} files from {TACO}", flush=True)

    upload_large_folder(
        repo_id=REPO,
        folder_path=str(PARENT),
        repo_type="dataset",
        num_workers=WORKERS,
        print_report=True,
        print_report_every=120,
    )
    print("UPLOAD FINISHED", flush=True)
    files = api.list_repo_files(REPO, repo_type="dataset")
    emit = [f for f in files if f.startswith("methaneset-emit/")]
    anidado = [f for f in emit if f.startswith("methaneset-emit/methaneset-emit/")]
    print(f"in the repo: {len(emit)} files under methaneset-emit/")
    print(f"nested (should be 0): {len(anidado)}")
    print("repo folders:", sorted({f.split('/')[0] for f in files}))


if __name__ == "__main__":
    main()
