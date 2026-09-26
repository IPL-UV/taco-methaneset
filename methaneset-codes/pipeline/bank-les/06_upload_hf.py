"""Step 6 · Uploads (or updates) methaneset-bank and methaneset-bank-les to Hugging Face.

Follows the pattern of `code/v2/26_upload_hf.py`: `upload_large_folder` from the parent folder, so
that the path in the repo is the folder name. With `allow_patterns` it is limited to the two TACOs
and the raw is not uploaded. It is resumable: if it stops, relaunch it and it continues.

Usage:
  python 06_upload_hf.py                 deletes methaneset-bank/ from the repo and uploads the two TACOs
  python 06_upload_hf.py --no-borrar     uploads without deleting anything
  nohup python 06_upload_hf.py > subida.log 2>&1 &
"""
import argparse
import pathlib
import sys

from huggingface_hub import HfApi, upload_large_folder

REPO = "tacofoundation/methaneset"
PARENT = pathlib.Path("/data/databases/METHANESET_TACOS")
TACOS = ("methaneset-bank", "methaneset-bank-les")
WORKERS = 12


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-borrar", action="store_true")
    ap.add_argument("--borrar", nargs="*", default=["methaneset-bank"])
    a = ap.parse_args()
    api = HfApi()
    print("user:", api.whoami().get("name"), flush=True)
    for t in TACOS:
        d = PARENT / t
        if not d.exists():
            sys.exit(f"ABORT: missing {d}")
        for req in ("README.md", "index.html", ".tacocat"):
            if not (d / req).exists():
                sys.exit(f"ABORT: missing {req} in {d}")
    if not a.no_borrar:
        for folder in a.borrar:
            print(f"deleting {folder}/ from the repo...", flush=True)
            api.delete_folder(folder, repo_id=REPO, repo_type="dataset",
                              commit_message=f"clear {folder} before the update")
    patrones = [f"{t}/**" for t in TACOS]
    print("uploading patterns:", patrones, flush=True)
    upload_large_folder(
        repo_id=REPO,
        folder_path=str(PARENT),
        repo_type="dataset",
        allow_patterns=patrones,
        num_workers=WORKERS,
        print_report=True,
        print_report_every=120,
    )
    print("UPLOAD FINISHED", flush=True)
    files = api.list_repo_files(REPO, repo_type="dataset")
    for t in TACOS:
        print(t, len([f for f in files if f.startswith(t + "/")]), "files")


if __name__ == "__main__":
    main()
