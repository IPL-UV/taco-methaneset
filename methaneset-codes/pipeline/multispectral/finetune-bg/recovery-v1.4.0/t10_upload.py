"""T10 · subir la v1.4.0 a Hugging Face, coleccion a coleccion.

Cada coleccion es un solo commit: si la subida se corta, no queda a medias.
No toca `.tacocat` de las demas colecciones ni el README raiz del repo.

Uso:
  python v140/t10_upload.py --coleccion l89
"""
import argparse
import pathlib

from huggingface_hub import HfApi

REPO = "tacofoundation/methaneset"
T7 = pathlib.Path("/data/databases/METHANESET_TACOS/_v140/T7")

MENSAJES = {
    "l89": "recover 17 Landsat samples that never made it into the release (v1.4.0)",
    "s2": "recover 53 Sentinel-2 samples that never made it into the release (v1.4.0)",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coleccion", required=True, choices=["l89", "s2"])
    a = ap.parse_args()

    carpeta = T7 / a.coleccion
    zips = sorted(carpeta.glob("*.tacozip"))
    tam = sum(p.stat().st_size for p in carpeta.rglob("*") if p.is_file()) / 1e9
    for req in ("README.md", "index.html", ".tacocat/COLLECTION.json",
                ".tacocat/level0.parquet", ".tacocat/level1.parquet"):
        if not (carpeta / req).exists():
            raise SystemExit(f"ABORT: falta {req}")
    print(f"{a.coleccion}: {len(zips)} zips, {tam:.2f} GB", flush=True)

    api = HfApi()
    print("user:", api.whoami().get("name"), flush=True)
    api.upload_folder(
        repo_id=REPO,
        repo_type="dataset",
        folder_path=str(carpeta),
        path_in_repo=f"methaneset-{a.coleccion}-finetune",
        commit_message=MENSAJES[a.coleccion],
        ignore_patterns=["*.tmp", "*.log", "__pycache__"],
    )
    print("SUBIDA TERMINADA", flush=True)


if __name__ == "__main__":
    main()
