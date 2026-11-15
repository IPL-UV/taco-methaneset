"""Sube una colección a Hugging Face en una rama, verifica el manifiesto y prueba la
descarga desde el commit exacto antes de tocar `main`.

  python t22_subir.py --fam l89 --rama rebuild-v140            # sube y verifica
  python t22_subir.py --fam l89 --rama rebuild-v140 --a-main   # tras verificar, pasa a main

La verificación compara nombre y tamaño de cada fichero remoto contra el local, y
descarga un zip del commit fijado para cargarlo con tacoreader.
"""

import argparse
import pathlib
import random
import shutil

import tacoreader
from huggingface_hub import HfApi, hf_hub_download

REPO = "tacofoundation/methaneset"
T16 = pathlib.Path("/data/databases/METHANESET_TACOS/_v140/T16")
PRUEBA = pathlib.Path("/tmp/opencode/prueba_hf")


def manifiesto_local(carpeta: pathlib.Path) -> dict:
    return {str(p.relative_to(carpeta)): p.stat().st_size
            for p in carpeta.rglob("*") if p.is_file() and not p.name.endswith(".tmp")}


def manifiesto_remoto(api: HfApi, ruta: str, revision: str) -> dict:
    fuera = {}
    for it in api.list_repo_tree(REPO, path_in_repo=ruta, repo_type="dataset",
                                 revision=revision, recursive=True):
        if getattr(it, "size", None) is not None:
            fuera[str(it.path)[len(ruta) + 1:]] = int(it.size)
    return fuera


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fam", required=True, choices=["l89", "s2"])
    ap.add_argument("--rama", required=True)
    ap.add_argument("--a-main", action="store_true")
    ap.add_argument("--sin-subir", action="store_true", help="solo verifica la rama ya subida")
    a = ap.parse_args()

    carpeta = T16 / a.fam
    ruta = f"methaneset-{a.fam}-finetune"
    api = HfApi()
    print("usuario:", api.whoami().get("name"), flush=True)

    if not a.sin_subir:
        api.create_branch(repo_id=REPO, branch=a.rama, repo_type="dataset", exist_ok=True)
        tam = sum(p.stat().st_size for p in carpeta.rglob("*") if p.is_file()) / 1e9
        print(f"subiendo {carpeta} ({tam:.2f} GB) a {a.rama}", flush=True)
        info = api.upload_folder(
            repo_id=REPO, repo_type="dataset", folder_path=str(carpeta),
            path_in_repo=ruta, revision=a.rama,
            commit_message=f"rebuild {a.fam} multispectral finetune: clean references, "
                           f"per-band stats and collection metadata (v1.4.0)",
            ignore_patterns=["*.tmp", "*.log", "__pycache__"],
        )
        print("commit:", info.commit_url, flush=True)
        sha = info.oid
    else:
        sha = api.repo_info(REPO, repo_type="dataset", revision=a.rama).sha
    print("sha de la rama:", sha, flush=True)

    local = manifiesto_local(carpeta)
    remoto = manifiesto_remoto(api, ruta, sha)
    faltan = sorted(set(local) - set(remoto))
    sobran = sorted(set(remoto) - set(local))
    distintos = sorted(k for k in set(local) & set(remoto) if local[k] != remoto[k])
    print(f"local: {len(local)} ficheros | remoto: {len(remoto)}")
    print(f"faltan en remoto: {len(faltan)} | sobran en remoto: {len(sobran)} | "
          f"tamaño distinto: {len(distintos)}")
    for k in (faltan + sobran + distintos)[:10]:
        print("   ", k)
    ok_manifiesto = not (faltan or sobran or distintos)

    # descarga de un zip desde el commit fijado
    if PRUEBA.exists():
        shutil.rmtree(PRUEBA)
    PRUEBA.mkdir(parents=True)
    zips = sorted(p.name for p in carpeta.glob("*.tacozip"))
    elegido = random.Random(140).choice(zips)
    print("probando descarga de:", elegido, flush=True)
    bajado = hf_hub_download(REPO, f"{ruta}/{elegido}", repo_type="dataset",
                             revision=sha, local_dir=str(PRUEBA))
    print("bytes:", pathlib.Path(bajado).stat().st_size, flush=True)
    tacoreader.use("pandas")
    ds = tacoreader.load(str(PRUEBA / ruta / elegido))
    filas = len(ds.data)
    hojas = len(ds.data.read(0))
    print(f"tacoreader: {filas} muestras | {hojas} hojas en la primera")
    ok_prueba = hojas == 8 and filas > 0

    print("MANIFIESTO OK" if ok_manifiesto else "MANIFIESTO CON DIFERENCIAS",
          "| PRUEBA OK" if ok_prueba else "| PRUEBA FALLIDA", flush=True)
    if not (ok_manifiesto and ok_prueba):
        return 1

    if a.a_main:
        print("pasando a main", flush=True)
        info = api.upload_folder(
            repo_id=REPO, repo_type="dataset", folder_path=str(carpeta),
            path_in_repo=ruta, revision="main",
            commit_message=f"rebuild {a.fam} multispectral finetune (v1.4.0)",
            ignore_patterns=["*.tmp", "*.log", "__pycache__"],
        )
        print("commit en main:", info.commit_url, flush=True)
        sha_main = api.repo_info(REPO, repo_type="dataset", revision="main").sha
        main_m = manifiesto_remoto(api, ruta, sha_main)
        igual = main_m == remoto
        print("main coincide con la rama:", igual)
        if not igual:
            dif = sorted(set(main_m) ^ set(remoto))
            print("  diferencias:", dif[:10])
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
