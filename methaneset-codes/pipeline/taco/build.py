"""Steps 7 and 8: write the TACO in FOLDER format and its documentation.

PREPARED, NOT RUN YET: INVENTARIO, ASSETS_DIR and OUTPUT still need to be set in
config.py, and it must run in an environment with tacotoolbox >= 0.25.0.

Order of main():
  1. load contexts             (contexts.load_contexts)
  2. check assets on disk      (contexts.check_context, fails early)
  3. build the hierarchy       (levels.build_root)
  4. identity                  (collection.build_taco)
  5. validate the schema       (taco.tortilla.export_metadata)
  6. write FOLDER              (tacotoolbox.create)
  7. COLLECTION.json           (if create does not consolidate it)
  8. README.md + index.html    (generate_markdown / generate_html)
"""
import json

import tacotoolbox
from tacotoolbox import create, generate_html, generate_markdown

from collection import build_taco
from config import BUILD, OUTPUT
from contexts import check_context, load_contexts
from levels import build_root


def main():
    contexts = load_contexts(limit=BUILD["sample_limit"])
    print(f"{len(contexts)} contexts")

    faltantes = {c["id"]: m for c in contexts if (m := check_context(c))}
    if faltantes:
        raise SystemExit(f"missing assets in {len(faltantes)} scenes: "
                         f"{list(faltantes)[:3]} ...")

    root = build_root(contexts)
    taco = build_taco(root)

    if BUILD["validate_schema"]:
        taco.tortilla.export_metadata()   # blows up here if the schema does not fit

    paths = create(
        taco=taco,
        output=str(OUTPUT),
        output_format=BUILD["output_format"],   # "folder"
        consolidate=BUILD["consolidate"],
    )
    print("written:", paths)

    collection_json = OUTPUT.parent / "COLLECTION.json"
    if not collection_json.exists():
        collection_json.write_text(
            json.dumps(taco.model_dump(exclude={"tortilla"}, mode="json"),
                       indent=2, default=str))

    generate_markdown(input=collection_json, output=OUTPUT.parent / "README.md")
    generate_html(input=collection_json, output=OUTPUT.parent / "index.html")


if __name__ == "__main__":
    main()
