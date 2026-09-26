# TACO builder

The shared builder. A TACO is a hierarchy of samples whose metadata lives in Parquet, so the
catalog can be queried with SQL without opening a single raster.

| Module | Step |
|---|---|
| `contexts.py` | Inventory: one dict per scene with its files and metadata |
| `extensions.py` | The metadata namespaces as pydantic models |
| `levels.py` | Files as leaves, scenes as nested samples, root tortilla |
| `collection.py` | Identity: id, version, description, license, providers |
| `build.py` | Writes the collection and generates the README and index |

The builder computes nothing: the assets already exist and it only organizes them. Before
writing, the schema is validated, so a mismatch fails there instead of halfway through the write.
