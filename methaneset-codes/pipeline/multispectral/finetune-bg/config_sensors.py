"""Config of the two finetunes (L89 and S2) for the pipeline scripts."""
from pathlib import Path

V = Path("/data/databases/METHANE_DATASETS_TACOv2")
R = Path("/data/databases/METHANESET_TACOS")

SENSORES = {
    "l89": {
        "fin": V / "methaneset-l89-finetune",
        "raw": R / "methaneset-l89-finetune-bg-raw",
        "bgprev": V / "methaneset-l89-bg-finetune",
        "id": "methaneset-l89-finetune",
        "plataformas": ["LC08", "LC09"],
        "rgn": (3, 2, 4),   # Landsat B4,B3,B5
        "swir": (5, 6),     # B6, B7
        "sensor_de_log": False,
    },
    "s2": {
        "fin": V / "methaneset-s2-finetune",
        "raw": R / "methaneset-s2-finetune-bg-raw",
        "bgprev": V / "methaneset-s2-bg-finetune",
        "id": "methaneset-s2-finetune",
        "plataformas": ["S2"],
        "rgn": (3, 2, 7),   # S2 B4,B3,B8
        "swir": (11, 12),   # B11, B12
        "sensor_de_log": True,
    },
}
