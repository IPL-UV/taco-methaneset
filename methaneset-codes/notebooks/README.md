# MethaneSET notebooks

Three small tours, one per component, plus the injection walkthrough. Each one opens a dataset,
shows one scene, computes the retrieval the paper describes and stops there: eight figures in
total, every one with units and a scale bar. They run locally (with `METHANESET_DATA` pointing at
a downloaded copy) or in Colab, where the bootstrap cell fetches `common.py` and the data comes
from Hugging Face.

| Notebook | Dataset(s) | What you do |
|---|---|---|
| [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/IPL-UV/taco-methaneset/blob/main/methaneset-codes/notebooks/01_s2.ipynb) [01_s2](01_s2.ipynb) | `methaneset-{s2,l89}-{pretraining,finetune}` | Query the catalog, open one plume chip with its mask, reproduce the MBMP enhancement and compare it with the MARS product. Set `SENSOR = "l89"` for Landsat |
| [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/IPL-UV/taco-methaneset/blob/main/methaneset-codes/notebooks/03_emit.ipynb) [03_emit](03_emit.ipynb) | `methaneset-emit` | One granule: radiance RGB, pixel spectra, wind and elevation; the two uint64 bitmask masks; and the four retrievals (mf, rmf, mag1c, wwmf) in ppm m on one scale |
| [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/IPL-UV/taco-methaneset/blob/main/methaneset-codes/notebooks/04_bank.ipynb) [04_bank](04_bank.ipynb) | `methaneset-bank`, `methaneset-bank-les` | Query the bank by solar geometry and wind, scale a map to any emission rate and reproject a raw 3D cube, which is how new geometries are generated |
| [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/IPL-UV/taco-methaneset/blob/main/methaneset-codes/notebooks/05_injection_emit.ipynb) [05_injection_emit](05_injection_emit.ipynb) | `methaneset-bank`, `methaneset-emit` | Inject one bank plume into a radiance cube and recover it with the matched filter |

## `common.py`

Shared helpers used by the notebooks: data resolution (local folder or Hugging Face), raster
reading with nodata as NaN, plot styling, the MBMP ratio, EMIT band helpers, bitmask decoding,
bank queries and segmentation metrics. The bootstrap cell downloads it when it is missing, so the
notebooks work on Colab without cloning the repository.
