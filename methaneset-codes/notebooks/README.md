# MethaneSET notebooks

Four small, self-contained tours. Each one opens a dataset, shows what is inside without
downloading imagery, reads a sample layer by layer, and finishes with one honest task you can
build on. They run locally (with `METHANESET_DATA` pointing at a downloaded copy) or directly in
Colab, where the bootstrap cell fetches `common.py` and the data comes from Hugging Face.

| Notebook | Dataset(s) | What you do |
|---|---|---|
| [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/IPL-UV/taco-methaneset/blob/main/methaneset-codes/notebooks/01_s2.ipynb) [01_s2](01_s2.ipynb) | `methaneset-s2-{pretraining,finetune}` | Schema and EDA, anatomy of a sample, reproduce the MBMP enhancement from B11/B12, minimal pixel classifier respecting the split |
| [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/IPL-UV/taco-methaneset/blob/main/methaneset-codes/notebooks/02_l89.ipynb) [02_l89](02_l89.ipynb) | `methaneset-l89-{pretraining,finetune}` | The same tour for Landsat 8/9 with the B6/B7 pair |
| [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/IPL-UV/taco-methaneset/blob/main/methaneset-codes/notebooks/03_emit.ipynb) [03_emit](03_emit.ipynb) | `methaneset-emit` | Radiance cube, the three matched filters, decoding the uint64 bitmask plumes, IMEO vs Carbon Mapper IoU, threshold baseline |
| [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/IPL-UV/taco-methaneset/blob/main/methaneset-codes/notebooks/04_bank.ipynb) [04_bank](04_bank.ipynb) | `methaneset-bank`, `methaneset-bank-les` | Query the bank by solar geometry and wind, scale a map to any emission rate, rebuild a column map from the 49 LES levels |

## `common.py`

Shared helpers used by the four notebooks: data resolution (local folder or Hugging Face),
raster reading with nodata as NaN, plot styling, the MBMP ratio, EMIT band helpers, bitmask
decoding and segmentation metrics. The bootstrap cell downloads it when it is missing, so the
notebooks work on Colab without cloning the repository.

The full engineering of the datasets is in [`../pipeline`](../pipeline); the citation and the
dataset cards are in the [repository README](../../README.md).
