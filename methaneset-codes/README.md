# MethaneSET

Processing code for **MethaneSET**, a family of seven [TACO](https://asterisk.coop/taco/spec)-compliant
datasets for satellite-based methane plume detection: Sentinel-2, Landsat 8/9, EMIT, and a
synthetic plume bank derived from WRF-LES simulations.

The datasets are published on Hugging Face. This folder holds the pipelines that build them,
from the raw catalogs to the final TACO collections.

## The datasets

| Dataset | Samples | Size | Sensor |
|---|---:|---:|---|
| [methaneset-s2-pretraining](https://huggingface.co/datasets/tacofoundation/methaneset/tree/main/methaneset-s2-pretraining) | 56,344 | 36.9 GB | Sentinel-2 |
| [methaneset-s2-finetune](https://huggingface.co/datasets/tacofoundation/methaneset/tree/main/methaneset-s2-finetune) | 3,552 | 6.4 GB | Sentinel-2 |
| [methaneset-l89-pretraining](https://huggingface.co/datasets/tacofoundation/methaneset/tree/main/methaneset-l89-pretraining) | 21,919 | 8.9 GB | Landsat 8/9 |
| [methaneset-l89-finetune](https://huggingface.co/datasets/tacofoundation/methaneset/tree/main/methaneset-l89-finetune) | 1,353 | 1.2 GB | Landsat 8/9 |
| [methaneset-emit](https://huggingface.co/datasets/tacofoundation/methaneset/tree/main/methaneset-emit) | 721 | 1.0 TB | EMIT |
| [methaneset-bank](https://huggingface.co/datasets/tacofoundation/methaneset/tree/main/methaneset-bank) | 238,545 | 5.1 GB | synthetic |
| [methaneset-bank-les](https://huggingface.co/datasets/tacofoundation/methaneset/tree/main/methaneset-bank-les) | 1,647 | 3.1 GB | synthetic |

## Layout

```
methaneset/
├── config/            paths and parameters (example versioned, the real one ignored)
├── catalogs/          dated snapshots of the IMEO and Carbon Mapper catalogs
├── pipeline/
│   ├── emit/          hyperspectral: consensus, selection, layers, TACO build
│   ├── multispectral/ Sentinel-2 and Landsat 8/9: download, select, chips, TACO build
│   ├── bank/          WRF-LES plume bank
│   ├── bank-les/      the raw 3D simulation cubes
│   └── taco/          the TACO builder shared by every product
├── injection/         plume bank into plume-free scenes (empty for now)
├── notebooks/         end-to-end usage examples
├── analysis/          one-off analyses kept for provenance
└── docs/              pipeline description and datacard
```

Modules are numbered and each one leaves the input of the next, so a run can be resumed at any
step. Long runs go with `nohup` and write a `.log` next to the script.

## Reproducing

Requirements: Python 3.11 or newer, GDAL, and the TACO tooling.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

Every path is read from the environment, with a default for the development server, so the same
code runs on a laptop, a cluster or a notebook. See `config/paths.py` for the full list. The main
ones:

| Variable | Meaning |
|---|---|
| `METHANESET_DATA` | Root of the TACO collections and intermediate products |
| `METHANESET_LES` | WRF-LES source simulations (Zenodo) |
| `METHANESET_WRF_CONF` | WRF configuration files |
| `METHANESET_BANK_RAW`, `METHANESET_BANK_FINAL` | Plume bank working folders |
| `METHANESET_INJECTION_ASSETS` | Profiles and masks used to build the bank |

Nothing is hardcoded in the modules, and no credential lives in the repository: LP DAAC reads
`EARTHDATA_USERNAME` and `EARTHDATA_PASSWORD` from the environment or `~/.netrc`.

Example, the EMIT consensus step:

```bash
nohup python pipeline/emit/00_cross_imeo_cm.py > pipeline/emit/00_cross_imeo_cm.log 2>&1 &
```

## Data and code availability

- Data: [tacofoundation/methaneset](https://huggingface.co/datasets/tacofoundation/methaneset) on
  Hugging Face, CC-BY-NC-SA 4.0 for the observational products and CC-BY 4.0 for the plume bank.
- Code: this repository, MIT.
- Website: [ipl-uv.github.io/taco-methaneset](https://ipl-uv.github.io/taco-methaneset/).

## Citation

```bibtex
@article{contreras2026methaneset,
  title   = {MethaneSET: Unified Multi-Sensor AI-Ready Datasets for
             Satellite-Based Methane Plume Detection},
  author  = {Contreras, Julio and Giner, Carlos and Aybar, Cesar
             and Mateo-Garcia, Gonzalo and Gorrono, Javier
             and Montero, David and Mahecha, Miguel D.
             and Guanter, Luis and Gomez-Chova, Luis},
  journal = {Scientific Data},
  year    = {2026}
}
```

## License

Code is MIT. The datasets keep their own licenses, inherited from the upstream sources.
