# MethaneSET

Public repository of the MethaneSET project. It hosts the project website and the processing code
that builds the datasets.

- Website: https://ipl-uv.github.io/taco-methaneset/
- Datasets: https://huggingface.co/datasets/tacofoundation/methaneset
- TACO specification: https://asterisk.coop/taco/spec

## What is in here

```
taco-methaneset/
├── src/, public/, patches/   the website (Astro)
├── scripts/                  site helpers
└── methaneset-codes/         the pipelines that build the datasets
```

The interesting part for other people is **`methaneset-codes/`**: the pipelines that produce the
seven TACO collections, from the raw catalogs to the final datasets. Start with
[`methaneset-codes/README.md`](methaneset-codes/README.md), and see
[`methaneset-codes/docs/pipeline.md`](methaneset-codes/docs/pipeline.md) for the end-to-end
description and the reasoning behind each decision.

```bash
cd methaneset-codes
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

Paths come from the environment, so the same code runs on a server, a laptop or a notebook.

## The datasets

| Dataset | Samples | Size | Sensor |
|---|---:|---:|---|
| methaneset-s2-pretraining | 56,344 | 36.9 GB | Sentinel-2 |
| methaneset-s2-finetune | 3,552 | 6.4 GB | Sentinel-2 |
| methaneset-l89-pretraining | 21,919 | 8.9 GB | Landsat 8/9 |
| methaneset-l89-finetune | 1,353 | 1.2 GB | Landsat 8/9 |
| methaneset-emit | 721 | 1.0 TB | EMIT |
| methaneset-bank | 238,545 | 5.1 GB | synthetic |
| methaneset-bank-les | 1,647 | 3.1 GB | synthetic |

All of them are TACO-compliant: Parquet catalogs with Cloud-Optimized GeoTIFFs.

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

Code is MIT. The datasets keep their own licenses, inherited from the upstream sources:
CC-BY-NC-SA 4.0 for the observational products and CC-BY 4.0 for the plume bank.
