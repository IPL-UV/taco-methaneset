# Pipeline

One folder per product. Modules are numbered and each one leaves the input of the next.

| Folder | Product | Collection |
|---|---|---|
| `emit/` | EMIT, consensus of IMEO and Carbon Mapper | `methaneset-emit` |
| `multispectral/finetune-bg/` | Sentinel-2 and Landsat 8/9 with verified plumes | `methaneset-s2-finetune`, `methaneset-l89-finetune` |
| `multispectral/pretraining/` | Sentinel-2 and Landsat 8/9, plume-free pairs | `methaneset-s2-pretraining`, `methaneset-l89-pretraining` |
| `bank/` | Projected WRF-LES plume bank | `methaneset-bank` |
| `bank-les/` | Raw 3D WRF-LES simulation cubes | `methaneset-bank-les` |
| `taco/` | The TACO builder shared by every product | all |

Paths come from `config/paths.py` (environment variables). Long runs go with `nohup`, and each script writes a log with the same name next to it:

```bash
nohup python pipeline/emit/04_nodata_scan.py > pipeline/emit/04_nodata_scan.log 2>&1 &
```
