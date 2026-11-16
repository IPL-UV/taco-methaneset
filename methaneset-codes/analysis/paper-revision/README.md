# Paper revision

Experiments added during peer review. Two questions are answered here:

1. does the band-ratio plume contrast change when the contrast is compared against an equal-area
   random mask (the review comment about the per-band figure);
2. do the 13 Sentinel-2 and 11 Landsat bands that the re-retrieval adds carry signal that the six
   bands of the original MARS-S2L release do not.

Both start from the published finetune TACOs and end in the numbers quoted in the paper.

## Environment

| Variable | Default | Meaning |
|---|---|---|
| `METHANESET_DATA` | required | folder with the seven `*.tacozip` collections, one subfolder per dataset |
| `METHANESET_WORK` | `./work` next to these scripts | cache and outputs |
| `METHANESET_LIMIT` | `0` (all) | samples per split, for a quick smoke test |

Dependencies are the `analysis` extra: `pip install -e ".[analysis]"`.

## Band contribution: from pixels to the paper numbers

Point `METHANESET_DATA` at a folder that holds the released `methaneset-s2-finetune` and
`methaneset-l89-finetune` subfolders. The numbers below were produced with version 1.4.0.

```bash
# 1. sample pixels from the finetune TACOs (target + 4 backgrounds per sample)
python ridge_01_extract.py                    # -> work/ridge_bands

# 2. same pixels, four references stored per pixel
python ridge_multiref_01_extract_multiref.py  # -> work/ridge_multiref

# 3. the comparison in the paper: six release bands vs all bands
python bench_08_mars_release.py --tag mr      # -> work/bench_nonlin/results_marsrelease_long_mr.csv

# 4. paired per-scene delta and its interval (the numbers in the paper)
python bootstrap_mars6.py mr

# 5. controls quoted in the paper
python bench_01_nonlin_vs_ridge.py --tag full
python bench_05_r0common.py --sensors s2  --tag s2c
python bench_05_r0common.py --sensors l89 --tag l89c
python bench_06_sitesplit.py --sensors s2  --tag s2site
python bench_06_sitesplit.py --sensors l89 --tag l89site
python bench_07_bootstrap.py                  # -> work/bench_nonlin/bootstrap_*.csv
```

`bench_08_mars_release.py` also accepts `--refs` and `--band-sets`, so a quick run can be limited to
the comparison the paper uses: `--refs 4 --band-sets mars6,all`.

Step 1 draws at most 500 scenes per split with a fixed seed and reads up to 2,000 pixels per sample
for each of three pixel seeds: 1,000 plume and 1,000 background, and when one class has fewer the
deficit goes to the other class, so a scene is never dropped for being plume-heavy. Step 2 re-uses that scene list, so the
two extractions draw the same pixels. Every later script reads the npz files and never touches the
imagery again, so the expensive part runs once.

### What each script is

| Script | Question |
|---|---|
| `ridge_01_extract.py` | pixel sampling, single reference |
| `ridge_multiref_01_extract_multiref.py` | pixel sampling, four references |
| `bench_01_nonlin_vs_ridge.py` | ridge vs histogram gradient boosting, band sets and reference counts |
| `bench_03_weak.py` | shared models (`fit_ridge_full`, `fit_hgb_full`); its own entry point is the weak-plume extension |
| `bench_05_r0common.py` | single reference on the four-reference pixel set, to separate bands from backgrounds |
| `bench_06_sitesplit.py` | split by `site:location_name`, no site shared between train and test |
| `bench_07_bootstrap.py` | bootstrap clustered by site over the per-scene deltas |
| `bench_08_mars_release.py` | the six release bands vs all bands |

`bench_03_weak.py` is imported by the others; its weak-plume run needs a flux table that is not kept
here, so it is a library for the rest of the folder.

### Numbers in the paper

Test split of the release, histogram gradient boosting, four references, three pixel seeds, AUC-ROC
pooled over test pixels, at most 500 scenes per split (500 test scenes in Sentinel-2, 490 in
Landsat 8/9). Measured on the published v1.4.0 collections:

| Sensor | Six release bands | All bands | Paired per-scene delta (95 % CI, by site) |
|---|---:|---:|---|
| Sentinel-2 | 0.7066 ± 0.0013 | 0.7434 ± 0.0016 | +0.0330 [+0.0246, +0.0406] |
| Landsat 8/9 | 0.7608 ± 0.0007 | 0.7983 ± 0.0003 | +0.0391 [+0.0296, +0.0492] |

`bootstrap_mars6.py` rebuilds the per-scene delta and its interval from the table
`bench_08_mars_release.py` leaves behind.

## Random-mask control for the per-band contrast

The per-band figure of the first submission was removed during review, and the text keeps only the
comparison against a control with the same area and shape as the plume mask, at random positions
inside the patch:

```bash
python compute_snr_control.py --sensor s2      # -> work/snr_control/snr_samples_s2.csv
python compute_snr_control.py --sensor l89
python summarize_snr_control.py               # real, control, ratio, paired sign test
```

`--limit N` keeps the first N samples, which is enough to check that the script runs. The numbers in
the paper are medians over the published 3,605 Sentinel-2 and 1,370 Landsat samples: SWIR-2 reaches
1.63 times its control in Sentinel-2 (B12) and 1.86 in Landsat 8/9 (B07), the Landsat thermal bands
stay at 0.86 and 0.82, the Sentinel-2 water-vapour band (B09) at 0.89, and the remaining bands
between 0.96 and 1.01.

## Smoke test

`METHANESET_LIMIT=8` runs the whole chain, bootstrap included, in about fifteen minutes on eight
scenes per split. The numbers are meaningless at that size; it is only a check that the scripts run
end to end on a laptop.
