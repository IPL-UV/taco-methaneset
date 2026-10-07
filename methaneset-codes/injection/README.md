# Injection

Convert plume-bank enhancements into radiometric perturbations and inject
them into plume-free EMIT scenes, for controlled data augmentation.

Skeleton seeded from Carlos' working code (5 oct 2026). Modules marked
"planned" below are handled by other workstreams and do **not** exist yet.

## Layout

```
injection/
  core/
    __init__.py
    config.py        physical constants, grid, paths (paths relative to this dir)
    bank.py          MethaneBank — TACO plume-bank lookup            (CARLOS/bank.py)
    reproject.py     footprint integration of the bank onto EMIT     (CARLOS/trans_enhmap.py)
    srf.py           EMIT Gaussian SRF matrix + coverage            (CARLOS/inject.py §2)
    inject_emit.py   EMIT spectral injection                        (CARLOS/inject.py §3)
    methanex.py      MARS retrieval engine, vendored untouched       (CARLOS/methanex.py)
    geometry.py      planned: meteo<->math, RAA, VAA parallax
    lut_emit.py      planned: hyperspectral LUT, log-T, AMF
    placement.py     planned: valid EMIT tiles, windward placement, Q_max
    inject_ms.py     planned: two-band S2/L89 injection
  scripts/
    run_injection_emit.py   main pipeline (imports adjusted to core.*)
  validation/        analysis scripts from CARLOS/analysis (verbatim)
  tests/
```

## Data files

`emit_wavelengths.npy` and `emit_fwhm.npy` are copied into `core/` so that
`config.EMIT_WAVELENGTHS_FILE` / `EMIT_FWHM_FILE` resolve against
`Path(__file__).resolve().parent` and not the current working directory.

`methanex.py` still expects `ch4_emit.safetensors` next to it (core/); it
downloads it on first use.

## Verify

```bash
cd 08-codes-github/methaneset-codes/injection
/data/users/julio/.conda/envs/deep/bin/python \
  -c "import core.config, core.bank, core.reproject, core.srf, core.inject_emit; print('imports ok')"
```
