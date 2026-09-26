#!/bin/bash
set -e
PY=/data/users/ceayca/.conda/envs/majortom/bin/python
cd /tmp/opencode/pre

echo "=== s2 build $(date) ==="
$PY 05_build_taco.py --ds methaneset-s2-pretraining --salida /data/databases/METHANESET_TACOS/methaneset-s2-pretraining

echo "=== l89 prepare $(date) ==="
$PY 01_prepare_table.py l89

echo "=== l89 ref scan $(date) ==="
$PY scan_ref_nulls.py l89

echo "=== l89 final table $(date) ==="
$PY 01_prepare_table.py l89

echo "=== l89 build $(date) ==="
$PY 05_build_taco.py --ds methaneset-l89-pretraining --salida /data/databases/METHANESET_TACOS/methaneset-l89-pretraining

echo "ALL_OK $(date)"
