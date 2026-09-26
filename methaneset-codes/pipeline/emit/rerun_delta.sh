#!/bin/bash
# Delta rerun after the IMEO flux filter (Aug 26): re-splits,
# layer generation for the new scenes (all resumable), masks,
# OBS, full verification, figures and the final table. Each module with its own log.
set -e
B=/data/users/julio/Notes/01-Projects/methanset/code/v2
P=/data/users/julio/.conda/envs/deep/bin/python
run() { echo "== $1 =="; $P $B/$1 > $B/${1%.py}.log 2>&1; tail -2 $B/${1%.py}.log; }
run 07_propose_splits.py
run 10_generate_radiance.py
run 11_generate_latlon_glt.py
run 13_generate_wind.py
run 12_generate_elevation.py
run 14_generate_retrievals.py
run 15a_download_cm_tifs.py
run 15b_cm_sources.py
run 15_generate_masks.py
run 17a_download_obs.py
run 16_verify_samples.py
run 06b_fig_before_after.py
run 06c_flux_vs_pixels.py
run 06d_map_selection.py
run 17b_build_metadata.py
echo "== FULL RERUN COMPLETE =="
