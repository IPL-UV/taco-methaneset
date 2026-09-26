"""Names and order of the columns of the methaneset-bank-les table. One row = one 3D volume
(49 x 90 x 120) of a physical plume in a snapshot.

Trim of 12 Sep: out with the constants, the duplicates and whatever can be derived from another
column. The constants and the conversion rules go in the TACO description and in the README, not in
the table.

Prefixes
  methane:  identity and mass (emitter, wind, snapshot, scale, peak, IME, source)
  plume:    shape in metres (mean height, width, length and height)
  edge:     maxima on the four edges of the interior domain; the downwind one carries slope,
            reference and width
  criteria: the flags of the projected bank, to know whether the volume would have made it in
  grid:     the real heights of the 49 levels
  qa:       zeros of the volume
"""

COLUMNAS = [
    "id", "path",
    "methane:sim_type", "methane:emitter", "methane:wind_speed", "methane:snapshot_index",
    "methane:scale_factor", "methane:peak_ppb",
    "plume:mean_height_m", "plume:width_max_m", "plume:length_downwind_m", "plume:height_vertical_m",
    "methane:ime_ppb_px", "methane:source_width_m", "methane:source_height_m",
    "edge:upwind_ppb", "edge:downwind_ppb", "edge:downwind_slope", "edge:downwind_ref_ppb",
    "edge:downwind_width_m", "edge:top_ppb", "edge:bottom_ppb", "edge:foreign_mass_frac",
    "edge:max_frac", "edge:touches_edge",
    "criteria:fades_out", "criteria:ent_ok", "criteria:lat_ok", "criteria:fuera_ok",
    "criteria:ancho_ok", "criteria:pasa",
    "grid:level_heights_m", "qa:zeros_frac",
]

DESCRIPCIONES = {
    "methane:sim_type": "LES simulation type: 'area' (area sources) or 'multi' (point sources)",
    "methane:emitter": "Emitter: a1 to a9 are area sources, from smallest to largest; p1 to p9 are point sources",
    "methane:wind_speed": "Geostrophic wind speed of the LES run [m/s]. The wind always blows towards +x (the right of the image), so there is no wind direction column",
    "methane:snapshot_index": "LES output index (one output every 30 s from 0 to 60 min); the valid range starts at 60 (minute 30), after the spin-up",
    "methane:scale_factor": "Factor applied to the arbitrary WRF tracer to bring the plume to 3000 kg/h",
    "methane:peak_ppb": "Maximum of the vertical column map [ppb] (the volume summed over levels). It is the reference of the criteria, which are relative to it",
    "methane:ime_ppb_px": "Integrated mass enhancement of the column map [ppb x pixel]. The IME of the plume. Total methane in mol = IME x 3.6055e5 x 400 / 1e9",
    "methane:source_width_m": "Size of the emitting rectangle along x, the wind direction [m]",
    "methane:source_height_m": "Size of the emitting rectangle along y [m]",
    "plume:mean_height_m": "Mass-weighted mean height of the methane [m], using the real grid:level_heights_m. Each level is shifted by mean_height x tan(SZA) when the plume is projected",
    "plume:width_max_m": "Widest crosswind extent of the column map above 50 ppb [m]. 50 ppb is the EMIT noise, so only the visible part of the plume is measured",
    "plume:length_downwind_m": "Downwind extent of the column map above 50 ppb [m]",
    "plume:height_vertical_m": "Vertical extent of the volume above 50 ppb [m]",
    "edge:upwind_ppb": "Maximum of the column map along the upwind edge (column 5, the first clean column), where the wind enters [ppb]",
    "edge:downwind_ppb": "Maximum of the column map along the downwind edge (column 114, the last clean column), where the wind leaves [ppb]. The plume is cut there",
    "edge:downwind_slope": "Slope of log(maximum per column) over the last 600 m before the downwind edge; negative means the plume fades towards it",
    "edge:downwind_ref_ppb": "Reference for the downwind edge: the lowest value of the maximum per column in the middle of the domain (columns 40 to 84) [ppb]",
    "edge:downwind_width_m": "Crosswind extent of the column map above 50 ppb at the downwind edge [m]. At most 60 m, one EMIT pixel, in the projected bank",
    "edge:top_ppb": "Maximum of the column map along the top edge (row 84, the +y side) [ppb]",
    "edge:bottom_ppb": "Maximum of the column map along the bottom edge (row 5, the -y side) [ppb]",
    "edge:foreign_mass_frac": "Estimated fraction of the mass of the column map that comes from the outer WRF domain, measured in the strip upwind of the source and extended over the domain. NaN when the source box leaves no upwind strip (a8, a9)",
    "edge:max_frac": "Largest of the four edge values (upwind, downwind, top, bottom) divided by methane:peak_ppb",
    "edge:touches_edge": "True when edge:max_frac is above 0.05: the plume reaches a domain boundary at more than 5% of its own peak",
    "criteria:fades_out": "Criterion of the projected bank: the plume fades before the downwind edge (edge:downwind_slope < 0 and edge:downwind_ppb < edge:downwind_ref_ppb)",
    "criteria:ent_ok": "Criterion of the projected bank: edge:upwind_ppb below 2% of the peak",
    "criteria:lat_ok": "Criterion of the projected bank: edge:top_ppb and edge:bottom_ppb below 2% of the peak",
    "criteria:fuera_ok": "Criterion of the projected bank: edge:foreign_mass_frac below 0.002",
    "criteria:ancho_ok": "Criterion of the projected bank: edge:downwind_width_m is 60 m or less",
    "criteria:pasa": "The five criteria true at once: this volume would be selected for the projected bank",
    "grid:level_heights_m": "Height above ground of the middle of each of the 49 levels [m], from (PH + PHB) / g. The levels are not exactly 20 m apart (mean 20.45 m, from 9.9 to 991 m), so the heights travel with the row",
    "qa:zeros_frac": "Fraction of cells that are exactly zero",
}
