"""Names and order of the columns of the bank table. One row = one physical plume seen with one
sun position = one GeoTIFF. The user-facing explanation is in DATASET_README.md.

Prefixes
  methane:  the physical plume (emitter, wind, snapshot, scale, peak, mean height, source)
  edge:     maximum on the four edges of the domain: upwind and downwind along the wind, top and
            bottom in the image (+y up). The downwind one also carries slope, reference and width
  sun:      the sun position of this row
  array:    the GeoTIFF (size, source pixel, how much it grows or shrinks relative to the domain)
  qa:       control of the zeroing step

Whatever has the same value in every row is not a column; it goes in the TACO description and in
the GeoTIFF tags: 3000 kg/h, ppb, 20 m pixel, view weight with VZA 0 (nadir).
"""


def emisor(tipo: str, p: int) -> str:
    """a1 to a9 the area sources, p1 to p9 the point sources."""
    return ("a" if tipo == "area" else "p") + str(p)


def uid(tipo: str, p: int, ws: float, snap: int) -> str:
    """Physical plume: emitter, wind and snapshot. Example: a1_w02.0_s078."""
    return f"{emisor(tipo, p)}_w{ws:04.1f}_s{snap:03d}"


COLUMNAS = [
    "id", "path",
    "methane:plume_uid", "methane:sim_type", "methane:emitter", "methane:wind_speed",
    "methane:snapshot_index", "methane:snapshot_minute", "methane:scale_factor",
    "methane:peak_ppb", "methane:mean_height_m", "methane:width_max_m",
    "methane:source_width_m", "methane:source_height_m",
    "sun:sza", "sun:raa",
    "edge:upwind_ppb", "edge:upwind_frac",
    "edge:downwind_ppb", "edge:downwind_frac", "edge:downwind_slope", "edge:downwind_ref_ppb",
    "edge:downwind_ref_frac", "edge:downwind_width_m",
    "edge:top_ppb", "edge:top_frac", "edge:bottom_ppb", "edge:bottom_frac",
    "array:width", "array:height", "array:source_col", "array:source_row", "array:peak_ppb",
    "array:grow_upwind_px", "array:grow_downwind_px", "array:grow_top_px", "array:grow_bottom_px",
    "qa:threshold_ppb", "qa:threshold_rel", "qa:ime_before", "qa:ime_after", "qa:foreign_mass_frac",
]

# User-facing descriptions (English): they go to the TACO and its generated README.md
DESCRIPCIONES = {
    "methane:plume_uid": "Physical plume: emitter, wind speed and snapshot (e.g. a1_w02.0_s078). Shared by the 855 rows (sun positions) of the plume",
    "methane:sim_type": "LES simulation type: 'area' (area sources) or 'multi' (point sources)",
    "methane:emitter": "Emitter: a1 to a9 are area sources, from smallest to largest; p1 to p9 are point sources",
    "methane:wind_speed": "Wind speed of the LES run [m/s]",
    "methane:snapshot_index": "LES output index (one output every 30 s)",
    "methane:snapshot_minute": "Simulation minute of the snapshot (index / 2)",
    "methane:scale_factor": "Factor that converts the arbitrary WRF tracer to an emission of 3000 kg/h",
    "methane:peak_ppb": "Maximum of the plume with the sun overhead (SZA = 0) [ppb]. The same for all rows of a plume; the reference for every _frac column",
    "methane:mean_height_m": "Mass-weighted mean height of the methane [m]. The solar-path image of the plume shifts by about mean_height * tan(SZA); the nearest sun position in the bank is off by at most ~42 m * mean_height / 330 m",
    "methane:source_width_m": "Size of the emitting rectangle along x, the wind direction [m]; 20 for point sources. The source at (0, 0) is its centre",
    "methane:source_height_m": "Size of the emitting rectangle along y [m]; 20 for point sources",
    "methane:width_max_m": "Widest crosswind extent of the plume above 50 ppb on the vertical column map [m], at 3000 kg/h",
    "edge:upwind_ppb": "Maximum along the upwind edge (column 5), where the wind enters [ppb]. Below 2% of the peak in every sample",
    "edge:downwind_ppb": "Maximum along the downwind edge (column 114), where the wind leaves [ppb]. The plume is cut there; below edge:downwind_ref_ppb in every sample",
    "edge:downwind_width_m": "Crosswind extent of the plume above 50 ppb at the downwind edge [m]. At most 60 m, one EMIT pixel, in every sample: the plume closes before leaving the domain",
    "edge:downwind_slope": "Slope of log(maximum per column) over the last 600 m before the downwind edge; a negative value means that the plume fades towards it. Negative in every sample",
    "edge:downwind_ref_ppb": "Reference for the downwind edge: the lowest value of the maximum per column in the middle of the domain, columns 40 to 84 [ppb]. Every sample has edge:downwind_ppb < edge:downwind_ref_ppb",
    "edge:bottom_ppb": "Maximum along the bottom edge (row 5, columns 5 to 114, the -y side) [ppb]. Below 2% of the peak in every sample",
    "edge:top_ppb": "Maximum along the top edge (row 84, columns 5 to 114, the +y side) [ppb]. Below 2% of the peak in every sample",
    "sun:sza": "Solar zenith angle [deg]",
    "sun:raa": "Azimuth of the sun relative to the wind direction [deg], counterclockwise from +x. The solar-path image of the plume is displaced towards raa + 180",
    "array:width": "Image width [pixels of 20 m]",
    "array:height": "Image height [pixels of 20 m]",
    "array:source_col": "Column of the source pixel in the image (0 = upwind side)",
    "array:source_row": "Row of the source pixel in the image (0 = top, +y)",
    "array:peak_ppb": "Maximum of this image [ppb]. It changes with the sun and equals methane:peak_ppb when sza = 0",
    "qa:threshold_ppb": "Zero threshold of this image [ppb]: lower values were set to 0. It is the highest threshold that removes at most 0.1% of the mass, capped at 1e-4 of the image maximum",
    "qa:threshold_rel": "qa:threshold_ppb / array:peak_ppb; equal to 1e-4 when the cap applies",
    "qa:ime_before": "Sum of the image before zeroing [ppb x pixel]",
    "qa:ime_after": "Sum of the image after zeroing [ppb x pixel]; ime_after / ime_before is the fraction of mass kept",
    "qa:foreign_mass_frac": "Estimated fraction of the mass of the vertical column map that comes from the outer WRF domain: the mass in the strip upwind of the source, where the plume itself cannot reach, extended over the 110 columns of the domain. Below 0.002 in every sample",
}
for _k in ("edge:upwind", "edge:downwind", "edge:downwind_ref", "edge:top", "edge:bottom"):
    DESCRIPCIONES[f"{_k}_frac"] = (f"{_k}_ppb / methane:peak_ppb. It does not change when the plume is "
                                   "rescaled to another emission rate")
for _lado, _nombre in (("upwind", "upwind, where the wind enters"), ("downwind", "downwind, where it leaves"),
                       ("top", "top (+y)"), ("bottom", "bottom (-y)")):
    DESCRIPCIONES[f"array:grow_{_lado}_px"] = (f"Number of pixels by which the image extends beyond (+) or falls "
                                               f"short of (-) the 110 x 80 LES domain on the {_nombre} side")
