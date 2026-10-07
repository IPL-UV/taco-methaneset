"""MODULE 18: assemble the definitive methaneset-emit TACO (FOLDER).

A single module, three ingredients (all verified beforehand):
  - samples: METHANSET_TACOS/emit/<granule>/ (721 x 10 layers)
  - table:   cross/<date>/metadata_level0.parquet (721 x 59)
  - collection: the v2 identity defined here (includes the DESCRIPTIONS of
    all columns: it is the official documentation of the dataset)

Key details:
  - HARDLINKS instead of copies: tacotoolbox copies with shutil.copy2 (+1.2 TB);
    it is patched to os.link (same filesystem): zero extra bytes, instant
    assembly. Safe: final artifacts are read-only.
  - Runs with the majortom environment (tacotoolbox 0.26.9):
      /data/users/ceayca/.conda/envs/majortom/bin/python

Output: /data/databases/METHANSET_TACOS/methaneset-emit/
Optional numeric argument = scene limit (test).

Usage:
  /data/users/ceayca/.conda/envs/majortom/bin/python code/v2/18_build_taco.py 3 \
        > code/v2/18_build_taco.log 2>&1          # test
  nohup /data/users/ceayca/.conda/envs/majortom/bin/python code/v2/18_build_taco.py \
        > code/v2/18_build_taco.log 2>&1 &        # full
"""
import os
import pathlib
import shutil
import sys

import pandas as pd
import pyarrow as pa

import tacotoolbox._writers.folder_writer as _fw
from tacotoolbox import create, generate_html, generate_markdown
from tacotoolbox.datamodel import Sample, Tortilla
from tacotoolbox.sample.datamodel import SampleExtension
from tacotoolbox.taco.datamodel import Taco, Provider, Curator, Extent
from tacotoolbox.taco.extensions.scientific import Publication, Publications

DATA = pathlib.Path(__file__).resolve().parent.parent.parent / "assets" / "data"
# samples live in the DATA of the current TACO (the emit/ workshop was deleted);
# with hardlinks, reading from there and writing the new FOLDER costs zero bytes
SRC = pathlib.Path("/data/databases/METHANSET_TACOS/methaneset-emit/DATA")
OUT = "/data/databases/METHANSET_TACOS/methaneset-emit"
LAYERS = ["radiance.tif", "latlon.tif", "glt.tif", "elevation.tif", "wind.tif",
          "mf.tif", "rmf.tif", "mag1c.tif", "plume_imeo.tif", "plume_cm.tif"]


def _link_or_copy(src, dst, **kw):
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


_fw.shutil.copy2 = _link_or_copy   # the hardlink patch

F, I, S, B = pa.float64(), pa.int64(), pa.string(), pa.bool_()
COLSPEC = {
    "detection": [
        ("n_imeo", I, "Number of distinct IMEO emitters with a plume in this granule"),
        ("n_cm", I, "Number of distinct CarbonMapper sources with a plume (upper bound: CM can register one plume under multiple source identities)"),
        ("coverage_imeo", F, "Percent of scene pixels covered by IMEO plume masks"),
        ("coverage_cm", F, "Percent of scene pixels covered by CarbonMapper plume masks"),
        ("coverage_union", F, "Percent of scene pixels covered by the union of both masks"),
        ("imeo_ids", S, "JSON list of IMEO emitter ids (source_name) present in the scene"),
        ("cm_ids", S, "JSON list of CarbonMapper source names present in the scene"),
        ("imeo_flux", S, "JSON dict IMEO emitter -> CH4 flux rate (kg/h); flux completeness enforced at selection"),
        ("cm_flux", S, "JSON dict CM source -> CH4 flux rate (kg/h); null = unquantified (structural in CM's catalog, not size-biased)"),
        ("sectors", S, "JSON dict sector -> {imeo: [...], cm: [...]}; CM IPCC codes mapped to IMEO names (97% cross-agreement on matched pairs)"),
        ("sector_list", S, "JSON list of sectors present in the scene"),
        ("sector", S, "Dominant sector by plume count"),
        ("n_imeo_weak", I, "IMEO plumes with 0 < flux < 1,000 kg/h (weak, following STARCOP)"),
        ("n_imeo_strong", I, "IMEO plumes with flux >= 1,000 kg/h (strong)"),
        ("n_cm_noflux", I, "CarbonMapper sources without quantified emission"),
        ("n_cm_dup_pairs", I, "CarbonMapper plume pairs from DIFFERENT sources overlapping with IoU >= 0.5 in this scene (duplicate identities). If > 0, treat n_cm and the CM source partition as an upper bound; IMEO is the reference catalog"),
        ("cm_dup_groups", S, "JSON list of GROUPS of CarbonMapper sources that describe the same plume in this scene (connected components of the IoU >= 0.5 overlap graph). A group can hold more than two sources: up to six in this dataset. Use it to check whether the CM source matched to your IMEO plume has twins, in which case its flux and mask area are a lower bound for that plume"),

        ("imeo_flux_total", F, "Sum of IMEO plume fluxes in the scene (kg/h)"),
        ("imeo_flux_max", F, "Largest IMEO plume flux in the scene (kg/h)"),
        ("cm_flux_total", F, "Sum of quantified CarbonMapper fluxes (kg/h)"),
        ("cm_flux_max", F, "Largest quantified CarbonMapper flux (kg/h)"),
        ("imeo_cod", S, "JSON dict IMEO emitter -> instance codes in plume_imeo.tif band 1 (a matched IMEO-CM pair shares the same code in both files)"),
        ("cm_cod", S, "JSON dict CM source -> instance codes in plume_cm.tif band 1"),
    ],
    "match": [
        ("pairs", S, "JSON list of matched IMEO-CM pairs (Hungarian assignment on geometry alone: overlap + source distance; flux and sector never used, keeping them free for cross-catalog studies)"),
        ("orphans", S, "JSON dict of unmatched plumes per catalog"),
        ("n_pairs", I, "Number of accepted IMEO-CM pairs"),
        ("n_perfect", I, "Pairs without same-catalog overlap ambiguity"),
        ("n_orph_imeo", I, "IMEO plumes without a CM counterpart"),
        ("n_orph_cm", I, "CM sources without an IMEO counterpart"),
    ],
    "sensor": [
        ("shape_rows", I, "Sensor-grid rows"),
        ("shape_cols", I, "Sensor-grid columns"),
        ("sza_mean", F, "Mean solar zenith angle (deg); intra-scene range is <1 deg"),
        ("vza_mean", F, "Mean view zenith angle (deg)"),
        ("saa_mean", F, "Mean solar azimuth angle (deg)"),
        ("vaa_mean", F, "Mean view azimuth angle (deg)"),
        ("amf_mean", F, "Two-way air mass factor 1/cos(SZA)+1/cos(VZA)"),
        ("phase_mean", F, "Mean solar phase angle between sun and view vectors (deg); sunglint indicator"),
        ("path_length_mean", F, "Mean sensor-to-ground path length (m)"),
        ("earth_sun_distance", F, "Earth-Sun distance (AU)"),
    ],
    "meteo": [
        ("wind_u", F, "Scene-mean 10 m eastward wind from the wind.tif layer (ERA5-Land, time-interpolated); per-pixel field in wind.tif"),
        ("wind_v", F, "Scene-mean 10 m northward wind (m/s), ERA5-Land"),
        ("wind_speed", F, "Scene-mean 10 m wind speed (m/s), ERA5-Land. NOTE: this is MethaneSET's own reanalysis wind, not the wind either catalog used to invert its fluxes; for that see imeo_wind and cm_wind"),
        ("imeo_wind", S, "JSON dict IMEO emitter -> {u, v, speed} (m/s): the PER-PLUME 10 m wind published by UNEP-IMEO, i.e. the wind IMEO used to invert its own flux rate. Independent of the ERA5-Land wind.tif layer shipped with the scene"),
        ("cm_wind", S, "JSON dict CarbonMapper source -> {speed, dir, u, v}: the PER-PLUME wind published by Carbon Mapper (wind_speed_avg_auto and wind_direction_avg_auto), i.e. the wind CM used to invert its own emission. dir follows the meteorological convention (degrees FROM which the wind blows); u and v are derived from speed and dir. Null when CM ships no wind for that plume"),
    ],
    "emit": [
        ("flight_line", S, "EMIT flight line identifier"),
        ("time_start", S, "Acquisition start (UTC)"),
        ("time_end", S, "Acquisition end (UTC)"),
    ],
    "radiance": [
        ("min", F, "Minimum radiance over the full 285-band cube (negative values are calibration noise in dark SWIR bands, kept unclipped)"),
        ("max", F, "Maximum radiance over the full cube"),
        ("elev_min_m", F, "Minimum elevation (m, Copernicus DEM GLO-30 2024)"),
        ("elev_max_m", F, "Maximum elevation (m)"),
    ],
    "site": [
        ("country", S, "Country of the scene (from the IMEO portal)"),
    ],
    "spatial": [
        ("bbox_west", F, "Bounding box west (deg, EPSG:4326)"),
        ("bbox_south", F, "Bounding box south (deg)"),
        ("bbox_east", F, "Bounding box east (deg)"),
        ("bbox_north", F, "Bounding box north (deg)"),
        ("imeo_points", S, "WKT MULTIPOINT of IMEO emission source points"),
        ("cm_points", S, "WKT MULTIPOINT of CarbonMapper source points"),
    ],
    "selection": [
        ("split", S, "Proposed train/val/test split (70/15/15, grouped by emitter: zero source leakage; users may re-split, everything ships)"),
        ("is_free", B, "True = verified plume-free granule from IMEO's full-tile test pool"),
        ("flux_bin", S, "has_weak / strong_only / free (weak = any IMEO plume under 1,000 kg/h, following STARCOP)"),
    ],
}


class NSExt(SampleExtension):
    model_config = {"arbitrary_types_allowed": True}
    ns: str
    values: dict

    def get_schema(self) -> pa.Schema:
        return pa.schema([pa.field(f"{self.ns}:{c}", t)
                          for c, t, _ in COLSPEC[self.ns]])

    def get_field_descriptions(self) -> dict:
        return {f"{self.ns}:{c}": d for c, _, d in COLSPEC[self.ns]}

    def _compute(self, sample) -> pa.Table:
        cols = {}
        for c, t, _ in COLSPEC[self.ns]:
            v = self.values.get(c)
            if v is not None and isinstance(v, float) and pd.isna(v):
                v = None
            cols[f"{self.ns}:{c}"] = [v]
        return pa.table(cols, schema=self.get_schema())


COLLECTION = dict(
    id="methaneset-emit",
    title="MethaneSET-EMIT: Hyperspectral Methane Plume Detection from EMIT",
    dataset_version="1.0.0",
    extent=Extent(spatial=[-123.6228, -47.2043, 152.1961, 50.7347],
                  temporal=["2022-08-10T06:49:57Z", "2025-11-12T17:40:08Z"]),
    description=(
        "methaneset-emit provides analysis-ready EMIT hyperspectral scenes for methane "
        "plume detection and quantification. 721 granules (700 with plumes + 21 "
        "verified plume-free), selected from the granules where the UNEP-IMEO and "
        "CarbonMapper catalogs BOTH report a plume (consensus), with 0% onboard "
        "nodata and complete IMEO flux quantification, balanced across flux class "
        "(all 812 weak plumes under 1,000 kg/h retained), year, region and sector. "
        "Each sample carries 10 sensor-geometry layers with 128x128 internal "
        "tiling: the full 285-band L1B radiance cube, per-pixel lat/lon, the GLT "
        "orthorectification lookup, Copernicus GLO-30 elevation, an ERA5-Land "
        "10 m wind field (u/v, time-interpolated to the acquisition minute), "
        "three matched-filter retrievals (mf, rmf, mag1c), and dual instance-coded "
        "plume masks from the IMEO portal and CarbonMapper analysts, where a "
        "matched pair shares the same code in both files. A leakage-free "
        "train/val/test split grouped by emitter is proposed; all metadata is "
        "queryable in level0.parquet without opening a raster."
    ),
    licenses=["cc-by-4.0"],
    providers=[
        Provider(name="NASA JPL", roles=["producer"]),
        Provider(name="UNEP IMEO", roles=["producer"]),
        Provider(name="Carbon Mapper", roles=["producer"]),
        Provider(name="Universitat de Valencia, ISP", roles=["processor"]),
    ],
    curators=[
        Curator(name="Julio Contreras",
                organization="Universitat de Valencia, Image and Signal Processing (ISP)",
                email="julio.contreras@uv.es"),
        Curator(name="Cesar Aybar",
                organization="Universitat de Valencia, Image and Signal Processing (ISP)",
                email="cesar.aybar@uv.es"),
    ],
    keywords=["methane", "EMIT", "hyperspectral", "plume detection",
              "matched filter", "IMEO", "CarbonMapper"],
    tasks=["segmentation", "regression"],
)

PUBS = Publications(publications=[
    Publication(doi="10.5067/EMIT/EMITL1BRAD.001",
                citation="Green, R. O., et al. (2023). EMIT L1B At-Sensor Calibrated Radiance and Geolocation Data 60 m V001. NASA LP DAAC.",
                summary="Source radiance and geolocation."),
    Publication(doi="10.1038/s41598-023-44918-6",
                citation="Ruzicka, V., et al. (2023). Semantic segmentation of methane plumes with hyperspectral machine learning models. Scientific Reports.",
                summary="Weak/strong plume convention at 1,000 kg/h."),
    Publication(doi="10.5270/ESA-c5d3d65",
                citation="European Space Agency (2024). Copernicus DEM GLO-30.",
                summary="Elevation layer source."),
    Publication(doi="10.24381/cds.e2161bac",
                citation="Munoz-Sabater, J., et al. (2021). ERA5-Land hourly data. Copernicus Climate Data Store.",
                summary="Wind layer source."),
])


def build_scene(row):
    d = SRC / row["id"]
    leaves = [Sample(id=name, path=d / name) for name in LAYERS]
    scene = Sample(id=row["id"], path=Tortilla(samples=leaves))
    for ns in COLSPEC:
        vals = {c: row.get(f"{ns}:{c}") for c, _, _ in COLSPEC[ns]}
        scene.extend_with(NSExt(ns=ns, values=vals))
    return scene


def main():
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
    cross_dir = sorted((DATA / "cross").iterdir())[-1]
    df = pd.read_parquet(cross_dir / "metadata_level0.parquet")
    if limit:
        df = df.head(limit)
    print(f"scenes: {len(df)}", flush=True)

    scenes = [build_scene(r) for r in df.to_dict("records")]
    root = Tortilla(samples=scenes)
    taco = Taco(tortilla=root, **COLLECTION)
    taco.extend_with(PUBS)

    print("validating schema...", flush=True)
    root.export_metadata()
    # NEVER write over the live folder: it is built in .new and
    # swapped by hand after verification (mv metanset-emit -> -old, .new -> live)
    out = OUT + ("-test" if limit else ".new")
    print(f"writing FOLDER (hardlinks) to {out} ...", flush=True)
    paths = create(taco=taco, output=out, output_format="folder")
    print(f"created: {len(paths)} path(s)")

    # the model recomputes extent from STAC (which we do not use) and falls back
    # to the global one: it is set by hand in the json, as v1's 80_update_extent did
    import json as _json
    cj = pathlib.Path(out) / "COLLECTION.json"
    c = _json.loads(cj.read_text())
    c["extent"] = {"spatial": [-123.6228, -47.2043, 152.1961, 50.7347],
                   "temporal": ["2022-08-10T06:49:57Z", "2025-11-12T17:40:08Z"]}
    cj.write_text(_json.dumps(c, indent=2))
    print("extent fixed in COLLECTION.json")

    try:
        generate_markdown(f"{out}/COLLECTION.json", output=f"{out}/README.md")
        generate_html(f"{out}/COLLECTION.json", output=f"{out}/index.html")
        print("README.md + index.html generated")
    except Exception as e:
        print(f"docs: {type(e).__name__}: {e} (not critical)")
    print("DONE")


if __name__ == "__main__":
    main()
