"""Steps 3 to 5: from loose files to the TACO hierarchy.

    file.tif     ->  Sample (leaf)
    9 leaves     ->  Tortilla (the scene inside)
    scene        ->  Sample(id=granule, path=tortilla) + extensions
    535 scenes   ->  root Tortilla

The root Tortilla is what tacotoolbox turns into DATA/ + METADATA/:
each nesting level produces its levelN.parquet.

PREPARED, NOT RUN YET. Requires tacotoolbox >= 0.25.0.
"""
from tacotoolbox.datamodel import Sample, Tortilla

from config import ASSETS
from extensions import (
    DetectionExtension, MeteoExtension, SensorExtension,
    SiteExtension, SpatialExtension,
)


def build_leaf_samples(ctx: dict) -> Tortilla:
    """Step 3: each file of the scene, wrapped in a leaf Sample."""
    hojas = [Sample(id=name, path=ctx["paths"][name]) for name in ASSETS]
    return Tortilla(samples=hojas)


def build_scene(ctx: dict) -> Sample:
    """Step 4: the scene = a Sample whose path is the tortilla of its leaves."""
    escena = Sample(id=ctx["id"], path=build_leaf_samples(ctx))
    # each extend_with adds namespace:field columns to level0.parquet
    escena.extend_with(DetectionExtension(**ctx["detection"]))
    escena.extend_with(SensorExtension(**ctx["sensor"]))
    escena.extend_with(SpatialExtension(**ctx["spatial"]))
    escena.extend_with(SiteExtension(**ctx["site"]))
    escena.extend_with(MeteoExtension(**ctx["meteo"]))
    return escena


def build_root(contexts: list[dict]) -> Tortilla:
    """Step 5: all scenes together. This IS the dataset."""
    return Tortilla(samples=[build_scene(c) for c in contexts])
