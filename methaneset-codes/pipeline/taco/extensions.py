"""Step 2: the metadata namespaces.

Each extension is a pydantic model. When you run sample.extend_with(ext),
its fields become columns of levelN.parquet with the namespace prefix:
DetectionExtension.n_imeo -> column "detection:n_imeo".

That way the parquet is queryable with SQL without opening a single raster:
    SELECT id FROM level0 WHERE "detection:n_imeo" > 3 AND "site:country" = 'Libya'

PREPARED, NOT RUN YET. Requires tacotoolbox >= 0.25.0.
"""
from typing import Optional

import pydantic


class DetectionExtension(pydantic.BaseModel):
    """detection: how many plumes and whose."""
    n_imeo: int
    n_cm: int
    imeo_ids: Optional[str] = None
    cm_ids: Optional[str] = None
    imeo_flux: Optional[str] = None
    cm_flux: Optional[str] = None
    sector: Optional[str] = None


class SensorExtension(pydantic.BaseModel):
    """sensor: viewing geometry (for injection and AMF)."""
    sza_mean: float
    vza_mean: float
    amf_mean: float
    sun_azimuth_mean: float
    sensor_azimuth_mean: float


class SpatialExtension(pydantic.BaseModel):
    """spatial: geographic bbox; EMIT has no affine grid, just a box."""
    bbox_west: float
    bbox_south: float
    bbox_east: float
    bbox_north: float


class SiteExtension(pydantic.BaseModel):
    """site: geographic context, shared with the multispectral datasets."""
    country: Optional[str] = None


class MeteoExtension(pydantic.BaseModel):
    """meteo: wind used for the flux, shared across datasets."""
    wind_u: Optional[float] = None
    wind_v: Optional[float] = None
    wind_speed: Optional[float] = None
