"""Step 6: the dataset identity (what ends up in COLLECTION.json).

The final Taco = Collection (who, what, license, extent, papers) + the
root Tortilla (the data). tacotoolbox only adds taco:pit_schema
(the 535 x 9 shape) and taco:field_schema on write.

PREPARED, NOT RUN YET. Requires tacotoolbox >= 0.25.0.
"""
from tacotoolbox.taco.datamodel import Curator, Provider, Taco
from tacotoolbox.datamodel import Tortilla

from config import COLLECTION_ID, COLLECTION_LICENSES, COLLECTION_VERSION

DESCRIPTION = (
    "Analysis-ready EMIT hyperspectral data for methane plume detection: "
    "full L1B radiance in sensor geometry, precomputed matched-filter "
    "retrievals, dual plume masks (UNEP-IMEO and Carbon Mapper), elevation "
    "and observation geometry."
)

PROVIDERS = [
    Provider(name="NASA JPL", roles=["producer"]),
    Provider(name="UNEP IMEO", roles=["producer"]),
    Provider(name="Carbon Mapper", roles=["producer"]),
]

CURATORS = [
    Curator(name="Julio Contreras",
            organization="Universitat de Valencia, ISP",
            email="julio.contreras@uv.es"),
]


def build_taco(root: Tortilla) -> Taco:
    return Taco(
        id=COLLECTION_ID,
        dataset_version=COLLECTION_VERSION,
        description=DESCRIPTION,
        licenses=COLLECTION_LICENSES,
        providers=PROVIDERS,
        curators=CURATORS,
        tortilla=root,
    )
