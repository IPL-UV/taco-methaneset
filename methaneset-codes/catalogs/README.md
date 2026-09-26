# Catalogs

Dated snapshots of the two annotation sources:

- `fetch_imeo.py` — downloads the files published by the IMEO Eye on Methane portal.
- `fetch_carbonmapper.py` — walks the public Carbon Mapper API.
- `explore_cm_api.py` — inventory of the Carbon Mapper endpoints and STAC collections.

Two things we learned the hard way, kept here so nobody repeats them:

- The IMEO portal blocks requests without a browser `User-Agent`.
- The Carbon Mapper API answers without a token, and its `gas` parameter is ignored: CH4 and CO2
  have to be separated by reading each record.

Snapshots are stored as `catalog_root/<source>/<YYYY-MM-DD>/` and are never overwritten.
