"""Downloads the IMEO (UNEP) public catalog and stores it dated in assets/data.

The portal is a SPA and blocks requests without a browser User-Agent, but the
files hang under /downloads/ and are served as octet-stream.

Usage:  python fetch_imeo.py
Output: assets/data/imeo/<YYYY-MM-DD>/<file>
"""
import datetime
import pathlib
import requests

BASE = "https://methanedata.unep.org/downloads"
FILES = [
    "unep_methanedata_detected_plumes.csv",
    "unep_methanedata_detected_plumes.geojson",
    "unep_methanedata_detected_sources.csv",
    "unep_methanedata_detected_sources.geojson",
]
UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/120 Safari/537.36"}
OUT = pathlib.Path(__file__).resolve().parent.parent / "assets" / "data" / "imeo"


def main():
    day = datetime.date.today().isoformat()
    dest = OUT / day
    dest.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        r = requests.get(f"{BASE}/{name}", headers=UA, timeout=300)
        if r.status_code != 200 or r.headers.get("content-type", "").startswith("text/html"):
            print(f"  FAILED {name}: HTTP {r.status_code} {r.headers.get('content-type')}")
            continue
        (dest / name).write_bytes(r.content)
        print(f"  ok {name}  {len(r.content)/1e6:.1f} MB")
    print("saved to", dest)


if __name__ == "__main__":
    main()
