"""Sentinel-1 IW passes at each repeat X-band site (Planetary Computer sentinel-1-rtc, metadata only).

For each repeat site in repeat_sites.csv, list every IW RTC pass over the site centroid from 30 days
before its first X-band day to its last. One row per pass (date, relative orbit, orbit direction);
frames of one pass are merged. Writes s1_passes.csv. pairs.py then matches passes to X-band pairs.
"""
from __future__ import annotations

import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
from pystac_client import Client

ROOT = Path(__file__).resolve().parent
FIELDS = {"include": ["properties.datetime", "properties.sat:relative_orbit", "properties.sat:orbit_state",
                      "properties.platform"], "exclude": ["assets", "links", "geometry"]}


def search(c, r):
    t0 = pd.Timestamp(str(r.first)) - pd.Timedelta(days=30)
    err = None
    for k in range(4):
        try:
            its = c.search(collections=["sentinel-1-rtc"], intersects={"type": "Point", "coordinates": [r.lon, r.lat]},
                           datetime=f"{t0:%Y-%m-%d}/{r.last}T23:59:59Z", fields=FIELDS).items_as_dicts()
            return [{"site_id": r.site_id, "t": i["properties"]["datetime"], "rel_orbit": i["properties"].get("sat:relative_orbit"),
                     "orbit_state": i["properties"].get("sat:orbit_state"), "platform": i["properties"].get("platform")}
                    for i in its]
        except Exception as e:
            time.sleep(5 * (k + 1))
            err = e
    print("failed", r.site_id, err, flush=True)
    return []


def main():
    sites = pd.read_csv(ROOT / "repeat_sites.csv")
    c = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")
    with ThreadPoolExecutor(6) as ex:
        rows = [x for lst in ex.map(lambda r: search(c, r), sites.itertuples()) for x in lst]
    p = pd.DataFrame(rows)
    p["t"] = pd.to_datetime(p.t, utc=True, format="ISO8601")
    p["date"] = p.t.dt.strftime("%Y-%m-%d")
    p = p.sort_values("t").drop_duplicates(["site_id", "date", "rel_orbit", "orbit_state"])
    p.to_csv(ROOT / "s1_passes.csv", index=False)
    print("S1 passes:", len(p), "at", p.site_id.nunique(), "of", len(sites), "repeat sites", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
    sys.stdout.flush()
    os._exit(0)
