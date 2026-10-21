"""Umbra STAC items point at GEC paths that were since moved in the bucket (e.g. 'tasks/ad hoc/...'
returns 404). List every *_GEC.tif key in the bucket and match items by platform and collect
time (file names are <start time>_UMBRA-NN_GEC.tif) within 60 s. Updates xband_hrefs.json.
"""
from __future__ import annotations

import json
import os
import re
import sys
import xml.etree.ElementTree as ET
from urllib.parse import quote

import geopandas as gpd
import pandas as pd
import requests

from common import PROJ, ROOT

BUCKET = "https://umbra-open-data-catalog.s3.amazonaws.com"
PAT = re.compile(r"(\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2})_(UMBRA-\d{2})_GEC\.tif$")


def list_gec():
    ns = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}
    s, token, rows = requests.Session(), None, []
    while True:
        p = {"list-type": "2", "prefix": "sar-data/"}
        if token:
            p["continuation-token"] = token
        root = ET.fromstring(s.get(BUCKET, params=p, timeout=120).content)
        for k in root.findall("s3:Contents/s3:Key", ns):
            m = PAT.search(k.text)
            if m:
                rows.append({"key": k.text, "t": pd.Timestamp(m.group(1)[:10] + "T" + m.group(1)[11:].replace("-", ":"), tz="UTC"),
                             "platform": m.group(2)})
        if root.findtext("s3:IsTruncated", namespaces=ns) != "true":
            return pd.DataFrame(rows)
        token = root.findtext("s3:NextContinuationToken", namespaces=ns)


def main():
    g = list_gec()
    print(f"{len(g)} GEC files in bucket", flush=True)
    um = gpd.read_file(PROJ / "exp0_overlap_atlas/s1sm_umbra/umbra_items.geojson")
    um["t"] = pd.to_datetime(um.datetime, utc=True, format="mixed")
    um["plat"] = um.platform.str.upper().str.replace("_", "-")
    hrefs = json.loads((ROOT / "xband_hrefs.json").read_text()) if (ROOT / "xband_hrefs.json").exists() else {}
    n = 0
    for r in um.itertuples():
        c = g[(g.platform == r.plat) & ((g.t - r.t).abs() <= pd.Timedelta(seconds=60))]
        if len(c):
            k = c.iloc[(c.t - r.t).abs().argmin()].key
            hrefs[r.xband_id] = f"{BUCKET}/{quote(k)}"
            n += 1
    (ROOT / "xband_hrefs.json").write_text(json.dumps(hrefs))
    print(f"matched {n} of {len(um)} Umbra items", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
    sys.stdout.flush()
    os._exit(0)
