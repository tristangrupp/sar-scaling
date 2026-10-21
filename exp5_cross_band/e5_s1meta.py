"""Orbit direction of each Sentinel-1 RTC scene paired with a held-out X-band tile (for conditioning)."""
from __future__ import annotations

import common  # noqa: F401
from common import ROOT, SPIKE, finish

import pandas as pd
import planetary_computer as pc
from pystac_client import Client


def main():
    man = pd.read_csv(SPIKE / "manifest.csv")
    man = man[(man.split != "train") & man.status.isin(["ok", "cached"])]
    c = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1", modifier=pc.sign_inplace)
    col = c.get_collection("sentinel-1-rtc")
    rows = []
    for sid in sorted(man.s1_id.dropna().unique()):
        try:
            it = col.get_item(sid)
            p = it.properties
            rows.append(dict(s1_id=sid, orbit=p.get("sat:orbit_state"), platform=p.get("platform"),
                             rel_orbit=p.get("sat:relative_orbit")))
        except Exception as e:
            rows.append(dict(s1_id=sid, orbit=None, error=str(e)[:100]))
        print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(ROOT / "s1_meta.csv", index=False)


if __name__ == "__main__":
    main()
    finish()
