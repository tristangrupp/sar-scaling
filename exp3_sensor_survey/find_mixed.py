"""Find NISAR GCOV mixed-mode (_M_) products and their same-datatake 40 MHz (_A_) partners."""
from config import DATA  # noqa: E402
import os, sys
import pandas as pd
import requests
from nisar_io import CMR, GCOV
rows = []
for pat in ["*_2005_*_M_2026*", "*_4005_*_A_2026*"]:
    page = 1
    sa = None
    while True:
        p = {"collection_concept_id": GCOV, "page_size": 2000, "readable_granule_name[]": pat,
             "options[readable_granule_name][pattern]": "true", "temporal": "2026-06-17T00:00:00Z,2026-10-17T00:00:00Z"}
        h = {"CMR-Search-After": sa} if sa else {}
        r = requests.get(CMR.replace("umm_json", "json"), params=p, headers=h, timeout=300)
        sa = r.headers.get("CMR-Search-After")
        e = r.json()["feed"]["entry"]
        for it in e:
            rows.append({"id": it["title"], "url": next(l["href"] for l in it["links"] if l["href"].endswith(".h5") and l["href"].startswith("https")),
                         "box": it.get("boxes", [None])[0], "poly": (it.get("polygons") or [[None]])[0][0]})
        print(pat, len(e), r.headers.get("CMR-Hits"), flush=True)
        if not e or not sa or len(e) < 2000:
            break
df = pd.DataFrame(rows)
f = df.id.str.split("_", expand=True)
df["cycle"], df["track"], df["dir"], df["frame"], df["bw"], df["pol"], df["mm"], df["t0"] = f[4], f[5], f[6], f[7], f[8], f[9], f[10], f[11]
df["full"] = f[15]
df.to_csv(str(DATA) + "/gcov_index.csv", index=False)
m = df[df.mm == "M"]; a = df[df.mm == "A"]
pairs = m.merge(a, on=["cycle", "track", "dir", "frame"], suffixes=("_m", "_a"))
pairs.to_csv(str(DATA) + "/mixed_pairs.csv", index=False)
print("M:", len(m), "A:", len(a), "pairs:", len(pairs), "track-frames:", pairs.groupby(["track", "frame"]).ngroups, flush=True)
sys.stdout.flush(); os._exit(0)
