"""Bundle Capella / Umbra / ICEYE x S1 SM overlap outputs into one JSON for the HTML page."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd

HERE = Path(__file__).parent
PROVIDERS = ["capella", "umbra", "iceye"]


def norm_pairs(df: pd.DataFrame, provider: str) -> pd.DataFrame:
    df = df.rename(columns={c: c.replace("capella_", "xband_") for c in df.columns if c.startswith("capella_")})
    df["provider"] = provider
    return df


out = {"providers": {}, "land": json.loads((HERE / "land-110m.json").read_text())}
for prov in PROVIDERS:
    d = HERE / f"s1sm_{prov}"
    sites_f, pairs_f = d / "sites.geojson", d / "pairs.csv"
    if not sites_f.exists():
        print(f"{prov}: no sites.geojson, skipped")
        continue
    sites = gpd.read_file(sites_f)
    if "n_capella" in sites:
        sites = sites.rename(columns={"n_capella": "n_xband"})
    c = sites.geometry.representative_point()
    pairs = norm_pairs(pd.read_csv(pairs_f), prov) if pairs_f.exists() else pd.DataFrame()

    site_rows = []
    for s, pt in zip(sites.itertuples(), c):
        p = pairs[pairs.site_id == s.site_id] if len(pairs) else pairs
        good = p[p.good] if len(p) else p
        site_rows.append({
            "id": int(s.site_id), "lon": round(pt.x, 4), "lat": round(pt.y, 4),
            "n_xband": int(s.n_xband), "n_sm": int(s.n_sm or 0) if pd.notna(s.n_sm) else None,
            "n_pairs": int(s.n_pairs), "n_good": int(s.n_good_pairs),
            "pols": sorted(set(good.co_pol.dropna())) if len(good) else [],
            "d_inc": [round(float(good.d_inc.min()), 1), round(float(good.d_inc.max()), 1)] if len(good) else None,
            "dt_days": int(good.dt_days.min()) if len(good) else None,
            "iw": int(good.iw_co_pol_in_window.max()) if len(good) and "iw_co_pol_in_window" in good
                  and good.iw_co_pol_in_window.notna().any() else None,
            "best": good.sort_values(["d_inc", "dt_days"]).head(5)[
                ["xband_id", "s1_id", "xband_dt", "s1_dt", "co_pol", "d_inc", "dt_days", "xband_cover"]
            ].round(2).to_dict("records") if len(good) else [],
        })
    out["providers"][prov] = {
        "n_items": int(sites.n_xband.sum()), "n_sites": len(sites),
        "n_sites_sm": int((sites.n_sm.fillna(0) > 0).sum()),
        "n_pairs": int(len(pairs)), "n_good": int(pairs.good.sum()) if len(pairs) else 0,
        "n_sites_good": int((sites.n_good_pairs > 0).sum()),
        "sites": site_rows,
    }
    print(prov, {k: v for k, v in out["providers"][prov].items() if k != "sites"})

tpl = (HERE / "overlap_page.template.html").read_text(encoding="utf-8")
tpl = tpl.replace("/*__DATA__*/null", json.dumps(out, separators=(",", ":"), default=str))
tpl = tpl.replace('<meta name="claude:viewport-fit-sides" content="cover">\n', "")

# Inline d3 and topojson so the page opens offline from disk.
for lib in ["d3.min.js", "topojson.min.js"]:
    tag = next(l for l in tpl.splitlines() if l.startswith("<script src=") and lib.split(".")[0] in l)
    tpl = tpl.replace(tag, "<script>\n" + (HERE / lib).read_text(encoding="utf-8") + "\n</script>")

# Template is a body fragment; wrap it in a full document.
head, body = tpl.split("</style>", 1)
page = ('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        + head + "</style>\n</head>\n<body>" + body + "\n</body>\n</html>\n")
(HERE / "sar_overlap_atlas.html").write_text(page, encoding="utf-8")
print("wrote", HERE / "sar_overlap_atlas.html")
