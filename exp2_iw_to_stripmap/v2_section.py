"""Write results_v2_section.html: the round 2 expanded set (map, WorldCover class mix, counts,
change mask). make_results.py inserts it into results_body.html.
"""
from __future__ import annotations

import base64
import io
import os
import sys
from html import escape

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import PROJ, ROOT, TILES

WC = {10: "Tree cover", 20: "Shrubland", 30: "Grassland", 40: "Cropland", 50: "Built-up", 60: "Bare / sparse",
      70: "Snow / ice", 80: "Water", 90: "Herbaceous wetland", 95: "Mangroves", 100: "Moss / lichen"}
WC_COL = {10: "#2e7d32", 20: "#a1887f", 30: "#c5e1a5", 40: "#f9a825", 50: "#c62828", 60: "#d7ccc8",
          70: "#e3f2fd", 80: "#1e88e5", 90: "#26a69a", 95: "#00695c", 100: "#cfd8dc"}
ROLE_COL = {"train": "#1f6fd1", "val": "#1f6fd1", "holdout_xband": "#c2410c", "holdout_near": "#0f8a6a",
            "holdout_far": "#a15c00", "holdout_paper": "#7c3aed", "holdout_near2": "#0f8a6a", "holdout_far2": "#a15c00"}


def fig64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def main():
    t = pd.read_csv(ROOT / "tiles_v2.csv")
    sp = pd.read_csv(ROOT / "splits_v2.csv")[["tile_id", "split", "used"]]
    pf = pd.read_csv(ROOT / "pairs_final_v2.csv")[["tile_id", "corr10"]]
    ch = pd.read_csv(ROOT / "change_v2.csv")
    d = t.merge(sp, on="tile_id").merge(pf, on="tile_id", how="left").merge(
        ch[ch.status == "ok"][["tile_id", "change_land", "gap_days"]], on="tile_id", how="left")
    d = d[d.used]
    # ---- map
    land = gpd.read_file(PROJ / "exp0_overlap_atlas/land-110m.json").set_crs(4326, allow_override=True)
    reg = d.groupby("region").agg(lon=("lon", "mean"), lat=("lat", "mean"), n=("tile_id", "size"),
                                  role=("role", "first"), rnd=("round", "first")).reset_index()
    fig, ax = plt.subplots(figsize=(11, 5.2))
    land.plot(ax=ax, color="#dfe3e8", edgecolor="none")
    for r in reg.itertuples():
        ax.scatter(r.lon, r.lat, s=18 + r.n * 2.2, color=ROLE_COL.get(r.role, "#555"), alpha=.85,
                   marker="o" if r.rnd == 1 else "D", edgecolor="k", linewidth=.6, zorder=3)
        ax.annotate(r.region, (r.lon, r.lat), xytext=(4, 4), textcoords="offset points", fontsize=7)
    ax.set_xlim(-180, 180)
    ax.set_ylim(-60, 75)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Stripmap tiles by region: circles = round 1 (pilot), diamonds = round 2 (inland); "
                 "blue train, orange/brown/green held out (size = tiles)", fontsize=8)
    map64 = fig64(fig)
    # ---- WorldCover class mix (pixel shares on the 5 m grid)
    rows = []
    for r in d.itertuples():
        wc = np.load(TILES / f"{r.tile_id}.npz")["wc"]
        v, c = np.unique(wc[wc > 0], return_counts=True)
        rows += [{"tile_id": r.tile_id, "round": r.round, "split": r.split, "cls": int(a), "px": int(b)} for a, b in zip(v, c)]
    px = pd.DataFrame(rows)
    share = px.pivot_table(index="round", columns="cls", values="px", aggfunc="sum").fillna(0)
    share.loc["both"] = share.sum()
    share = share.div(share.sum(axis=1), axis=0)
    share.index = ["Round 1 (pilot)", "Round 2 (inland)", "Combined"]
    fig, ax = plt.subplots(figsize=(10, 2.6))
    left = np.zeros(len(share))
    for c in share.columns:
        ax.barh(share.index, share[c], left=left, color=WC_COL.get(c, "#999"), label=WC.get(c, c))
        left += share[c].to_numpy()
    ax.set_xlim(0, 1)
    ax.invert_yaxis()
    ax.legend(ncol=6, fontsize=7, loc="upper center", bbox_to_anchor=(.5, -.18), frameon=False)
    ax.set_xlabel("share of tile pixels (ESA WorldCover 2021)", fontsize=8)
    lc64 = fig64(fig)
    tab = share.T.rename(index=lambda c: WC.get(c, str(c)))
    lc_rows = "".join(f"<tr><td>{escape(i)}</td>" + "".join(f"<td class='n'>{100 * v:.1f}%</td>" for v in r) + "</tr>"
                      for i, r in tab.iterrows())
    lc_tbl = ("<div class='tw'><table><thead><tr><th>WorldCover class</th>" +
              "".join(f"<th class='n'>{escape(c)}</th>" for c in tab.columns) + f"</tr></thead><tbody>{lc_rows}</tbody></table></div>")
    # ---- round 2 counts
    r2 = d[d["round"] == 2]
    g = r2.groupby("region").agg(theme=("theme", "first"), role=("role", "first"), scenes=("slc", "nunique"),
                                 tiles=("tile_id", "size"), train=("split", lambda s: (s == "train").sum()),
                                 val=("split", lambda s: (s == "val").sum()), test=("split", lambda s: (s == "test").sum()),
                                 corr10=("corr10", "median"), change=("change_land", "median")).reset_index()
    g = g.sort_values(["role", "region"])
    cnt = "".join(f"<tr><td>{r.region}</td><td>{escape(str(r.theme))}</td><td>{r.role}</td><td class='n'>{r.scenes}</td>"
                  f"<td class='n'>{r.tiles}</td><td class='n'>{r.train}</td><td class='n'>{r.val}</td><td class='n'>{r.test}</td>"
                  f"<td class='n'>{r.corr10:.2f}</td><td class='n'>{'' if pd.isna(r.change) else f'{100 * r.change:.1f}%'}</td></tr>"
                  for r in g.itertuples())
    cnt_tbl = ("<div class='tw'><table><thead><tr><th>Region</th><th>Land cover</th><th>Role</th><th class='n'>Scenes</th>"
               "<th class='n'>Tiles used</th><th class='n'>Train</th><th class='n'>Val</th><th class='n'>Test</th>"
               "<th class='n'>10 m corr</th><th class='n'>Changed land</th></tr></thead><tbody>" + cnt + "</tbody></table></div>")
    # ---- change mask summary
    cm = d.groupby("round").agg(tiles=("tile_id", "size"), masked=("change_land", lambda s: s.notna().sum()),
                                med=("change_land", "median"), p90=("change_land", lambda s: s.quantile(.9)),
                                gap=("gap_days", "median"))
    cm_txt = "; ".join(f"round {i}: {int(r.masked)} of {int(r.tiles)} tiles have a mask, median {100 * r.med:.1f}% of land "
                       f"flagged (90th percentile {100 * r.p90:.1f}%), passes {r.gap:.0f} days apart"
                       for i, r in cm.iterrows())
    tot = {"r1": int((d["round"] == 1).sum()), "r2": int((d["round"] == 2).sum())}
    notes = (ROOT / "results_v2_notes.html").read_text(encoding="utf-8") if (ROOT / "results_v2_notes.html").exists() else ""
    html = f"""<section class='card' style='border-left:3px solid var(--a3)'><h2>Round 2: inland scenes and a change mask</h2>
<p>Round 2 adds {r2.slc.nunique()} inland Stripmap scenes ({tot['r2']} usable tiles) to the {tot['r1']} round 1 tiles. The pilot
set leaned on islands and coasts, because Stripmap since 2023 is acquired mostly there. Round 2 searches the whole ASF archive
(19,180 Stripmap SLCs since 2014) for cells more than 25 km inland with a same-direction IW pass within 6 days on Planetary
Computer. Chicago, São Paulo, the Amazon and Campinas have many Stripmap scenes but no same-direction IW within 6 days, so they
could not be used.</p>{notes}
<figure><img src='{map64}' alt='Map of tile regions'></figure>
<h3>Round 2 regions</h3>{cnt_tbl}
<h3>Land-cover mix (ESA WorldCover pixel shares)</h3><figure><img src='{lc64}' alt='Land-cover mix by round'></figure>{lc_tbl}
<h3>Change mask</h3><p>{cm_txt}.</p></section>"""
    (ROOT / "results_v2_section.html").write_text(html, encoding="utf-8")
    print("wrote results_v2_section.html", flush=True)
    print(share.round(3).to_string())


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
    sys.stdout.flush()
    os._exit(0)
