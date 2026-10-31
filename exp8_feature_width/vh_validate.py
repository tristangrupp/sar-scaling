"""Round 2, after scoring: check each rule against the 60 spot-check verdicts and draw figures/vh_vs_vv.png.

Reads spotcheck.csv (read only), segments_vh_vv.csv, vh_vv_widths.csv, vh_vv_paired.csv. Writes vh_validate.json.
The reviewer saw Capella, single-date VV and the multi-date VV and VH means, so the multi-date columns are the
fair comparison; single-date VH was never shown.
"""
import json
import os
import sys
import traceback
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import analyze as A  # noqa: E402

# reviewer notes, coded by which Sentinel-1 image they name as clearer (panel number -> class)
FAVOR = {**{p: "vh" for p in (0, 4, 9, 11, 12, 16, 19, 21, 30, 31, 42, 52, 56)},
         **{p: "vv" for p in (25, 32, 34, 41, 43)},
         **{p: "both_named" for p in (6, 22, 23)},
         45: "vv_weak", 38: "other"}
RULES = {"original": "", "amended": "am_", "ls": "ls_"}
RNAME = {"original": "original rule", "amended": "amended rule", "ls": "line or step (secondary)"}
COLS = ["multi_vv", "multi_vh", "multi_either", "single_vv", "single_vh", "single_either"]
CVV, CVH = "#2a78d6", "#eb6834"


def main():
    sp = pd.read_csv(HERE / "spotcheck.csv")
    sp["pnum"] = sp.panel.str[:2].astype(int)
    sp["favor"] = sp.pnum.map(FAVOR).fillna("")
    seg = pd.read_csv(HERE / "segments_vh_vv.csv").set_index("seg_id")
    d = sp.join(seg, on="seg_id", rsuffix="_s")
    assert d.cap_width.notna().all()
    out = {"n_panels": len(d), "verdicts": d.verdict.value_counts().to_dict(),
           "favor_counts": d.favor.replace("", "no note").value_counts().to_dict()}
    dec = d[d.verdict.isin(["both", "capella_only"])].copy()
    seen = (dec.verdict == "both").astype(float)
    ag = []
    for rule, pre in RULES.items():
        for c in COLS:
            col = f"{pre}{c}_det"
            det = dec[col].map({True: 1.0, False: 0.0, "True": 1.0, "False": 0.0, 1.0: 1.0, 0.0: 0.0}).astype(float)
            ag.append({"rule": rule, "product": c, "n": len(dec), "agree": int((det == seen).sum()),
                       "agree_share": round(float((det == seen).mean()), 3),
                       "hit_on_seen": f"{int(det[seen == 1].sum())}/{int((seen == 1).sum())}",
                       "flag_on_unseen": f"{int(det[seen == 0].sum())}/{int((seen == 0).sum())}"})
    out["agreement"] = ag
    fav = []
    for rule, pre in RULES.items():
        for g in ("vh", "vv", "both_named"):
            s = d[d.favor == g]
            rec = {"rule": rule, "notes_favor": g, "n": len(s)}
            for c in ("multi_vv", "multi_vh"):
                rec[f"{c}_detected"] = int(s[f"{pre}{c}_det"].astype(str).isin(["True", "1.0"]).sum())
                rec[f"{c}_median_abs_z"] = round(float(s[f"{'am_' if rule != 'original' else ''}{c}_z"].abs().median()), 2)
            if g in ("vh", "vv"):
                zv = s[f"{'am_' if rule != 'original' else ''}multi_vv_z"].abs()
                zh = s[f"{'am_' if rule != 'original' else ''}multi_vh_z"].abs()
                rec["vh_z_above_vv_z"] = int((zh > zv).sum())
            fav.append(rec)
    out["notes_favor"] = fav
    (HERE / "vh_validate.json").write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps(out, indent=1, default=str), flush=True)

    # ---------------- figure
    seg = seg.reset_index()
    meta = pd.read_csv(HERE / "scene_meta.csv")
    seg["wbin"] = pd.cut(seg.cap_width, A.EDGES, labels=A.BLAB, right=False)
    wt = pd.read_csv(HERE / "vh_vv_widths.csv")
    pr = pd.read_csv(HERE / "vh_vv_paired.csv")
    mids = np.sqrt(np.array(A.EDGES[:-1]) * np.array(A.EDGES[1:]))
    ws = np.exp(np.linspace(np.log(1.5), np.log(40), 200))
    fig, axs = plt.subplots(2, 2, figsize=(13.5, 9))
    for ax, prod in zip(axs[0], ("multi", "single")):
        for q, c in (("vv", CVV), ("vh", CVH)):
            col = f"am_{prod}_{q}_det"
            g = seg.groupby("wbin", observed=False)[col].agg(["mean", "size"]).reindex(A.BLAB)
            m = g["size"].values >= 10
            ax.plot(mids[m] * (0.97 if q == "vv" else 1.03), g["mean"].values[m], "o", ms=8, color=c,
                    markeredgecolor="white", markeredgewidth=1.5)
            r = wt[(wt.rule == "amended") & (wt["product"] == prod) & (wt.pol == q)].iloc[0]
            ax.plot(ws, A.curve((r.a, r.b), r.false_alarm, ws), color=c, lw=2, ls="-" if q == "vv" else "--",
                    label=f"{q.upper()}: {r.det_rate:.0%} detected, 50% at {r.w50_txt} m")
        ax.axhline(0.5, color="#ccc", lw=0.8)
        ax.set_xscale("log")
        ax.set_xticks([2, 3, 5, 8, 12, 20, 30, 40])
        ax.set_xticklabels(["2", "3", "5", "8", "12", "20", "30", "40"])
        ax.set_ylim(0, 0.8)
        ax.set_xlabel("feature width on Capella (m)")
        ax.set_ylabel("share detected, one polarization")
        ax.set_title(f"{'Multi-date mean' if prod == 'multi' else 'Single date'}, amended rule (bins with 10+ segments)", fontsize=10)
        ax.legend(fontsize=8, loc="upper left", frameon=False)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    # paired shares
    ax = axs[1, 0]
    labs, y = [], 0
    for rule in ("original", "amended", "ls"):
        for prod in ("multi", "single"):
            r = pr[(pr.rule == rule) & (pr["product"] == prod)].iloc[0]
            n = r.both + r.vv_only + r.vh_only + r.neither
            left = 0
            for k, c, a in (("both", "#8a8a85", 1), ("vv_only", CVV, 1), ("vh_only", CVH, 1)):
                ax.barh(y, r[k] / n, left=left, color=c, height=0.7, edgecolor="white", linewidth=2)
                left += r[k] / n
            ax.text(left + 0.01, y, f"VH − VV {r.vh_minus_vv * 100:+.1f} pts ({r.vh_minus_vv_lo * 100:+.1f} to "
                    f"{r.vh_minus_vv_hi * 100:+.1f}),\nMcNemar p = {r.mcnemar_exact_p:.2f}", va="center", fontsize=7.5, color="#333")
            labs.append(f"{RNAME[rule]}, {'multi-date' if prod == 'multi' else 'single date'}")
            y += 1
    ax.set_yticks(range(len(labs)))
    ax.set_yticklabels(labs, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.25)
    ax.set_xlabel("share of the 803 segments")
    ax.set_title("Same segments: detected by both (gray), VV only (blue), VH only (orange)", fontsize=10)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    # human agreement
    ax = axs[1, 1]
    agd = pd.DataFrame(ag)
    show = [("multi_vv", "VV multi-date"), ("multi_vh", "VH multi-date"), ("multi_either", "VV or VH multi-date")]
    xs = np.arange(len(show))
    for k, (rule, c) in enumerate((("original", "#b8b6ad"), ("amended", "#5c5b55"), ("ls", "#1baf7a"))):
        v = [agd[(agd.rule == rule) & (agd["product"] == p)].agree_share.iloc[0] for p, _ in show]
        b = ax.bar(xs + (k - 1) * 0.26, v, width=0.24, color=c, edgecolor="white", linewidth=2, label=RNAME[rule])
        for xx, vv in zip(xs + (k - 1) * 0.26, v):
            ax.text(xx, vv + 0.01, f"{vv:.0%}", ha="center", fontsize=7.5, color="#333")
    ax.set_xticks(xs)
    ax.set_xticklabels([s for _, s in show], fontsize=9)
    ax.set_ylim(0, 1)
    ax.set_ylabel(f"agreement with the reviewer (n = {len(dec)} panels)")
    ax.set_title("Spot check: agreement with the reviewer's\n'visible on Sentinel-1' verdict (unsure, no feature excluded)", fontsize=9.5)
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.suptitle("VH against VV: same rule, same segments", fontsize=12)
    fig.tight_layout()
    fig.savefig(HERE / "figures" / "vh_vs_vv.png", dpi=120)
    plt.close(fig)
    print("figure written", flush=True)


if __name__ == "__main__":
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            main()
    except Exception:
        traceback.print_exc()
    sys.stdout.flush()
    os._exit(0)
