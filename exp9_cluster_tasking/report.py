"""Experiment 9 figures and spot-check panels.

Reads grid_tune.csv, chosen_rule.json, results_test.json, test_looks.csv. Writes fig_hit_rates.png,
fig_topk.png, fig_grid.png, and spotcheck/look_XX.png with spotcheck/looks.csv.
"""
from __future__ import annotations

import json
import os
import sys
import traceback

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import core as C

sys.path.insert(0, str(C.ROOT.parent))
from panel_roles import legend, outline  # noqa: E402

COL = {"cluster": "#1f6fb4", "bright": "#d0791b", "random": "#8a8a8a"}


def err(v, ci):
    return [[v - ci[0]], [ci[1] - v]]


def fig_rates(ch, te):
    keys = [f"{f}m_{h}m2" for f in C.FOOTPRINTS for h in C.HIT_M2]
    fig, ax = plt.subplots(1, 2, figsize=(14, 5), gridspec_kw={"width_ratios": [1, 1.5]})
    k = "500m_200m2"
    for j, (half, s) in enumerate((("tune", ch["tune"][k]), ("test", te["test"][k]))):
        for i, (name, v, ci) in enumerate((("Sentinel-1 clusters", s["hit_rate"], s["hit_rate_ci"]),
                                           ("brightest pixels", s["bright_rate"], s["bright_rate_ci"]),
                                           ("random looks", s["rand_rate"], s["rand_rate_ci"]))):
            x = j * 4 + i
            c = list(COL.values())[i]
            ax[0].bar(x, v, color=c, alpha=0.85, label=name if j == 0 else None)
            ax[0].errorbar(x, v, yerr=err(v, ci), color="k", capsize=3, lw=1)
            ax[0].text(x, ci[1] + 0.01, f"{v:.0%}", ha="center", fontsize=9)
        ax[0].text(j * 4 + 1, -0.08, f"{half} half\n{s['looks']} looks at {s['sites_with_looks']} of {s['sites']} sites", ha="center", fontsize=9)
    ax[0].set_xticks([])
    ax[0].set_ylim(0, 1.25)
    ax[0].set_ylabel("share of looks that contain Capella change")
    ax[0].set_title("A. Hit rate, 500 m look, hit = 200 m² of change", loc="left")
    ax[0].legend(frameon=False, fontsize=8.5, loc="upper center", ncol=3)
    for j, (half, src, off) in enumerate((("tune", ch["tune"], -0.17), ("test", te["test"], 0.17))):
        xs = np.arange(len(keys)) + off
        v = np.array([src[q]["lift"] for q in keys])
        lo = np.maximum(np.array([src[q]["lift_ci"][0] for q in keys]), 0.1)
        hi = np.array([src[q]["lift_ci"][1] for q in keys])
        ax[1].errorbar(xs, v, yerr=[v - lo, hi - v], fmt="o", capsize=3, label=f"{half} half",
                       color=("#1f6fb4", "#7a2a8c")[j])
    ax[1].axhline(1, color="#999", lw=1, ls="--")
    ax[1].set_yscale("log")
    ax[1].set_xticks(range(len(keys)), [q.replace("m_", " m look\n").replace("m2", " m² hit") for q in keys], fontsize=8)
    ax[1].set_ylabel("lift: cluster hits / expected random hits (log)")
    ax[1].set_title("B. Lift over random looks, every look and hit setting", loc="left")
    ax[1].legend(frameon=False, fontsize=9)
    fig.text(0.5, 0.005, f"Rule: {ch['rule_name']}. Bars and whiskers: 95% intervals, sites resampled.", ha="center",
             fontsize=8.5, color="#555")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(C.ROOT / "fig_hit_rates.png", dpi=130)
    plt.close(fig)


def fig_topk(ch, te):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
    for a, foot in zip(ax, ("500m", "250m")):
        for j, (half, src, off) in enumerate((("tune", ch["tune_topk"][foot], -0.15), ("test", te["test_topk"][foot], 0.15))):
            ks = sorted(src, key=int)
            xs = np.arange(len(ks)) + off
            v = np.array([src[k]["hit_rate"] for k in ks])
            ci = np.array([src[k]["hit_rate_ci"] for k in ks])
            r = np.array([src[k]["rand_rate"] for k in ks])
            col = ("#1f6fb4", "#7a2a8c")[j]
            a.errorbar(xs, v, yerr=[v - ci[:, 0], ci[:, 1] - v], fmt="o", color=col, capsize=3, label=f"{half}: top clusters")
            a.plot(xs, r, "_", ms=18, mew=2, color=col, alpha=0.5, label=f"{half}: random, same scenes")
        a.set_xticks(range(3), ["top 1", "top 3", "top 10"])
        a.set_ylim(0, 1.05)
        a.set_title(f"{foot.replace('m', ' m')} look, hit = 200 m²", loc="left")
    ax[0].set_ylabel("hit rate of the top-scored clusters per scene")
    ax[0].legend(frameon=False, fontsize=8, loc="lower right")
    fig.text(0.5, 0.005, "Score = cluster area x mean absolute change. Whiskers: 95% intervals, sites resampled.",
             ha="center", fontsize=8.5, color="#555")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(C.ROOT / "fig_topk.png", dpi=130)
    plt.close(fig)


def fig_grid(ch):
    G = pd.read_csv(C.ROOT / "grid_tune.csv")
    P = G[(G.foot == 500) & (G.hit_m2 == 200)]
    fig, ax = plt.subplots(figsize=(8, 5))
    mk = {"vv": "o", "vh": "s", "any": "^", "both": "D"}
    for pol, d in P.groupby("pol"):
        ok = d.looks >= C.MIN_LOOKS
        ax.scatter(d.looks[ok], d.lift[ok], marker=mk[pol], s=22, alpha=0.7, label=pol)
        ax.scatter(d.looks[~ok], d.lift[~ok], marker=mk[pol], s=22, facecolor="none", edgecolor="#aaa")
    c = P[P.rule == ch["rule_name"]].iloc[0]
    ax.scatter([c.looks], [c.lift], s=160, facecolor="none", edgecolor="crimson", lw=2, label="chosen")
    ax.axhline(1, color="#999", ls="--", lw=1)
    ax.set_xscale("log")
    ax.set_xlabel("looks on the tune half")
    ax.set_ylabel("tune lift over random (500 m, 200 m²)")
    ax.set_title("All 180 rules on the tune half (hollow grey: under 30 looks)", loc="left", fontsize=10)
    ax.legend(frameon=False, fontsize=8, title="polarization")
    fig.tight_layout()
    fig.savefig(C.ROOT / "fig_grid.png", dpi=130)
    plt.close(fig)


def spotcheck(te):
    out = C.ROOT / "spotcheck"
    out.mkdir(exist_ok=True)
    rule = te["rule"]
    L = pd.read_csv(C.ROOT / "test_looks.csv")
    L = L[L.foot == 500].copy()
    L["hit"] = L.change_m2 >= 200
    rng = np.random.default_rng(90)
    # the test half yields only 28 cluster looks (25 hits, 3 misses at 500 m / 200 m2): show all of them,
    # then fill to 40 with random looks
    pick = [{"kind": "cluster", **r} for r in L.to_dict("records")]
    _, meta = C.load_pairs("test")
    tags = meta.tag.tolist()
    pairs = {}
    for t in rng.choice(tags, max(40 - len(pick), 10), replace=True):
        sid = int(meta.set_index("tag").site_id[t])
        p = pairs.setdefault(t, C.Pair(t, sid))
        rr, cc = np.nonzero(p.domain)
        i = int(rng.integers(len(rr)))
        pick.append({"kind": "random", "tag": t, "site_id": sid, "row": int(rr[i]), "col": int(cc[i]),
                     "change_m2": float(p.area_map(500)[rr[i], cc[i]])})
    order = rng.permutation(len(pick))
    rows = []
    for n, i in enumerate(order, 1):
        lk = pick[i]
        p = pairs.setdefault(lk["tag"], C.Pair(lk["tag"], lk["site_id"]))
        r0, c0, w = p.look_box(int(lk["row"]), int(lk["col"]), 500)
        z = p._z
        sl2 = (slice(r0 * C.K, (r0 + w) * C.K), slice(c0 * C.K, (c0 + w) * C.K))
        sl1 = (slice(r0, r0 + w), slice(c0, c0 + w))
        a = z["a_db"][sl2].astype("float32")
        b = z["b_db"][sl2].astype("float32")
        mg = z["mg"][sl2].astype("float32")
        lab = z["lab"][sl2] > 0
        cl, clab = p.clusters(rule)
        _, d = p.flags(rule)
        inc = np.isin(clab, cl.label.values) if len(cl) else np.zeros_like(clab, bool)
        fig, ax = plt.subplots(1, 4, figsize=(16, 4.6))
        lo, hi = np.nanpercentile(np.r_[a[np.isfinite(a)], b[np.isfinite(b)]], [2, 98]) if np.isfinite(a).any() else (-30, 0)
        ax[0].imshow(a, cmap="gray", vmin=lo, vmax=hi)
        ax[0].set_title("Capella before (2 m)")
        ax[1].imshow(b, cmap="gray", vmin=lo, vmax=hi)
        ax[1].set_title("Capella after (2 m)")
        ax[2].imshow(np.where(np.isfinite(mg), mg, np.nan), cmap="RdBu_r", vmin=-8, vmax=8)
        if lab.any():
            ax[2].contour(lab.astype(float), levels=[0.5], colors="#7e3fc0", linewidths=0.9)
        ax[2].set_title("Capella change, patches outlined")
        outline(ax[2], "truth")
        ax[3].imshow(d[sl1], cmap="RdBu_r", vmin=-8, vmax=8)
        if inc[sl1].any():
            ax[3].contour(inc[sl1].astype(float), levels=[0.5], colors="#1f9d55", linewidths=1.4)
        ax[3].set_title("Sentinel-1 change (10 m), clusters")
        outline(ax[3], "model")
        cy, cx = int(lk["row"]) - r0, int(lk["col"]) - c0
        for j, a_ in enumerate(ax):
            f = C.K if j < 3 else 1
            a_.plot([cx * f + f / 2], [cy * f + f / 2], "+", color="yellow", ms=12, mew=1.5)
            a_.set_xticks([])
            a_.set_yticks([])
        fig.suptitle(f"look {n:02d}   500 m square, + = look center   (color scale +-8 dB)", fontsize=11)
        legend(fig)
        fig.tight_layout(rect=(0, 0.04, 1, 0.95))
        fn = f"look_{n:02d}.png"
        fig.savefig(out / fn, dpi=100)
        plt.close(fig)
        from pyproj import Transformer
        x = p.x0 + (int(lk["col"]) + 0.5) * C.RES
        y = p.y1 - (int(lk["row"]) + 0.5) * C.RES
        lon, lat = Transformer.from_crs(p.epsg, 4326, always_xy=True).transform(x, y)
        rows.append({"file": fn, "look": n, "site_id": lk["site_id"], "pair": lk["tag"], "lon": round(lon, 5),
                     "lat": round(lat, 5), "look_type": lk["kind"], "capella_change_m2": round(float(lk["change_m2"])),
                     "auto_hit_200m2": bool(lk["change_m2"] >= 200), "verdict": ""})
    pd.DataFrame(rows).to_csv(out / "looks.csv", index=False)
    print("spotcheck panels:", len(rows), flush=True)


def main():
    ch = json.loads((C.ROOT / "chosen_rule.json").read_text())
    te = json.loads((C.ROOT / "results_test.json").read_text())
    fig_rates(ch, te)
    fig_topk(ch, te)
    fig_grid(ch)
    spotcheck(te)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
    sys.stdout.flush()
    os._exit(0)
