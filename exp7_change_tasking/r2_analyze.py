"""Round 2 analysis: Sentinel-1 catch rate of Capella change patches versus patch area.

Reads r2_pair_results.csv and data/r2/*_patches.csv from r2_run.py. Writes r2_curve.csv (binned recall
with site-bootstrap CIs), r2_summary.json (logistic fits, false alarm rates, region precision) and the
figures r2_curve.png and r2_curve_splits.png.

Recall per area bin pools the patches of all pairs. CIs resample sites with replacement (1,000
draws). The logistic fit is P(caught) = 1 / (1 + exp(-(a + b log10 area))), fitted by maximum
likelihood; the areas at 50% and 90% follow from a and b. The chance curve fits the same model to
each patch's chance-catch probability (S1 flag map shifted at random).
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "data" / "r2"
EDGES = np.array([8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 16384, 65536, 1e8])
NB = 1000
FLAGSETS = ("vv", "vh", "any")


def land(r):
    if r.near_water and r.wc_class != 80:
        return "water edge"
    return {50: "urban", 40: "farm", 60: "bare/mine", 80: "water"}.get(int(r.wc_class), "vegetation/other")


def load():
    pr = pd.read_csv(ROOT / "r2_pair_results.csv")
    ok = pr[pr.status == "ok"].copy()
    parts = []
    for r in ok.itertuples():
        f = OUT / f"{r.tag}_patches.csv"
        if f.exists() and f.stat().st_size > 0:
            p = pd.read_csv(f)
            if len(p):
                p["tag"], p["site_id"] = r.tag, r.site_id
                parts.append(p)
    P = pd.concat(parts, ignore_index=True)
    P["land"] = [land(r) for r in P.itertuples()]
    P["bin"] = np.digitize(P.area_m2, EDGES) - 1
    return pr, ok, P


def fit_logistic(x, y, w=None):
    """y may be 0/1 or a probability (soft labels); w = bootstrap weights."""
    w = np.ones_like(x) if w is None else w

    def nll(t):
        z = t[0] + t[1] * x
        return np.sum(w * (np.logaddexp(0, z) - y * z))
    return minimize(nll, [0.0, 1.0], method="L-BFGS-B").x


def fit_corrected(x, y, c, w=None):
    """P(caught) = c + (1 - c) * sigmoid(a + b x), c = the patch's chance-catch probability. The sigmoid is
    detection beyond chance: the share of patches Sentinel-1 catches because it saw them."""
    c = np.clip(c, 0, 0.999)
    w = np.ones_like(x) if w is None else w

    def nll(t):
        q = 1 / (1 + np.exp(-np.clip(t[0] + t[1] * x, -50, 50)))
        p = np.clip(c + (1 - c) * q, 1e-9, 1 - 1e-9)
        return -np.sum(w * (y * np.log(p) + (1 - y) * np.log(1 - p)))
    return minimize(nll, [0.0, 1.0], method="L-BFGS-B").x


def area_at(t, p):
    a, b = t
    if b <= 0:
        return np.nan
    return 10 ** ((np.log(p / (1 - p)) - a) / b)


def boot_weights(site_ids, rng, nb=NB):
    """Site-bootstrap draws as per-row weights: each row gets its site's multiplicity in the draw."""
    codes, uniq = pd.factorize(np.asarray(site_ids))
    S = len(uniq)
    for _ in range(nb):
        m = np.bincount(rng.integers(0, S, S), minlength=S).astype(float)
        yield m[codes]


def wbinned(bins, y, w):
    nb = len(EDGES) - 1
    ok = (bins >= 0) & (bins < nb) & np.isfinite(y)
    sw = np.bincount(bins[ok], weights=w[ok], minlength=nb)
    sy = np.bincount(bins[ok], weights=(w * y)[ok], minlength=nb)
    with np.errstate(invalid="ignore", divide="ignore"):
        return sy / sw


def curve_table(P, fs, rng):
    rows = []
    P = P.copy()
    P[f"{fs}_excess"] = (P[f"{fs}_loose"].astype(float) - P[f"{fs}_chance"]) / (1 - P[f"{fs}_chance"].clip(upper=0.999))
    cols = [f"{fs}_loose", f"{fs}_strict", f"{fs}_chance", f"{fs}_excess"]
    bins = P.bin.values
    Y = {c: P[c].values.astype(float) for c in cols}
    one = np.ones(len(P))
    est = {c: wbinned(bins, Y[c], one) for c in cols}
    n = P.groupby("bin").size().reindex(range(len(EDGES) - 1)).fillna(0).astype(int)
    nsite = P.groupby("bin").site_id.nunique().reindex(range(len(EDGES) - 1)).fillna(0).astype(int)
    bs = np.array([[wbinned(bins, Y[c], w) for c in cols] for w in boot_weights(P.site_id, rng)])
    for i in range(len(EDGES) - 1):
        row = {"flagset": fs, "area_lo_m2": EDGES[i], "area_hi_m2": EDGES[i + 1], "n_patches": n[i], "n_sites": nsite[i]}
        for k, c in enumerate(cols):
            v = bs[:, k, i]
            key = c.split("_")[1]
            row[key] = round(float(est[c][i]), 3) if np.isfinite(est[c][i]) else np.nan
            row[key + "_lo"] = round(float(np.nanpercentile(v, 2.5)), 3) if np.isfinite(v).any() else np.nan
            row[key + "_hi"] = round(float(np.nanpercentile(v, 97.5)), 3) if np.isfinite(v).any() else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def fits(P, col, rng, boot=True, corrected=False, nb=NB):
    x = np.log10(P.area_m2.values)
    y = P[col].values.astype(float)
    c = P[col.replace("loose", "chance")].values if corrected else None
    f = (lambda w: fit_corrected(x, y, c, w)) if corrected else (lambda w: fit_logistic(x, y, w))
    t = f(None)
    res = {"a": round(float(t[0]), 3), "b": round(float(t[1]), 3), "A50": area_at(t, 0.5), "A90": area_at(t, 0.9),
           "A10": area_at(t, 0.1), "max_area_m2": float(P.area_m2.max()), "p99_area_m2": float(np.percentile(P.area_m2, 99))}
    if boot:
        bs = [f(w) for w in boot_weights(P.site_id, rng, nb)]
        for k, p in (("A50", .5), ("A90", .9), ("A10", .1)):
            v = np.array([area_at(t2, p) for t2 in bs])
            res[k + "_n_undefined"] = int((~np.isfinite(v)).sum())
            v = v[np.isfinite(v)]
            res[k + "_ci"] = [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] if len(v) else None
    return res


def main():
    rng = np.random.default_rng(0)
    pr, ok, P = load()
    summ = {"pairs_run": int(len(pr)), "pairs_ok": int(len(ok)), "sites_ok": int(ok.site_id.nunique()),
            "status": pr.status.value_counts().to_dict(), "mb_read_total": round(float(pr.mb_read.sum()), 1),
            "n_patches": int(len(P)), "sites_with_patches": int(P.site_id.nunique()),
            "patches_per_site": P.site_id.value_counts().describe().round(1).to_dict()}
    print(json.dumps(summ, indent=1, default=str), flush=True)
    tabs = []
    for fs in FLAGSETS:
        if f"{fs}_loose" not in P or P[f"{fs}_loose"].isna().all():
            continue
        Q = P[P[f"{fs}_loose"].notna()]
        tabs.append(curve_table(Q, fs, rng))
        summ[fs] = {"n_patches": int(len(Q)), "sites": int(Q.site_id.nunique()),
                    "recall_loose_all": round(float(Q[f"{fs}_loose"].mean()), 3),
                    "recall_strict_all": round(float(Q[f"{fs}_strict"].mean()), 3),
                    "chance_all": round(float(Q[f"{fs}_chance"].mean()), 3),
                    "fit_loose": fits(Q, f"{fs}_loose", rng, boot=fs == "vv" or fs == "any"),
                    "fit_strict": fits(Q, f"{fs}_strict", rng, boot=fs == "vv"),
                    "fit_chance": fits(Q, f"{fs}_chance", rng, boot=fs == "vv"),
                    "fit_corrected": fits(Q, f"{fs}_loose", rng, boot=True, corrected=True)}
        if f"{fs}_sign_agree" in Q:
            c = Q[Q[f"{fs}_loose"].astype(bool)]
            summ[fs]["sign_agree_of_caught"] = round(float(c[f"{fs}_sign_agree"].mean()), 3) if len(c) else None
        k = ok[ok[f"{fs}_fa_rate"].notna()] if f"{fs}_fa_rate" in ok else ok.iloc[:0]
        if len(k):
            w = k.stable10_frac
            fa, wt = k[f"{fs}_fa_rate"].values, k.stable10_frac.values
            bs = [float(np.sum(w * wt * fa) / np.sum(w * wt)) for w in boot_weights(k.site_id, rng)]
            summ[fs]["fa_rate_pooled"] = round(float(np.sum(wt * fa) / np.sum(wt)), 4)
            summ[fs]["fa_rate_ci"] = [round(float(np.nanpercentile(bs, 2.5)), 4), round(float(np.nanpercentile(bs, 97.5)), 4)]
            summ[fs]["fa_rate_median_pair"] = round(float(k[f"{fs}_fa_rate"].median()), 4)
            summ[fs]["flag_frac_median_pair"] = round(float(k[f"{fs}_flag_frac"].median()), 4)
            for mp in (1, 20):
                kk = k[k[f"{fs}_nreg{mp}"] > 0]
                summ[fs][f"regprec{mp}_mean"] = round(float(kk[f"{fs}_regprec{mp}"].mean()), 3)
                summ[fs][f"regnull{mp}_mean"] = round(float(kk[f"{fs}_regnull{mp}"].mean()), 3)
                summ[fs][f"regprec{mp}_pooled_regions"] = int(kk[f"{fs}_nreg{mp}"].sum())
                summ[fs][f"regprec{mp}_weighted"] = round(float(np.average(kk[f"{fs}_regprec{mp}"], weights=kk[f"{fs}_nreg{mp}"])), 3)
                summ[fs][f"regnull{mp}_weighted"] = round(float(np.average(kk[f"{fs}_regnull{mp}"], weights=kk[f"{fs}_nreg{mp}"])), 3)
                summ[fs][f"regprec{mp}_pairs_beat_null"] = f"{int((kk[f'{fs}_regprec{mp}'] > kk[f'{fs}_regnull{mp}']).sum())} of {len(kk)}"
    T = pd.concat(tabs, ignore_index=True)
    T.to_csv(ROOT / "r2_curve.csv", index=False)
    pd.set_option("display.width", 250)
    print(T.to_string(index=False), flush=True)

    # splits (VV, loose)
    splits = {}
    for key in ("land", "sign"):
        for v, d in P.groupby(key):
            if d.site_id.nunique() < 3 or len(d) < 30:
                splits[f"{key}={v}"] = {"n": int(len(d)), "sites": int(d.site_id.nunique()), "note": "too few"}
                continue
            f = fits(d, "vv_loose", rng, boot=True, corrected=True, nb=300)
            fc = fits(d, "vv_loose", rng, boot=False)
            splits[f"{key}={v}"] = {"n": int(len(d)), "sites": int(d.site_id.nunique()), "recall": round(float(d.vv_loose.mean()), 3),
                                    "chance": round(float(d.vv_chance.mean()), 3), "A50": f["A50"], "A50_ci": f.get("A50_ci"),
                                    "A90": f["A90"], "A90_ci": f.get("A90_ci"), "raw_A50": fc["A50"], "raw_A90": fc["A90"]}
    summ["splits_vv_loose"] = splits
    # site-weighted check: every site counts once (mean of per-site binned recall)
    sw = P.groupby(["site_id", "bin"]).vv_loose.mean().groupby("bin").mean()
    summ["site_weighted_vv_loose_by_bin"] = {f"{EDGES[i]:.0f}-{EDGES[i+1]:.0f}": round(float(v), 3) for i, v in sw.items()}
    # dominance
    summ["top_site_share_of_patches"] = round(float(P.site_id.value_counts(normalize=True).iloc[0]), 3)
    (ROOT / "r2_summary.json").write_text(json.dumps(summ, indent=1, default=lambda o: o if not isinstance(o, float) else round(o, 4)))
    print(json.dumps({k: v for k, v in summ.items() if k in FLAGSETS or k == "splits_vv_loose"}, indent=1, default=str), flush=True)
    figures(T, P, summ)


def _sig(f, xs):
    return 1 / (1 + np.exp(-(f["a"] + f["b"] * np.log10(xs))))


def figures(T, P, summ):
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(1, 3, figsize=(18, 5.4), gridspec_kw={"width_ratios": [1.25, 1.25, 0.7]})
    d = T[(T.flagset == "vv") & (T.n_patches >= 10)]
    hi = d.area_hi_m2.clip(upper=4 * d.area_lo_m2)
    mid = np.sqrt(d.area_lo_m2 * hi)
    xs = np.logspace(np.log10(8), np.log10(max(mid.max() * 2, 1e3)), 200)
    colors = {"loose": "#1f6fb4", "strict": "#d0791b", "chance": "#8a8a8a", "excess": "#7a2a8c"}
    labels = {"loose": "caught: a flag within 10 m", "strict": "caught: >= 50% of patch under flags",
              "chance": "chance: flag map shifted at random"}
    for k in ("loose", "strict", "chance"):
        ax[0].errorbar(mid, d[k], yerr=[d[k] - d[k + "_lo"], d[k + "_hi"] - d[k]], fmt="o", color=colors[k], capsize=3,
                       label=labels[k], ms=5)
        ax[0].plot(xs, _sig(summ["vv"][f"fit_{k}"], xs), color=colors[k], lw=1.2, alpha=0.8)
    ax[0].set_title("A. Raw catch rate, Sentinel-1 VV", loc="left")
    ax[0].legend(loc="upper left", fontsize=8.5, frameon=False)
    k = "excess"
    ax[1].axhline(0, color="#bbb", lw=0.8)
    ax[1].errorbar(mid, d[k], yerr=[d[k] - d[k + "_lo"], d[k + "_hi"] - d[k]], fmt="o", color=colors[k], capsize=3, ms=5,
                   label="binned (caught - chance) / (1 - chance)")
    f = summ["vv"]["fit_corrected"]
    ax[1].plot(xs, _sig(f, xs), color=colors[k], lw=1.6, label="fit: detection beyond chance")
    top = mid.max() * 2
    for p, key in ((0.5, "A50"), (0.9, "A90")):
        v = f[key]
        if np.isfinite(v) and v <= top:
            ax[1].axvline(v, color=colors[k], ls=":", lw=1)
            ax[1].text(v, p + 0.03, f" {p:.0%} at {v:,.0f} m²", color=colors[k], fontsize=9)
        else:
            ax[1].text(xs[5], p + 0.03, f"{p:.0%} not reached within observed areas", color=colors[k], fontsize=8.5)
    ax[1].set_title("B. Chance-corrected detection, Sentinel-1 VV", loc="left")
    ax[1].legend(loc="upper left", fontsize=8.5, frameon=False, bbox_to_anchor=(0, 0.9))
    for a in ax[:2]:
        a.set_xscale("log")
        a.set_ylim(-0.1, 1.02)
        a.set_xlabel("Capella change patch area (m², 2 m pixels)")
        for i, r in enumerate(d.itertuples()):
            a.text(mid.iloc[i], -0.095, f"{r.n_patches}" + "\n" + f"{r.n_sites}s", ha="center", va="bottom", fontsize=6.5, color="#555")
    ax[0].set_ylabel("share of Capella change patches")
    fs = [s_ for s_ in FLAGSETS if s_ in summ and "fa_rate_pooled" in summ[s_]]
    vals = [summ[s_]["fa_rate_pooled"] for s_ in fs]
    ci = [summ[s_]["fa_rate_ci"] for s_ in fs]
    ax[2].bar(range(len(fs)), vals, color="#1f6fb4", alpha=0.8,
              yerr=[[v - c[0] for v, c in zip(vals, ci)], [c[1] - v for v, c in zip(vals, ci)]], capsize=4)
    ax[2].set_xticks(range(len(fs)), [{"vv": "VV", "vh": "VH", "any": "VV or VH"}[s_] for s_ in fs])
    ax[2].set_ylabel("flagged share of Capella-stable 10 m pixels")
    ax[2].set_title("C. False alarm rate", loc="left")
    for i, v in enumerate(vals):
        ax[2].text(i, v, f"{v:.1%}", ha="center", va="bottom", fontsize=9)
    fig.text(0.5, 0.005, "Bars: 95% intervals, sites resampled. Numbers under the points: patches and sites per bin "
             "(bins with >= 10 patches).", ha="center", fontsize=8.5, color="#555")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(ROOT / "r2_curve.png", dpi=130)
    plt.close(fig)

    sp = summ["splits_vv_loose"]
    P = P.copy()
    P["excess"] = (P.vv_loose.astype(float) - P.vv_chance) / (1 - P.vv_chance.clip(upper=0.999))
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
    for a, key in zip(ax, ("land", "sign")):
        for v, dd in P.groupby(key):
            if sp.get(f"{key}={v}", {}).get("note"):
                continue
            g = dd.groupby("bin")
            m = g.excess.mean()
            n = g.size()
            keep = (n >= 10).values
            mids = np.sqrt(EDGES[:-1] * np.minimum(EDGES[1:], 4 * EDGES[:-1]))[m.index[keep]]
            a.plot(mids, m.values[keep], "o-", ms=4, label=f"{v} ({len(dd)} patches, {dd.site_id.nunique()} sites)")
        a.axhline(0, color="#bbb", lw=0.8)
        a.set_xscale("log")
        a.set_ylim(-0.3, 1.02)
        a.set_xlabel("Capella change patch area (m²)")
        a.set_title({"land": "By land cover (ESA WorldCover)", "sign": "By sign of the Capella change"}[key], loc="left")
        a.legend(fontsize=8, frameon=False, loc="upper left")
    ax[0].set_ylabel("chance-corrected detection, Sentinel-1 VV")
    fig.text(0.5, 0.005, "Points: bins with >= 10 patches. Values pool patches; one busy site can dominate a class.",
             ha="center", fontsize=8.5, color="#555")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(ROOT / "r2_curve_splits.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
    sys.stdout.flush()
    os._exit(0)
