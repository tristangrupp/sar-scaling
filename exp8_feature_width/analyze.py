"""Step 4: detection curves, scene-bootstrap intervals, logistic widths at 50% / 90%, figures.

Reads data/<capella_id>/segments.csv and meta.json. Writes segments_all.csv, curve_bins.csv,
width_table.csv, summary.json and figures/*.png.
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
from scipy.optimize import minimize

HERE = Path(__file__).resolve().parent
FIG = HERE / "figures"
EDGES = [0.75, 2, 3, 5, 8, 12, 20, 30, 40]
BLAB = ["<2", "2-3", "3-5", "5-8", "8-12", "12-20", "20-30", "30-40"]
CEDGES = [1, 3, 6, 10, 100]
CLAB = ["1-3 dB", "3-6 dB", "6-10 dB", ">10 dB"]
TYPES = ["tree_line", "road", "track", "ditch_canal", "field_edge"]
TNAME = {"tree_line": "hedgerows / tree lines", "road": "roads", "track": "tracks and paths",
         "ditch_canal": "ditches and canals", "field_edge": "field-edge strips"}
PRODS = [("multi", "either"), ("single", "either"), ("multi", "vv"), ("multi", "vh"), ("single", "vv"), ("single", "vh")]
NB = 1000
RNG = np.random.default_rng(8)
VAR = os.environ.get("EXP8_VARIANT", "")      # "" = pre-set rule; "db" = post-hoc dB-domain contrast (rescore.py)
SUF = f"_{VAR}" if VAR else ""


def load():
    segs, metas = [], []
    for d in sorted((HERE / "data").iterdir()):
        if (d / "meta.json").exists():
            segs.append(pd.read_csv(d / "segments.csv"))
            metas.append(json.loads((d / "meta.json").read_text()))
    return pd.concat(segs, ignore_index=True), pd.DataFrame(metas)


def det_col(prod, pol):
    return f"{prod}_{pol}_det"


def far(df, prod, pol):
    return df[f"{prod}_{pol}_far_n"].sum() / max(1, df[f"{prod}_{pol}_far_m"].sum())


def fit(w, y, f):
    """MLE of p = f + (1-f) * sigmoid(a + b log2 w). Returns (a, b) or None."""
    x = np.log2(w)
    if len(y) < 15 or y.min() == y.max():
        return None

    def nll(t):
        q = 1 / (1 + np.exp(-(t[0] + t[1] * x)))
        p = np.clip(f + (1 - f) * q, 1e-6, 1 - 1e-6)
        return -np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))

    best = None
    for b0 in (0.5, 1.5, 3.0):
        r = minimize(nll, [-b0 * 3, b0], method="Nelder-Mead", options={"xatol": 1e-4, "fatol": 1e-6, "maxiter": 4000})
        if best is None or r.fun < best.fun:
            best = r
    return best.x


def width_at(ab, f, p):
    if ab is None or ab[1] <= 0 or p <= f:
        return np.nan
    q = (p - f) / (1 - f)
    return float(2 ** ((np.log(q / (1 - q)) - ab[0]) / ab[1]))


def curve(ab, f, ws):
    if ab is None:
        return np.full(len(ws), np.nan)
    return f + (1 - f) / (1 + np.exp(-(ab[0] + ab[1] * np.log2(ws))))


def boot(df, prod, pol, f, nb=NB):
    """Scene bootstrap: bin rates and fitted widths."""
    col = det_col(prod, pol)
    scenes = df.capella_id.unique()
    groups = {s: g for s, g in df.groupby("capella_id")}
    rates, w50, w90 = [], [], []
    for _ in range(nb):
        pick = RNG.choice(scenes, len(scenes), replace=True)
        b = pd.concat([groups[s] for s in pick], ignore_index=True)
        rates.append(b.groupby("wbin", observed=False)[col].mean().reindex(BLAB).values)
        ab = fit(b.cap_width.values, b[col].values.astype(float), f)
        w50.append(width_at(ab, f, 0.5))
        w90.append(width_at(ab, f, 0.9))
    return np.array(rates, dtype=float), np.array(w50, dtype=float), np.array(w90, dtype=float)


def fmt_w(point, lo, hi):
    def one(v):
        if not np.isfinite(v):
            return "n/a"
        return ">40" if v > 40 else f"{v:.1f}"
    return f"{one(point)} ({one(lo)}–{one(hi)})"


def summarize(df, label, prod, pol, f, nb=NB, rows=None):
    col = det_col(prod, pol)
    ab = fit(df.cap_width.values, df[col].values.astype(float), f)
    p50, p90 = width_at(ab, f, 0.5), width_at(ab, f, 0.9)
    _, b50, b90 = boot(df, prod, pol, f, nb)
    q = lambda a: (np.nanpercentile(a, 2.5), np.nanpercentile(a, 97.5)) if np.isfinite(a).sum() > 20 else (np.nan, np.nan)
    l50, h50 = q(np.where(np.isfinite(b50), b50, 1e3))
    l90, h90 = q(np.where(np.isfinite(b90), b90, 1e3))
    rec = {"group": label, "product": prod, "pol": pol, "n_segments": len(df), "n_scenes": df.capella_id.nunique(),
           "det_rate": round(df[col].mean(), 3), "false_alarm": round(f, 3),
           "w50": p50, "w50_lo": l50, "w50_hi": h50, "w90": p90, "w90_lo": l90, "w90_hi": h90,
           "w50_txt": fmt_w(p50, l50, h50), "w90_txt": fmt_w(p90, l90, h90),
           "boot_w50_undefined_frac": round(float(np.mean(~np.isfinite(b50) | (b50 > 40))), 3),
           "a": None if ab is None else round(ab[0], 3), "b": None if ab is None else round(ab[1], 3)}
    if rows is not None:
        rows.append(rec)
    return rec, ab


def main():
    FIG.mkdir(exist_ok=True)
    allseg, meta = load()
    if VAR:
        allseg = pd.read_csv(HERE / f"segments_all{SUF}.csv")
    else:
        allseg.to_csv(HERE / "segments_all.csv", index=False)
        meta.to_csv(HERE / "scene_meta.csv", index=False)
    bad = meta[~meta.osm_refined.astype(bool)].capella_id.tolist()   # amendment 3: no ground-feature check possible
    ok = allseg[(allseg.status == "ok") & ~allseg.capella_id.isin(bad)].copy()
    for c in [c for c in ok.columns if c.endswith("_det")]:
        ok[c] = ok[c].map({True: 1.0, False: 0.0, "True": 1.0, "False": 0.0}).astype(float)
    ok["wbin"] = pd.cut(ok.cap_width, EDGES, labels=BLAB, right=False)
    ok["abs_contrast"] = ok.cap_contrast.abs()
    ok["cbin"] = pd.cut(ok.abs_contrast, CEDGES, labels=CLAB, right=False)
    ok["bright"] = ok.cap_contrast > 0
    ok["orient"] = np.where(ok.angle_to_flight <= 30, "along flight (azimuth)",
                            np.where(ok.angle_to_flight >= 60, "across flight (range)", "oblique"))
    print("scenes", meta.capella_id.nunique(), "bad align", bad, "ok segments", len(ok), flush=True)
    status = allseg.groupby(["type", "status"]).size().unstack(fill_value=0)
    status.to_csv(HERE / f"segment_status{SUF}.csv")
    print(status.to_string(), flush=True)

    fars = {(p, q): far(ok, p, q) for p, q in PRODS}
    summary = {"n_scenes_measured": int(meta.capella_id.nunique()), "excluded_alignment": bad,
               "n_segments_total": int(len(allseg)), "n_segments_ok": int(len(ok)),
               "n_ok_by_type": ok.type.value_counts().to_dict(), "false_alarm": {f"{p}_{q}": round(v, 4) for (p, q), v in fars.items()},
               "capella_mb": float(meta.capella_mb.sum()), "s1_mb": float(meta.s1_mb.sum())}

    # bin table with scene-bootstrap CIs
    bins = []
    for p, q in PRODS:
        col = det_col(p, q)
        rates, _, _ = boot(ok, p, q, fars[(p, q)], nb=300)
        for g, lab in [(ok, "all")] + [(ok[ok.type == t], t) for t in TYPES]:
            gb = g.groupby("wbin", observed=False)[col].agg(["mean", "size"]).reindex(BLAB)
            for i, b in enumerate(BLAB):
                rec = {"product": p, "pol": q, "group": lab, "wbin": b, "n": int(gb.loc[b, "size"]) if pd.notna(gb.loc[b, "size"]) else 0,
                       "rate": round(gb.loc[b, "mean"], 3) if pd.notna(gb.loc[b, "mean"]) else np.nan}
                if lab == "all":
                    rec["lo"] = round(np.nanpercentile(rates[:, i], 2.5), 3) if np.isfinite(rates[:, i]).any() else np.nan
                    rec["hi"] = round(np.nanpercentile(rates[:, i], 97.5), 3) if np.isfinite(rates[:, i]).any() else np.nan
                bins.append(rec)
    bins = pd.DataFrame(bins)
    # per-group bin CIs for type split (multi/single either)
    extra = []
    for p, q in PRODS[:2]:
        col = det_col(p, q)
        for t in TYPES:
            g = ok[ok.type == t]
            if g.capella_id.nunique() < 3:
                continue
            r, _, _ = boot(g, p, q, fars[(p, q)], nb=300)
            for i, b in enumerate(BLAB):
                extra.append({"product": p, "pol": q, "group": t, "wbin": b,
                              "lo2": np.nanpercentile(r[:, i], 2.5) if np.isfinite(r[:, i]).any() else np.nan,
                              "hi2": np.nanpercentile(r[:, i], 97.5) if np.isfinite(r[:, i]).any() else np.nan})
    if extra:
        bins = bins.merge(pd.DataFrame(extra), on=["product", "pol", "group", "wbin"], how="left")
        bins["lo"] = bins["lo"].fillna(bins["lo2"]).round(3)
        bins["hi"] = bins["hi"].fillna(bins["hi2"]).round(3)
        bins = bins.drop(columns=["lo2", "hi2"])
    bins.to_csv(HERE / f"curve_bins{SUF}.csv", index=False)

    # widths at 50% / 90%
    rows, fits = [], {}
    for p, q in PRODS:
        f = fars[(p, q)]
        _, fits[(p, q, "all")] = summarize(ok, "all", p, q, f, rows=rows)
        print(rows[-1]["product"], rows[-1]["pol"], rows[-1]["w50_txt"], rows[-1]["w90_txt"], flush=True)
        if q != "either":
            continue
        for t in TYPES:
            g = ok[ok.type == t]
            if len(g) >= 15 and g.capella_id.nunique() >= 3:
                _, fits[(p, q, t)] = summarize(g, t, p, q, f, nb=400, rows=rows)
        for c in CLAB:
            g = ok[ok.cbin == c]
            if len(g) >= 15:
                _, fits[(p, q, c)] = summarize(g, "contrast " + c, p, q, f, nb=400, rows=rows)
        for lab, g in [("bright", ok[ok.bright]), ("dark", ok[~ok.bright]),
                       ("along flight (azimuth)", ok[ok.orient == "along flight (azimuth)"]),
                       ("across flight (range)", ok[ok.orient == "across flight (range)"])]:
            if len(g) >= 15:
                _, fits[(p, q, lab)] = summarize(g, lab, p, q, f, nb=400, rows=rows)
        for t in TYPES:
            for lab, g in [("bright", ok[(ok.type == t) & ok.bright]), ("dark", ok[(ok.type == t) & ~ok.bright])]:
                if len(g) >= 15 and g.capella_id.nunique() >= 3:
                    summarize(g, f"{t} {lab}", p, q, f, nb=300, rows=rows)
    wt = pd.DataFrame(rows)
    wt.to_csv(HERE / f"width_table{SUF}.csv", index=False)

    # 2-D: width x contrast
    heat = {}
    for p in ("multi", "single"):
        col = det_col(p, "either")
        h = ok.groupby(["cbin", "wbin"], observed=False)[col].agg(["mean", "size"]).reset_index()
        h["product"] = p
        heat[p] = h
    pd.concat(heat.values()).to_csv(HERE / f"width_contrast_grid{SUF}.csv", index=False)

    # logistic with width and contrast (multi either)
    two = {}
    for p in ("multi", "single"):
        col = det_col(p, "either")
        f = fars[(p, "either")]
        x1, x2, y = np.log2(ok.cap_width.values), ok.abs_contrast.values, ok[col].values.astype(float)

        def nll(t):
            qq = 1 / (1 + np.exp(-(t[0] + t[1] * x1 + t[2] * x2)))
            pp = np.clip(f + (1 - f) * qq, 1e-6, 1 - 1e-6)
            return -np.sum(y * np.log(pp) + (1 - y) * np.log(1 - pp))

        r = minimize(nll, [-3, 1, 0.1], method="Nelder-Mead", options={"maxiter": 8000, "xatol": 1e-5, "fatol": 1e-7})
        a, b, c = r.x
        w50 = {}
        for ct in (2, 4, 8, 12):
            qq = (0.5 - f) / (1 - f)
            w50[f"{ct} dB"] = round(float(2 ** ((np.log(qq / (1 - qq)) - a - c * ct) / b)), 1) if b > 0 else None
        two[p] = {"a": round(a, 3), "b_log2w": round(b, 3), "c_per_dB": round(c, 3), "w50_at_contrast": w50}
    summary["two_factor_logistic"] = two

    # sign agreement
    sa = {}
    for p in ("single", "multi"):
        for q in ("vv", "vh"):
            d = ok[ok[det_col(p, q)] == 1]
            sa[f"{p}_{q}"] = round(float((np.sign(d[f"{p}_{q}_C"]) == np.sign(d.cap_contrast)).mean()), 3) if len(d) else None
    summary["sign_agreement_detected"] = sa
    # sensitivity: all 30 scenes, including the 8 without a ground-feature check (point estimates only)
    allok = allseg[allseg.status == "ok"].copy()
    sens = {}
    for p, q in PRODS[:2]:
        col = det_col(p, q)
        y = allok[col].map({True: 1.0, False: 0.0, "True": 1.0, "False": 0.0}).astype(float).values
        ab = fit(allok.cap_width.values, y, fars[(p, q)])
        sens[f"{p}_{q}"] = {"n": int(len(allok)), "w50": width_at(ab, fars[(p, q)], 0.5), "w90": width_at(ab, fars[(p, q)], 0.9)}
    summary["sensitivity_all_scenes"] = sens
    summary["width_median_by_type"] = ok.groupby("type").cap_width.median().round(2).to_dict()
    summary["contrast_median_by_type"] = ok.groupby("type").cap_contrast.median().round(2).to_dict()
    summary["bright_share_by_type"] = ok.groupby("type").bright.mean().round(3).to_dict()
    (HERE / f"summary{SUF}.json").write_text(json.dumps(summary, indent=1, default=str))

    # ---------------- figures
    ws = np.exp(np.linspace(np.log(0.8), np.log(40), 200))
    mids = np.sqrt(np.array(EDGES[:-1]) * np.array(EDGES[1:]))
    colors = {"multi": "#1f5fa8", "single": "#d07a1e"}
    # 1. headline curve
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for p in ("single", "multi"):
        b = bins[(bins["product"] == p) & (bins.pol == "either") & (bins.group == "all")]
        yerr = np.array([b.rate - b.lo, b.hi - b.rate])
        ax.errorbar(mids * (1.04 if p == "multi" else 0.96), b.rate, yerr=yerr, fmt="o", color=colors[p], capsize=3,
                    label=f"{'multi-date mean' if p == 'multi' else 'single date'} (bins, 95% scene bootstrap)")
        ax.plot(ws, curve(fits[(p, "either", "all")], fars[(p, "either")], ws), color=colors[p], lw=2)
        for x, n in zip(mids, b.n):
            if p == "multi":
                ax.annotate(f"n={n}", (x, 0.02), ha="center", fontsize=7, color="#555")
    ax.axhline(fars[("multi", "either")], color="#888", ls=":", lw=1)
    ax.text(0.85, fars[("multi", "either")] + 0.015, "false-alarm floor", fontsize=8, color="#666")
    for yv in (0.5, 0.9):
        ax.axhline(yv, color="#ccc", lw=0.8)
    ax.set_xscale("log")
    ax.set_xticks([1, 2, 3, 5, 8, 12, 20, 30, 40])
    ax.set_xticklabels(["1", "2", "3", "5", "8", "12", "20", "30", "40"])
    ax.set_xlabel("feature width on Capella (m, full width at half contrast)")
    ax.set_ylabel("share detected on Sentinel-1 (VV or VH)")
    ax.set_ylim(0, 1.02)
    ax.set_title("How wide a linear feature must be for Sentinel-1 to show it", fontsize=11)
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIG / f"detection_curve{SUF}.png", dpi=130)
    plt.close(fig)

    # 2. by type
    fig, axs = plt.subplots(1, len(TYPES), figsize=(17, 3.9), sharey=True)
    for ax, t in zip(axs, TYPES):
        for p in ("single", "multi"):
            b = bins[(bins["product"] == p) & (bins.pol == "either") & (bins.group == t)]
            m = b.n >= 5
            if m.any():
                yerr = np.array([(b.rate - b.lo)[m].clip(lower=0), (b.hi - b.rate)[m].clip(lower=0)]) if b.lo.notna().any() else None
                ax.errorbar(mids[m.values] * (1.04 if p == "multi" else 0.96), b.rate[m], yerr=yerr, fmt="o", ms=4,
                            color=colors[p], capsize=2)
            if (p, "either", t) in fits:
                ax.plot(ws, curve(fits[(p, "either", t)], fars[(p, "either")], ws), color=colors[p], lw=1.8,
                        label="multi-date" if p == "multi" else "single date")
        ax.set_xscale("log")
        ax.set_xticks([1, 2, 5, 10, 20, 40])
        ax.set_xticklabels(["1", "2", "5", "10", "20", "40"])
        ax.set_title(f"{TNAME[t]} (n={int((ok.type == t).sum())})", fontsize=9)
        ax.set_xlabel("width (m)")
        ax.axhline(0.5, color="#ccc", lw=0.8)
        ax.axhline(0.9, color="#ccc", lw=0.8)
    axs[0].set_ylabel("share detected (VV or VH)")
    axs[0].legend(fontsize=8)
    fig.suptitle("Detection by feature type (bins with at least 5 segments)", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG / f"detection_by_type{SUF}.png", dpi=120)
    plt.close(fig)

    # 3. width x contrast heat map
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.2))
    for ax, p in zip(axs, ("single", "multi")):
        h = heat[p]
        M = h.pivot(index="cbin", columns="wbin", values="mean").reindex(index=CLAB, columns=BLAB)
        Nn = h.pivot(index="cbin", columns="wbin", values="size").reindex(index=CLAB, columns=BLAB).fillna(0)
        Mm = np.where(Nn.values >= 5, M.values, np.nan)
        im = ax.imshow(Mm, cmap="viridis", vmin=0, vmax=1, aspect="auto", origin="lower")
        for i in range(len(CLAB)):
            for j in range(len(BLAB)):
                if Nn.values[i, j] > 0:
                    txt = f"{M.values[i, j]:.0%}\nn={int(Nn.values[i, j])}" if Nn.values[i, j] >= 5 else f"n={int(Nn.values[i, j])}"
                    ax.text(j, i, txt, ha="center", va="center", fontsize=7,
                            color="white" if (np.isfinite(Mm[i, j]) and Mm[i, j] < 0.6) else "black")
        ax.set_xticks(range(len(BLAB)))
        ax.set_xticklabels(BLAB)
        ax.set_yticks(range(len(CLAB)))
        ax.set_yticklabels(CLAB)
        ax.set_xlabel("width on Capella (m)")
        ax.set_ylabel("|contrast| on Capella")
        ax.set_title(f"{'Multi-date mean' if p == 'multi' else 'Single date'}: share detected", fontsize=10)
    fig.colorbar(im, ax=axs, shrink=0.8, label="share detected")
    fig.savefig(FIG / f"width_by_contrast{SUF}.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    # 4. polarization and orientation
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.2))
    for p, q, c, ls in [("multi", "vv", "#1f5fa8", "-"), ("multi", "vh", "#1f5fa8", "--"), ("single", "vv", "#d07a1e", "-"),
                        ("single", "vh", "#d07a1e", "--")]:
        axs[0].plot(ws, curve(fits[(p, q, "all")], fars[(p, q)], ws), color=c, ls=ls, label=f"{p} {q.upper()}")
    axs[0].set_title("By polarization (|z| > 2 in that polarization)", fontsize=10)
    for lab, ls in [("along flight (azimuth)", "-"), ("across flight (range)", "--")]:
        for p in ("multi", "single"):
            if (p, "either", lab) in fits:
                axs[1].plot(ws, curve(fits[(p, "either", lab)], fars[(p, "either")], ws), color=colors[p], ls=ls,
                            label=f"{p}: feature {lab}")
    axs[1].set_title("By feature direction relative to the Sentinel-1 flight path", fontsize=10)
    for ax in axs:
        ax.set_xscale("log")
        ax.set_xticks([1, 2, 5, 10, 20, 40])
        ax.set_xticklabels(["1", "2", "5", "10", "20", "40"])
        ax.set_xlabel("width (m)")
        ax.set_ylim(0, 1)
        ax.axhline(0.5, color="#ccc", lw=0.8)
        ax.legend(fontsize=8)
    axs[0].set_ylabel("share detected")
    fig.tight_layout()
    fig.savefig(FIG / f"polarization_orientation{SUF}.png", dpi=120)
    plt.close(fig)
    print(json.dumps(summary, indent=1, default=str), flush=True)
    print(wt[["group", "product", "pol", "n_segments", "det_rate", "w50_txt", "w90_txt"]].to_string(), flush=True)


if __name__ == "__main__":
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            main()
    except Exception:
        traceback.print_exc()
    sys.stdout.flush()
    os._exit(0)
