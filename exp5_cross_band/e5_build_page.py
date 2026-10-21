"""Write exp5_cross_band/page_body.html and exp5_cross_band/page_scripts.html from the result CSVs (site builder wraps them).

Narrative text lives in narrative.py (written after the numbers are in) so this file stays mechanical.
"""
from __future__ import annotations

import base64
import json
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
PROJ = ROOT.parent

NAMES = {"bicubic4": "Bicubic", "bicubic": "Bicubic", "exp1_cnn": "Exp 1 CNN (real S1→X pairs)",
         "x4_cond": "X-synthetic 4x, conditioned", "x4_unc": "X-synthetic 4x, unconditioned",
         "x2_cond": "X-synthetic 2x, conditioned", "x2_unc": "X-synthetic 2x, unconditioned",
         "c4": "C-band-only 4x (exp 4 means)", "c2": "C-band-only 2x (exp 4 means)",
         "xband_other_collect": "Another real X-band collect (ceiling)", "input (10 m)": "Input, no filter (10 m)",
         "null_other_tile": "Unrelated tile's S1 (chance level)", "x4_aug": "X-synthetic 4x, band-augmented",
         "bicubic4+adapt": "Bicubic + band adapt", "x4_cond+adapt": "X-synthetic 4x cond. + band adapt",
         "x4_unc+adapt": "X-synthetic 4x unc. + band adapt", "x4_aug+adapt": "X-synthetic 4x band-aug. + band adapt",
         "x2_cond+adapt": "X-synthetic 2x cond. + band adapt", "x2_unc+adapt": "X-synthetic 2x unc. + band adapt"}
ORDER = ["xband_other_collect", "null_other_tile", "input (10 m)", "bicubic4", "bicubic", "bicubic4+adapt", "exp1_cnn",
         "x4_cond", "x4_cond+adapt", "x4_unc", "x4_unc+adapt", "x4_aug", "x4_aug+adapt", "x2_cond", "x2_cond+adapt",
         "x2_unc", "x2_unc+adapt", "c4", "c2"]


def img64(p, max_w=1200):
    from PIL import Image
    im = Image.open(p)
    if im.width > max_w:
        im = im.resize((max_w, int(im.height * max_w / im.width)), Image.LANCZOS)
    im = im.convert("RGB").quantize(128)
    b = BytesIO()
    im.save(b, "PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(b.getvalue()).decode()


def fmt(v, d=3):
    if isinstance(v, (float, np.floating)):
        if np.isnan(v):
            return "–"
        return f"{v:.{d}f}"
    return str(v)


def table(df, cols, labels, first="model", first_label="Model", best=None, digits=None):
    """best: {col: 'min'|'max'} to bold the best non-reference row."""
    digits = digits or {}
    h = "<div class='tw'><table><tr><th>" + first_label + "</th>" + "".join(f"<th>{labels[c]}</th>" for c in cols) + "</tr>"
    bests = {}
    if best:
        cand = df[~df[first].isin(["xband_other_collect", "input (10 m)", "null_other_tile"])]
        for c, how in best.items():
            if c in cand and cand[c].notna().any():
                bests[c] = cand[c].min() if how == "min" else cand[c].max()
    for _, r in df.iterrows():
        h += f"<tr><td>{NAMES.get(r[first], r[first])}</td>"
        for c in cols:
            v = r.get(c, np.nan)
            s = fmt(v, digits.get(c, 3))
            if c in bests and not pd.isna(v) and v == bests[c]:
                s = f"<b>{s}</b>"
            h += f"<td class='n'>{s}</td>"
        h += "</tr>"
    return h + "</table></div>"


def ordered(df, col="model"):
    df = df.copy()
    df["_o"] = df[col].map({k: i for i, k in enumerate(ORDER)}).fillna(99)
    return df.sort_values("_o").drop(columns="_o")


def main():
    import narrative as N
    met_lbl = {"mae": "MAE (std units)", "ssim": "SSIM", "detail_r": "detail_r", "hf_ratio": "hf_ratio",
               "chamfer_m": "Edge chamfer (m)", "ch_out2true_m": "out→truth (m)", "ch_true2out_m": "truth→out (m)",
               "edgeF_5m": "Edge F @5 m", "edgeF_10m": "Edge F @10 m", "edgeF_20m": "Edge F @20 m", "n": "Tiles",
               "chamfer_m_s10": "Chamfer, outline scale (m)", "edgeF_10m_s10": "Edge F @10 m, outline", "edgeF_20m_s10": "Edge F @20 m, outline"}
    best = {"mae": "min", "ssim": "max", "detail_r": "max", "chamfer_m": "min", "ch_out2true_m": "min",
            "ch_true2out_m": "min", "edgeF_5m": "max", "edgeF_10m": "max", "edgeF_20m": "max",
            "chamfer_m_s10": "min", "edgeF_10m_s10": "max", "edgeF_20m_s10": "max"}
    ex = pd.read_csv(ROOT / "eval_x_scores.csv")
    main_ = ex[ex.split.isin(["test", "ceiling"])]
    agg = main_.groupby(["input", "grid", "model"]).agg(
        **{c: (c, "median") for c in ["mae", "ssim", "detail_r", "hf_ratio", "chamfer_m", "ch_out2true_m",
                                       "ch_true2out_m", "edgeF_5m", "edgeF_10m", "chamfer_m_s10", "edgeF_10m_s10",
                                       "edgeF_20m_s10"]}, n=("tile", "nunique")).reset_index()
    cols = ["n", "mae", "detail_r", "hf_ratio", "chamfer_m", "ch_out2true_m", "ch_true2out_m", "edgeF_10m",
            "chamfer_m_s10", "edgeF_20m_s10"]
    dig = {"n": 0, "chamfer_m": 1, "ch_out2true_m": 1, "ch_true2out_m": 1, "chamfer_m_s10": 1}

    def tab(inp, grid, extra_ref=True):
        d = agg[(agg.input == inp) & (agg.grid == grid)]
        if extra_ref:
            d = pd.concat([agg[(agg.input == "other_xband") & (agg.grid == grid)], d])
        return table(ordered(d), cols, met_lbl, best=best, digits=dig)

    T = {}
    T["syn25"], T["syn5"] = tab("synthetic", "2.5m"), tab("synthetic", "5m")
    T["real25"], T["real5"] = tab("real_s1", "2.5m"), tab("real_s1", "5m")

    # per provider / polarization (real S1 and synthetic, 2.5 m grid for 4x models, 5 m for all)
    keep = ["bicubic4", "exp1_cnn", "x4_cond", "x4_cond+adapt", "x4_aug", "c4"]
    pp = ex[(ex.split.isin(["test", "test_xpol"])) & (ex.grid == "2.5m") & ex.model.isin(keep)].copy()
    pp["group"] = pp.provider + " " + pp.xpol + np.where(pp.split == "test_xpol", " (S1 VV input, cross-pol truth)", "")
    g = pp.groupby(["input", "group", "model"]).agg(detail_r=("detail_r", "median"), edgeF_10m=("edgeF_10m", "median"),
                                                     chamfer_m=("chamfer_m", "median"), n=("tile", "nunique")).reset_index()
    rows = ""
    for (inp, grp), d in g.groupby(["input", "group"]):
        d = d.set_index("model")
        rows += f"<tr><td>{'synthetic' if inp == 'synthetic' else 'real S1'}</td><td>{grp}</td><td class='n'>{int(d.n.max())}</td>"
        for m in keep:
            if m in d.index:
                rows += f"<td class='n'>{d.loc[m, 'edgeF_10m']:.3f} / {d.loc[m, 'detail_r']:.3f}</td>"
            else:
                rows += "<td class='n'>–</td>"
        rows += "</tr>"
    T["prov"] = ("<div class='tw'><table><tr><th>Input</th><th>Provider, X-band pol</th><th>Tiles</th>" +
                 "".join(f"<th>{NAMES[m]}</th>" for m in keep) + "</tr>" + rows + "</table></div>")

    # domain-gap chart data: synthetic vs real, per model, 2.5 m (4x) and 5 m (2x)
    gap = []
    for _, r in agg[agg.input.isin(["synthetic", "real_s1"])].iterrows():
        gap.append(dict(model=NAMES.get(r.model, r.model), key=r.model, input=r.input, grid=r.grid,
                        edgeF_10m=r.edgeF_10m, detail_r=r.detail_r, chamfer_m=r.chamfer_m, hf_ratio=r.hf_ratio))
    ceil = agg[agg.input == "other_xband"].set_index("grid")

    # calibration
    cal = json.loads((ROOT / "calib.json").read_text())
    calcsv = pd.read_csv(ROOT / "calib.csv")
    real = calcsv[calcsv.source == "real_s1"]
    T["calib"] = ("<div class='tw'><table><tr><th>Source</th><th>Sites or tiles</th><th>ENL (median)</th><th>ENL range</th>"
                  "<th>Lag-1 autocorrelation of speckle</th></tr>"
                  f"<tr><td>Real Sentinel-1 RTC VV, single date (exp 4 stacks)</td><td class='n'>{len(real)} sites</td>"
                  f"<td class='n'>{real.enl.median():.2f}</td><td class='n'>{real.enl.min():.1f}–{real.enl.max():.1f}</td>"
                  f"<td class='n'>{real.lag1.median():.2f}</td></tr>" +
                  "".join(f"<tr><td>Synthetic, K = {int(s['k'])} looks</td><td class='n'>39 tiles</td><td class='n'>{s['enl']:.2f}</td>"
                          f"<td class='n'>–</td><td class='n'>{s['lag1']:.2f}</td></tr>" for s in cal["synthetic"]) +
                  "</table></div>")

    # despeckling
    ds = pd.read_csv(ROOT / "despeckle_scores.csv")
    dsum = ds.groupby("model")[["mean_ratio_raw", "enl10", "enl_native", "mean_ratio", "mean_ratio_h", "ratio_enl", "ratio_struct",
                                "mae_ref_db"]].median().reset_index()
    dl = {"mean_ratio_raw": "Mean ratio, naive de-standardization", "enl10": "ENL at 10 m", "enl_native": "ENL, output grid", "mean_ratio": "Mean ratio (all)",
          "mean_ratio_h": "Mean ratio (homog.)", "ratio_enl": "Ratio-image ENL", "ratio_struct": "Ratio-image structure r",
          "mae_ref_db": "MAE vs temporal mean (dB)"}
    T["desp"] = table(ordered(dsum), list(dl), dl, digits={"enl10": 1, "enl_native": 1, "ratio_enl": 1})

    # stripmap
    smf = ROOT / "eval_sm_scores.csv"
    T["sm"] = ""
    if smf.exists():
        sm = pd.read_csv(smf)
        if len(sm):
            ssum = sm.groupby(["variant", "model"])[["mae", "ssim", "detail_r", "hf_ratio", "chamfer_m", "edgeF_10m", "edgeF_20m"]].median().reset_index()
            for v, d in ssum.groupby("variant"):
                info = sm[sm.variant == v].iloc[0]
                T["sm"] += (f"<h3>{v.capitalize()} position (offset {info.offset_row_m:.0f} m N-S, {info.offset_col_m:.0f} m E-W; "
                            f"IW-Stripmap r at 10 m = {info.corr10_reg:.2f}, {sm.tile.nunique()} tile)</h3>" +
                            table(ordered(d), ["mae", "ssim", "detail_r", "hf_ratio", "chamfer_m", "edgeF_10m", "edgeF_20m"],
                                  met_lbl, best=best, digits={"chamfer_m": 1}))

    # transfer stats
    tsf = ROOT / "transfer_stats.csv"
    T["transfer"] = ""
    if tsf.exists():
        ts = pd.read_csv(tsf)
        tsum = ts.groupby("model")[["consistency_r", "added_detail"]].median().reset_index()
        T["transfer"] = table(ordered(tsum), ["consistency_r", "added_detail"],
                              {"consistency_r": "Input consistency r", "added_detail": "Added fine detail (std units)"})

    figs = {}
    for d in ["panels_transfer", "panels_sm", "panels_x"]:
        p = ROOT / d
        figs[d] = sorted(p.glob("*.png")) if p.exists() else []

    import results_text as R
    import torch
    ck = torch.load(ROOT / "ckpt/x4_cond.pt", map_location="cpu", weights_only=False)
    prov = ", ".join(f"{v} {k.capitalize() if k != 'iceye' else 'ICEYE'}" for k, v in ck["providers"].items())
    pols = ", ".join(f"{v} {k}" for k, v in ck["pols"].items())
    real = ex[(ex.input == "real_s1") & (ex.split == "test") & (ex.grid == "2.5m")]
    w = real.pivot_table(index="tile", columns="model", values="edgeF_20m_s10")
    dsi = dsum.set_index("model")
    ctx = dict(T=T, gap=gap, ceil=ceil, cal=cal, agg=agg, ds=dsum, figs=figs, img64=img64, NAMES=NAMES, root=ROOT,
               lede=R.LEDE, exp1_which="the spike's Capella retrain, exp1_gan_spike/ckpt/l1.pt",
               n_tiles_txt=f"Training uses {ck['n_tiles']} X-band tiles ({prov}; {pols}) and no Sentinel-1 at all.",
               fine_shift_med=float(real[real.model == "bicubic4"].fine_shift_m.median()),
               win_aug=float((w["x4_aug+adapt"] > w["bicubic4"]).mean()),
               win_exp1=float((w["exp1_cnn"] > w["bicubic4"]).mean()),
               ds_x4=float(dsi.loc["x4_cond", "mae_ref_db"]), ds_bic=float(dsi.loc["bicubic4", "mae_ref_db"]),
               ds_exp1=float(dsi.loc["exp1_cnn", "mae_ref_db"]),
               panel_ids=["644f8836", "XN73DX", "C09_SM_GEO_VV_20241123", "ce922bc0"])
    sm = pd.read_csv(smf) if smf.exists() else pd.DataFrame()
    if len(sm):
        nom, srch = sm[sm.variant == "nominal"].iloc[0], sm[sm.variant == "searched"]
        ctx.update(sm_r_nom=float(nom.corr10_reg), sm_r_srch=float(srch.iloc[0].corr10_reg) if len(srch) else float("nan"),
                   sm_off=(f"{abs(srch.iloc[0].offset_row_m):.0f} m {'south' if srch.iloc[0].offset_row_m > 0 else 'north'} and "
                           f"{abs(srch.iloc[0].offset_col_m):.0f} m {'east' if srch.iloc[0].offset_col_m > 0 else 'west'}") if len(srch) else "nowhere")
    import r2_text
    import r3_section
    import r4_section
    ctx["results_html"] = r4_section.section(ctx) + r3_section.section(ctx) + r2_text.section(ctx) + R.results(ctx)
    body = N.body(ctx)
    (ROOT / "page_body.html").write_text(body, encoding="utf-8")
    d3 = (PROJ / "exp0_overlap_atlas" / "d3.min.js").read_text(encoding="utf-8")
    scripts = "<script>" + d3 + "</script>\n<script>\nconst GAP=" + json.dumps(gap, default=float) + ";\n" + N.SCRIPT + "\n</script>"
    if "r2_json" in ctx:
        import r2_results
        scripts += "\n<script>\nconst R2=" + ctx["r2_json"] + ";\n" + r2_results.SCRIPT + "\n</script>"
    if "r3_json" in ctx:
        import r3_section
        scripts += "\n<script>\nconst R3=" + ctx["r3_json"] + ";\n" + r3_section.SCRIPT + "\n</script>"
    (ROOT / "page_scripts.html").write_text(scripts, encoding="utf-8")
    print("page written", len(body) // 1024, "KB body", flush=True)


if __name__ == "__main__":
    main()
