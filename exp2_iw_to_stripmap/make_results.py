"""Write results_body.html (body fragment for site/exp2.html) and progress.json from the CSVs.

Narrative text that depends on judgment lives in results_notes.json (keys: geoloc, verdict,
caveats, stage1, armb); everything numeric is read from the result files.
"""
from __future__ import annotations

import base64
import json
import os
import sys
from html import escape

import numpy as np
import pandas as pd

from common import ROOT

PAPER = {"ssim": 70.14, "psnr": 26.07, "enl": 201.51, "moi": 1.022, "mor": 1.027, "cx": 0.073}
PAPER_MERLIN = {"ssim": 67.30, "psnr": 20.92, "enl": 329.50}
LABEL = {"bicubic": "Bicubic IW", "lee_bicubic": "Lee-filtered IW + bicubic", "esrt_paper": "ESRT, paper recipe (SSIM + HM)", "esrt_a": "ESRT arm A (+ consistency)",
         "cnn_a": "Residual CNN arm A (+ consistency)", "esrt_ctrl": "ESRT arm A, fine-tuned (control)",
         "esrt_b05": "ESRT arm B, edge weight 0.05", "esrt_b20": "ESRT arm B, edge weight 0.2",
         "stripmap": "Stripmap itself", "stripmap_lee": "Stripmap, Lee", "stripmap_sm": "Stripmap (raw)"}


def img64(p, width=1200):
    """Embed a figure as JPEG, downsampled to `width` px, to keep the page small."""
    import io
    from PIL import Image
    im = Image.open(p).convert("RGB")
    if im.width > width:
        im = im.resize((width, round(im.height * width / im.width)), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=82, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def f(v, d=2):
    return "" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{v:.{d}f}"


def table(df, cols, heads, num=None):
    num = num or {}
    th = "".join(f"<th{' class=n' if c in num else ''}>{h}</th>" for c, h in zip(cols, heads))
    tr = ""
    for r in df.itertuples(index=False):
        tds = ""
        for c in cols:
            v = getattr(r, c)
            tds += f"<td class='n'>{f(v, num[c])}</td>" if c in num else f"<td>{escape(str(v))}</td>"
        tr += f"<tr>{tds}</tr>"
    return f"<div class='tw'><table><thead><tr>{th}</tr></thead><tbody>{tr}</tbody></table></div>"


def main():
    notes = json.loads((ROOT / "results_notes.json").read_text(encoding="utf-8"))
    out = []

    # ---------------------------------------------------------------- verdict
    out.append(f"<section class='card' style='border-left:3px solid var(--a2)'><h2>Result so far</h2>{notes['verdict']}</section>")
    r3 = ROOT / "results_r3_section.html"           # round 3: training on the expanded set (r3_section.py)
    if r3.exists():
        out.append(r3.read_text(encoding="utf-8"))
    v2 = ROOT / "results_v2_section.html"           # round 2 expanded set (v2_section.py)
    if v2.exists():
        out.append(v2.read_text(encoding="utf-8"))

    # ---------------------------------------------------------------- data
    tiles = pd.read_csv(ROOT / "tiles.csv")
    pf = pd.read_csv(ROOT / "pairs_final.csv")
    sp = pd.read_csv(ROOT / "splits.csv")[["tile_id", "split"]]
    d = tiles.merge(pf[["tile_id", "use", "corr10", "scene_spread_m", "off_e_m", "off_n_m"]], on="tile_id", how="left").merge(sp, on="tile_id", how="left")
    g = d.groupby("region").agg(role=("role", "first"), scenes=("slc", "nunique"), tiles=("tile_id", "size"),
                                used=("use", "sum"), corr10=("corr10", "median"), spread=("scene_spread_m", "median"),
                                train=("split", lambda s: (s == "train").sum()), val=("split", lambda s: (s == "val").sum()),
                                test=("split", lambda s: (s == "test").sum())).reset_index()
    g["off"] = [float(np.median(np.hypot(x.off_e_m, x.off_n_m))) for _, x in d.groupby("region")]
    g = g.sort_values(["role", "region"])
    out.append("<section class='card'><h2>Tiles built: Stripmap 5 m ↔ IW 10 m, 3.2 km each</h2>"
               f"<p>{notes['data']}</p>" +
               table(g, ["region", "role", "scenes", "tiles", "used", "train", "val", "test", "corr10", "off", "spread"],
                     ["Region", "Role", "Scenes", "Tiles", "Used", "Train", "Val", "Test", "10 m corr (median)",
                      "|offset| m", "Within-scene spread m"],
                     {"scenes": 0, "tiles": 0, "used": 0, "train": 0, "val": 0, "test": 0, "corr10": 2, "off": 1, "spread": 1})
               + "</section>")
    out.append(f"<section class='card'><h2>Geolocation: the flipped tile, and what the offsets show</h2>{notes['geoloc']}</section>")

    # ---------------------------------------------------------------- arm A vs paper
    s = pd.read_csv(ROOT / "scores_summary.csv")
    s["Model"] = s.model.map(lambda m: LABEL.get(m, m))
    order = ["bicubic", "lee_bicubic", "esrt_paper", "esrt_a", "cnn_a", "esrt_ctrl", "esrt_b05", "esrt_b20", "stripmap_sm", "stripmap_lee"]
    s["o"] = s.model.map({m: i for i, m in enumerate(order)})
    s = s.sort_values(["group", "o"])
    cols = ["Model", "ssim_lee", "psnr_lee", "ssim_raw", "psnr_raw", "mae_db_raw", "detail_r_raw", "hf_ratio_raw",
            "chamfer_p2t_m_raw", "chamfer_t2p_m_raw", "enl", "moi", "mor", "cx"]
    heads = ["Model", "SSIM vs Lee", "PSNR vs Lee", "SSIM vs raw", "PSNR vs raw", "MAE dB", "detail r", "HF ratio",
             "edge→SM m", "SM edge→out m", "ENL", "MoI", "MoR", "Cx"]
    num = {c: 2 for c in cols[1:]}
    num.update(ssim_lee=1, ssim_raw=1, enl=1)
    blocks = ""
    for grp in ["all pilot holdouts", "houston (X-band holdout)", "owens (near)", "w_australia (far)", "paper test scenes"]:
        x = s[s.group == grp]
        if x.empty:
            continue
        blocks += f"<h3>{escape(grp)}</h3>" + table(x, cols, heads, num)
    paper = (f"<p>Published (Amieva et al. 2026, best: ESRT, SSIM loss, histogram matching, their 3 test scenes): SSIM "
             f"{PAPER['ssim']}, PSNR {PAPER['psnr']} dB, ENL {PAPER['enl']}, MoI {PAPER['moi']}, MoR {PAPER['mor']}, "
             f"Cx {PAPER['cx']}. Their MERLIN + bicubic baseline: SSIM {PAPER_MERLIN['ssim']}, PSNR {PAPER_MERLIN['psnr']} dB.</p>")
    out.append("<section class='card'><h2>Arm A against bicubic and the published baseline (held-out Stripmap)</h2>"
               f"{notes['arma']}{paper}{blocks}<p class='lede' style='font-size:.85rem'>{notes['metrics']}</p></section>")

    # ---------------------------------------------------------------- stage 1
    st = pd.read_csv(ROOT / "stage1_summary.csv")
    st["Model"] = st.model.map(lambda m: LABEL.get(m, m))
    sb = ""
    for set_, title in [("sites", "X-band sites with IW only (> 150 km from training)"),
                        ("tiles", "Held-out Stripmap tiles with X-band")]:
        x = st[st.set == set_]
        if x.empty:
            continue
        sb += f"<h3>{title}</h3>" + table(x, ["prov_pol", "Model", "n", "p2x_m", "x2p_m", "osm_prec"],
                                          ["Provider", "Image", "Pairs", "Output edge → X-band edge (m)",
                                           "X-band edge → output edge (m)", "Edges on OSM"],
                                          {"n": 0, "p2x_m": 1, "x2p_m": 1, "osm_prec": 2})
    out.append(f"<section class='card'><h2>Stage 1: X-band geometry check, per provider</h2>{notes['stage1']}{sb}</section>")

    # ---------------------------------------------------------------- arm B
    tune = []
    for n in ["esrt_ctrl", "esrt_b05", "esrt_b20"]:
        j = ROOT / "ckpt" / f"{n}.json"
        if j.exists():
            v = json.loads(j.read_text())
            tune.append({"Model": LABEL[n], "w": v.get("edge_w", 0.0), "val": v["best_val_ssim"]})
    tb = table(pd.DataFrame(tune), ["Model", "w", "val"], ["Run", "Edge weight", "Validation SSIM (Lee)"],
               {"w": 2, "val": 4}) if tune else ""
    out.append(f"<section class='card'><h2>Arm B: arm A plus the X-band edge loss</h2>{notes['armb']}{tb}</section>")

    # ---------------------------------------------------------------- panels
    figs = ""
    for p in sorted((ROOT / "panels").glob("*.png")):
        figs += f"<figure class='card'><img src='{img64(p)}' alt='{p.stem}'><figcaption>{escape(p.stem.replace('_', ' '))}</figcaption></figure>"
    if figs:
        out.append(f"<section><h2>Transfer panels</h2><p>{notes['panels']}</p><div class='grid' style='grid-template-columns:1fr'>{figs}</div></section>")
    out.append(f"<section class='card'><h2>Caveats</h2>{notes['caveats']}</section>")
    (ROOT / "results_body.html").write_text("\n".join(out), encoding="utf-8")
    print("wrote results_body.html", flush=True)


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    os._exit(0)
