"""Evaluate on held-out X-band tiles (the spike's test regions, never trained on).

Three comparisons, all on the same held-out tiles:
  synthetic  input = held-out X-band degraded to S1-like 10 m (fixed seed, K=6 looks);
             truth = the same X-band at 2.5 m / 5 m. The X-trained models' home ground.
  real       input = the real Sentinel-1 RTC tile paired with the X-band (spike pairs, coarse
             co-registered at 10 m by the spike, then fine-registered here at 2.5 m against the
             bicubic upsample); truth = real X-band. Brightness differs by band, so the geometry
             metrics (edge chamfer, edge F) carry the transfer question; MAE/SSIM/detail_r are reported too.
  ceiling    a second X-band collect of the same place (exp1_gan_spike/ceiling pairs in held-out
             regions): how well one real X-band image's edges match another's.

Models: bicubic, exp1 CNN (real S1 -> X pairs), X-synthetic 4x/2x (conditioned / not), C-band-only
4x/2x. 4x outputs are also scored at 5 m (2x2 mean of standardized dB).
Writes eval_x_scores.csv and panels in panels_x/.
"""
from __future__ import annotations

import common  # noqa: F401
from common import (adapt, DEV, ROOT, SPIKE, S1_TO_SENSOR_AZ, bicubic, chamfer, degrade, edges, finish, fine_register,
                    load_exp1, load_model, meta_vec, norm, norm_stats, run_model, scores, tile_metadata)

import json
import zlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy import ndimage

MODELS = ["x4_cond", "x4_unc", "x4_aug", "x2_cond", "x2_unc", "c4", "c2"]


def pool2(a):
    n = a.shape[0] // 2
    return a[:n * 2, :n * 2].reshape(n, 2, n, 2).mean(axis=(1, 3))


def pool2m(m):
    n = m.shape[0] // 2
    return m[:n * 2, :n * 2].reshape(n, 2, n, 2).all(axis=(1, 3))


def score_all(preds, truth, m, extra):
    """preds: name -> (array, scale). Score 4x at 2.5 m and at 5 m; 2x at 5 m."""
    rows = []
    t5, m5 = pool2(np.where(m, truth, 0)), pool2m(m)
    te25 = edges(truth, m, 2.5)
    te5 = edges(t5, m5, 5.0)
    te25b = edges(truth, m, 2.5, sigma_m=10.0)
    te5b = edges(t5, m5, 5.0, sigma_m=10.0)

    def coarse(p, te, mm, px):
        c = chamfer(edges(p, mm, px, sigma_m=10.0), te, mm, px, tols=(10.0, 20.0))
        return {k + "_s10": v for k, v in c.items()}

    for k, (p, sc) in preds.items():
        if sc == 4:
            r = dict(extra, model=k, grid="2.5m", **scores(p, truth, m, 4))
            r.update(chamfer(edges(p, m, 2.5), te25, m, 2.5))
            r.update(coarse(p, te25b, m, 2.5))
            rows.append(r)
            p5 = pool2(p)
        else:
            p5 = p
        r = dict(extra, model=k, grid="5m", **scores(p5, t5, m5, 2))
        r.update(chamfer(edges(p5, m5, 5.0), te5, m5, 5.0))
        r.update(coarse(p5, te5b, m5, 5.0))
        rows.append(r)
    return rows


def main():
    meta = tile_metadata()
    s1m = pd.read_csv(ROOT / "s1_meta.csv")
    s1_orbit = dict(zip(s1m.s1_id, s1m.orbit))
    models = {}
    for k in MODELS:
        f = ROOT / "ckpt" / f"{k}.pt"
        if f.exists():
            models[k] = load_model(f)[0]
    exp1, which = load_exp1()
    print("models:", list(models), "| exp1:", which, flush=True)

    man = pd.read_csv(SPIKE / "manifest.csv")
    man = man[(man.split.isin(["test", "test_xpol"])) & man.status.isin(["ok", "cached"])]
    sel = pd.read_csv(SPIKE / "selection.csv")
    s1pol = dict(zip(sel.xband_id, sel.s1_pol.fillna("vv")))
    gen = torch.Generator(device=DEV)
    rows, panel_store = [], []
    prev_bic = None
    for r in man.itertuples():
        fid = f"{r.provider}_{r.xband_id}"
        f = SPIKE / "data" / f"{fid}.npz"
        if not f.exists():
            continue
        z = np.load(f)
        x = z["x"].astype("float32")
        m = np.isfinite(x)
        if m.mean() < 0.8:
            continue
        md = meta.get(fid, dict(inc=35.0, az=270.0, pol="VV"))
        xpol = md["pol"]
        truth = np.nan_to_num(norm(x))
        base = dict(tile=fid, provider=r.provider, xpol=xpol, split=r.split, region=r.test_region,
                    corr10=r.corr10)

        # ---------------- synthetic: degrade the held-out X-band
        gen.manual_seed(zlib.crc32(fid.encode()))
        xf = np.where(m, x, np.nanmedian(x))
        with torch.no_grad():
            sdb = degrade(torch.from_numpy(xf)[None, None].to(DEV), [md["az"]], looks=6, gen=gen)[0, 0].cpu().numpy()
        s_syn = norm(sdb)
        mv = meta_vec(md["az"], md["inc"], xpol, "X")
        preds = {"bicubic4": (bicubic(s_syn, 4), 4), "exp1_cnn": (run_model(exp1, s_syn, kind="exp1"), 4)}
        for k, G in models.items():
            preds[k] = (run_model(G, s_syn, mv), G.scale)
        rows += score_all(preds, truth, m, dict(base, input="synthetic"))

        # ---------------- real Sentinel-1
        s1 = z["s1"].astype("float32")
        s1 = np.where(np.isfinite(s1), s1, np.nanmedian(s1))
        s_real = norm(s1)
        bic = bicubic(s_real, 4)
        sh = fine_register(bic, truth, m)
        truth_r = ndimage.shift(truth, sh, order=1, mode="nearest")
        m_r = ndimage.shift(m.astype("float32"), sh, order=0, mode="constant") > 0.5
        pol_in = "HH" if str(s1pol.get(r.xband_id, "vv")).lower() == "hh" else "VV"
        az_s1 = S1_TO_SENSOR_AZ.get(str(s1_orbit.get(r.s1_id)), 258.0)
        mv_real = meta_vec(az_s1, 37.0, pol_in, "X")
        preds_r = {"bicubic4": (bic, 4), "exp1_cnn": (run_model(exp1, s_real, kind="exp1"), 4)}
        s_ad = adapt(s_real)
        if prev_bic is not None:       # chance level: edges of an unrelated tile's S1
            preds_r["null_other_tile"] = (prev_bic, 4)
        prev_bic = bic
        preds_r["bicubic4+adapt"] = (bicubic(s_ad, 4), 4)
        for k, G in models.items():
            preds_r[k] = (run_model(G, s_real, mv_real), G.scale)
            if k.startswith("x"):
                preds_r[k + "+adapt"] = (run_model(G, s_ad, mv_real), G.scale)
        rows += score_all(preds_r, truth_r, m_r, dict(base, input="real_s1", s1_pol=pol_in,
                                                      fine_shift_m=float(np.hypot(*sh) * 2.5)))
        panel_store.append((fid, r.test_region, r.split, s_syn, preds, truth, s_real, preds_r, truth_r))
        print(fid, r.split, "shift", sh, flush=True)

    # ---------------- ceiling: X-band vs X-band of the same place, held-out regions only
    pairs = pd.read_csv(SPIKE / "ceiling/pairs_selected.csv")
    pairs = pairs[pairs.region != "train"]
    for r in pairs.itertuples():
        hits = list((SPIKE / "ceiling/tiles").glob(f"*{r.id_a[-12:]}*{r.id_b[-12:]}*.npz"))
        if not hits:
            continue
        zc = np.load(hits[0])
        A, B = zc["A"].astype("float32"), zc["B"].astype("float32")
        m = np.isfinite(A) & np.isfinite(B)
        if m.mean() < 0.5:
            continue
        a, b = np.nan_to_num(norm(A)), np.nan_to_num(norm(B))
        sh = fine_register(a, b, m, max_px=16)
        b = ndimage.shift(b, sh, order=1, mode="nearest")
        m = m & (ndimage.shift(m.astype("float32"), sh, order=0, mode="constant") > 0.5)
        base = dict(tile=hits[0].stem, provider=r.combo, xpol=r.pol, split="ceiling", region=r.region,
                    input="other_xband", d_inc=r.d_inc, d_look_az=r.d_look_az)
        rows += score_all({"xband_other_collect": (a, 4)}, b, m, base)

    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "eval_x_scores.csv", index=False)
    met = ["mae", "ssim", "detail_r", "hf_ratio", "chamfer_m", "ch_out2true_m", "ch_true2out_m", "edgeF_5m", "edgeF_10m", "chamfer_m_s10", "edgeF_10m_s10", "edgeF_20m_s10"]
    main_ = df[df.split.isin(["test", "ceiling"])]
    summ = main_.groupby(["input", "grid", "model"])[met].median().round(3)
    print(summ.to_string(), flush=True)
    summ.to_csv(ROOT / "eval_x_summary.csv")

    # ---------------- panels: one 256 px crop per tile (most structured truth), synthetic + real rows
    out = ROOT / "panels_x"
    out.mkdir(exist_ok=True)
    order = [("bicubic4", "Bicubic"), ("exp1_cnn", "Exp 1 CNN (real pairs)"), ("x4_cond", "X-synthetic 4x, cond."),
             ("x4_cond+adapt", "X-synth 4x cond. + band adapt"), ("c4", "C-band-only 4x")]
    meta_p = []
    for fid, reg, split, s_syn, P, truth, s_real, Pr, truth_r in panel_store:
        best, rc = -1, (0, 0)
        for rr in range(0, 1280 - 256, 128):
            for cc in range(0, 1280 - 256, 128):
                w = truth[rr:rr + 256, cc:cc + 256]
                if w.std() > best:
                    best, rc = w.std(), (rr, cc)
        rr, cc = rc
        sl = (slice(rr, rr + 256), slice(cc, cc + 256))
        fig, ax = plt.subplots(2, 7, figsize=(17.5, 5.4), dpi=80)
        for row, (s, PP, tr, lab) in enumerate([(s_syn, P, truth, "synthetic input"), (s_real, Pr, truth_r, "real S1")]):
            lo, hi = np.percentile(tr[sl], [2, 98])
            ims = [("Input: " + lab, np.kron(s, np.ones((4, 4)))[sl])]
            ims += [(t, PP[k][0][sl]) if k in PP else ("", None) for k, t in order]
            ims += [("X-band truth (2.5 m)", tr[sl])]
            for j in range(7):
                a = ax[row, j]
                a.axis("off")
                if j < len(ims) and ims[j][1] is not None:
                    a.imshow(ims[j][1], cmap="gray", vmin=lo, vmax=hi, interpolation="nearest")
                    a.set_title(ims[j][0], fontsize=8)
        fig.suptitle(f"{fid[:40]}  ({reg}, {split})  640 m crop", fontsize=9)
        fig.tight_layout()
        fig.savefig(out / f"{fid[:60]}.png")
        plt.close(fig)
        meta_p.append(dict(id=fid, region=reg, split=split))
    (out / "panels.json").write_text(json.dumps(meta_p))
    print("panels", len(meta_p), flush=True)


if __name__ == "__main__":
    main()
    finish()
