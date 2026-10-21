"""Score bicubic and trained models on held-out Stripmap tiles, and cache the predictions.

Metrics (per tile, on valid land pixels):
  ssim, psnr      in the paper's space: [0, 1] min-max normalized dB, against the histogram-matched
                  reference, raw (_raw) and Lee-filtered (_lee). SSIM reported x100 as in the paper.
  mae_db          |prediction - reference| in dB (histogram-matched reference)
  detail_r        correlation of the 5 m detail layer (image minus its 10 m block mean) with the
                  reference's: does added detail land where Stripmap detail is?
  hf_ratio        detail-layer energy of prediction / reference (1 = as sharp, < 1 blurrier)
  chamfer_p2t_m   mean distance from prediction edges to the nearest reference edge (m, capped 50)
  chamfer_t2p_m   the reverse: reference edges missed by the prediction
  enl, moi, mor, cx  despeckling metrics on the two most homogeneous 32 x 32 windows per tile
                  (single WorldCover class, lowest Lee coefficient of variation), linear intensity:
                  ENL = mean^2 / var, MoI = mean(pred) / mean(IW), MoR = mean(IW / pred), Cx = std / mean
Writes scores_tiles.csv, scores_summary.csv, DATA/preds/<model>/<tile_id>.npy.

  python score.py esrt_a cnn_a esrt_b ...
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy import ndimage
from skimage.metrics import structural_similarity

from common import DATA, ROOT, block_mean, lee_filter
from sr_data import Norm, load_manifest, load_split
from train_sr import CKPT, DEV, make_model, predict
from xband_prep import edges

PRED = DATA / "preds"
GROUP = {"houston": "houston (X-band holdout)", "owens": "owens (near)", "w_australia": "w_australia (far)",
         "amieva_test": "paper test scenes", "yemen_west": "yemen_west (near, inland)",
         "bangladesh": "bangladesh (far, inland)"}


def bicubic(iw_n):
    t = torch.from_numpy(np.nan_to_num(iw_n, nan=0.0).astype("float32"))[None, None]
    return F.interpolate(t, scale_factor=2, mode="bicubic", align_corners=False)[0, 0].numpy()


def detail(a):
    return a - np.kron(block_mean(a, 2), np.ones((2, 2)))


def chamfer(pe, te, m, cap=50.0):
    pe, te = pe & m, te & m
    if pe.sum() < 20 or te.sum() < 20:
        return np.nan, np.nan
    dt = ndimage.distance_transform_edt(~te) * 5.0
    dp = ndimage.distance_transform_edt(~pe) * 5.0
    return float(np.minimum(dt[pe], cap).mean()), float(np.minimum(dp[te], cap).mean())


def homogeneous_windows(t, k=2, w=32):
    lee = 10 ** (t["lee"] / 10)
    wc, m = t["wc"], t["mask"]
    best = []
    for r in range(0, 640 - w + 1, w // 2):
        for c in range(0, 640 - w + 1, w // 2):
            mm = m[r:r + w, c:c + w]
            cls = wc[r:r + w, c:c + w]
            if not mm.all() or cls.min() != cls.max() or cls[0, 0] not in (10, 20, 30, 40, 60, 90, 100):
                continue
            v = lee[r:r + w, c:c + w]
            best.append((float(v.std() / v.mean()), r, c))
    best.sort()
    out = []
    for cv, r, c in best:
        if all(abs(r - r2) >= w or abs(c - c2) >= w for _, r2, c2 in out):
            out.append((cv, r, c))
        if len(out) == k:
            break
    return [(r, c) for _, r, c in out]


def despeckle_metrics(pred_db, iw_db, wins, w=32):
    P = 10 ** (pred_db / 10)
    I = np.kron(10 ** (iw_db / 10), np.ones((2, 2)))
    res = []
    for r, c in wins:
        p, i = P[r:r + w, c:c + w], I[r:r + w, c:c + w]
        res.append({"enl": p.mean() ** 2 / p.var(), "moi": p.mean() / i.mean(), "mor": (i / p).mean(),
                    "cx": p.std() / p.mean()})
    return pd.DataFrame(res).mean().to_dict() if res else {}


def tile_scores(name, p_n, t, norm, wins):
    m = t["mask"]
    out = {"model": name}
    p_db = norm.inv(p_n)
    for ref in ("sm_hm", "lee_hm"):
        tag = "raw" if ref == "sm_hm" else "lee"
        y = t[ref]
        mm = m & np.isfinite(y)
        y_n = norm(y)
        _, S = structural_similarity(np.where(mm, y_n, 0), np.where(mm, p_n, 0), data_range=1.0, full=True)
        out[f"ssim_{tag}"] = 100 * float(S[mm].mean())
        for c in np.unique(t["wc"][mm]):            # per WorldCover class: SSIM sum and pixel count
            cm = mm & (t["wc"] == c)
            out.setdefault("_cls", []).append({"ref": tag, "cls": int(c), "ssim_sum": float(S[cm].sum()), "px": int(cm.sum())})
        mse = float(((np.clip(p_n, 0, 1) - np.clip(y_n, 0, 1))[mm] ** 2).mean())
        out[f"psnr_{tag}"] = 10 * np.log10(1.0 / max(mse, 1e-12))
        out[f"mae_db_{tag}"] = float(np.abs(p_db - y)[mm].mean())
        m4 = block_mean(mm.astype("float32"), 2) == 1
        m4 = np.kron(m4, np.ones((2, 2), bool))
        dp, dy = detail(np.where(mm, p_db, 0))[m4], detail(np.where(mm, y, 0))[m4]
        out[f"detail_r_{tag}"] = float(np.corrcoef(dp, dy)[0, 1]) if dp.std() > 0 else 0.0
        out[f"hf_ratio_{tag}"] = float(dp.std() / dy.std())
        pe = edges(p_db, mm)
        te = edges(y, mm)
        out[f"chamfer_p2t_m_{tag}"], out[f"chamfer_t2p_m_{tag}"] = chamfer(pe, te, mm)
    out.update(despeckle_metrics(p_db, t["iw"], wins))
    return out


def main(argv):
    """python score.py [--splits v2] [--tag r3] model ...
    Each model predicts in its own normalized space (its checkpoint's min-max); predictions are
    converted to dB and every metric is computed in one common normalization (the first model's),
    so models trained on different tile sets compare on equal terms."""
    splits, tag = "v1", ""
    if "--splits" in argv:
        i = argv.index("--splits"); splits = argv[i + 1]; argv = argv[:i] + argv[i + 2:]
    if "--tag" in argv:
        i = argv.index("--tag"); tag = "_" + argv[i + 1]; argv = argv[:i] + argv[i + 2:]
    models = argv
    man = load_manifest(splits)
    test = load_split(["test"], man)
    nets, norms = {}, {}
    for name in models:
        ck = torch.load(CKPT / f"{name}.pt", map_location=DEV)
        net = make_model(ck["model"]).to(DEV)
        net.load_state_dict(ck["state"])
        nets[name], norms[name] = net, Norm(*ck["norm"])
    norm = norms[models[0]]
    rows, crows = [], []
    for k, t in enumerate(test):
        wins = homogeneous_windows(t)
        preds = {"bicubic": bicubic(norm(t["iw"]))}
        # despeckle-then-interpolate baseline (the paper's is MERLIN + bicubic; Lee 7x7 here)
        iw_lee = np.where(np.isfinite(t["iw"]), 10 * np.log10(np.maximum(lee_filter(10 ** (t["iw"] / 10), 7)[0], 1e-6)), np.nan)
        preds["lee_bicubic"] = bicubic(norm(iw_lee))
        for name, net in nets.items():
            preds[name] = norm(norms[name].inv(predict(net, norms[name](t["iw"]))))
        for name, p in preds.items():
            (PRED / name).mkdir(parents=True, exist_ok=True)
            np.save(PRED / name / f"{t['tile_id']}.npy", norm.inv(p).astype("float16"))
            sc = tile_scores(name, p, t, norm, wins)
            for c in sc.pop("_cls", []):
                crows.append({"tile_id": t["tile_id"], "group": GROUP.get(t["region"], t["region"]), "model": name, **c})
            rows.append({"tile_id": t["tile_id"], "region": t["region"], "group": GROUP.get(t["region"], t["region"]), **sc})
        for ref in ("sm", "lee"):
            d = despeckle_metrics(t[ref], t["iw"], wins)
            rows.append({"tile_id": t["tile_id"], "region": t["region"], "group": GROUP.get(t["region"], t["region"]),
                         "model": f"stripmap_{ref}", **d})
        print(f"[{k + 1}/{len(test)}] {t['tile_id']}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(ROOT / f"scores_tiles{tag}.csv", index=False)
    num = df.select_dtypes("number").columns
    old_g = ["houston", "owens", "w_australia"]
    summ = pd.concat([df.groupby(["group", "model"])[num].mean().reset_index(),
                      df[df.region.isin(old_g)].assign(group="all pilot holdouts").groupby(["group", "model"])[num].mean().reset_index(),
                      df[df.region.isin(["yemen_west", "bangladesh"])].assign(group="all inland holdouts").groupby(["group", "model"])[num].mean().reset_index()])
    summ.to_csv(ROOT / f"scores_summary{tag}.csv", index=False)
    cdf = pd.DataFrame(crows)
    cdf.to_csv(ROOT / f"scores_class_tiles{tag}.csv", index=False)
    cls = cdf.groupby(["ref", "cls", "model"])[["ssim_sum", "px"]].sum().reset_index()
    cls["ssim"] = 100 * cls.ssim_sum / cls.px
    cls.to_csv(ROOT / f"scores_class{tag}.csv", index=False)
    pd.set_option("display.width", 250)
    cols = ["group", "model", "ssim_lee", "psnr_lee", "ssim_raw", "psnr_raw", "mae_db_raw", "detail_r_raw",
            "hf_ratio_raw", "chamfer_p2t_m_raw", "chamfer_t2p_m_raw", "enl", "moi", "mor", "cx"]
    print(summ[cols].round(3).to_string(), flush=True)
    print(cls[cls.ref == "lee"].pivot_table(index="cls", columns="model", values="ssim").round(2).to_string(), flush=True)


if __name__ == "__main__":
    try:
        main(sys.argv[1:])
    except Exception:
        import traceback
        traceback.print_exc()
    sys.stdout.flush()
    os._exit(0)
