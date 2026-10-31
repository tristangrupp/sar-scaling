"""Evaluate trained models on validation or test sites and apply the pre-set decision rule (DESIGN.md).

  python eval.py --names esrt_r5_s0 [--split test] [--device cpu] [--feature-test] [--rejects]
  python eval.py --names esrt_r5_s0 esrt_r5_s1 esrt_r5_s2       # seeds: rule on the seed mean, and per seed

Per model writes eval_<name>_<split>.csv (chip scores, plus shuffled-input rows "shuffle_<name>"). Then compares
with baselines_chips.csv (best baseline per metric chosen on validation, best_baselines.json) and
ceiling_summary.csv, and writes decision_<names>_<split>.json. --feature-test adds experiment 8's linear-feature
test (feature_<name>.csv). --rejects scores the align_reject chips too (sensitivity check).
CPU by default; on the GPU only inside a coordinator slot.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import traceback

import numpy as np
import pandas as pd
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import (HERE as HEREP, RES_LIST, SC, crop_c, finish, flat_mask, load_item, res_index, scores,  # noqa: E402
                    shuffle_partners)
from train import CKPT, Norm, make_model, predict_chips, predict_large  # noqa: E402

RULE = {"ssim": 0.02, "hp_r": 0.05}
FLAT_MAX, SHUF_MAX = 1.5, 0.05


def load_model(name, dev):
    ck = torch.load(CKPT / f"{name}.pt", map_location=dev, weights_only=False)
    m = make_model(ck["arch"]).to(dev).eval()
    m.load_state_dict(ck["model"])
    return m, Norm(ck["norm"]), float(ck["res"]), ck


def chip_table(split, rejects):
    ch = pd.read_csv(HEREP / "chips.csv")
    ch = ch[ch.split == split]
    if rejects:
        al = pd.read_csv(HEREP / "chips_all.csv")
        al = al[(al.status == "align_reject") & (al.valid_frac >= 0.8)].merge(pd.read_csv(HEREP / "splits.csv")[["site_id", "split"]], on="site_id")
        ch = pd.concat([ch, al[al.split == split]], ignore_index=True)
    return ch


def score_model(name, split, dev, rejects=False):
    model, norm, R, _ = load_model(name, dev)
    ri = res_index(R)
    ch = chip_table(split, rejects)
    partners = shuffle_partners(ch[ch.status == "ok"] if "status" in ch else ch)
    cache = {}

    def item(c):
        if c not in cache:
            cache[c] = load_item(c)
        return cache[c]

    rows = []
    for cid, g in ch.groupby("capella_id"):
        z = item(cid)
        pos = {c: i for i, c in enumerate(z["chip_id"])}
        idx = [pos[c] for c in g.chip_id]
        X = np.stack([norm.x(z["s1"][i].astype("float32")) for i in idx])
        P = predict_chips(model, X, dev)
        # shuffled input: the partner chip's input, scored against this chip's target
        Xs, ok_s = [], []
        for c in g.chip_id:
            if c in partners:
                pc, pchip = partners[c]
                pz = item(pc)
                j = list(pz["chip_id"]).index(pchip)
                Xs.append(norm.x(pz["s1"][j].astype("float32")))
                ok_s.append((c, np.kron(pz["valid10"][j], np.ones((SC, SC), bool))))
        Ps = dict(zip([o[0] for o in ok_s], predict_chips(model, np.stack(Xs), dev))) if Xs else {}
        pv = dict(ok_s)
        for r, i, p in zip(g.itertuples(), idx, P):
            t = z["tgt"][i, ri].astype("float32")
            up = np.kron(z["valid10"][i], np.ones((SC, SC), bool))
            m = z["tvalid"][i] & up & np.isfinite(t)
            fl = flat_mask(np.where(m, t, np.nan), m)
            st = getattr(r, "status", "ok")
            s = scores(crop_c(norm.t_inv(p, R)), crop_c(t), crop_c(m), flat=crop_c(fl))
            base = {"chip_id": r.chip_id, "capella_id": cid, "site_id": r.site_id, "split": split, "cover": r.cover,
                    "pol": r.pol, "R": R, "status": st}
            if s:
                rows.append({**base, "method": name, **s})
            if r.chip_id in Ps:
                mm = m & pv[r.chip_id]
                s2 = scores(crop_c(norm.t_inv(Ps[r.chip_id], R)), crop_c(t), crop_c(mm), flat=crop_c(fl & mm))
                if s2:
                    rows.append({**base, "method": "shuffle_" + name, **s2})
    df = pd.DataFrame(rows)
    df.to_csv(HEREP / f"eval_{name}_{split}{'_rej' if rejects else ''}.csv", index=False)
    return df, R


def site_diff(model_df, base_df, metric, n=1000, seed=0):
    """Paired chip difference (model - baseline), site median, then median over sites with a bootstrap 95% CI."""
    a = model_df[["chip_id", "site_id", metric]].merge(base_df[["chip_id", metric]], on="chip_id", suffixes=("", "_b"))
    a["d"] = a[metric] - a[metric + "_b"]
    s = a.groupby("site_id").d.median().dropna().values
    if len(s) == 0:
        return {"diff": np.nan, "lo": np.nan, "hi": np.nan, "n_sites": 0}
    rng = np.random.default_rng(seed)
    bs = [np.median(rng.choice(s, len(s))) for _ in range(n)]
    return {"diff": float(np.median(s)), "lo": float(np.percentile(bs, 2.5)), "hi": float(np.percentile(bs, 97.5)),
            "n_sites": int(len(s)), "site_diffs": s.tolist()}


def site_median(df, metric):
    return float(df.groupby("site_id")[metric].median().median())


def decide(names, frames, R, split):
    bc = pd.read_csv(HEREP / "baselines_chips.csv")
    bc = bc[(bc.split == split) & (bc.R == R)]
    best = json.loads((HEREP / "best_baselines.json").read_text())[str(R)]
    ceil = pd.read_csv(HEREP / "ceiling_summary.csv") if (HEREP / "ceiling_summary.csv").exists() else None
    out = {"R": R, "split": split, "models": names, "best_baselines": best, "per_seed": {}}
    seed_pass = []
    diffs = {k: [] for k in ("ssim", "hp_r", "edge_f1")}
    for name, df in zip(names, frames):
        mdf = df[(df.method == name) & (df.status == "ok")]
        sdf = df[(df.method == "shuffle_" + name) & (df.status == "ok")]
        res = {}
        for mt in ("ssim", "hp_r", "edge_f1"):
            res[mt] = site_diff(mdf, bc[bc.method == best[mt]], mt)
            diffs[mt].append(res[mt]["diff"])
        res["flat_ratio"] = site_median(mdf, "flat_ratio")
        res["shuffle_hp_r"] = site_median(sdf, "hp_r") if len(sdf) else np.nan
        res["model_medians"] = {mt: site_median(mdf, mt) for mt in ("psnr", "ssim", "detail_r", "hp_r", "edge_f1", "gms")}
        ok = (res["ssim"]["diff"] >= RULE["ssim"] and res["ssim"]["lo"] > 0 and
              res["hp_r"]["diff"] >= RULE["hp_r"] and res["hp_r"]["lo"] > 0 and
              res["edge_f1"]["diff"] >= 0 and res["flat_ratio"] <= FLAT_MAX and res["shuffle_hp_r"] <= SHUF_MAX)
        res["passes"] = bool(ok)
        if ceil is not None and (ceil.R == R).any():
            c = ceil[ceil.R == R].iloc[0]
            res["gap_closed"] = {}
            for mt in ("ssim", "hp_r"):
                b = site_median(bc[bc.method == best[mt]], mt)
                res["gap_closed"][mt] = float((res["model_medians"][mt] - b) / max(c[mt] - b, 1e-9))
        for mt in ("ssim", "hp_r", "edge_f1"):
            res[mt].pop("site_diffs", None)
        out["per_seed"][name] = res
        seed_pass.append(ok)
    mean_ok = (np.mean(diffs["ssim"]) >= RULE["ssim"] and np.mean(diffs["hp_r"]) >= RULE["hp_r"]
               and np.mean(diffs["edge_f1"]) >= 0)
    out["mean_diffs"] = {k: float(np.mean(v)) for k, v in diffs.items()}
    out["passes"] = bool(mean_ok and all(seed_pass))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--names", nargs="+", required=True)
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--feature-test", action="store_true")
    ap.add_argument("--rejects", action="store_true")
    a = ap.parse_args()
    dev = torch.device(a.device)
    frames, Rs = [], set()
    for name in a.names:
        df, R = score_model(name, a.split, dev, a.rejects)
        frames.append(df)
        Rs.add(R)
        print(name, df[df.method == name][["psnr", "ssim", "detail_r", "hp_r", "edge_f1", "gms", "flat_ratio"]]
              .median().round(3).to_dict(), flush=True)
        if a.feature_test:
            import feature_test as ft
            model, norm, R_, _ = load_model(name, dev)
            fn = lambda s1: norm.t_inv(predict_large(model, norm.x(s1), dev), R_)  # noqa: E731
            fr = pd.DataFrame(ft.run(fn, name))
            fr.to_csv(HEREP / f"feature_{name}.csv", index=False)
            print(f"feature test {name}: detected {fr.det.mean():.3f} of {len(fr)}", flush=True)
    assert len(Rs) == 1, "all models in one decision must share a target resolution"
    R = Rs.pop()
    dec = decide(a.names, frames, R, a.split)
    tag = "+".join(a.names)
    (HEREP / f"decision_{tag}_{a.split}.json").write_text(json.dumps(dec, indent=1))
    print(json.dumps(dec, indent=1), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
    finish()
