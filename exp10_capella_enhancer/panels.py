"""Example panels: Sentinel-1 inputs, baselines and the degraded Capella targets (DESIGN.md).

  python panels.py [--split test] [--n 6] [--model esrt_r5_s0]

Outlines follow panel_roles.py: targets (the truth every score uses) purple, model outputs green, inputs and
baselines none. Each tile shows dB with its own 2-98 percentile stretch. Writes figures/panels_<split>.png.
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "4")

import common  # noqa: F401
from common import HERE, PROJ, RES_LIST, SC, baseline_preds, bicubic_up, finish, load_item

import argparse
import sys
import traceback

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(PROJ))
from panel_roles import legend, outline  # noqa: E402


def show(ax, a, title, mask=None):
    v = a[np.isfinite(a)] if mask is None else a[mask & np.isfinite(a)]
    lo, hi = np.percentile(v, [2, 98]) if v.size else (0, 1)
    ax.imshow(np.where(np.isfinite(a), a, np.nan), cmap="gray", vmin=lo, vmax=hi, interpolation="nearest")
    ax.set_title(title, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--model", default=None)
    a = ap.parse_args()
    ch = pd.read_csv(HERE / "chips.csv")
    ch = ch[(ch.split == a.split) & (ch.valid_frac >= 0.95)]
    # one chip per cover class first, then the best-aligned remaining chips, never two from one site
    picks, sites = [], set()
    for cov in ["built", "crop", "tree", "grass", "bare", "shrub", "wetland"]:
        g = ch[(ch.cover == cov) & ~ch.site_id.isin(sites)].sort_values("r_final", ascending=False)
        if len(g) and len(picks) < a.n:
            picks.append(g.iloc[0])
            sites.add(g.iloc[0].site_id)
    for _, r in ch.sort_values("r_final", ascending=False).iterrows():
        if len(picks) >= a.n:
            break
        if r.site_id not in sites:
            picks.append(r)
            sites.add(r.site_id)
    model = None
    if a.model:
        import torch
        from eval import load_model
        from train import predict_chips
        cpu = torch.device("cpu")
        model, norm, mR, _ = load_model(a.model, cpu)
    cols = ["S1 VV, one date", "S1 VV, multi-date mean", "Lee, one date", "Despeckler, multi-date",
            "Exp. 2 ESRT, one date"] + ([f"Model {a.model}"] if a.model else []) + \
           [f"Capella at {R:g} m" for R in sorted(RES_LIST, reverse=True)]
    fig, axes = plt.subplots(len(picks), len(cols), figsize=(2.1 * len(cols), 2.3 * len(picks) + 0.4))
    axes = np.atleast_2d(axes)
    for i, r in enumerate(picks):
        z = load_item(r.capella_id)
        j = list(z["chip_id"]).index(r.chip_id)
        s1 = z["s1"][j].astype("float32")
        b = baseline_preds(s1)
        m = z["tvalid"][j] & np.kron(z["valid10"][j], np.ones((SC, SC), bool))
        tiles = [(bicubic_up(s1[0]), None), (b["bicubic_multi"], None), (b["lee_single"], None),
                 (b["n2n_multi"], None), (b.get("esrt_exp2", np.full((256, 256), np.nan)), None)]
        if model is not None:
            tiles.append((norm.t_inv(predict_chips(model, norm.x(s1)[None], cpu)[0], mR), "model"))
        for R in sorted(RES_LIST, reverse=True):
            tiles.append((np.where(m, z["tgt"][j, RES_LIST.index(R)].astype("float32"), np.nan), "truth"))
        for k, (img, role) in enumerate(tiles):
            ax = axes[i, k]
            show(ax, img, cols[k] if i == 0 else "")
            if role:
                outline(ax, role)
        axes[i, 0].set_ylabel(f"{r.country[:14]}\n{r.cover}, {r.pol}\n{r.chip_id}", fontsize=8)
    fig.suptitle(f"Experiment 10, {a.split} chips (640 m). Inputs and baselines shown at 2.5 m by bicubic "
                 "interpolation; targets are Capella degraded to each effective resolution", fontsize=10)
    legend(fig)
    fig.tight_layout(rect=(0, 0.02, 1, 0.97))
    (HERE / "figures").mkdir(exist_ok=True)
    out = HERE / "figures" / f"panels_{a.split}{'_' + a.model if a.model else ''}.png"
    fig.savefig(out, dpi=110)
    print("wrote", out, flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
    finish()
