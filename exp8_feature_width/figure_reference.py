"""Headline figure with the blurred-Capella reference: figures/detection_vs_reference.png.

Reads curve_bins.csv, width_table.csv, summary.json (from analyze.py) and blur_reference.csv.
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
from analyze import BLAB, EDGES, curve, fit  # noqa: E402


def main():
    bins = pd.read_csv(HERE / "curve_bins.csv")
    wt = pd.read_csv(HERE / "width_table.csv")
    sm = json.loads((HERE / "summary.json").read_text())
    br = pd.read_csv(HERE / "blur_reference.csv")
    br["wbin"] = pd.cut(br.cap_width, EDGES, labels=BLAB, right=False)
    mids = np.sqrt(np.array(EDGES[:-1]) * np.array(EDGES[1:]))
    ws = np.exp(np.linspace(np.log(0.8), np.log(40), 200))
    fig, ax = plt.subplots(figsize=(8, 5))
    rb = br.groupby("wbin", observed=False).blur_det.agg(["mean", "size"]).reindex(BLAB)
    m = rb["size"] >= 5
    ax.plot(mids[m.values], rb["mean"][m], "s", color="#7e3fc0", label="Capella blurred to 20 m, same test (reference)")
    ab = fit(br.cap_width.values, br.blur_det.astype(float).values, 0.05)
    ax.plot(ws, curve(ab, 0.05, ws), color="#7e3fc0", lw=1.5, ls="--")
    for p, c, lab in [("multi", "#1f5fa8", "Sentinel-1 multi-date mean"), ("single", "#d07a1e", "Sentinel-1 single date")]:
        b = bins[(bins["product"] == p) & (bins.pol == "either") & (bins.group == "all")]
        mm = b.n >= 5
        ax.errorbar(mids[mm.values], b.rate[mm], yerr=np.array([(b.rate - b.lo)[mm], (b.hi - b.rate)[mm]]), fmt="o",
                    color=c, capsize=3, label=f"{lab} (95% scene bootstrap)")
        r = wt[(wt.group == "all") & (wt["product"] == p) & (wt.pol == "either")].iloc[0]
        ax.plot(ws, curve((r.a, r.b), sm["false_alarm"][f"{p}_either"], ws), color=c, lw=2)
    ax.axhline(sm["false_alarm"]["multi_either"], color="#888", ls=":", lw=1)
    ax.text(0.85, sm["false_alarm"]["multi_either"] + 0.015, "false-alarm floor", fontsize=8, color="#666")
    for yv in (0.5, 0.9):
        ax.axhline(yv, color="#ddd", lw=0.8, zorder=0)
    ax.set_xscale("log")
    ax.set_xticks([1, 2, 3, 5, 8, 12, 20, 30, 40])
    ax.set_xticklabels(["1", "2", "3", "5", "8", "12", "20", "30", "40"])
    ax.set_xlim(0.8, 42)
    ax.set_ylim(0, 1.03)
    ax.set_xlabel("feature width on Capella (m, full width at half contrast)")
    ax.set_ylabel("share of 200 m segments detected")
    ax.set_title("Sentinel-1 detects about half the features at any width;\nresolution alone would allow far more", fontsize=11)
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(HERE / "figures" / "detection_vs_reference.png", dpi=130)
    plt.close(fig)
    print("ok", flush=True)


if __name__ == "__main__":
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            main()
    except Exception:
        traceback.print_exc()
    sys.stdout.flush()
    os._exit(0)
