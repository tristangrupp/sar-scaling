"""Experiment 9, tune half: score every candidate rule, log all of them, and pick one by DESIGN.md.

Writes grid_tune.csv (every rule x footprint x hit area), chosen_rule.json, tune_looks.csv (the chosen
rule's looks) and tune_pairs.csv (its per-pair counts).
"""
from __future__ import annotations

import json
import os
import sys
import time
import traceback

import numpy as np
import pandas as pd

import core as C


def main():
    t0 = time.time()
    pairs, meta = C.load_pairs("tune")
    print(f"tune half: {len(pairs)} pairs, {meta.site_id.nunique()} sites; VH present in "
          f"{sum(p.has_vh for p in pairs)}", flush=True)
    rows = []
    for i, rule in enumerate(C.all_rules()):
        df = C.score_rule(pairs, rule)
        for (foot, h), d in df.groupby(["foot", "hit_m2"]):
            primary = foot == C.PRIMARY["foot"] and h == C.PRIMARY["hit_m2"]
            s = C.summarize(d, nb=1000 if primary else 200, seed=i)
            rows.append({"rule": C.rule_name(rule), **rule, "foot": foot, "hit_m2": h, **{k: s[k] for k in (
                "looks", "hits", "sites", "sites_with_looks", "pairs", "hit_rate", "rand_rate", "lift", "bright_rate",
                "lift_bright", "lift_vs_bright")}, "lift_lo": s["lift_ci"][0], "lift_hi": s["lift_ci"][1]})
        if i % 10 == 0:
            print(f"{i + 1}/180 {C.rule_name(rule)} {time.time() - t0:.0f}s", flush=True)
            pd.DataFrame(rows).to_csv(C.ROOT / "grid_tune.csv", index=False)
    G = pd.DataFrame(rows)
    G.to_csv(C.ROOT / "grid_tune.csv", index=False)

    P = G[(G.foot == C.PRIMARY["foot"]) & (G.hit_m2 == C.PRIMARY["hit_m2"])].copy()
    P["eligible"] = (P.looks >= C.MIN_LOOKS) & (P.sites_with_looks >= C.MIN_SITES) & P.lift.notna()
    P["complexity"] = P.pol.map(C.COMPLEXITY) + P.smooth.map(C.COMPLEXITY)
    E = P[P.eligible].sort_values("lift", ascending=False)
    best = E.lift.iloc[0]
    tied = E[E.lift >= best * (1 - C.TIE_REL)].sort_values(["complexity", "lift"], ascending=[True, False])
    ch = tied.iloc[0]
    rule = {"pol": ch.pol, "thr": float(ch.thr), "smooth": ch.smooth, "min_ha": float(ch.min_ha)}
    pd.set_option("display.width", 250)
    cols = ["rule", "looks", "sites_with_looks", "hit_rate", "rand_rate", "lift", "lift_lo", "lift_hi", "bright_rate", "complexity"]
    print(f"\neligible rules {len(E)} of {len(P)}; best lift {best:.3f}; tied within {C.TIE_REL:.0%}: {len(tied)}")
    print("top 15 eligible by tune lift (500 m, 200 m2):\n", E[cols].head(15).round(3).to_string(index=False), flush=True)
    print("tied set:\n", tied[cols].round(3).to_string(index=False), flush=True)
    print("ineligible (too few looks):", int((~P.eligible).sum()), flush=True)

    df, looks = C.score_rule(pairs, rule, keep_looks=True)
    df.to_csv(C.ROOT / "tune_pairs.csv", index=False)
    looks.to_csv(C.ROOT / "tune_looks.csv", index=False)
    out = {"rule": rule, "rule_name": C.rule_name(rule), "chosen_from": int(len(E)), "tied": tied.rule.tolist(),
           "best_lift_rule": E.rule.iloc[0], "tune": {}}
    for (foot, h), d in df.groupby(["foot", "hit_m2"]):
        out["tune"][f"{foot}m_{h}m2"] = C.summarize(d, nb=2000, seed=1)
    out["tune_topk"] = {f"{foot}m": C.topk(looks, df, foot=foot) for foot in C.FOOTPRINTS}
    (C.ROOT / "chosen_rule.json").write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps({"rule": out["rule_name"], "tune_primary": out["tune"]["500m_200m2"], "topk": out["tune_topk"]["500m"]},
                     indent=1, default=float), flush=True)
    print(f"done {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
    sys.stdout.flush()
    os._exit(0)
