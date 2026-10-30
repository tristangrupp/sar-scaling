"""Experiment 9, test half: score the one rule chosen on the tune half, once.

Refuses to run if results_test.json already exists. Writes results_test.json, test_pairs.csv and
test_looks.csv.
"""
from __future__ import annotations

import json
import os
import sys
import traceback

import core as C


def main():
    out_f = C.ROOT / "results_test.json"
    if out_f.exists():
        print("results_test.json exists: the test half has been scored; refusing to score it again", flush=True)
        return
    ch = json.loads((C.ROOT / "chosen_rule.json").read_text())
    rule = ch["rule"]
    pairs, meta = C.load_pairs("test")
    print(f"test half: {len(pairs)} pairs, {meta.site_id.nunique()} sites; rule {ch['rule_name']}", flush=True)
    df, looks = C.score_rule(pairs, rule, keep_looks=True)
    df.to_csv(C.ROOT / "test_pairs.csv", index=False)
    looks.to_csv(C.ROOT / "test_looks.csv", index=False)
    out = {"rule": rule, "rule_name": ch["rule_name"], "test": {}}
    for (foot, h), d in df.groupby(["foot", "hit_m2"]):
        out["test"][f"{foot}m_{h}m2"] = C.summarize(d, nb=2000, seed=2)
    out["test_topk"] = {f"{foot}m": C.topk(looks, df, foot=foot) for foot in C.FOOTPRINTS}
    p = out["test"]["500m_200m2"]
    out["holds"] = bool(p["lift_ci"][0] > 1)
    out_f.write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps(out, indent=1, default=float), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
    sys.stdout.flush()
    os._exit(0)
