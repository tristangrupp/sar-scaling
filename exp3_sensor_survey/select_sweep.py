"""Round 2 step 1: pick the consistency setting on VALIDATION only.
Rule (fixed before seeing all results): beat bicubic on as many of MAE, SSIM, detail_r, ch_p2t, ch_t2p as
possible; ties go to the lowest grid ratio (least input-grid imprint). Writes chosen.json."""
import json, sys, os
import pandas as pd
SET = {"s_w0": (0.0, 0), "s_w005": (0.05, 0), "s_w02": (0.2, 0), "s_w05": (0.5, 0), "s_w05_40": (0.5, 40), "s_w2_40": (2.0, 40)}
d = pd.concat([pd.read_csv(f"eval2/{f}") for f in os.listdir("eval2") if f.startswith("sweep_") and f.endswith("_summary.csv")])
d = d[d.split == "val"].drop_duplicates("model").set_index("model")
b = d.loc["bicubic"]
rows = []
for m, (w, at) in SET.items():
    if m not in d.index:
        continue
    r = d.loc[m]
    wins = int(r.mae < b.mae) + int(r.ssim > b.ssim) + int(r.detail_r > b.detail_r) + int(r.ch_p2t < b.ch_p2t) + int(r.ch_t2p < b.ch_t2p)
    rows.append({"run": m, "weight": w, "cons_at_m": at or 20, "wins_vs_bicubic": wins, **r[["mae", "ssim", "detail_r", "hf_ratio", "cons", "ch_p2t", "ch_t2p", "grid"]].to_dict()})
t = pd.DataFrame(rows).sort_values(["wins_vs_bicubic", "grid"], ascending=[False, True])
t.to_csv("eval2/sweep_table.csv", index=False)
best = t.iloc[0]
json.dump({"run": best.run, "cons": float(best.weight), "cons_at": int(best.cons_at_m) if best.cons_at_m != 20 else 0}, open("chosen.json", "w"))
print(t.round(3).to_string(), "\nchosen:", best.run, flush=True)
