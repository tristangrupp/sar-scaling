"""Round 3 step 2: pick the loss for the multi-date target on VALIDATION (target mt10).
Rule, fixed in advance: wins over plain L1 (m_multi) on MAE, SSIM, detail_r, ch_p2t, ch_t2p; ties -> higher hf_ratio."""
import json, sys
import pandas as pd
d = pd.read_csv("eval2/r3_loss_val_summary.csv")
d = d[d.split == "val"].set_index("model")
b = d.loc["m_multi"]
rows = []
for m in [x for x in d.index if x.startswith("r3_g")]:
    r = d.loc[m]
    w = int(r.mae < b.mae) + int(r.ssim > b.ssim) + int(r.detail_r > b.detail_r) + int(r.ch_p2t < b.ch_p2t) + int(r.ch_t2p < b.ch_t2p)
    rows.append({"run": m, "wins_vs_L1": w, **r.to_dict()})
t = pd.DataFrame(rows).sort_values(["wins_vs_L1", "hf_ratio"], ascending=[False, False])
t.to_csv("eval2/r3_loss_table.csv", index=False)
w = t.iloc[0].run
json.dump({"run": w, "grad_w": 0.5 if "g05" in w else 1.0}, open("chosen_loss.json", "w"))
print(t.round(3).to_string(), "\nchosen:", w, flush=True)
