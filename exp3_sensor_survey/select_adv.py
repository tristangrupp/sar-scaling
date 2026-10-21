"""Round 4: choose the adversarial weight on VALIDATION (target mt10), relative to the adv 0 control (r4_c0_s0).
Rule fixed before any run:
  eligible: MAE <= control + 0.01 dB, ch_p2t <= control + 0.6 m, grid <= control + 0.05   (round 3 seed noise)
  sharpness gains: hf_ratio > control + 0.004, ch_t2p < control - 0.6 m
  winner: eligible with >= 1 sharpness gain; most gains, ties -> higher hf_ratio. Else no winner."""
import json
import pandas as pd
d = pd.read_csv("eval2/r4_adv_val_summary.csv")
d = d[d.split == "val"].set_index("model")
c = d.loc["e3r4_c0_s0"]
rows = []
for m in [x for x in d.index if x.startswith("e3r4_a")]:
    r = d.loc[m]
    elig = (r.mae <= c.mae + 0.01) and (r.ch_p2t <= c.ch_p2t + 0.6) and (r.grid <= c.grid + 0.05)
    gains = int(r.hf_ratio > c.hf_ratio + 0.004) + int(r.ch_t2p < c.ch_t2p - 0.6)
    rows.append({"run": m, "eligible": elig, "sharpness_gains": gains, "d_mae": r.mae - c.mae, "d_ch_p2t": r.ch_p2t - c.ch_p2t,
                 "d_grid": r.grid - c.grid, "d_hf": r.hf_ratio - c.hf_ratio, "d_ch_t2p": r.ch_t2p - c.ch_t2p, **r.to_dict()})
t = pd.DataFrame(rows).sort_values(["eligible", "sharpness_gains", "hf_ratio"], ascending=[False, False, False])
t.to_csv("eval2/r4_adv_table.csv", index=False)
ok = t[t.eligible & (t.sharpness_gains >= 1)]
win = ok.iloc[0].run if len(ok) else None
json.dump({"run": win}, open("chosen_adv.json", "w"))
print(t.round(4).to_string(), "\nchosen:", win, flush=True)
