"""Match X-band collect pairs to Sentinel-1 pass pairs at each repeat site.

The tasking chain in retrospect: X-band "before" image x1, then Sentinel-1 passes s_b < s_a on one
track (same relative orbit and direction), then X-band "after" image x2, with
    t(x1) < t(s_b) < t(s_a) < t(x2)        (acquisition times, not dates)
so any lasting change Sentinel-1 flags between s_b and s_a happened inside the X-band interval.
Per track, take the last two passes inside the interval; across tracks, prefer the shortest S1 gap,
then the shortest latency.

X-band pairs: any two collects at a site on different days, MIN_GAP to MAX_GAP days apart, with
matched geometry: same polarization, same look direction (Umbra azimuth within 20 deg, else same
orbit direction when known) and incidence within 5 deg. Mixed geometry makes X-band change maps
useless.

Writes tasking_pairs.csv (one row per qualifying X-band pair, best S1 pair) and updates
repeat_sites.csv with counts. Latency = x2 - s_a (days from the S1 flag to the X-band look).
"""
from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent
MIN_GAP, MAX_GAP, DINC, DAZ = 6, 60, 5.0, 20.0


def main():
    col = pd.read_csv(ROOT / "xband_collects.csv")
    col["d"] = pd.to_datetime(col.datetime, utc=True, format="ISO8601").dt.tz_localize(None)
    sites = pd.read_csv(ROOT / "repeat_sites.csv")
    sites = sites[[c for c in sites.columns if not c.startswith("n_") or c in ("n_collects", "n_days")]]
    p = pd.read_csv(ROOT / "s1_passes.csv")
    p["d"] = pd.to_datetime(p.t, utc=True, format="ISO8601").dt.tz_localize(None)
    rows, loose = [], {}
    for sid in sites.site_id:
        s = col[col.site_id == sid].sort_values("d").reset_index(drop=True)
        ps = p[p.site_id == sid]
        tracks = [(k, np.array(sorted(g.d.unique()), dtype="datetime64[s]"))
                  for k, g in ps.groupby(["rel_orbit", "orbit_state"])]
        d = s.d.values.astype("datetime64[s]")
        day = s.d.values.astype("datetime64[D]")
        gap = (day[None, :] - day[:, None]).astype(int)
        I, J = np.where((gap >= MIN_GAP) & (gap <= MAX_GAP))
        n_loose = 0
        for i, j in zip(I, J):
            best = None
            for (ro, st), td in tracks:
                inside = td[(td > d[i]) & (td < d[j])]
                if len(inside) >= 2:
                    sb, sa = inside[-2], inside[-1]
                    key = ((sa - sb).astype(int), (d[j] - sa).astype(int))   # seconds
                    if best is None or key < best[0]:
                        best = (key, sb, sa, ro, st)
            if best is None:
                continue
            n_loose += 1
            a, b = s.iloc[i], s.iloc[j]
            same_pol = str(a.pol) == str(b.pol)
            daz = abs((a.azimuth - b.azimuth + 180) % 360 - 180) if pd.notna(a.azimuth) and pd.notna(b.azimuth) else np.nan
            if np.isfinite(daz):
                same_look = daz <= DAZ
            elif pd.notna(a.orbit) and pd.notna(b.orbit):
                same_look = a.orbit == b.orbit
            else:
                same_look = True
            dinc = abs(a.inc - b.inc) if pd.notna(a.inc) and pd.notna(b.inc) else np.nan
            if not (same_pol and same_look and np.isfinite(dinc) and dinc <= DINC):
                continue
            _, sb, sa, ro, st = best
            rows.append({"site_id": sid, "provider": a.provider if a.provider == b.provider else f"{a.provider}+{b.provider}",
                         "xid1": a.xband_id, "xid2": b.xband_id, "x1": a.day, "x2": b.day, "x_gap_d": int(gap[i, j]),
                         "pol": a.pol, "d_inc": round(float(dinc), 2), "d_az": round(float(daz), 1) if np.isfinite(daz) else None,
                         "x1_time": str(d[i]), "x2_time": str(d[j]),
                         "s1_before": str(sb)[:10], "s1_after": str(sa)[:10], "s1_before_time": str(sb), "s1_after_time": str(sa),
                         "s1_gap_d": round((sa - sb).astype(int) / 86400), "rel_orbit": int(ro), "orbit_state": st,
                         "latency_d": round((d[j] - sa).astype(int) / 86400, 2),
                         "lon": a.lon, "lat": a.lat})
        loose[sid] = n_loose
    t = pd.DataFrame(rows)
    t.to_csv(ROOT / "tasking_pairs.csv", index=False)
    sites["n_s1_passes"] = sites.site_id.map(p.groupby("site_id").size()).fillna(0).astype(int)
    sites["n_pairs_s1_inside"] = sites.site_id.map(loose).fillna(0).astype(int)
    sites["n_pairs_matched"] = sites.site_id.map(t.groupby("site_id").size()).fillna(0).astype(int)
    sites["n_after_days_matched"] = sites.site_id.map(t.groupby("site_id").x2.nunique()).fillna(0).astype(int)
    sites.to_csv(ROOT / "repeat_sites.csv", index=False)
    print(f"repeat sites: {len(sites)}; with any S1 pass: {(sites.n_s1_passes > 0).sum()}")
    print(f"sites with an X-band pair ({MIN_GAP}-{MAX_GAP} d) bracketing an S1 pass pair, any geometry: "
          f"{(sites.n_pairs_s1_inside > 0).sum()} ({sum(loose.values())} pairs)")
    print(f"... and with matched X-band geometry: {(sites.n_pairs_matched > 0).sum()} sites, {len(t)} pairs; "
          f"sites with >= 5 distinct after-dates: {(sites.n_after_days_matched >= 5).sum()}")
    print("by provider (sites):", t.groupby("provider").site_id.nunique().to_dict())
    print("by provider (pairs):", t.groupby("provider").size().to_dict())
    print("latency days (x2 - s_a) median / p90:", t.latency_d.median(), t.latency_d.quantile(.9))
    print("S1 gap days median:", t.s1_gap_d.median())
    pd.set_option("display.width", 250)
    print(sites.sort_values("n_after_days_matched", ascending=False).head(30).to_string(index=False))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
    sys.stdout.flush()
    os._exit(0)
