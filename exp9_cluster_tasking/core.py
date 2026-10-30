"""Experiment 9 core: Sentinel-1 change clusters as Capella tasking looks, scored against experiment 7's
Capella change truth. Rules, looks, hits and baselines are defined here and fixed in DESIGN.md.
"""
from __future__ import annotations

from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage

ROOT = Path(__file__).resolve().parent
E7 = ROOT.parent / "exp7_change_tasking"
S1DIR = ROOT / "data" / "s1"
N, K, RES = 256, 5, 10.0            # 10 m Sentinel-1 grid, 5 x 5 Capella 2 m pixels per S1 pixel

POLS = ("vv", "vh", "any", "both")
THRESHOLDS = (2.0, 3.0, 4.0)
SMOOTH = ("m3", "m5", "lr5")
MIN_HA = (0.1, 0.2, 0.5, 1.0, 2.0)
FOOTPRINTS = (500, 250)              # look side, m
HIT_M2 = (200, 50, 1000)             # Capella change area inside a look that makes it a hit
PRIMARY = {"foot": 500, "hit_m2": 200}
BRIGHT_SEP = 5                       # brightest-pixel baseline: looks at least 5 px (50 m) apart
MIN_LOOKS, MIN_SITES = 30, 5         # eligibility of a rule on the tune half
TIE_REL = 0.02                       # rules within 2% of the best tune lift count as tied
COMPLEXITY = {"vv": 0, "vh": 0, "any": 1, "both": 1, "m3": 0, "m5": 1, "lr5": 1}


def all_rules():
    return [dict(pol=p, thr=t, smooth=s, min_ha=a) for p, t, s, a in product(POLS, THRESHOLDS, SMOOTH, MIN_HA)]


def rule_name(r):
    return f"{r['pol']}_{r['smooth']}_{r['thr']:g}dB_{r['min_ha']:g}ha"


# ------------------------------------------------------------------ per pair inputs
class Pair:
    def __init__(self, tag, site_id):
        self.tag, self.site_id = tag, int(site_id)
        z = np.load(E7 / "data" / "r2" / f"{tag}.npz")
        s = np.load(S1DIR / f"{tag}.npz")
        self.epsg, self.x0, self.y1 = int(z["epsg"]), float(z["x0"]), float(z["y1"])
        self.lab = z["lab"]
        self.wc = z["wc"]
        land = (self.wc != 80) & (self.wc != 0) if self.wc.any() else np.ones((N, N), bool)
        capvalid10 = z["use"].reshape(N, K, N, K).mean(axis=(1, 3)) >= 0.5
        self.domain = z["s1ok"] & land & capvalid10          # where looks may point and flags count
        self.lin = {k: s[k] for k in ("vv_b", "vv_a", "vh_b", "vh_a")}
        self.has_vh = bool(np.isfinite(self.lin["vh_b"]).mean() > 0.5 and np.isfinite(self.lin["vh_a"]).mean() > 0.5)
        pm = (self.lab > 0).reshape(N, K, N, K).sum(axis=(1, 3)) * 4.0   # Capella change m^2 per 10 m pixel
        self.patch10 = pm
        self.integral = np.pad(pm.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
        self._area = {}
        self._d = {}
        # keep the Capella images for panels on demand
        self._z = z

    def area_map(self, foot):
        """Capella change m^2 inside a look of side `foot` m centered on each 10 m pixel. Looks near the
        window edge slide inward so they stay whole inside the 2.56 km window."""
        if foot not in self._area:
            w = int(round(foot / RES))
            st = np.clip(np.arange(N) - w // 2, 0, N - w)
            I = self.integral
            r0, c0 = st[:, None], st[None, :]
            self._area[foot] = I[r0 + w, c0 + w] - I[r0, c0 + w] - I[r0 + w, c0] + I[r0, c0]
        return self._area[foot]

    def look_box(self, r, c, foot):
        w = int(round(foot / RES))
        r0 = int(np.clip(r - w // 2, 0, N - w))
        c0 = int(np.clip(c - w // 2, 0, N - w))
        return r0, c0, w

    def diff(self, pol, smooth):
        """Median-corrected change in dB (after minus before) for one polarization, and its valid mask."""
        key = (pol, smooth)
        if key in self._d:
            return self._d[key]
        b, a = self.lin[f"{pol}_b"], self.lin[f"{pol}_a"]
        m = np.isfinite(a) & np.isfinite(b) & (a > 0) & (b > 0)
        size, ero = (3, 1) if smooth == "m3" else (5, 2)
        cnt = ndimage.uniform_filter(m.astype(float), size)
        with np.errstate(divide="ignore", invalid="ignore"):
            if smooth == "lr5":           # ratio of local mean intensities: the speckle (gamma) likelihood-ratio statistic
                ma = ndimage.uniform_filter(np.where(m, a, 0), size) / cnt
                mb = ndimage.uniform_filter(np.where(m, b, 0), size) / cnt
                d = 10 * np.log10(ma / mb)
            else:                         # mean of dB (the experiment 7 rule for m3)
                da = ndimage.uniform_filter(np.where(m, 10 * np.log10(a), 0), size) / cnt
                db = ndimage.uniform_filter(np.where(m, 10 * np.log10(b), 0), size) / cnt
                d = da - db
        ok = ndimage.binary_erosion(m, iterations=ero) & self.domain
        if ok.sum() > 100:
            d = d - float(np.median(d[ok]))
        d = np.where(ok, d, np.nan)
        self._d[key] = (d, ok)
        return self._d[key]

    def flags(self, rule):
        """Flag map and the change map (dB) a rule scores clusters with."""
        pol, t, sm = rule["pol"], rule["thr"], rule["smooth"]
        if pol in ("vh", "any", "both") and not self.has_vh:
            return None, None
        if pol in ("vv", "vh"):
            d, ok = self.diff(pol, sm)
            return ok & (np.abs(np.nan_to_num(d)) > t), d
        dv, okv = self.diff("vv", sm)
        dh, okh = self.diff("vh", sm)
        ok = okv & okh
        dv0, dh0 = np.nan_to_num(dv), np.nan_to_num(dh)
        if pol == "any":
            f = ok & ((np.abs(dv0) > t) | (np.abs(dh0) > t))
            d = np.where(np.abs(dv0) >= np.abs(dh0), dv0, dh0)
        else:                             # both: VV and VH flagged with the same sign
            f = ok & (np.abs(dv0) > t) & (np.abs(dh0) > t) & (np.sign(dv0) == np.sign(dh0))
            d = 0.5 * (dv0 + dh0)
        return f, np.where(ok, d, np.nan)

    def clusters(self, rule):
        """Connected flag regions (8-connected) of at least rule['min_ha']. One look per cluster, at the
        rounded centroid. Score = area (ha) x mean |change| (dB)."""
        f, d = self.flags(rule)
        if f is None:
            return None
        lab, n = ndimage.label(f, structure=np.ones((3, 3)))
        if n == 0:
            return pd.DataFrame(columns=["row", "col", "area_ha", "mean_abs_db", "score", "label"]), lab
        idx = np.arange(1, n + 1)
        size = ndimage.sum(np.ones_like(lab), lab, idx)
        keep = size * RES * RES / 1e4 >= rule["min_ha"] - 1e-9
        if not keep.any():
            return pd.DataFrame(columns=["row", "col", "area_ha", "mean_abs_db", "score", "label"]), lab
        idx = idx[keep]
        com = ndimage.center_of_mass(np.ones_like(lab), lab, idx)
        mabs = ndimage.mean(np.abs(np.nan_to_num(d)), lab, idx)
        area = size[keep] * RES * RES / 1e4
        cl = pd.DataFrame({"row": [int(round(r)) for r, _ in com], "col": [int(round(c)) for _, c in com],
                           "area_ha": area, "mean_abs_db": mabs, "score": area * mabs, "label": idx})
        return cl, lab

    def brightest(self, rule, n):
        """n highest-|change| pixels in the domain, at least BRIGHT_SEP px apart (no clustering)."""
        _, d = self.flags(rule)
        if d is None or n == 0:
            return np.zeros((0, 2), int)
        v = np.abs(np.nan_to_num(d, nan=-1))
        order = np.argsort(v.ravel())[::-1]
        taken = np.zeros((N, N), bool)
        out = []
        for i in order:
            r, c = divmod(int(i), N)
            if v[r, c] < 0:
                break
            if taken[r, c]:
                continue
            out.append((r, c))
            taken[max(r - BRIGHT_SEP, 0):r + BRIGHT_SEP + 1, max(c - BRIGHT_SEP, 0):c + BRIGHT_SEP + 1] = True
            if len(out) == n:
                break
        return np.array(out, int).reshape(-1, 2)

    def random_rate(self, foot, hit_m2):
        """Exact hit probability of a look centered on a uniform random domain pixel."""
        A = self.area_map(foot)
        return float((A[self.domain] >= hit_m2).mean()) if self.domain.any() else np.nan


def load_pairs(half=None):
    pr = pd.read_csv(E7 / "r2_pair_results.csv")
    ok = pr[pr.status == "ok"]
    sp = pd.read_csv(ROOT / "sites_split.csv")
    ok = ok.merge(sp[["site_id", "half", "land"]], on="site_id")
    if half:
        ok = ok[ok.half == half]
    return [Pair(r.tag, r.site_id) for r in ok.itertuples()], ok


def score_rule(pairs, rule, feet=FOOTPRINTS, hits=HIT_M2, with_bright=True, keep_looks=False):
    """Per-pair counts for one rule: looks, cluster hits, expected random hits, brightest-pixel hits."""
    rows, looks = [], []
    for p in pairs:
        res = p.clusters(rule)
        if res is None:
            continue
        cl, _ = res
        n = len(cl)
        bright = p.brightest(rule, n) if with_bright else np.zeros((0, 2), int)
        for foot in feet:
            A = p.area_map(foot)
            ca = A[cl.row.values.astype(int), cl.col.values.astype(int)] if n else np.zeros(0)
            ba = A[bright[:, 0], bright[:, 1]] if len(bright) else np.zeros(0)
            for h in hits:
                rows.append({"tag": p.tag, "site_id": p.site_id, "foot": foot, "hit_m2": h, "looks": n,
                             "hits": int((ca >= h).sum()), "p_rand": p.random_rate(foot, h),
                             "bright_looks": len(bright), "bright_hits": int((ba >= h).sum())})
            if keep_looks and n:
                c2 = cl.copy()
                c2["tag"], c2["site_id"], c2["foot"], c2["change_m2"] = p.tag, p.site_id, foot, ca
                c2["rank"] = c2.score.rank(ascending=False, method="first").astype(int)
                looks.append(c2)
    df = pd.DataFrame(rows)
    df["exp_rand"] = df.looks * df.p_rand
    return (df, pd.concat(looks, ignore_index=True) if looks else pd.DataFrame()) if keep_looks else df


def summarize(df, nb=2000, seed=0):
    """Pooled hit rate, random rate, lift, brightest-pixel rate, with site-bootstrap 95% intervals."""
    g = df.groupby("site_id")[["looks", "hits", "exp_rand", "bright_looks", "bright_hits"]].sum()
    tot = g.sum()

    def stats(t):
        L = t["looks"]
        return {"hit_rate": t["hits"] / L if L else np.nan, "rand_rate": t["exp_rand"] / L if L else np.nan,
                "lift": t["hits"] / t["exp_rand"] if t["exp_rand"] > 0 else np.nan,
                "bright_rate": t["bright_hits"] / t["bright_looks"] if t["bright_looks"] else np.nan,
                "lift_bright": (t["bright_hits"] / t["bright_looks"]) / (t["exp_rand"] / L) if L and t["exp_rand"] > 0 else np.nan,
                "lift_vs_bright": (t["hits"] / L) / (t["bright_hits"] / t["bright_looks"]) if L and t["bright_hits"] > 0 else np.nan}
    out = {"looks": int(tot["looks"]), "hits": int(tot["hits"]), "sites": int(len(g)), "sites_with_looks": int((g.looks > 0).sum()),
           "pairs": int(df.tag.nunique()), **stats(tot)}
    rng = np.random.default_rng(seed)
    G = g.values
    bs = []
    for _ in range(nb):
        t = pd.Series(G[rng.integers(0, len(G), len(G))].sum(0), index=g.columns)
        bs.append(stats(t))
    bs = pd.DataFrame(bs)
    for k in ("hit_rate", "rand_rate", "lift", "bright_rate", "lift_bright", "lift_vs_bright"):
        v = bs[k].replace([np.inf, -np.inf], np.nan).dropna()
        out[k + "_ci"] = [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] if len(v) else [np.nan, np.nan]
    out["hits_per10"] = 10 * out["hit_rate"]
    out["rand_per10"] = 10 * out["rand_rate"]
    return out


def topk(looks, pairs_df, ks=(1, 3, 10), foot=500, hit_m2=200, nb=2000, seed=0):
    """Ranking view: per scene, the top k clusters by score; hit rate against the scene's random rate."""
    L = looks[looks.foot == foot].copy()
    L["hit"] = L.change_m2 >= hit_m2
    pr = pairs_df[(pairs_df.foot == foot) & (pairs_df.hit_m2 == hit_m2)].set_index("tag").p_rand
    L["p_rand"] = L.tag.map(pr)
    res = {}
    for k in ks:
        s = L[L["rank"] <= k]
        g = s.groupby("site_id").agg(looks=("hit", "size"), hits=("hit", "sum"), exp_rand=("p_rand", "sum"))
        t = g.sum()
        r = {"k": k, "looks": int(t.looks), "scenes": int(s.tag.nunique()), "hit_rate": t.hits / t.looks,
             "rand_rate": t.exp_rand / t.looks, "lift": t.hits / t.exp_rand}
        rng = np.random.default_rng(seed)
        G = g.values
        b = []
        for _ in range(nb):
            u = G[rng.integers(0, len(G), len(G))].sum(0)
            b.append((u[1] / u[0], u[2] / u[0], u[1] / u[2] if u[2] > 0 else np.nan))
        b = np.array(b, float)
        for j, key in enumerate(("hit_rate", "rand_rate", "lift")):
            v = b[:, j][np.isfinite(b[:, j])]
            r[key + "_ci"] = [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
        res[k] = r
    return res
