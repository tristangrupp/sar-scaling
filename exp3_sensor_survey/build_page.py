"""Write exp3_sensor_survey/page_body.html (body fragment) from the survey text and the results on disk."""
from __future__ import annotations
import base64, html, io, json, os, sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parent
EV = ROOT / "eval"


def b64png(path, width="100%"):
    p = Path(path)
    if not p.exists():
        return ""
    return f'<img alt="{html.escape(p.stem)}" style="width:{width};height:auto;border:1px solid var(--line)" src="data:image/png;base64,{base64.b64encode(p.read_bytes()).decode()}">'


def small_png(path, max_w=1100):
    """downscale big PNGs before embedding"""
    from PIL import Image
    p = Path(path)
    if not p.exists():
        return ""
    im = Image.open(p).convert("RGB")
    if im.width > max_w:
        im = im.resize((max_w, int(im.height * max_w / im.width)), Image.LANCZOS)
    im = im.quantize(colors=64, method=Image.Quantize.MEDIANCUT)
    buf = io.BytesIO(); im.save(buf, "PNG", optimize=True)
    return f'<img alt="{html.escape(p.stem)}" style="width:100%;height:auto;border:1px solid var(--line)" src="data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}">'


def a(url, text):
    return f'<a href="{url}">{text}</a>'


SURVEY = [
    # sensor, band, openness, coverage/revisit, resolution, coarse+fine same place?, practicality, verdict
    ("NISAR L-band (NASA/ISRO)", "L, 24 cm",
     "Free, no EULA block. Calibrated PROVISIONAL products public since 20 Jul 2026, acquisitions from 17 Jun 2026; BETA pre-calibration set from Oct 2025 to Jan 2026.",
     "All land, every 12 days, ascending and descending. 2 to 4 tracks per site in our sample.",
     "GCOV 10 m (40 MHz), 20 m (20 MHz), 80 m (5 MHz band B); GSLC down to 5 m x 5 m.",
     "Yes, simultaneous. Every land acquisition carries a 5 MHz band B beside the 20 or 40 MHz band A. On 670 track-frames JPL also issues a 20 MHz mixed-mode product cut from the same 40 MHz echoes (4,674 pairs since June).",
     "HTTPS from ASF with the Earthdata token. HDF5 in 512 x 512 gzip chunks, so a windowed read takes about 2 s; no need to pull the 1.2 to 7 GB files.",
     ("done", "1: best for training")),
    ("Sentinel-1A/C/D", "C, 5.6 cm",
     "Free (Copernicus). 1C operational since Mar 2025, 1D data open since 17 Apr 2026, 1A retired Jun 2026.",
     "Global, 12 days (6 days over Europe).",
     "IW about 5 x 20 m (GRD posted at 10 m); Stripmap 5 m.",
     "Only through Stripmap, which is rare (small islands, emergencies), on different dates and orbits from IW. Free Stripmap GRD is 23 m, so only the SLC (4.4 GB) holds the detail.",
     "Easy for IW (Planetary Computer RTC). Stripmap needs SLC download and sarsen processing (Experiment 2).",
     ("run", "2: still the C-band choice")),
    ("NISAR S-band (ISRO)", "S, 10 cm",
     "Released through ISRO Bhoonidhi to registered users; licence terms not stated. Operational products from cycle 25 (8 Jul 2026).",
     "India plus selected global science sites, not systematic global land.",
     "Not published by ISRO; same GCOV/GSLC product family.",
     "Possibly, but the coverage limits the pool.",
     "Separate portal and account; API only for authorised users.",
     ("blocked", "3: limited coverage")),
    ("ALOS-2 PALSAR-2", "L, 24 cm",
     "ScanSAR L2.2 (25 m) and L1.1 free (AWS, Earth Engine, ASF); Stripmap 3 to 10 m is commercial.",
     "Global ScanSAR, released about 42 days after acquisition; partial archive.",
     "Open: 25 m. Fine modes not open.",
     "No open fine mode, so no same-sensor target.",
     "ScanSAR zips are 28 to 56 GB each.",
     ("neg", "4: no open fine mode")),
    ("RADARSAT Constellation Mission", "C",
     "Open only at 16 m and coarser on land (archive, EODMS public account); finer modes go to vetted users.",
     "Canada-focused (30 m compact-pol ARD every 12 days); limited elsewhere.",
     "Open: 16 to 30 m.",
     "No open fine mode; not global.",
     "EODMS ordering.",
     ("neg", "5")),
    ("BIOMASS (ESA)", "P, 70 cm",
     "Open since the Dec 2025 to Jan 2026 release.",
     "Forest belts only; no acquisitions over Europe and North America (radar-tracking restriction).",
     "Tens of metres (6 MHz bandwidth).",
     "No fine mode.",
     "ESA portal.",
     ("neg", "6")),
    ("ALOS-4 PALSAR-3", "L, 24 cm",
     "Commercial (PASCO since 10 Nov 2025, Tenchijin); research access through JAXA announcements only.",
     "Global by plan.", "3 m Stripmap and wide ScanSAR.", "Yes in principle, but not open.", "Purchase.",
     ("neg", "not open")),
    ("SAOCOM-1A/1B", "L, 24 cm",
     "Not open: ESA announcement-of-opportunity proposals (6 to 8 week review, capped product counts); ASI holds Europe.",
     "Global by request.", "Stripmap 10 m; TOPSAR 30 to 100 m.", "Yes in principle, but not open.", "Proposal, then order.",
     ("neg", "not open")),
]

SOURCES = [
    ("NASA Earthdata: NISAR L-band data now publicly available", "https://www.earthdata.nasa.gov/data/alerts-outages/nisar-l-band-data-now-publicly-available"),
    ("ASF: NISAR available data (PROVISIONAL from 17 Jun 2026; BETA set; data gap)", "https://nisar-docs.asf.alaska.edu/availability-overview/"),
    ("ASF: NISAR GCOV spec (10 / 20 / 80 m by bandwidth; HDF5 paths)", "https://nisar-docs.asf.alaska.edu/gcov"),
    ("ASF: NISAR GSLC spec (5 x 5 m at 40 MHz, 5 x 2.5 m at 77 MHz)", "https://nisar-docs.asf.alaska.edu/gslc"),
    ("ASF: NISAR observation plan (mode maps, polarization codes)", "https://nisar-docs.asf.alaska.edu/observation-plan"),
    ("NASA CMR: NISAR_L2_GCOV_PROVISIONAL_V1 (queried directly for modes and mixed-mode pairs)", "https://cmr.earthdata.nasa.gov/search/concepts/C2854338529-ASF.html"),
    ("ISRO: NISAR S-band data products release (Bhoonidhi)", "https://www.isro.gov.in/ISRO_EN/NISARS_Band_SAR_Data_Products_Release.html"),
    ("ASF: Sentinel-1 mission (1C/1D timeline)", "https://hyp3-docs.asf.alaska.edu/sentinel1/"),
    ("Copernicus Data Space: Sentinel-1D user data opening 17 Apr 2026", "https://dataspace.copernicus.eu/node/3210"),
    ("ESA: Sentinel-1 instrument payload (Stripmap 5 m)", "https://sentinel.esa.int/web/sentinel/missions/sentinel-1/instrument-payload"),
    ("AWS registry: PALSAR-2 ScanSAR CARD4L L2.2 (free, 42-day latency)", "https://registry.opendata.aws/jaxa-alos-palsar2-scansar/"),
    ("NASA catalog: ALOS-2 ScanSAR open collection (28 to 56 GB granules)", "https://catalog.data.gov/dataset/alos-2-palsar-2-level-1-scansar-product"),
    ("JAXA EORC: ALOS-4 data provision (data service providers)", "https://www.eorc.jaxa.jp/ALOS/en/alos-4/a4_data_e.htm"),
    ("PASCO: ALOS-4 PALSAR-3 distribution started", "https://satpf.jp/spf_atl/"),
    ("ESA: SAOCOM data access (AO, ASI zone)", "https://earth.esa.int/eogateway/missions/saocom/data"),
    ("CSA at CEOS LSI-VC-14: RCM open data at 16 m and coarser", "https://ceos.org/document_management/Virtual_Constellations/LSI/Meetings/LSI-VC-14/Presentations/2.6_CSA_Yves.pdf"),
    ("NRCan: RCM CEOS ARD (30 m compact pol, Canada)", "https://registry.opendata.aws/rcm-ceos-ard/"),
    ("ESA Biomass data opening", "https://www.hjkc.de/_blog/26758-raumfahrt-esa-s-biomass-goes-live-with-data-now-open-to-all/"),
]


def tbl(df, fmt=None, n_cols=()):
    fmt = fmt or {}
    h = "<div class='tw'><table><tr>" + "".join(f"<th>{html.escape(str(c))}</th>" for c in df.columns) + "</tr>"
    for _, r in df.iterrows():
        h += "<tr>"
        for c in df.columns:
            v = r[c]
            if isinstance(v, (float, np.floating)):
                s = "" if not np.isfinite(v) else (fmt.get(c, "{:.3f}").format(v))
                h += f"<td class='n'>{s}</td>"
            elif isinstance(v, (int, np.integer)):
                h += f"<td class='n'>{v}</td>"
            else:
                h += f"<td>{v}</td>"
        h += "</tr>"
    return h + "</table></div>"


def bar_svg(summary, metric, label, better):
    """grouped horizontal bars: bicubic vs CNN per arm/role"""
    rows = summary.reset_index()
    groups = [(r.arm, r.role) for r in rows[["arm", "role"]].drop_duplicates().itertuples()]
    rows = rows[rows.model != "truth"]
    vals = {(r.arm, r.role, r.model): r[metric] for _, r in rows.iterrows()}
    vmax = max(v for v in vals.values() if np.isfinite(v)) * 1.15
    W, rowh, left = 560, 22, 140
    H = 30 + len(groups) * (3 * rowh + 10)
    s = f'<svg viewBox="0 0 {W} {H}" style="width:100%;max-width:{W}px;height:auto" role="img" aria-label="{html.escape(label)}">'
    s += f'<text x="0" y="14" fill="var(--fg)" font-size="12">{html.escape(label)} ({better})</text>'
    y = 26
    for g in groups:
        s += f'<text x="0" y="{y + rowh + 2}" fill="var(--muted)" font-size="11">{g[0]} {g[1]}</text>'
        for j, (m, col) in enumerate([("bicubic", "var(--muted)"), ("cnn", "var(--accent)"), ("cnn_cons0", "var(--a2)")]):
            v = vals.get((g[0], g[1], m), np.nan)
            if not np.isfinite(v):
                continue
            w = (W - left - 60) * v / vmax
            yy = y + j * rowh
            s += f'<rect x="{left}" y="{yy}" width="{w:.1f}" height="{rowh - 6}" fill="{col}" rx="2"/>'
            s += f'<text x="{left + w + 4:.1f}" y="{yy + rowh - 10}" fill="var(--fg)" font-size="11">{v:.3f} {m}</text>'
        y += 3 * rowh + 10
    return s + "</svg>"


def sweep_svg(t):
    """horizontal bars: grid ratio per sweep setting (validation), with the bicubic reference line"""
    W, rowh, left = 560, 24, 150
    H = 40 + len(t) * rowh
    vmax = max(t.grid.max(), 1.4) * 1.15
    sx = lambda v: left + (W - left - 70) * v / vmax
    s = f'<svg viewBox="0 0 {W} {H}" style="width:100%;max-width:{W}px;height:auto" role="img" aria-label="grid ratio per setting">'
    s += '<text x="0" y="14" fill="var(--fg)" font-size="12">Grid ratio on validation (1 = no input-grid imprint; lower is better)</text>'
    y = 26
    for r in t.itertuples():
        lab = f"w {r.weight:g} at {int(r.cons_at_m)} m"
        col = "var(--a2)" if r.Index == t.index[0] else "var(--accent)"
        s += f'<text x="0" y="{y + 14}" fill="var(--muted)" font-size="11">{lab}</text>'
        s += f'<rect x="{left}" y="{y + 3}" width="{sx(r.grid) - left:.1f}" height="{rowh - 8}" fill="{col}" rx="2"/>'
        s += f'<text x="{sx(r.grid) + 4:.1f}" y="{y + 14}" fill="var(--fg)" font-size="11">{r.grid:.2f} ({r.wins_vs_bicubic}/5 wins)</text>'
        y += rowh
    xb = sx(1.38)
    s += f'<line x1="{xb:.1f}" y1="22" x2="{xb:.1f}" y2="{y}" stroke="var(--muted)" stroke-dasharray="3 3"/>'
    s += f'<text x="{xb + 3:.1f}" y="{y + 10}" fill="var(--muted)" font-size="10">bicubic 1.38</text>'
    return s + "</svg>"


def round4():
    """Round 4 section (adversarial term), placed above round 3."""
    E2 = ROOT / "eval2"
    tj = ROOT / "results_text4.json"
    if not tj.exists():
        return []
    T = json.loads(tj.read_text(encoding="utf-8"))
    rd = lambda f: pd.read_csv(E2 / f) if (E2 / f).exists() else None
    fmt = {"n": "{:.0f}", "ch_p2t": "{:.1f}", "ch_t2p": "{:.1f}", "grid": "{:.2f}"}
    figs = lambda pat: "".join(f"<figure>{small_png(q, 1000)}<figcaption>{html.escape(q.stem)}</figcaption></figure>"
                               for q in sorted((E2 / "panels_r4").glob(pat)))
    out = ['<section class="card"><h2>Round 4 (latest): an adversarial term for sharpness</h2>' + T["lede"] + "</section>"]
    sec = '<section class="card">' + T["choice"]
    d = rd("r4_adv_table.csv")
    if d is not None:
        keep = ["run", "eligible", "sharpness_gains", "d_mae", "d_ch_p2t", "d_grid", "d_hf", "d_ch_t2p", "mae", "ssim", "detail_r", "hf_ratio", "ch_p2t", "ch_t2p", "grid"]
        sec += tbl(d[[c for c in keep if c in d]], fmt | {"sharpness_gains": "{:.0f}", "d_mae": "{:+.3f}", "d_ch_p2t": "{:+.2f}",
                                                          "d_grid": "{:+.3f}", "d_hf": "{:+.3f}", "d_ch_t2p": "{:+.2f}"})
        sec += '<p style="color:var(--muted)">Validation, against the multi-date target; d_* = change from the control (same fine-tune, adversarial weight 0).</p>'
    out.append(sec + "</section>")
    sec = '<section class="card">' + T["hold"]
    for f, cap in [("r4_hold_mt10_summary.csv", "Holdouts, against the multi-date target"), ("r4_hold_a10_summary.csv", "Holdouts, against the single-date target")]:
        d = rd(f)
        if d is not None:
            sec += f"<p><b>{cap}</b></p>" + tbl(d, fmt)
    out.append(sec + figs("tile_*.png") + "</section>")
    sec = '<section class="card">' + T["xband"]
    x = rd("r4_xb_summary.csv")
    if x is not None:
        sec += tbl(x, {c: "{:.1f}" for c in x.columns if c.endswith(("p2x", "x2p"))} | {"n": "{:.0f}"})
    out.append(sec + figs("xband_*.png") + "</section>")
    out.append('<section class="card"><h2>Round 4 caveats</h2>' + T["caveats"] + "</section>")
    return out


def round3():
    """Round 3 section, placed at the top of the page (newest first)."""
    E2 = ROOT / "eval2"
    tj = ROOT / "results_text3.json"
    if not tj.exists():
        return []
    T = json.loads(tj.read_text(encoding="utf-8"))
    rd = lambda f: pd.read_csv(E2 / f) if (E2 / f).exists() else None
    fmt = {"n": "{:.0f}", "ch_p2t": "{:.1f}", "ch_t2p": "{:.1f}", "grid": "{:.2f}"}
    figs = lambda pat: "".join(f"<figure>{small_png(q, 1000)}<figcaption>{html.escape(q.stem)}</figcaption></figure>"
                               for q in sorted((E2 / "panels_r3").glob(pat)))
    out = ['<section class="card"><h2>Round 3</h2>' + T["lede"] + "</section>"]
    sec = '<section class="card">' + T["gslc"]
    cv = ROOT / "logs" / "gslc2_coverage_summary.txt"
    if cv.exists():
        sec += "<pre style='font-size:0.8em;white-space:pre-wrap'>" + html.escape(cv.read_text()) + "</pre>"
    for f, cap in [("r3_g2_old_on_v2_summary.csv", "Round 2 model scored against the deeper target"),
                   ("r3_g2v2_summary.csv", "Retrained on the deeper target (r3_g2v2) vs the round 2 model (g_g2)")]:
        d = rd(f)
        if d is not None:
            sec += f"<p><b>{cap}</b></p>" + tbl(d, fmt)
    out.append(sec + figs("g2v2_*.png") + "</section>")
    sec = '<section class="card">' + T["loss"]
    for f, cap in [("r3_loss_table.csv", "Validation, against the multi-date target: wins over plain L1"),
                   ("r3_seeds_mt10_summary.csv", "Two seeds each, against the multi-date target"),
                   ("r3_seeds_a10_summary.csv", "Two seeds each, against the single-date target")]:
        d = rd(f)
        if d is not None:
            sec += f"<p><b>{cap}</b></p>" + tbl(d, fmt | {"wins_vs_L1": "{:.0f}"})
    out.append(sec + figs("loss_*.png") + "</section>")
    sec = '<section class="card">' + T["long"]
    for f, cap in [("r3_long_a10_summary.csv", "Against the single-date target"), ("r3_long_mt10_summary.csv", "Against the multi-date target")]:
        d = rd(f)
        if d is not None:
            sec += f"<p><b>{cap}</b> (r3_long_*: 8,000 steps; s_w05_40 and m_multi: the 3,000-step round 2 models)</p>" + tbl(d, fmt)
    out.append(sec + "</section>")
    out.append('<section class="card"><h2>Round 3 caveats</h2>' + T["caveats"] + "</section>")
    return out


def round2():
    E2 = ROOT / "eval2"
    tj = ROOT / "results_text2.json"
    if not tj.exists():
        return []
    T = json.loads(tj.read_text(encoding="utf-8"))
    rd = lambda f: pd.read_csv(E2 / f) if (E2 / f).exists() else None
    fmt = {"n": "{:.0f}", "ch_p2t": "{:.1f}", "ch_t2p": "{:.1f}", "grid": "{:.2f}"}
    out = ['<section class="card"><h2>Round 2</h2>' + T["lede"] + "</section>"]
    t = rd("sweep_table.csv")
    h = rd("sweep_hold_summary.csv")
    sec = '<section class="card">' + T["sweep"]
    if t is not None:
        sec += sweep_svg(t) + tbl(t.drop(columns=["run"]), {"weight": "{:g}", "cons_at_m": "{:.0f}", "wins_vs_bicubic": "{:.0f}", **fmt})
        sec += '<p style="color:var(--muted)">Validation medians, 120 tiles. Orange bar: the chosen setting. cons_at_m: grid the consistency loss is measured on.</p>'
    sec += T["sweep_hold"]
    if h is not None:
        sec += tbl(h, fmt) + '<p style="color:var(--muted)">Holdout medians. s_w05_40 is the chosen setting; s_w0 and s_w05 bracket it (no consistency, and the strict round 1 setting, both retrained the round 2 way).</p>'
    out.append(sec + "</section>")
    sec = '<section class="card">' + T["multi"]
    for f, cap in [("multi_vs_mt10_summary.csv", "Scored against the multi-date target"), ("multi_vs_a10_summary.csv", "Scored against the single-date target")]:
        d = rd(f)
        if d is not None:
            sec += f"<p><b>{cap}</b> (m_multi: trained on the multi-date target; s_w05_40: trained on single dates)</p>" + tbl(d, fmt)
    figs = "".join(f"<figure>{small_png(q, 1000)}<figcaption>{html.escape(q.stem)}</figcaption></figure>" for q in sorted((E2 / "panels").glob("multi_*.png")))
    out.append(sec + figs + "</section>")
    sec = '<section class="card">' + T["g2"]
    d = rd("g2_summary.csv")
    if d is not None:
        sec += tbl(d, fmt) + '<p style="color:var(--muted)">Chamfer here is in metres on the 5 m grid; the grid ratio refers to the 10 m input grid.</p>'
    figs = "".join(f"<figure>{small_png(q, 1000)}<figcaption>{html.escape(q.stem)}</figcaption></figure>" for q in sorted((E2 / "panels").glob("g2_*.png")))
    out.append(sec + figs + "</section>")
    sec = '<section class="card">' + T["armb"]
    d = rd("armb_summary.csv")
    if d is not None:
        sec += "<p><b>Near / far holdouts and validation</b> (b_ref: no X-band windows; b_a: arm A′; b_e005 and b_e02: arm B)</p>" + tbl(d, fmt)
    x = rd("xb_summary.csv")
    if x is not None:
        sec += ("<p><b>Chamfer against X-band edges on infrastructure</b>, metres, per provider. holdout_near / holdout_far: the round 1 "
                "held-out X-band collects; xtest: test sites inside training frames, never trained on; xval: the tuning split.</p>"
                + tbl(x, {c: "{:.1f}" for c in x.columns if c.endswith(("p2x", "x2p"))} | {"n": "{:.0f}"}))
    figs = "".join(f"<figure>{small_png(q, 1000)}<figcaption>{html.escape(q.stem)}: orange = X-band edges on infrastructure</figcaption></figure>" for q in sorted((E2 / "panels").glob("armb_*.png")))
    out.append(sec + figs + "</section>")
    out.append('<section class="card"><h2>Round 2 caveats</h2>' + T["caveats"] + "</section>")
    return out


def main():
    P = []
    P.append("""<header><h1>Experiment 3: a better open sensor for SAR super-resolution</h1>
<p class="lede">NISAR's L-band radar beats Sentinel-1 as a source of <em>training pairs</em>. Every NISAR land
acquisition records a 5 MHz band beside its 20 or 40 MHz band. On 670 track-frames, JPL also issues a 20 MHz
product cut from the same 40 MHz echoes. That gives coarse and fine images of the same ground, at the same
instant, from the same orbit, with no co-registration or change problem. We ran Experiment 2's design on those
pairs, with the coarse mode as input, the fine mode as target, input consistency, and near and far held-out
regions, plus an X-band geometry check.</p></header>""")

    P.extend(round4())
    P.extend(round3())
    # ---- survey
    rows = "".join(
        f"<tr><td><b>{html.escape(s[0])}</b><br><span style='color:var(--muted)'>{s[1]}</span></td><td>{s[2]}</td><td>{s[3]}</td><td>{s[4]}</td><td>{s[5]}</td><td>{s[6]}</td>"
        f"<td><span class='pill {s[7][0]}'>{s[7][1]}</span></td></tr>" for s in SURVEY)
    P.append(f"""<section class="card"><h2>Phase 1: sensor survey</h2>
<p>Same-sensor coarse-to-fine training needs four things: open data, global coverage, a coarse mode and a fine
mode over the same ground, and access that works from this machine. The table rates each candidate on those
terms. We checked NISAR's claims against NASA's catalog (CMR) directly: which bandwidth modes fly where, which
products exist, file sizes, and whether ranged reads work.</p>
<div class="tw"><table><tr><th>Sensor</th><th>Openness and licence</th><th>Coverage and revisit</th><th>Resolution</th>
<th>Coarse and fine over the same place?</th><th>Access from here</th><th>Rank</th></tr>{rows}</table></div>
<h2>What the catalog shows for NISAR</h2>
<p>We sampled 24 cities and farm regions for the four months since 17 June 2026. The 40+5 MHz mode (10 m GCOV)
covers most of the Americas, Africa, South and East Asia and Australia. Lagos, São Paulo, Sydney, Jakarta,
Nairobi, Houston and Denver all get it. Europe and Chicago get the 20+5 MHz mode (20 m GCOV); Chicago's is
quad-pol. Each site had 17 to 53 GCOV products in four months.</p>
<p>The decisive find is in the product metadata. A GCOV whose name carries <code>_M_</code> has
<code>isMixedMode=True</code>, the same <code>plannedDatatakeId</code> as its 40 MHz twin,
<code>acquiredRangeBandwidth</code> = 40 MHz and <code>processedRangeBandwidth</code> = 20 MHz, centred on the
lower half of the band (1229 vs 1239 MHz). JPL makes it so 20 MHz and 40 MHz tracks can be stacked. For
super-resolution, that makes it an official 20 MHz image of the same echoes as the 10 m product. Both products
also carry the 5 MHz band B (80 m). The catalog lists 6,297 mixed-mode products since 17 June; 4,674 have a
40 MHz twin, across 670 track-frames on every continent, and 109 of those frames contain open X-band collects.</p>
<h2>Recommendation</h2>
<p><b>For training super-resolution, NISAR L-band is the better open sensor.</b> It removes the confounds that sank
Experiment 1 and that Experiment 2 must still fight: different dates, different orbits, single-shift
co-registration, and the rarity of Stripmap. It also gives three real resolution levels (80, 20 and 10 m), so we can
test how the scale factor matters with nothing else changing.</p>
<p><b>Sentinel-1 is still the right sensor for C-band products.</b> L-band sees different scatterers: it passes
through leaves and responds to trunks, ground and buildings. A NISAR model does not sharpen Sentinel-1, and
moving between bands is what failed in Experiment 1. NISAR also already delivers 10 m GCOV over most land. So the
practical targets are (a) Europe's 20 m GCOV to 10 m, which this experiment tests, and (b) 10 m to the 5 m GSLC
grid, a natural next step. NISAR S-band, PALSAR-2, RCM and BIOMASS lack an open fine mode or global coverage.
ALOS-4 and SAOCOM are not open.</p>
<p style="color:var(--muted);font-size:0.9em">Sources: {"; ".join(a(u, html.escape(t)) for t, u in SOURCES)}.</p></section>""")

    # ---- design
    P.append("""<section class="card"><h2>Phase 2: Experiment 2's design, adapted to NISAR</h2>
<div class="grid">
<div><h2>Pairs</h2><p><b>Input:</b> the 20 MHz mixed-mode GCOV, HH gamma0, 20 m. <b>Target:</b> the 40 MHz GCOV of
the same datatake, HH gamma0, 10 m. Both are radiometrically terrain corrected by JPL on grids with the same
origin, so a target pixel sits exactly inside its input pixel. No change mask is needed because both come from
the same echoes. A second arm (x4) uses the 5 MHz band B (80 m) as input and the 20 MHz product (20 m) as
target.</p>
<p><b>Tiles</b> are 5.12 km squares, read by HDF5 window straight from ASF (512 x 512 target pixels). Each frame
contributes two dates, the first and last in the PROVISIONAL archive, giving two seasons and independent
speckle.</p></div>
<div><h2>Splits</h2><p><b>Train:</b> 18 track-frames on six continents (Tokyo, Dubai, south Taiwan, Marrakesh,
Shandong, Alaska, plus the busiest frame in each of 12 other regions). <b>Near holdout:</b> Abu Dhabi (beside the
Dubai training frame) and north Taiwan, with tiles kept outside every training footprint. <b>Far holdout:</b>
Buenos Aires, Melbourne and Johannesburg, each over 1,000 km from any training frame and rich in X-band. Training
tiles also avoid every holdout footprint.</p>
<p><b>Model and losses:</b> Experiment 1's residual CNN (16 blocks, bicubic skip), one fixed dB normalisation so
brightness stays physical, masked L1 plus input consistency (output averaged back to 20 m in linear power must
match the input; weight 0.5). Flips only, no 90-degree rotations, because range and azimuth resolution
differ.</p></div></div>
<p><b>Scoring</b> follows Experiments 1 and 2: MAE and SSIM in dB, detail correlation, and fine-detail energy
ratio. We added two edge chamfer distances, since a 2.5 m shift drops a perfect copy to detail_r 0.37:
predicted edges to the nearest true edge (invented or misplaced edges), and true edges to the nearest
predicted edge (missed edges). The <b>geometry check</b> registers each Umbra, ICEYE or Capella collect to the
40 MHz truth, first on 40 m block means and then at 10 m with sub-pixel refinement. It then measures edge chamfer
only on ESA WorldCover built-up pixels, reported per provider. X-band never enters training.</p></section>""")

    # ---- QA
    qa = pd.read_csv(ROOT / "qa_pairs.csv") if (ROOT / "qa_pairs.csv").exists() else None
    if qa is not None:
        q = qa.describe().T.loc[["r_m_a", "bias_m_a", "l1_m_a", "shift_m_a", "r_b_m", "bias_b_m", "l1_b_m", "b_identical"], ["25%", "50%", "75%"]]
        q.index = ["corr: 20 MHz vs 40 MHz averaged to 20 m", "bias (dB): 20 MHz minus averaged 40 MHz",
                   "mean abs diff (dB): 20 MHz vs averaged 40 MHz", "phase-correlation shift (m): 20 vs 40 MHz",
                   "corr: 5 MHz band B vs 20 MHz averaged to 80 m", "bias (dB): band B minus averaged 20 MHz",
                   "mean abs diff (dB): band B vs averaged 20 MHz", "max diff (dB): band B in the 40 MHz vs mixed-mode file"]
        q = q.reset_index().rename(columns={"index": "check (per tile)", "25%": "p25", "50%": "median", "75%": "p75"})
        P.append(f"""<section class="card"><h2>Pair quality ({len(qa)} tiles)</h2>
<p>The 20 MHz and 40 MHz products line up exactly: phase correlation finds zero shift on every tile. The 20 MHz
image matches the 40 MHz image averaged to 20 m with a median correlation of {qa.r_m_a.median():.2f} and a
{qa.bias_m_a.median():+.2f} dB offset. The remaining {qa.l1_m_a.median():.2f} dB mean difference is the floor
for the input-consistency loss: even the true 10 m image does not average back exactly to the 20 MHz input,
because the two use different parts of the spectrum and so carry different speckle. The 5 MHz band B is noisier.
The 5 MHz band B differs by a few tenths of a dB between the two files, so JPL processes it separately in each.</p>
{tbl(q)}</section>""")

    # ---- results
    sm = EV / "summary.csv"
    if sm.exists():
        s = pd.read_csv(sm)
        win = pd.read_csv(EV / "winrate.csv") if (EV / "winrate.csv").exists() else None
        s2 = s.rename(columns={"ch_p2t": "chamfer pred→truth (m)", "ch_t2p": "chamfer truth→pred (m)", "cons": "input consistency (dB)"})
        res = json.loads((ROOT / "results_text.json").read_text()) if (ROOT / "results_text.json").exists() else {}
        charts = "".join(f"<div>{bar_svg(s.set_index(['arm', 'role', 'model']), m, lab, b)}</div>" for m, lab, b in
                         [("detail_r", "detail correlation", "higher is better"), ("ssim", "SSIM", "higher is better"),
                          ("mae", "MAE dB", "lower is better"), ("ch_p2t", "chamfer pred→truth, m", "lower is better")])
        P.append(f"""<section class="card"><h2>Results on held-out tiles</h2>
{res.get('results_lede', '')}
{tbl(s2, {"n": "{:.0f}", "chamfer pred→truth (m)": "{:.1f}", "chamfer truth→pred (m)": "{:.1f}"})}
<p style="color:var(--muted)">Medians over tiles. x2: 20 MHz (20 m) to 40 MHz (10 m). x4: 5 MHz band B (80 m) to 20 MHz (20 m). cnn: arm A (L1 plus input consistency, weight 0.5). cnn_cons0: the same network trained without input consistency (ablation, 5,000 steps). truth: the target scored against itself, a reference for the input-consistency floor and the grid ratio. grid: mean pixel step across input-pixel boundaries divided by the step inside them (1 = no input-grid imprint).</p>
<div class="grid">{charts}</div>
{('<h2>Share of tiles where the CNN beats bicubic</h2>' + tbl(win)) if win is not None else ''}
{res.get('results_more', '')}</section>""")
        pans = sorted((EV / "panels").glob("x2_*.png"))
        if pans:
            figs = "".join(f"<figure>{small_png(p, 900)}<figcaption>{html.escape(p.stem)}</figcaption></figure>" for p in pans[:6])
            P.append(f"""<section class="card"><h2>Side by side, held-out tiles (1.92 km crops)</h2>
<p>Each panel shows the 1.92 km crop with the most structure in its tile, in one shared gray stretch: the 20 MHz input, bicubic, the CNN, and the real 40 MHz image.</p>{figs}</section>""")
    xs = EV / "xband_summary.csv"
    if xs.exists():
        x = pd.read_csv(xs)
        res = json.loads((ROOT / "results_text.json").read_text()) if (ROOT / "results_text.json").exists() else {}
        xr = pd.read_csv(EV / "xband_scores.csv")
        st = xr.groupby(["provider", "status"]).size().unstack(fill_value=0).reset_index()
        P.append(f"""<section class="card"><h2>X-band geometry check (stage 1, built-up pixels only)</h2>
{res.get('xband_text', '')}
{tbl(x, {c: "{:.1f}" for c in x.columns if c.endswith(('p2x', 'x2p'))} | {"n": "{:.0f}"})}
<p style="color:var(--muted)">Median chamfer distance in metres. p2x: from the image's edges to the nearest X-band edge. x2p: from X-band edges to the nearest image edge. truth40 is the real 40 MHz image, the best a 10 m L-band image can do against X-band geometry. Collect outcomes:</p>
{tbl(st)}
{"".join(f"<figure>{small_png(f, 900)}<figcaption>{html.escape(f.stem)}: one example site per provider</figcaption></figure>" for f in sorted((EV / "xband_figs").glob("*.png")))}</section>""")
    tj = EV / "transfer" / "transfer.json"
    if tj.exists():
        res = json.loads((ROOT / "results_text.json").read_text()) if (ROOT / "results_text.json").exists() else {}
        meta = json.loads(tj.read_text())
        nb = [m for m in meta if m.get("kind") == "neighbor" and m.get("png")]
        cx = [m for m in meta if m.get("kind") == "context" and m.get("png")]
        fn = lambda ms: "".join(f"<figure>{small_png(EV / "transfer" / m["png"], 900)}<figcaption>{html.escape(m['name'])}"
                                f"{' (' + ('native 20 MHz' if m.get('native') else 'mixed-mode') + ')' if m.get('kind') == 'context' else ''}</figcaption></figure>" for m in ms)
        P.append(f"""<section class="card"><h2>Transfer: beyond the scored tiles</h2>
{res.get('transfer_text', '')}
<h2>9.6 km windows around held-out tiles</h2><p>The orange square marks the scored tile and the blue square marks the zoom, which lies outside it.</p>{fn(nb)}
<h2>Landscapes the training set barely covers, from native 20 MHz acquisitions</h2>{fn(cx)}</section>""")
    P.extend(round2())
    res = json.loads((ROOT / "results_text.json").read_text()) if (ROOT / "results_text.json").exists() else {}
    P.append(f"""<section class="card"><h2>Caveats</h2>{res.get('caveats', '<p>Pending results.</p>')}</section>""")
    P.append(f"""<section class="card"><h2>Status</h2>{res.get('status', '<p><span class="pill run">running</span> Tiles are being built.</p>')}</section>""")
    (ROOT / "page_body.html").write_text("\n".join(P), encoding="utf-8")
    print("page written", len("".join(P)) // 1024, "KB", flush=True)


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    os._exit(0)
