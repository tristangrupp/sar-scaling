"""Round 3 prose (Stripmap anchor)."""
from __future__ import annotations

import pandas as pd


def render(c, s, lcm, table):
    G = "all pilot holdouts"

    def v(model, col, group=G):
        d = s[(s.group == group) & (s.model == model)]
        return float(d[col].iloc[0]) if len(d) else float("nan")

    def L(init, frac, col, group=G):
        d = lcm[(lcm.group == group) & (lcm.init == init) & (lcm.frac == frac)]
        return float(d[col].iloc[0]) if len(d) else float("nan")

    cols = ["ssim_lee", "psnr_lee", "mae_db_raw", "detail_r_raw", "hf_ratio_raw", "chamfer_p2t_m_raw", "chamfer_t2p_m_raw", "enl"]
    dig = {"ssim_lee": 1, "psnr_lee": 2, "chamfer_p2t_m_raw": 1, "chamfer_t2p_m_raw": 1, "enl": 0}
    keep = ["bicubic", "cnn_a", "esrt_a", "zeroshot_x2_unc", "zeroshot_joint_r2_target_NISAR", "zeroshot_joint_r2_target_X_spotlight",
            "sm_scratch_f1.0_s0", "sm_x2_unc_f1.0_s0", "joint3_target_stripmap", "joint3_target_NISAR", "joint3_target_X_spotlight"]
    main_tabs = ""
    for grp, title in [(G, "All pilot holdouts (Houston, Owens near, W Australia far)"), ("paper test scenes", "Paper test scenes (Amieva et al.)"),
                       ("houston (X-band holdout)", "Houston"), ("owens (near)", "Owens (near)"), ("w_australia (far)", "W Australia (far)")]:
        d = s[(s.group == grp) & s.model.isin(keep)]
        main_tabs += f"<h4>{title}</h4>" + table(d, cols, dig)

    fr = sorted(lcm.frac.unique())
    rows = ""
    for grp in [G, "paper test scenes"]:
        for f in fr:
            rows += f"<tr><td>{grp}</td><td class='n'>{f:.0%}</td><td class='n'>{int(round(f * 200))}</td>"
            for col, hi in [("ssim_lee", True), ("mae_db_raw", False), ("detail_r_raw", True), ("chamfer_t2p_m_raw", False)]:
                a, b = L("scratch", f, col, grp), L("x2_unc", f, col, grp)
                better = b > a if hi else b < a
                fmt = ".2f" if col in ("ssim_lee", "chamfer_t2p_m_raw") else ".3f"
                rows += f"<td class='n'>{a:{fmt}}</td><td class='n'>{'<b>' if better else ''}{b:{fmt}}{'</b>' if better else ''}</td>"
            rows += "</tr>"
    lc_tab = ("<div class='tw'><table><tr><th>Group</th><th>Fine-tune share</th><th>Tiles</th><th>SSIM, scratch</th><th>SSIM, X-pretrained</th>"
              "<th>MAE dB, scratch</th><th>MAE dB, X-pretrained</th><th>detail_r, scratch</th><th>detail_r, X-pretrained</th>"
              "<th>Edges ref→out m, scratch</th><th>Edges ref→out m, X-pretrained</th></tr>" + rows + "</table></div>")

    return f"""<section class="card"><h2>Round 3: the Stripmap anchor (same sensor, same band, 5 m) <span class="pill done">done</span></h2>
<p>Experiment 2 finished its IW to Stripmap pairs: 396 tiles, Sentinel-1 IW RTC at 10 m against Stripmap SLC
processed to 5 m. It fixed its geolocation bug (south-up output written north-up, plus a geoid shift), and per-scene
offsets are now under about 16 m. This is the truth round 1 lacked: same sensor, same band, a real 5 m target. We used
Experiment 2's splits exactly (200 training, 44 validation, 152 test tiles) and its reference, the Lee-filtered,
histogram-matched Stripmap. We scored with Experiment 2's own <code>score.py</code> functions (<code>tile_scores</code>,
homogeneous windows, its normalization), averaged per holdout group as Experiment 2 does, so every number below sits
directly beside its arm A. Our recomputed bicubic matches Experiment 2's table to the last digit.</p>
<p>Our models use the round 1 residual CNN at 2x with masked L1 on per-tile standardized dB for 3,000 steps at batch 16.
Experiment 2's arm A trained 8,000 steps with an SSIM loss (plus input consistency) in min-max space. The SSIM columns
therefore favor arm A by construction, and MAE and PSNR favor L1 training. The learning curve compares like with like:
scratch against X-band-pretrained, same recipe, same budget, on 5, 10, 25, 50 and 100 % of the training tiles (two
seeds at 5, 10 and 25 %).</p>

<h3>Zero-shot: X-band-only models help on real Stripmap truth</h3>
<p>Round 1 found that X-band-trained models lose to bicubic on real Sentinel-1 when scored against real X-band. Against
the same-band Stripmap reference the picture changes. Round 1's X-band 2x model, never shown a Stripmap or real IW
pixel, beats bicubic on every pilot-holdout average:</p>
<ul>
<li>SSIM {v('zeroshot_x2_unc','ssim_lee'):.1f} against {v('bicubic','ssim_lee'):.1f};</li>
<li>MAE {v('zeroshot_x2_unc','mae_db_raw'):.2f} dB against {v('bicubic','mae_db_raw'):.2f};</li>
<li><code>detail_r</code> {v('zeroshot_x2_unc','detail_r_raw'):.3f} against {v('bicubic','detail_r_raw'):.3f};</li>
<li>output-to-reference edge distance {v('zeroshot_x2_unc','chamfer_p2t_m_raw'):.1f} m against {v('bicubic','chamfer_p2t_m_raw'):.1f} m.</li></ul>
<p>That SSIM is what a Stripmap model trained from scratch on 10 tiles reaches
({L('scratch', 0.05, 'ssim_lee'):.1f} at 5 %). The cost is the same as in round 1. The zero-shot model misses more
reference edges ({v('zeroshot_x2_unc','chamfer_t2p_m_raw'):.1f} m against {v('bicubic','chamfer_t2p_m_raw'):.1f} m),
smooths hard (ENL {v('zeroshot_x2_unc','enl'):.0f}) and brightens slightly (mean ratio {v('zeroshot_x2_unc','moi'):.2f}).
Round 1's negative verdict was partly an artifact of scoring against another band: X-band truth punishes C-band output
for being C-band.</p>

<h3>Learning curve: does X-band pretraining help once Stripmap data is available?</h3>
<div id="r3curve" style="margin:8px 0"></div>
{lc_tab}
<p>Yes, but not where a shared-trunk argument predicts. At 5 and 10 % of the tiles (10 and 20), pretrained and scratch
models are indistinguishable. Seed-to-seed spread is about 1 SSIM point there, larger than any difference between
starts. From 25 % up, the pretrained model is ahead on every seed-averaged fidelity metric. At 100 % it scores SSIM
{v('sm_x2_unc_f1.0_s0','ssim_lee'):.2f} against {v('sm_scratch_f1.0_s0','ssim_lee'):.2f}, MAE
{v('sm_x2_unc_f1.0_s0','mae_db_raw'):.3f} against {v('sm_scratch_f1.0_s0','mae_db_raw'):.3f} dB, and <code>detail_r</code>
{v('sm_x2_unc_f1.0_s0','detail_r_raw'):.3f} against {v('sm_scratch_f1.0_s0','detail_r_raw'):.3f}. At equal steps the
gain grows with data rather than shrinking. The X-band start works as a better initialization for a model that has
enough real data to use it, not as a substitute for real data.</p>
<p>Pretraining also carries its X-band habit. Pretrained models smooth more: ENL {v('sm_x2_unc_f1.0_s0','enl'):.0f} against
{v('sm_scratch_f1.0_s0','enl'):.0f}, and they miss more reference edges
({v('sm_x2_unc_f1.0_s0','chamfer_t2p_m_raw'):.1f} m against {v('sm_scratch_f1.0_s0','chamfer_t2p_m_raw'):.1f} m). On the
paper test scenes the gains shrink to a few tenths of an SSIM point.</p>

<h3>Against Experiment 2's arm A</h3>
<p>Experiment 2's CNN and ESRT (8,000 steps, SSIM loss) stay ahead on SSIM: {v('cnn_a','ssim_lee'):.1f} and
{v('esrt_a','ssim_lee'):.1f} against {v('sm_x2_unc_f1.0_s0','ssim_lee'):.1f} for our best. They also lead on
<code>detail_r</code> ({v('esrt_a','detail_r_raw'):.3f} against {v('sm_x2_unc_f1.0_s0','detail_r_raw'):.3f}) and on
output-to-reference edges ({v('esrt_a','chamfer_p2t_m_raw'):.1f} against {v('sm_x2_unc_f1.0_s0','chamfer_p2t_m_raw'):.1f} m).
Our L1 models lead on MAE ({v('sm_x2_unc_f1.0_s0','mae_db_raw'):.2f} against {v('cnn_a','mae_db_raw'):.2f} dB) and PSNR
({v('sm_x2_unc_f1.0_s0','psnr_lee'):.2f} against {v('esrt_a','psnr_lee'):.2f}). With the loss and a 2.7x step budget both
different, this does not rank the architectures. It does show that X-band pretraining closes part of the gap: at
3,000 L1 steps, half of the SSIM distance from scratch to arm A.</p>

<h3>The joint X-band, L-band and C-band model</h3>
<p>Round 2's shared trunk, retrained with a third source: one batch each of synthetic X-band, NISAR and Stripmap per
step, for 3,000 steps. Stripmap exposure therefore equals the 100 % per-band runs. Stripmap batches carry the input
code "Sentinel-1 IW" and the new target code "Sentinel-1 Stripmap". On the holdouts we ran the same weights with each
target code:</p>
<ul>
<li><b>Stripmap code:</b> matches the per-band Stripmap model trained from scratch. SSIM
{v('joint3_target_stripmap','ssim_lee'):.2f} against {v('sm_scratch_f1.0_s0','ssim_lee'):.2f}, MAE
{v('joint3_target_stripmap','mae_db_raw'):.3f} against {v('sm_scratch_f1.0_s0','mae_db_raw'):.3f} dB, <code>detail_r</code>
{v('joint3_target_stripmap','detail_r_raw'):.3f} against {v('sm_scratch_f1.0_s0','detail_r_raw'):.3f}. It trails the
X-pretrained per-band model slightly. Sharing the trunk with two other bands costs nothing measurable on C-band.</li>
<li><b>NISAR code:</b> applies the L-band style to C-band input. The output is conservative: SSIM
{v('joint3_target_NISAR','ssim_lee'):.1f}, ENL {v('joint3_target_NISAR','enl'):.0f}, close to bicubic on edge recall. It
improves on bicubic but is clearly worse than the matching target code. In round 2 this code was the safest choice
for real Sentinel-1 scored against X-band. With a same-band truth, the matching code wins.</li>
<li><b>X spotlight code:</b> the best output-to-reference edges ({v('joint3_target_X_spotlight','chamfer_p2t_m_raw'):.1f} m)
and a slightly higher SSIM than the Stripmap code ({v('joint3_target_X_spotlight','ssim_lee'):.2f}). Its brightness is
less faithful: MAE {v('joint3_target_X_spotlight','mae_db_raw'):.2f} dB against
{v('joint3_target_stripmap','mae_db_raw'):.2f} dB, and PSNR {v('joint3_target_X_spotlight','psnr_lee'):.2f} against
{v('joint3_target_stripmap','psnr_lee'):.2f}.</li></ul>
<p>The target code works as a style control. One network holds three output styles. The matching Stripmap code gives the
most faithful brightness, the X code slightly crisper structure, and the NISAR code the most conservative output.</p>
{main_tabs}

<h3>Round 3 verdict</h3>
<p>With a same-band truth, the case for a multi-sensor radar model is stronger than rounds 1 and 2 suggested. X-band
knowledge transfers to C-band zero-shot, worth about 10 Stripmap training tiles. X-band pretraining improves a
Stripmap model at equal budget, more as the real data grows. A shared X+L+C trunk with sensor codes loses nothing on
C-band against a per-band model. What X-band does not supply is C-band's own texture: pretrained and zero-shot models
smooth more and miss more fine edges than Stripmap-trained ones. A multi-sensor model is worth building as a shared
trunk with per-sensor input and target codes, pretrained on everything and fine-tuned on each band's real pairs. It is
not worth building as a replacement for those pairs.</p>
<p style="color:var(--muted)">Caveats: one seed at 50 and 100 %; seed spread at small shares is about 1 SSIM point.
L1 training against arm A's SSIM training makes the cross-experiment SSIM comparison loss-dependent. Experiment 2's
tiles are coastal and island scenes plus three inland holdouts, so the land-cover mix is narrow.</p>
</section>"""
