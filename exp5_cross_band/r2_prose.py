"""Round 2 prose. render(c) uses c['r2_parts'] (tables and lookups) and the round 2 CSVs."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent


def render(c):
    P = c["r2_parts"]
    d = pd.read_csv(ROOT / "r2_nisar_scores.csv")
    med = d.groupby(["role", "model"])[["mae", "detail_r", "chamfer_m", "edgeF_10m"]].median()
    q = lambda role, m, col: float(med.loc[(role, m), col])
    seed = lambda role, init, f, col: (q(role, f"ft_{init}_f{f}_s0", col) + q(role, f"ft_{init}_f{f}_s1", col)) / 2
    x = pd.read_csv(ROOT / "r2_x_scores.csv")
    x = x[x.grid == "5m"].groupby(["input", "model"])[["detail_r", "chamfer_m", "edgeF_10m", "edgeF_20m_s10"]].median()
    X = lambda inp, m, col: float(x.loc[(inp, m), col])
    cr = pd.read_csv(ROOT / "r2_cross.csv").groupby("model").cross.median()
    jn, jx = "joint (target: NISAR style)", "joint (target: X spotlight)"
    h = f"""<section class="card"><h2>Round 2: pretraining plus fine-tuning, and a shared multi-band trunk <span class="pill done">done</span></h2>
<p>Round 1 showed that X-band training alone does not transfer to real Sentinel-1. Round 2 asks the question a
multi-sensor model depends on: once a model sees real data of a new band, does X-band pretraining still help, most of
all when that real data is scarce? Experiment 2's Stripmap tiles were still waiting on their geolocation fix when this
round ran, so the anchor is Experiment 3's NISAR L-band pairs: 20 MHz GCOV HH at 20 m to 40 MHz GCOV HH at 10 m, from
the same datatake. Experiment 3's three validation frames were excluded from training, and its near (2 frames, 102
tiles) and far (3 frames, 125 tiles) holdouts were used only for scoring.</p>

<h3>Part 1: learning curve, X-band pretrained against from scratch</h3>
<p>Every run uses the same budget: 1,500 steps at batch 16 with the same schedule, on 1, 3, 10, 25 or 100 % of the 600
NISAR training tiles (nested subsets, two seeds at 1, 3 and 10 %). The pretrained start is round 1's unconditioned
X-band 2x model, all weights. Every fine-tuned model, from either start, beats bicubic on MAE and <code>detail_r</code>.
On the far holdout <code>detail_r</code> rises from {q('far','bicubic','detail_r'):.3f} to about 0.32. All of them roughly
match bicubic on edge F, which is already high at 10 m for NISAR.</p>
<div id="r2curve" style="margin:8px 0"></div>
{P['lc_tab']}
<p>Results of the NISAR transfer test:</p>
<ol>
<li><b>Zero-shot X-band does not work on L-band at all.</b> The X-band model applied to NISAR without fine-tuning
scores MAE {q('far','zeroshot_x2_unc','mae'):.2f} (bicubic {q('far','bicubic','mae'):.2f}) and edge F
{q('far','zeroshot_x2_unc','edgeF_10m'):.2f}. NISAR GCOV is heavily multilooked and nearly speckle-free, far outside the
ENL-7 inputs the X-band model learned to clean, so it shifts the whole image by about 3 standard deviations. Fine-tuning
removes this within a few hundred steps.</li>
<li><b>After fine-tuning, pretraining helps a little and consistently on the far holdout.</b> There the pretrained model
leads on <code>detail_r</code> at every share. At 1 % (6 tiles) it scores {seed('far','x2_unc','0.01','detail_r'):.3f}
against {seed('far','scratch','0.01','detail_r'):.3f}; at 100 % {q('far','ft_x2_unc_f1.0_s0','detail_r'):.3f} against
{q('far','ft_scratch_f1.0_s0','detail_r'):.3f}. Its edge chamfer is also shorter at most shares
({q('far','ft_x2_unc_f1.0_s0','chamfer_m'):.2f} m against {q('far','ft_scratch_f1.0_s0','chamfer_m'):.2f} m at 100 %).
Pretrained at 10 % ({seed('far','x2_unc','0.1','detail_r'):.3f}) roughly matches scratch at 100 % with the same steps
({q('far','ft_scratch_f1.0_s0','detail_r'):.3f}). On the near holdout the gain appears only at 1 %
({seed('near','x2_unc','0.01','detail_r'):.3f} against {seed('near','scratch','0.01','detail_r'):.3f}) and is within
seed noise elsewhere.</li>
<li><b>The effects are small.</b> Gaps of 0.002 to 0.005 in <code>detail_r</code> are similar to the spread between two
seeds, so the pretraining benefit is real in direction but modest in size. With the budget cut to 300 steps, the
pretrained model keeps its <code>detail_r</code> lead on the far holdout
({q('far','ft_x2_unc_f0.03_b300','detail_r'):.3f} against {q('far','ft_scratch_f0.03_b300','detail_r'):.3f} at 3 %).
MAE and edge F come out even, because part of the short budget goes into undoing the X-band brightness offset.
NISAR's 2x task saturates on very little data: 18 tiles get within 0.01 of the full set. That leaves little room
for any prior to show.</li></ol>

<h3>Part 2: one shared trunk for X-band and L-band</h3>
<p>One network (the same trunk) trained jointly for 3,000 steps. Each step takes one batch of synthetic Sentinel-1-like
input from X-band, with round 1's band augmentation, and one NISAR batch. FiLM embeddings carry the input sensor and,
separately, the target sensor. The separate target code is the explicit point-spread-function handling: the X-band
spotlight look, with sharp points and sidelobes, belongs to the "X spotlight" target code. At inference the network
can be asked for a different target style. NISAR exposure equals the per-band NISAR model trained 3,000 steps on 100 %
of tiles.</p>
<h4>On NISAR's holdouts</h4>
{P['nis_tab']}
<p>The joint model gives up a little against the per-band NISAR model. On the far holdout it scores <code>detail_r</code>
{q('far','joint','detail_r'):.3f} against {q('far','ft_scratch_f1.0_3k','detail_r'):.3f}, and MAE
{q('far','joint','mae'):.4f} against {q('far','ft_scratch_f1.0_3k','mae'):.4f}. That is a small cost of sharing
capacity with a second band. No transfer gain shows up on L-band.</p>
<h4>On held-out X-band tiles (5 m)</h4>
{P['x_tab']}
<p>Asked for X-band output, the joint model trails round 1's X-band model on synthetic input (edge F
{X('synthetic', jx, 'edgeF_10m'):.3f} against {X('synthetic','x2_unc','edgeF_10m'):.3f}). It saw 3,000 X-band steps with
band augmentation, against 8,000 without. On real Sentinel-1 it does better: {X('real_s1', jx, 'edgeF_10m'):.3f} against
{X('real_s1','x2_unc','edgeF_10m'):.3f}. Switching only the target code to the NISAR style, with the same weights and the
same input, lifts real-Sentinel-1 edge F to {X('real_s1', jn, 'edgeF_10m'):.3f} (outline scale
{X('real_s1', jn, 'edgeF_20m_s10'):.3f}). That puts this model level with bicubic ({X('real_s1','bicubic','edgeF_10m'):.3f}
and {X('real_s1','bicubic','edgeF_20m_s10'):.3f}), with the best edge chamfer of any model
({X('real_s1', jn, 'chamfer_m'):.1f} m against {X('real_s1','bicubic','chamfer_m'):.1f} m). It is the first learned model in
this experiment to reach bicubic's edge placement on real Sentinel-1, and it does so while cleaning speckle. It does
not exceed bicubic: <code>detail_r</code> stays at {X('real_s1', jn, 'detail_r'):.3f}. Changing the target code changes the output:
the X-spotlight style sharpens and invents detail, the NISAR style stays conservative, and real C-band rewards
conservative.</p>
<h4>Sidelobe check: inconclusive</h4>
<p>We tried to measure the "+" streaks seen in round 1 with a cross index: at the brightest peaks of each 5 m output on
real Sentinel-1, standardized dB along the four axis-aligned arms minus along the diagonals, 10 to 30 m out. Bicubic
scores highest ({cr['bicubic']:.2f}; models {cr.drop('bicubic').min():.2f} to {cr.drop('bicubic').max():.2f}). The
index therefore picks up Sentinel-1's own range and azimuth sidelobes and the north-up street grids of ports, not
learned X-band artifacts, so it cannot rank the models. Telling learned sidelobes apart from real ones needs peaks
with known point-spread functions, such as corner reflectors.</p>
{P['cross_tab']}

<h3>Stripmap (C-band) arm</h3>
<p>Experiment 2's Stripmap tiles were still being fixed when round 2 ran, so the C-band arm moved to round 3
(above).</p>

<h3>Round 2 conclusion</h3>
<p>Round 2 found small gains from X-band pretraining after NISAR fine-tuning, and small losses from sharing one network across bands. Changing the target-sensor code improved the shared model’s Sentinel-1 edge scores against X-band references to about bicubic level. These results support testing a shared model, but do not show that X-band data improves the final C-band model. Rounds 3 and 4 address that question using Stripmap references.</p>
</section>"""
    return h
