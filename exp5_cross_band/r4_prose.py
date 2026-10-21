"""Round 4 prose: the matched-recipe headline test."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent


def render(c, vs, dec, T, dec_tab, paper_tab):
    def v(arch, init, frac, col, grp="all pilot holdouts"):
        d = vs[(vs.group == grp) & (vs.arch == arch) & (vs.init == init) & (vs.frac == frac)]
        return float(d[col].iloc[0]) if len(d) else float("nan")

    pan = sorted((ROOT / "panels_r4").glob("neighbors_*.png")) if (ROOT / "panels_r4").exists() else []
    ctxp = sorted((ROOT / "panels_r4").glob("context_*.png")) if (ROOT / "panels_r4").exists() else []
    img = c["img64"]
    figs = "".join(f'<figure><img src="{img(p)}" alt="{p.stem}"><figcaption>{p.stem.replace("_", " ")}: IW input, bicubic, '
                   f'exp 2 arm A (ESRT, CNN) and the best X-band-pretrained CNN; bottom row zooms into the truth tile next to Stripmap.</figcaption></figure>'
                   for p in pan)
    figs_c = "".join(f'<figure><img src="{img(p)}" alt="{p.stem}"><figcaption>{p.stem.replace("_", " ")} (no Stripmap truth)</figcaption></figure>' for p in ctxp)
    n_beats = int(dec.beats.sum())
    return f"""<section class="card"><h2>Round 4: does pretraining beat the published baseline at a matched recipe?
<span class="pill neg">no, within seed noise</span> <span class="pill run">25 % ESRT pretrained rows pending</span></h2>
<p><b>Decision:</b> at Experiment 2's arm A recipe and budget, neither X-band nor X-band + NISAR pretraining beats arm A
beyond seed noise, in the near group (Owens), the far group (W Australia), Houston or the paper's test scenes, for
either architecture. {n_beats} of {len(dec)} group-by-variant tests pass the rule. Over all pilot holdouts at 100 % of
the data, the seed means are close. ESRT: scratch {v('esrt','scratch',1.0,'ssim_lee'):.2f}, X-pretrained
{v('esrt','x',1.0,'ssim_lee'):.2f}, X+L-pretrained {v('esrt','xl',1.0,'ssim_lee'):.2f} SSIM. CNN: scratch
{v('cnn','scratch',1.0,'ssim_lee'):.2f}, X {v('cnn','x',1.0,'ssim_lee'):.2f}, X+L {v('cnn','xl',1.0,'ssim_lee'):.2f}.
Seed SDs are about 0.1. The round 3 gain (+0.7 SSIM from X-band pretraining) came from models trained with an L1
loss for 3,000 steps. With arm A's SSIM loss and 8,000 steps, scratch training reaches the same place, and the head
start disappears.</p>

<h3>What was matched</h3>
<ul>
<li><b>Fine-tune:</b> Experiment 2's <code>train_sr.py</code> loop, imported rather than copied, so nothing drifts. Loss 0.85 x (1 - SSIM)
against the Lee-filtered, histogram-matched Stripmap, plus 0.15 x input consistency. Patches 48 to 96 px with
rotation and flip, min-max normalization from the full training split, OneCycleLR (ESRT 1e-3, CNN 5e-4), gradient clip
1, 8,000 steps at batch 16, best validation checkpoint every 1,000 steps. Exp 2's own <code>esrt_a</code> and <code>cnn_a</code> are
the scratch seed 0; we trained seeds 1 and 2. At 25 % of the 200 training tiles, every start has seeds 0 and 1.</li>
<li><b>Pretraining (new work, equal budget for every pretrained variant):</b> the same model code, loss, patches,
normalization and schedule, 8,000 steps at batch 16. X: 598 synthetic X-band tiles (round 1 Sentinel-1-like
degradation, 3 speckle draws each), with the target built like exp 2's (X-band at 5 m, Lee-filtered, histogram-matched
to the input). X+L: 8 X-band and 8 NISAR patches per batch (596 NISAR 20 to 40 MHz pairs, same target recipe), so both
variants see 128,000 patches. No Stripmap or real IW enters pretraining.</li>
<li><b>Scoring:</b> exp 2's <code>score.py</code> functions on its 152 test tiles, recomputed for every model including arm A and
bicubic (predictions on CPU while the GPU was reserved for other work). Paired tests are per tile against arm A of the same
architecture, using the variant's seed-averaged score.</li></ul>
<p><b>Decision rule</b> (fixed before scoring): a pretrained variant beats arm A in a group only if its seed-mean SSIM
(Lee reference, arm A's own objective) exceeds the scratch seed-mean by more than twice the larger seed SD, and its
paired Wilcoxon test against arm A gives p &lt; 0.05 with a positive mean.</p>
{dec_tab}
<p>The CNN rows show why both parts are needed. Pretrained CNNs beat the <code>cnn_a</code> checkpoint per tile in the near and far
groups (+0.27 to +0.43 SSIM, p &lt; 0.001). Fresh scratch seeds beat it as well: <code>cnn_a</code> is a slightly weak seed. Against
the scratch seed-mean the gain is +0.07 to +0.19, inside two seed SDs. For ESRT the pretrained models land slightly
below arm A in the far group (−0.23 SSIM per tile against <code>esrt_a</code>), again inside the scratch seed spread.</p>

<h3>Per group (seed mean ± seed SD)</h3>
<h4>Owens (near)</h4>{T['owens (near)']}
<h4>W Australia (far)</h4>{T['w_australia (far)']}
<h4>Houston</h4>{T['houston (X-band holdout)']}
<h4>All pilot holdouts</h4>{T['all pilot holdouts']}
<p style="color:var(--muted)">Pending: ESRT X-pretrained and X+L-pretrained at 25 % (2 seeds each). The GPU slot was
reassigned to other projects before they ran, and they are held until it reopens. The CNN 25 % rows are final. They
show the same picture: pretraining does not separate from scratch, and seed spread at 25 % is large (SD up to 1.1 SSIM in
W Australia).</p>

<h3>The paper's metric set on its three test scenes</h3>
{T['paper test scenes']}
{paper_tab}
<p>This is as close to Amieva et al. as our data allows, and it is not close. The paper reports SSIM 70.14 and PSNR
26.07 dB on whole scenes. We score 24 tiles of 3.2 km cut from the same three scenes, chosen for structure rather than
covering the scene. Water is masked. The reference is our own sarsen processing of the Stripmap SLC, raw or
Lee-filtered and histogram-matched, not their reference. SSIM is the masked mean of the SSIM map. Whole scenes dominated
by homogeneous fields score far higher on both metrics. Our absolute numbers are therefore not comparable with the
paper's. The ranking is: every model beats bicubic by about 5 SSIM points, and every learned variant sits within 0.1
SSIM of the others. Despeckling metrics land in the paper's range: ENL 118 to 143 against its 202, mean of intensity
ratio 1.02 to 1.06 against 1.022.</p>

<h3>Transfer panels (exp 2's neighborhoods)</h3>
<p>Same windows as Experiment 2's panels: 9.6 km of IW around one held-out tile per group, and the landscapes it uses for
context. The best pretrained model by all-pilot SSIM (CNN, X-pretrained, seed 2) is shown next to both arm A models.
They are hard to tell apart by eye, which matches the numbers.</p>
{figs}{figs_c}

<h3>Round 4 verdict</h3>
<p>Pretraining on X-band, or on X-band plus L-band, buys nothing measurable once the Stripmap model gets the published
recipe and its full budget. It does not beat arm A in the near or far group beyond seed noise, for ESRT or the CNN.
Earlier rounds showed X-band priors are real: zero-shot transfer works and pretraining speeds up short training. At
200 real pairs and 8,000 steps, though, the C-band data alone reaches the same optimum. A multi-sensor trunk remains
useful for convenience (one network, sensor codes, a good start when compute or data is scarce), not as a route to
better Stripmap-level output. A future round would need either far fewer real pairs or a harder target to show a
benefit.</p>
</section>"""
