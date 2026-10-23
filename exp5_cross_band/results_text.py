"""Results prose for the Experiment 5 page (numbers come from the CSVs via ctx)."""
from __future__ import annotations

import json

from narrative import g

LEDE = ("X-band pretraining helped some short runs, but did not improve the final Sentinel-1 Stripmap model "
        "with the training settings and budget held constant. Round 4 compared ESRT and CNN models using "
        "SSIM loss, 200 real training tiles and 8,000 steps. None of 40 comparisons met the improvement rule. "
        "Earlier rounds below explain the direct-transfer result and why the evaluation reference matters.")


def results(c):
    T, agg = c["T"], c["agg"]
    G = lambda inp, grid, m, col: g(agg, inp, grid, m, col)
    figs, img = c["figs"], c["img64"]
    gj = json.loads((c["root"] / "input_gap.json").read_text())
    rat = gj["spectrum_ratio_real_over_syn"]
    mid_lo, mid_hi = min(rat[2:5]), max(rat[2:5])
    h = []

    h.append(f"""<section class="card"><h2>Round 1: results on synthetic X-band inputs</h2>
<p>This is a check on synthetic inputs. Degrade held-out X-band the same way as in training, and ask the models to undo
it. The X-trained 4x models raise <code>detail_r</code> from {G('synthetic','2.5m','bicubic4','detail_r'):.3f}
(bicubic) to {G('synthetic','2.5m','x4_cond','detail_r'):.3f}, and edge F within 10 m from
{G('synthetic','2.5m','bicubic4','edgeF_10m'):.3f} to {G('synthetic','2.5m','x4_cond','edgeF_10m'):.3f}. At the
outline scale (20 m tolerance) they reach {G('synthetic','2.5m','x4_cond','edgeF_20m_s10'):.3f}, against
{G('synthetic','2.5m','bicubic4','edgeF_20m_s10'):.3f} for bicubic. Output-to-reference edge distance also improves: the mean
distance from an output edge to the nearest true edge falls from {G('synthetic','2.5m','bicubic4','ch_out2true_m'):.1f} m
to {G('synthetic','2.5m','x4_cond','ch_out2true_m'):.1f} m. The 2x models do the same at 5 m (edge F
{G('synthetic','5m','x2_cond','edgeF_10m'):.3f} against {G('synthetic','5m','bicubic4','edgeF_10m'):.3f}). The C-band-only
control, trained on Sentinel-1 temporal means and never on X-band, also beats bicubic on these X-band inputs
(<code>detail_r</code> {G('synthetic','2.5m','c4','detail_r'):.3f}). When the degradation is the one the model was trained
on, geometry and speckle priors carry across bands in both directions.</p>
<h3>4x models, 2.5 m grid, synthetic input from held-out X-band</h3>{T['syn25']}
<h3>All models at 5 m (4x outputs averaged 2 x 2), synthetic input</h3>{T['syn5']}
<p style="color:var(--muted)">The ceiling row compares two different real X-band collects of the same place, so it is
not an upper bound for the synthetic case, where the truth is the very image that was degraded.</p></section>""")

    nreal = int(agg[(agg.input == 'real_s1') & (agg.model == 'bicubic4') & (agg.grid == '2.5m')].n.iloc[0])
    h.append(f"""<section class="card"><h2>Round 1: Sentinel-1 evaluated against X-band references</h2>
<p>Same held-out tiles, but the input is the real Sentinel-1 RTC VV tile paired with each X-band collect. Truth is
the real X-band at 2.5 m, fine-registered to the input (median residual shift {c['fine_shift_med']:.1f} m). Two reference
rows bound the scale. Another real X-band collect of the same place reaches edge F
{G('other_xband','2.5m','xband_other_collect','edgeF_10m'):.3f} (texture scale) and
{G('other_xband','2.5m','xband_other_collect','edgeF_20m_s10'):.3f} (outline scale). An unrelated tile's Sentinel-1, which
shares no geometry with the truth, still reaches {G('real_s1','2.5m','null_other_tile','edgeF_10m'):.3f} and
{G('real_s1','2.5m','null_other_tile','edgeF_20m_s10'):.3f}, because quantile-thresholded edges are dense and a 10 to 20 m
tolerance forgives a lot. Bicubic sits at {G('real_s1','2.5m','bicubic4','edgeF_10m'):.3f} and
{G('real_s1','2.5m','bicubic4','edgeF_20m_s10'):.3f}.</p>
<p>The X-trained model falls to {G('real_s1','2.5m','x4_cond','edgeF_10m'):.3f} at the texture scale, barely above
chance. It recovers to {G('real_s1','2.5m','x4_cond','edgeF_20m_s10'):.3f} at the outline scale, still under bicubic.
The Experiment 1 CNN, trained on real Sentinel-1 paired with real X-band, does no better
({G('real_s1','2.5m','exp1_cnn','edgeF_10m'):.3f} and {G('real_s1','2.5m','exp1_cnn','edgeF_20m_s10'):.3f}). The C-band-only
control sits at chance ({G('real_s1','2.5m','c4','edgeF_10m'):.3f}).</p>
<p>The two chamfer directions explain it. Output-to-truth distance measures whether a model's edges are real. Every
learned model improves on bicubic here: {G('real_s1','2.5m','x4_cond','ch_out2true_m'):.1f} m for the X-trained model,
{G('real_s1','2.5m','exp1_cnn','ch_out2true_m'):.1f} m for the Experiment 1 CNN, against
{G('real_s1','2.5m','bicubic4','ch_out2true_m'):.1f} m for bicubic, {G('real_s1','2.5m','null_other_tile','ch_out2true_m'):.1f} m
for chance, and {G('other_xband','2.5m','xband_other_collect','ch_out2true_m'):.1f} m for a second X-band collect.
Truth-to-output distance measures how much real structure a model leaves out. It gets far worse:
{G('real_s1','2.5m','x4_cond','ch_true2out_m'):.1f} m against {G('real_s1','2.5m','bicubic4','ch_true2out_m'):.1f} m. The
models keep only the strongest structures and smooth the rest away (<code>hf_ratio</code>
{G('real_s1','2.5m','x4_cond','hf_ratio'):.2f}, against {G('synthetic','2.5m','x4_cond','hf_ratio'):.2f} on synthetic input).
Where an X-trained model does draw an edge on real C-band, the edge is as well placed as the supervised model's.
Edge precision improves, while edge recall is lower in this comparison.</p>
<h3>4x models, 2.5 m grid, real Sentinel-1 input</h3>{T['real25']}
<h3>All models at 5 m, real Sentinel-1 input</h3>{T['real5']}
<p style="color:var(--muted)">Medians over {nreal} held-out tiles. Capella HH tiles with VV input are scored as
cross-polarization in the per-provider table instead. Bold marks the best row other than the references. Result 3
explains the "+ band adapt" rows.</p></section>""")

    h.append(f"""<section class="card"><h2>Round 1: input statistics and adaptation</h2>
<p>The chart puts each model's synthetic and real scores side by side on the same tiles. Bicubic loses little between
synthetic and real input (edge F {G('synthetic','2.5m','bicubic4','edgeF_10m'):.3f} to {G('real_s1','2.5m','bicubic4','edgeF_10m'):.3f}),
a drop of about 0.03 in this comparison. The X-trained model loses
{G('synthetic','2.5m','x4_cond','edgeF_10m') - G('real_s1','2.5m','x4_cond','edgeF_10m'):.3f}, and its
<code>detail_r</code> falls from {G('synthetic','2.5m','x4_cond','detail_r'):.3f} to {G('real_s1','2.5m','x4_cond','detail_r'):.3f}.
The larger drop suggests sensitivity to the difference between synthetic training inputs and real Sentinel-1.</p>
<div id="gapchart" style="margin:8px 0 4px"></div>
<p>We measured how the inputs differ, on 316 training tiles, as unpaired statistics: the real Sentinel-1 tile against
the synthetic input made from its X-band, both standardized. Real C-band at 10 m has {100 * (1 - mid_hi):.0f} to
{100 * (1 - mid_lo):.0f} % less power at 20 to 50 m scales than degraded X-band (spectrum ratio {mid_lo:.2f} to {mid_hi:.2f}).
It has more power at the pixel scale ({rat[-2]:.2f} to {rat[-1]:.2f}) and a heavier bright tail (skewness
{gj['skew_real']:.2f} against {gj['skew_syn']:.2f}). Speckle is not the difference, since its ENL and correlation
match. The difference is the scene. C-band sees fewer mid-sized bright scatterers and a smoother background, so isolated
strong returns stand out. X-band fills the same cell with texture. A model that learned "texture at 20 to 50 m is
structure" reads the C-band background as speckle and smooths it.</p>
<p>Two fixes use no real training pairs. <b>Band adaptation at test time</b> maps the real input's value distribution
onto the synthetic one and equalizes its radial spectrum. It is a fixed transform built from 8 spectral gains and a
quantile curve measured on training tiles. It lifts the X-trained model from {G('real_s1','2.5m','x4_cond','edgeF_10m'):.3f}
to {G('real_s1','2.5m','x4_cond+adapt','edgeF_10m'):.3f} (texture scale) and from {G('real_s1','2.5m','x4_cond','edgeF_20m_s10'):.3f}
to {G('real_s1','2.5m','x4_cond+adapt','edgeF_20m_s10'):.3f} (outline scale). The same transform does nothing for bicubic
({G('real_s1','2.5m','bicubic4+adapt','edgeF_10m'):.3f}), so the gain comes from the model, not from the edge detector.
<b>Band augmentation during training</b> perturbs each synthetic input with a random spectral tilt and a random contrast
curve, in generic ranges not fitted to C-band. It costs a little on synthetic input
({G('synthetic','2.5m','x4_aug','edgeF_10m'):.3f} against {G('synthetic','2.5m','x4_cond','edgeF_10m'):.3f}). On real input it
reaches {G('real_s1','2.5m','x4_aug','edgeF_10m'):.3f} without adaptation, and {G('real_s1','2.5m','x4_aug+adapt','edgeF_10m'):.3f}
and {G('real_s1','2.5m','x4_aug+adapt','edgeF_20m_s10'):.3f} with it. That is the best learned result on real
Sentinel-1, still short of bicubic's {G('real_s1','2.5m','bicubic4','edgeF_10m'):.3f} and
{G('real_s1','2.5m','bicubic4','edgeF_20m_s10'):.3f}. It beats bicubic at the outline scale on {c['win_aug']:.0%} of tiles;
the Experiment 1 CNN does so on {c['win_exp1']:.0%}.</p>
<p><b>Metadata conditioning did not matter.</b> Conditioned and unconditioned models score within 0.01 of each other
everywhere, on synthetic and real inputs alike. Rotation augmentation already spreads look directions evenly, and a
20 m resolution cell blurs most layover and shadow detail. Conditioning may matter once training mixes sensors whose
brightness differs systematically. A model trained on one band never sees the band flag change, so the flag cannot do
anything yet.</p></section>""")

    h.append(f"""<section class="card"><h2>Per provider and polarization</h2>
<p>Each cell shows edge F within 10 m (texture scale) / <code>detail_r</code>, median per group, 4x models at 2.5 m.
On real input, ICEYE tiles come closest; at the outline scale the band-augmented and adapted model ties bicubic there
(0.644 each). Capella HH truth is hardest. On synthetic HH input the X-trained models fall below bicubic, because HH
made up only 40 of 650 training tiles and several HH test cells are polar ice, which the training set never covered.</p>
{T['prov']}</section>""")

    h.append(f"""<section class="card"><h2>Despeckling on real Sentinel-1 (Experiment 4's held-out sites)</h2>
<p>Six held-out sites (cropland, forest, urban, mountain, desert, wetland), three dates each, scored against each
site's temporal mean with Experiment 4's ENL definition. Each output was mapped back to dB with a gain and offset
fitted against its own input at 10 m. The naive de-standardization (first column) inflates contrast and brightens the
mean by up to 70 %. After the fit, every model's mean ratio is 0.91 to 0.93. That matches the known bias of averaging
in dB at ENL 7 (0.929), so brightness is preserved once that bias is corrected. Ratio images keep an ENL close to the
input's (7 to 9), so the models remove speckle rather than signal. The X-trained models reach ENL 190 to 270 at 10 m
and bring the image closer to the temporal mean than bicubic does (MAE {c['ds_x4']:.2f} dB against {c['ds_bic']:.2f} dB).
The Experiment 1 CNN gets closest ({c['ds_exp1']:.2f} dB), and the C-band-only control smooths least (ENL about 62).
Bicubic removes no speckle, so its ratio-image figures carry no meaning.</p>
{T['desp']}</section>""")

    tf = figs["panels_transfer"]
    ctx_f = [p for p in tf if p.stem.startswith("context")]
    nb_f = [p for p in tf if p.stem.startswith("neighbor")]
    h.append(f"""<section class="card"><h2>Transfer panels: 9.6 km windows with no truth</h2>
<p>The wider comparisons show smoother fields, roads, shorelines and urban blocks. Output averaged back to 10 m correlates with IW at 0.85–0.90, versus 0.999 for bicubic. Bright targets sometimes develop cross-shaped streaks; round 2’s attempted sidelobe metric could not establish their source. The models retain only one-third to one-half of bicubic’s fine-detail energy. Adaptation adds outlines, but the unscored panels cannot establish whether those outlines are correct.</p>
{T['transfer']}
<h3>Around held-out X-band sites</h3>
{''.join(f'<figure><img src="{img(p)}" alt="{p.stem}"><figcaption>{p.stem.replace("neighbor_", "").replace("_", " ")}: top, the full 9.6 km window (orange box = the X-band tile); bottom, a zoom on the blue box outside it.</figcaption></figure>' for p in nb_f)}
<h3>Unseen landscapes</h3>
{''.join(f'<figure><img src="{img(p)}" alt="{p.stem}"><figcaption>{p.stem.replace("context_", "").replace("_", " ")}</figcaption></figure>' for p in ctx_f)}
</section>""")

    sel = [p for p in figs["panels_x"] if any(k in p.stem for k in c["panel_ids"])]
    h.append(f"""<section class="card"><h2>Held-out tile panels</h2>
<p>Top row of each panel: the synthetic input made from the held-out X-band. Bottom row: the real Sentinel-1 tile.
The same model sharpens building outlines and point targets from the synthetic input and returns a smooth picture from
the real one. 640 m crops, all on the truth's grey scale.</p>
{''.join(f'<figure><img src="{img(p)}" alt="{p.stem}"><figcaption>{p.stem[:60]}</figcaption></figure>' for p in sel)}</section>""")

    h.append("""<section class="card"><h2>Stripmap check (round 1): superseded by round 3</h2>
<p>In round 1 the only Stripmap tile was Experiment 2's Houston processing test, and it sat about 750 m from where
Sentinel-1 IW put the same ground. Experiment 2 traced that to its own processing: sarsen wrote south-up output as
north-up without a flip, plus an EGM2008 geoid shift. It has since fixed both (per-scene offsets now under about 16 m)
and built 396 paired tiles. Round 3, above, scores every model on those tiles.</p></section>""")

    h.append("""<section class="card"><h2>Caveats</h2><ul>
<li>X-band truth differs from Sentinel-1 in date (days to weeks), look direction and incidence, as well as band. The
shift-tolerant edge metrics and the bicubic and chance rows bound how much that matters, but a perfect C-band
super-resolver would still not score 1.0 here.</li>
<li>The C-band-only control is weak. Experiment 4's temporal means are at 10 m, so it learned at a physical scale 4x
coarser. Its failure on real Sentinel-1 says less about C-band training than a Stripmap-trained control would.</li>
<li>The synthetic degradation matches speckle ENL and correlation, but not the thermal noise floor or Sentinel-1's
own sidelobes. X-band tiles are uncalibrated (Umbra 8-bit, ICEYE 16-bit amplitude), so we could not place an absolute
noise floor.</li>
<li>We fixed the incidence angle for real Sentinel-1 at 37° (mid-swath) and took look azimuth from orbit direction alone.</li>
<li>Training ran 8,000 steps on a shared GPU, roughly half the spike's budget. Longer training would raise synthetic
scores, but is unlikely to close a real-data gap of 0.09 to 0.17 in edge F.</li>
<li>Band adaptation and augmentation use summary statistics of real Sentinel-1. Adaptation uses 8 spectral gains and a
quantile curve from training tiles; augmentation uses generic ranges only. Neither uses held-out data or real image pairs.</li>
<li>We trained no GAN variant. The spike's discriminator overpowered its generator, and the L1 models already answer
the transfer question. A GAN would raise <code>hf_ratio</code>, not edge placement.</li></ul></section>""")

    h.append("""<section class="card"><h2>Round 1 conclusion, followed up in rounds 2–4</h2>
<p><span class="pill neg">round 1: no gain against X-band references</span> <span class="pill done">some transfer in synthetic tests</span></p>
<p>In round 1, models applied to real Sentinel-1 did not beat bicubic edge scores when evaluated against X-band references. Their output edges were often close to reference edges, but many reference edges were missing. Input-statistics adaptation reduced part of that gap. This conclusion is specific to the cross-band evaluation: round 3 later found a zero-shot benefit against same-band Stripmap references, and round 4 tested the final model with matched training settings.</p>
<p>Round 1 suggested three requirements for a shared model. Later rounds test these ideas; the final result is reported in round 4 above.</p>
<ol><li>Anchor each band in real data of that band: same-sensor coarse and fine pairs (Stripmap for Sentinel-1,
NISAR's dual-bandwidth modes for L-band from Experiment 3), or self-supervised objectives on real single-date images.
Synthetic degradation of another band's images is not enough.</li>
<li>Share the trunk, and make brightness statistics an explicit per-band input: band and sensor embeddings that vary
during training, plus band augmentation. A flag that never varies does nothing.</li>
<li>Remove sensor-specific point-spread functions from the targets, or condition on them, so X-band sidelobes do not
become a "radar prior".</li></ol>
<p>The proposed Stripmap follow-up is complete. Round 3 found a benefit in short training runs; round 4 found no improvement beyond seed variation with the full training method and budget.</p></section>""")

    h.append("""<section class="card"><h2>Files</h2><ul>
<li><code>exp5_cross_band/common.py</code>: degradation, conditioned generator, metrics, and the single path constant for Experiment 4 data</li>
<li><code>exp5_cross_band/e5_calib.py</code>, <code>calib.csv</code>: speckle calibration against real Sentinel-1</li>
<li><code>exp5_cross_band/e5_train.py</code>, <code>ckpt/*.pt</code>: x4_cond, x4_unc, x4_aug, x2_cond, x2_unc, c4, c2</li>
<li><code>exp5_cross_band/e5_eval_x.py</code>, <code>eval_x_scores.csv</code>: synthetic, real and ceiling scores per tile</li>
<li><code>exp5_cross_band/e5_input_gap.py</code>, <code>input_gap.json</code>: real against synthetic input statistics and the adaptation curve</li>
<li><code>exp5_cross_band/e5_despeckle.py</code>, <code>despeckle_scores.csv</code>; <code>exp5_cross_band/e5_transfer.py</code>, <code>transfer_stats.csv</code></li>
<li><code>exp5_cross_band/e5_eval_sm.py</code>, <code>eval_sm_scores.csv</code>: the Stripmap evaluation, ready for Experiment 2's tiles</li></ul></section>""")
    return "\n".join(h)
