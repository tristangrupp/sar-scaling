"""Write results_text.json: the prose blocks build_page.py drops into the results sections."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
T = {}
T["results_lede"] = """<p><b>Arm A beats bicubic on structure, consistently but modestly.</b> On far-holdout tiles
(Buenos Aires, Melbourne, Johannesburg; 125 tiles), detail correlation rises from 0.273 to 0.333 and SSIM from 0.487
to 0.505. The CNN wins on detail correlation and SSIM in every far tile. Missed-edge chamfer (truth edges to the
nearest predicted edge) drops from 12.1 m to 10.1 m, and the CNN wins on 93% of tiles. MAE does not move (1.76 dB),
and invented-edge chamfer stays the same (11.3 vs 11.5 m). Near-holdout tiles (Abu Dhabi desert and north Taiwan;
102 tiles) are flatter and harder: detail correlation goes from 0.141 to 0.153 and SSIM from 0.350 to 0.369.</p>
<p>The detail correlation is much higher than Experiment 1’s roughly 0.05, but the tasks differ. NISAR uses the same sensor and acquisition with a 2× grid factor; bicubic already reaches 0.27. The learned model reaches about 0.33.</p>"""
T["results_more"] = """<h2>What the extra metrics show</h2>
<p><b>The input-consistency loss is too strict for this data.</b> The real 10 m image, averaged back to 20 m, sits
1.0 dB from the 20 MHz input, because the two products use different parts of the spectrum and carry different
speckle. Arm A matches its input to 0.02 dB, far tighter than the truth does. The cost shows in the <b>grid</b>
ratio: arm A's pixel steps across input-pixel boundaries are 3.1 times its steps inside them on far tiles, and 6.0
times on near tiles. The truth scores 1.0 and bicubic 1.4. The model imprints the 20 m input grid.</p>
<p><b>The ablation without consistency</b> (same network, 5,000 steps) trades the other way. MAE falls to 1.68 dB
on far tiles and beats bicubic on every tile. The grid ratio falls to 2.0 and invented-edge chamfer to 8.5 m.
Detail correlation holds at 0.335. But SSIM drops to 0.462, below bicubic, and missed-edge chamfer rises to 13.4 m.
Neither weight is right. Round 2 below tests settings that set the consistency weight so that the model's consistency error
lands near the truth's 1 dB floor, or apply the loss to block means at 40 m rather than 20 m.</p>
<p><b>Scale factor.</b> The x4 arm (5 MHz band B at 80 m to 20 MHz at 20 m, also simultaneous) helps less:
detail correlation goes from 0.265 to 0.278 on far tiles and from 0.087 to 0.115 on near tiles. MAE improves on
82% of far tiles, and missed-edge chamfer drops from 64 m to 48 m. Band B is the weak link (QA above). Even with
perfect pairing, a 4x jump recovers little detail, in line with Experiment 1's lesson that factor size matters
on its own.</p>
<p><b>Sharpness.</b> All models stay much smoother than the truth (fine-detail energy ratio 0.16 to 0.45). L1
training against a target with about 5 looks of speckle predicts the conditional median, not the speckle. The
detail metrics therefore measure structure that L1 can recover, against a target that is itself noisy.</p>"""
T["xband_text"] = """<p>We scored 68 X-band collects over built-up ground: 55 Umbra, 7 ICEYE and 6 Capella (HH, geometry only).
Registration converged with a median shift of 11 to 20 m and 40 m correlation of 0.50 to 0.65. We dropped 44
collects with under 10% built-up land, 6 below the 0.2 correlation bar and 2 outside the NISAR swath. Open X-band
over the held-out frames is thin, so we also scored collects inside training frames but at least 3 km from every
training tile ("train_frame_unseen"). All 6 usable Capella collects fall in that group; the far frames (Buenos
Aires and the Philippines) yielded only low-correlation or non-urban Capella scenes.</p>
<p>The CNN reduces the distance from X-band edges to predicted edges. Umbra distances fall from 25.9 m with bicubic to 23.7 m with the CNN; the real 40 MHz image scores 22.7 m. Corresponding values are 34.0, 31.8 and 29.0 m for Capella, and 32.0, 30.4 and 28.7 m for ICEYE. Distances in the reverse direction remain similar. The comparison supports improved edge recall at this scale, while differences in band and viewing geometry limit stronger claims.</p>"""
T["transfer_text"] = """<p>The wider windows show modest sharpening of roads, shorelines and field edges beyond bicubic, alongside smooth fields and a faint input-grid pattern. Street grids and building texture in the 40 MHz reference remain unresolved. Native 20 MHz acquisitions show similar behavior, but have no matching higher-bandwidth reference here; those comparisons are visual checks only.</p>"""
T["caveats"] = """<ul>
<li><b>Mixed-mode, not native, 20 MHz in training.</b> The model trained on 20 MHz products cut from 40 MHz echoes.
Native 20+5 MHz acquisitions use a different centre frequency and pulse, and possibly a different noise floor.
Transfer to them is visual only (no truth exists there).</li>
<li><b>Shared echoes.</b> Input and target share half their spectrum, so their speckle is partly correlated. That
holds for any simultaneous pair and matches the deployment question: what would a 40 MHz radar have seen at that
moment? It also makes this an easier problem than two separate acquisitions would be.</li>
<li><b>PROVISIONAL data.</b> NISAR L-band products are calibrated but only partly validated. JPL will reprocess
them, with validated versions due by the end of 2026. Band B differed by a few tenths of a dB between the two files of
one datatake.</li>
<li><b>Round 1 sample.</b> 720 training tiles from 18 frames, 227 held-out tiles from 5 frames, one training seed, no
tuning. The consistency weight was not tuned; the ablation shows it matters.</li>
<li><b>Speckled target.</b> The 10 m GCOV has about 5 looks. Detail metrics against it are capped well below 1, and
L1 cannot reproduce speckle. Rounds 2 and 3 test multi-date references.</li>
<li><b>X-band sample.</b> Only 13 ICEYE and Capella collects passed, and Capella appears only on unseen ground
inside training frames. The X-band check is evidence on edge placement, not a calibrated accuracy.</li>
<li><b>L-band is not C-band.</b> None of this sharpens Sentinel-1. It shows the method works when pairing is clean.</li>
</ul>"""
T["status"] = """<p><span class="pill done">round 1 done</span> Survey; 947 paired tiles from remote HDF5 reads; arm A (x2)
and the x4 scale arm on near and far holdouts; consistency ablation; X-band geometry check (Umbra, ICEYE, Capella);
transfer panels.</p>
<p><span class="pill done">round 2 done</span> Consistency sweep chosen on validation (weight 0.5 at 40 m), now beating
bicubic on all five far-holdout metrics; multi-date 40 MHz target (more accurate, smoother); 10 m GCOV to 5 m GSLC
(same pattern one level finer); arm B with the X-band edge loss (no gain over arm A′, so X-band stays a validation
reference).</p>
<p><span class="pill done">rounds 3 and 4 done</span> Additional GSLC dates, gradient loss, longer runs and GAN fine-tuning are reported above. The gradient term gave a small accuracy gain; the GAN increased texture while reducing detail correlation.</p>"""
(ROOT / "results_text.json").write_text(json.dumps(T, indent=1), encoding="utf-8")
print("ok")
