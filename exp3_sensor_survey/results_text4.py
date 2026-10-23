"""Write results_text4.json: prose for the Round 4 section (adversarial term), placed at the top of the page."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
T = {}
T["lede"] = """<p>Round 4 tested a conditional PatchGAN after the gradient-loss experiments. Starting from model r3_g05_s0, we fine-tuned for 2,000 steps at adversarial weights 0.001, 0.003 and 0.01. A weight-zero control received the same extra training. Runs used batches of 16, one job at a time; evaluation ran on CPU.</p>
<p>At weight 0.01, the GAN increased fine-detail energy and reduced the visible input-grid pattern. Detail correlation fell and more reference edges were missed, especially in the nearby test regions. X-band edge distances barely changed. We kept the model without the GAN term.</p>"""
T["choice"] = """<h2>Choice on validation</h2>
<p>The rule, fixed before any run: compared with the control on validation (multi-date target), a weight is
<b>eligible</b> only if MAE rises by at most 0.01 dB, invented-edge chamfer by at most 0.6 m and grid imprint by at most
0.05 (round 3 seed noise). It <b>wins</b> if it also gains sharpness: fine-detail energy up by more than 0.004, or
missed-edge chamfer down by more than 0.6 m. Weights 0.001 and 0.003 changed almost nothing (all differences within
noise). <b>Weight 0.01 passed the selection rule</b> on fine-detail energy (+0.024) and also cut the grid imprint by 0.32. Two warning
signs sat outside the rule: missed edges rose by 1.0 m and detail correlation fell from 0.30 to 0.28.</p>"""
T["hold"] = """<h2>Holdouts, two seeds each</h2>
<p>Against the multi-date target, both GAN seeds behave the same way, and so do both control seeds:</p>
<ul>
<li><b>Sharper, less gridded:</b> fine-detail energy 0.407 to 0.418 vs 0.394 to 0.395 (far) and 0.353 to 0.368 vs
0.329 (near). Grid imprint falls to 1.83 to 1.94 from 2.10 (far) and to 1.86 to 2.03 from 3.08 (near).</li>
<li><b>Less faithful:</b> detail correlation drops (far 0.332 to 0.338 vs 0.345 to 0.346; near 0.273 to 0.278 vs
0.293), SSIM drops, and MAE rises by about 0.01 dB. Missed edges rise slightly on far tiles (+0.2 to 0.3 m) and
sharply on near tiles (36.6 to 38.6 m vs 32.8 to 33.1 m), though invented edges fall there.</li>
<li>Against the single-date target the pattern is the same, more starkly on near tiles: detail correlation 0.084
to 0.089 vs 0.120, and missed edges 37.6 to 40.4 m vs 30.9 to 31.1 m.</li>
</ul>
<p>The GAN added high-frequency variation that agreed less well with the reference. The panels show fine grain over smooth fields, without recovering the street grids or building rows visible in the 40 MHz image.</p>"""
T["xband"] = """<h2>Hallucination check against X-band</h2>
<p>The X-band check uses the same infrastructure masks and sites as round 2. Differences between GAN and control are under 1 m: at most 0.3 m for output-to-X-band edges and 0.8 m in the reverse direction. Capella test distances are 31.8 versus 32.6 m; far Umbra distances are 22.8 versus 22.9 m. This check detected no meaningful change in edge placement.</p>"""
T["caveats"] = """<ul><li>One adversarial design (conditional PatchGAN, LSGAN, 2,000 fine-tune steps); a perceptual loss or a longer GAN
schedule could behave differently.</li>
<li>Two seeds each for the winner and the control. The seed spread is small (≤0.006 in detail correlation, ≤2 m in
near missed edges), well below the GAN-vs-control gaps reported above.</li>
<li>The selection rule did not include detail correlation, so the validation winner was allowed to lower it; the
holdouts then confirmed that drop.</li>
<li>Incident: while stopping our first round 4 attempt (which ran beside Experiment 5 and slowed it), an over-broad
process filter also killed two Experiment 5 runs. Experiment 5's queue restarted both from scratch within the
minute. Our filter now matches only Experiment 3's training script, and Round 4 then waited for Experiment 5 to
finish.</li></ul>"""
(ROOT / "results_text4.json").write_text(json.dumps(T, indent=1), encoding="utf-8")
print("ok")
