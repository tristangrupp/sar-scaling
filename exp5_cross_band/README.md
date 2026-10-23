# Experiment 5: pretraining across radar bands

Test direct transfer from X-band and whether X-band or X-band + NISAR pretraining improves a Sentinel-1 Stripmap model.

**Result:** An X-band-only model beat bicubic when evaluated against same-band Stripmap references, and pretraining helped short L1 runs. The final comparison matched the training method and budget: 200 real tiles, SSIM loss and 8,000 steps. None of 40 comparisons met the improvement rule. That result limits the earlier claims about pretraining.

[Full results, figures and limitations](https://tristangrupp.github.io/sar-scaling/exp5.html).

## Main files

| File | Purpose |
|---|---|
| `common.py, e5_calib.py, e5_train.py` | Synthetic degradation, speckle calibration and initial training. |
| `e5r3_score.py, r3_prose.py` | Stripmap evaluation and short-run learning curves. |
| `e5r4_pretrain.py, e5r4_train.py, e5r4_score.py` | Pretraining and the final comparison with matched settings. |
| `narrative.py, results_text.py, r2_prose.py, r3_prose.py, r4_prose.py` | Write-up sources, organized by round. |
| `e5_build_page.py` | Assemble the report from saved results. |

## Reproducing the experiment

Read the [project README](../README.md) for environments and data paths. Data, checkpoints and logs are excluded from Git. The scripts record several rounds of work; running the page builder requires their saved result tables and figures. It does not rerun training.
