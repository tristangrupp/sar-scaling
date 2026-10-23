# Experiment 2: Sentinel-1 IW to Stripmap

Train on 5 m Stripmap references, then test whether X-band edge guidance improves the result.

**Result:** The model based on the published ESRT method beat bicubic, mostly through despeckling. X-band edge loss reduced held-out SSIM by 0.21 points. Adding inland data and a change mask made little difference. The comparison uses our own processing and tiles, so its absolute scores cannot be compared directly with the paper.

[Full results, figures and limitations](https://tristangrupp.github.io/sar-scaling/exp2.html).

## Main files

| File | Purpose |
|---|---|
| `DESIGN.md` | Study design, final methods and limitations. |
| `common.py, process_slc.py, finalize_pairs.py` | Data paths, Stripmap processing and image pairing. |
| `train_sr.py, score.py` | Training and evaluation. |
| `make_results.py, v2_section.py, r3_section.py` | Generate the report and follow-up results. |
| `change_mask.py` | Flag backscatter changes using bracketing passes on the same track. |

## Reproducing the experiment

Read the [project README](../README.md) for environments and data paths. Data, checkpoints and logs are excluded from Git. The scripts record several rounds of work; running the page builder requires their saved result tables and figures. It does not rerun training.
