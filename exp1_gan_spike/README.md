# Experiment 1: Sentinel-1 to X-band

Test whether Sentinel-1 IW can predict X-band imagery on a 2.5 m output grid.

**Result:** The tested residual CNN and GAN mainly smoothed the image. Detail correlation was 0.04, versus 0.03 for bicubic. The repeat-image check in `ceiling/` shows that alignment and viewing geometry also limit these scores; this is a negative result for the tested setup, not a universal limit on super-resolution.

[Full results, figures and limitations](https://tristangrupp.github.io/sar-scaling/spike.html).

## Main files

| File | Purpose |
|---|---|
| `prep.py, train.py, evaluate.py` | Prepare paired tiles, train the models and score them. |
| `report.py, verdict.html` | Build the report and supply its written conclusion. |
| `ceiling.py, ceiling/notes.html` | Compare real X-band images of the same place and explain the result. |
| `infer_neighbors.py` | Make comparisons around test tiles and on other landscapes. |

## Reproducing the experiment

Read the [project README](../README.md) for environments and data paths. Data, checkpoints and logs are excluded from Git. The scripts record several rounds of work; running the page builder requires their saved result tables and figures. It does not rerun training.
