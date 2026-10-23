# Experiment 4: Sentinel-1 despeckling

Compare learned speckle filters with standard filters, then test land-cover classification and field-boundary detection.

**Result:** The VV+VH Noise2Noise CNN trained on date pairs from 24 sites gained about 0.7 dB PSNR over Lee. WorldCover classification accuracy was 0.806, versus 0.770 for raw VV+VH and 0.779 for Lee. Despeckling did not improve field-boundary detection in the FTW test. Textured desert remained a weakness.

[Full results, figures and limitations](https://tristangrupp.github.io/sar-scaling/exp4.html).

## Main files

| File | Purpose |
|---|---|
| `common.py, fetch.py, stack.py` | Configure paths, fetch images and build temporal references. |
| `train4.py, train_r2.py` | Original and follow-up training. |
| `eval4.py, eval_r2.py, downstream.py, downstream_r2.py` | Image metrics and mapping tests. |
| `text.py, findings.py, findings_r2.py` | Methods, results and limitations. |
| `build_page.py, build_r2.py` | Build the report and follow-up section. |

## Reproducing the experiment

Read the [project README](../README.md) for environments and data paths. Data, checkpoints and logs are excluded from Git. The scripts record several rounds of work; running the page builder requires their saved result tables and figures. It does not rerun training.
