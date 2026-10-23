# Experiment 3: NISAR bandwidth pairs

Survey open radar sources and test super-resolution with NISAR images from the same acquisition.

**Result:** The tuned 20 MHz to 40 MHz model beat bicubic on all five metrics in the distant test regions. Multi-date targets reduced error but made outputs smoother. X-band edge loss did not help; the GAN fine-tune added texture while reducing detail correlation. This experiment evaluates L-band NISAR and does not sharpen Sentinel-1.

[Full results, figures and limitations](https://tristangrupp.github.io/sar-scaling/exp3.html).

## Main files

| File | Purpose |
|---|---|
| `config.py, mixed.py, prep.py` | Configure paths, find bandwidth pairs and prepare tiles. |
| `train.py, evaluate.py` | Train and evaluate the models. |
| `results_text*.py` | Prose for the original results and later rounds. |
| `build_page.py` | Combine the survey, results and figures into the report. |

## Reproducing the experiment

Read the [project README](../README.md) for environments and data paths. Data, checkpoints and logs are excluded from Git. The scripts record several rounds of work; running the page builder requires their saved result tables and figures. It does not rerun training.
