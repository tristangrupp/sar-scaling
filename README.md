# SAR Bench

Experiments on sharpening Sentinel-1 radar images and reducing speckle with open SAR data.

**Main result:** X-band data did not improve the final Sentinel-1 model under the settings tested. Training on Sentinel-1 Stripmap and reducing speckle both helped. NISAR provided a separate test with low- and high-resolution images from the same acquisition.

[Read the results and view the image comparisons](https://tristangrupp.github.io/sar-scaling/).

## Experiments

Experiments 0–5 are complete. The table gives the final result; individual pages retain earlier rounds and their limitations.

| Experiment | Question | Result |
|---|---|---|
| [0: Overlap atlas](exp0_overlap_atlas/) | Where does open X-band imagery overlap Sentinel-1 Stripmap? | Umbra: 1,069 eligible pairs at 12 sites; ICEYE: 211 at 7; Capella: 110 at 4. Eligibility does not guarantee a usable training tile. |
| [1: Direct X-band targets](exp1_gan_spike/) | Can Sentinel-1 IW predict a 2.5 m X-band image? | The tested CNN and GAN mainly smoothed the input. Detail correlation was 0.04, versus 0.03 for bicubic interpolation. Viewing geometry and alignment also limited this comparison. |
| [2: IW to Stripmap](exp2_iw_to_stripmap/) | Does X-band edge guidance improve a model trained on 5 m Sentinel-1 Stripmap? | The same-sensor model beat bicubic, largely through despeckling. Adding X-band edge loss reduced held-out SSIM by 0.21 points. More inland data and a change mask made little difference. |
| [3: NISAR bandwidth pairs](exp3_sensor_survey/) | What happens when the input and target come from one acquisition? | A tuned NISAR model beat bicubic on all five metrics in the distant test regions. X-band edge loss did not help. A GAN added texture without improving structural accuracy. These results apply to NISAR, not Sentinel-1. |
| [4: Despeckling](exp4_despeckling/) | Does a learned speckle filter improve images and mapping? | A VV+VH Noise2Noise model gained about 0.7 dB PSNR over Lee. WorldCover classification accuracy rose from 0.770 on raw VV+VH to 0.806. Field-boundary detection did not improve. |
| [5: Pretraining across bands](exp5_cross_band/) | Does X-band or X-band + L-band pretraining improve the final Stripmap model? | Pretraining helped some short runs. With 200 real training tiles and the full 8,000-step recipe, none of 40 comparisons met the rule for an improvement over training from scratch. |

## Reading the results

Sentinel-1 IW is sampled on a 10 m grid, but its spatial resolution is about 20 m. An output grid of 5 m or 2.5 m does not by itself show that the model recovered features at that scale.

X-, C- and L-band radars can show the same ground differently. Different dates, viewing angles and residual alignment errors also affect image comparisons. The pages report these limits alongside the scores. Real radar references contain speckle; they are not noise-free ground truth.

Experiments 2 and 5 adapt the method of Amieva, Ayala and Galar. They do not reproduce the paper's processing and whole-scene evaluation exactly, so the absolute scores are not directly comparable. See [related work](related_work/NOTES.md).

## Repository contents

The repository includes experiment code, result tables, figures and the published HTML pages. Raw scenes, processed image tiles, checkpoints and run logs stay local and are excluded by `.gitignore`.

Each experiment folder has a README with its question, result and main scripts. `related_work/NOTES.md` summarizes the papers used to design the tests. `site/` contains the published pages; `build_site.py` assembles them from the experiment reports.

## Running the code

This is a record of research runs, not a packaged training library. Fetching data, processing scenes, training and evaluating are separate scripts. Read the relevant experiment README before running them. The scripts require data and results that are not included in a fresh clone.

The original runs used two conda environments:

- `ftwtools`: PyTorch with CUDA, rasterio, geopandas, NumPy, pandas and Markdown. Used for training, evaluation, figures and the site builder.
- `s1slc`: sarsen 0.9.6 and xarray-sentinel 0.9.6, with OpenBLAS. Used to process Sentinel-1 Stripmap SLCs.

Some Windows GDAL jobs stalled when attached to a foreground output pipe. The original runs sent their output to log files in the background.

Data paths are relative to each experiment folder:

| Experiment | Configuration | Environment override |
|---|---|---|
| 2 | `DATA` in `exp2_iw_to_stripmap/common.py` | `S1SM_DATA` |
| 3 | `DATA` in `exp3_sensor_survey/config.py` | `EXP3_DATA` |
| 4 | `DATA` in `exp4_despeckling/common.py` | — |
| 5 | `EXP3_DATA`, `EXP4_DATA`, `SM_TILES` in `exp5_cross_band/common.py` | See the script |

Keep the folder names: several experiments read one another's data through the project root. ASF downloads require Earthdata access. Keep credentials outside the repository.

## Building the site

On the original machine, with the experiment reports and results present:

```sh
python build_site.py
```

The experiment builders produce the reports consumed by this command. Edit their prose sources when changing an experiment write-up; editing only `site/*.html` will be overwritten on rebuild. The overview and result summaries live in `build_site.py`.

GitHub Pages publishes `site/` when changes to that folder reach `main`.
