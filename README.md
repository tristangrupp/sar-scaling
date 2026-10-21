# SAR super-resolution and despeckling

Can open synthetic aperture radar (SAR) data make Sentinel-1 sharper or cleaner? This folder holds every experiment, its code, its data and
the project site. Open `site\index.html` in a browser for the write-ups, charts and panels; every page links to the
others.

## Layout

| Folder | What it holds |
|---|---|
| `site\` | The project site, one page per experiment. Rebuild with `python build_site.py` (ftwtools env). |
| `build_site.py` | Builds the site from each experiment folder's results. |
| `related_work\` | `NOTES.md` on prior work (Amieva et al. 2026; Ahn et al. 2025). The papers themselves are not in the repository. |
| `exp0_overlap_atlas\` | Where open X-band (Capella, Umbra, ICEYE) overlaps Sentinel-1 Stripmap. Scripts `s1sm_*_overlap.py`, item tables in `s1sm_*\`, the atlas page, and the d3, topojson and land files every map uses. |
| `exp1_gan_spike\` | Sentinel-1 IW to X-band at 2.5 m (negative), plus `ceiling\`, the real X-band against real X-band check. |
| `exp2_iw_to_stripmap\` | Sentinel-1 IW to Stripmap at 5 m, with X-band as a geometry check. `data\` holds the Stripmap SLCs (about 207 GB), tiles and predictions. |
| `exp3_sensor_survey\` | Survey of open SAR sensors, then a NISAR L-band experiment. `data\` holds the NISAR tiles. |
| `exp4_despeckling\` | Sentinel-1 despeckling and downstream mapping. `data\` holds the Sentinel-1 stacks. |
| `exp5_cross_band\` | Cross-band transfer: X-band pretraining, joint multi-sensor model. |

## What the repository leaves out

The GitHub repository holds code, result tables, figures and the built site. It leaves out the `data\` folders
(about 250 GB of scenes and tiles), model checkpoints, run logs and lock files; `.gitignore` lists them. Rerunning an
experiment means fetching the data again with its fetch scripts.

## Paths

Each experiment finds its own data through one constant, relative to its folder:

- Experiment 2: `DATA` in `exp2_iw_to_stripmap\common.py` (env override `S1SM_DATA`)
- Experiment 3: `DATA` in `exp3_sensor_survey\config.py` (env override `EXP3_DATA`)
- Experiment 4: `DATA` in `exp4_despeckling\common.py`
- Experiment 5: `EXP3_DATA`, `EXP4_DATA` and `SM_TILES` in `exp5_cross_band\common.py`

Scripts in one experiment read another's files through the project root (`ROOT.parent`), so keep the folder names
as they are. The Earthdata token stays outside the project at `C:\Users\grupp\Desktop\earthdata.txt`.

## Environments

- `ftwtools` conda env: torch with CUDA, rasterio, geopandas. Run GDAL scripts in the background with output sent to
  a log; a foreground pipe can hang GDAL.
- `s1slc` conda env: sarsen 0.9.6 and xarray-sentinel 0.9.6 for Stripmap SLC processing.

Local log files (`*.log`, not in the repository) record past runs and still show the old paths (`C:\Users\grupp\Desktop\sar-xband-s1sm-overlap`,
`E:\s1_stripmap_slc`, `E:\exp3_sar`, `E:\exp4_s1`). The code and tables use the paths above.
