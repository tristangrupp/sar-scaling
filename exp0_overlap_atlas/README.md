# Experiment 0: overlap atlas

Find open Capella, Umbra and ICEYE scenes that overlap Sentinel-1 Stripmap.

**Result:** Umbra supplied the most eligible pairs: 1,069 at 12 sites, followed by ICEYE (211 at 7) and Capella (110 at 4). The filters check polarization, incidence, footprint overlap and time gap. Each candidate still needs image-level quality checks.

[Full results, figures and limitations](https://tristangrupp.github.io/sar-scaling/overlap.html).

## Main files

| File | Purpose |
|---|---|
| `s1sm_capella_overlap.py, s1sm_umbra_overlap.py, s1sm_iceye_overlap.py` | Search and pair each provider with Sentinel-1 Stripmap. |
| `overlap_page.template.html, build_page.py` | Map template and atlas builder. |

## Reproducing the experiment

Read the [project README](../README.md) for environments and data paths. Data, checkpoints and logs are excluded from Git. The scripts record several rounds of work; running the page builder requires their saved result tables and figures. It does not rerun training.
