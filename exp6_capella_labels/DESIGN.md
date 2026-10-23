# Experiment 6: labels from Capella

Status: planned (2026-10-23). The CPU data check ran; no GPU work has started.

## Question

Does a Sentinel-1 field-boundary model at 10 m get better when its training labels come from sub-meter
Capella X-band, compared with the labels already available from optical data?

Experiments 1–5 used X-band as a pixel target or loss, and that failed. This experiment uses X-band only for
what it does well: sharp geometry that a person or a segmenter can trace.

## Hypothesis

Labels traced or corrected on Capella place field edges more accurately than labels from Sentinel-2 at 10 m.
They also catch narrow lanes and small fields and match the date of the radar. A Sentinel-1 model trained
with them predicts boundaries closer to the truth on held-out sites. The test fails if Capella-refined labels
and the existing labels agree almost everywhere at 10 m, or if the model can't use the difference.

## Prior work

De, Cazcarra-Bes, Goncharenko, McDaniel and Stringham (Capella Space), "A Deep Learning Approach to Field
Boundary Segmentation Using Very High-Resolution X-Band SAR Data", IGARSS 2025, pp. 8037–8041,
doi:10.1109/IGARSS55030.2025.11314044. They segmented field boundaries directly on Capella Spotlight
imagery (sub-meter X-band). They combined modern segmentation networks with classical iterative region
merging. Their references include U-Net, Mask R-CNN, ViT, SAM and SAM 2, seeded region growing and snakes.
The abstract gives no dataset size or scores, and the full text is behind IEEE's paywall. Their result
suggests Capella Spotlight shows field edges clearly enough to segment, which is the premise here. They did
not try to transfer labels to a coarser sensor.

## Data (from scenes.csv and summary.txt)

The open Capella GEO catalog holds 2,285 items at 480 sites (2021-02 to 2026-09): 1,741 Spotlight,
291 Stripmap and 253 Sliding Spotlight. Of these, 2,088 are HH. A Sentinel-1 IW GRD scene within ±12 days
covers at least half the footprint for 2,189 items (2,168 with VV+VH, median 4 acquisitions per window).
Polarization doesn't matter here, because only the geometry of the labels moves between sensors.

Label coverage at the Capella footprints:

- **FTW benchmark (human-verified, 25 countries):** 2 items touch it (Slovenia, 8 chips; Luxembourg,
  28 chips). Practically none.
- **FTW Global** (Robinson et al. 2026, a Sentinel-2 model map, 2024–2025): covers every land footprint.
  192 items at 60 sites have at least 10% cropland (WorldCover) and at least 10% field share; 190 have
  Sentinel-1. 105 of the 192 date from 2024; items from 2021–2023 predate the map's years.
- **Open national parcels** (LPIS type, an independent truth) exist for about 13 of the 60 field sites:
  Germany 3, Spain 3, Italy 2, Poland 2, Austria, Belgium and the Netherlands 1 each. The 8 US field sites
  have USDA Crop Sequence Boundaries, though those come from 30 m data.
- **Buildings:** 1,536 items at 413 sites hold at least 200 Overture buildings. Those come from
  OpenStreetMap, Google and Microsoft, all traced on optical imagery under 1 m.

## Label route

Route (a) in its strict form, human labels already at Capella sites, doesn't exist: FTW's benchmark misses
Capella almost entirely. Route (b), tracing from scratch on Capella, wastes the FTW Global map.

The chosen route is a hybrid: take FTW Global polygons as the starting labels at the 60 field sites, then
correct them on Capella. SAM 2 prompted by the FTW polygons snaps edges to Capella, and a person accepts,
splits, merges or deletes each field. This keeps the work to checking rather than drawing.

Buildings are a control, not the main route. Their existing labels already come from sub-meter imagery, so
Capella has little to add.

Yield: the 60 field sites cover 20,500 km² of unique footprint with Sentinel-1, or 1,905 chips of 256 px
(9,924 of 128 px). Weighted by cropland share, that is about 3,600 km², or roughly 550 cropland-dominated
256-px chips (about 2,200 of 128 px). Large Stripmap scenes supply most of this: 19 Stripmap sites give
1,660 of the 1,905 chips. Spotlight scenes give only 3 or 4 chips each.

## Model

U-Net (FTW baseline style) on a Sentinel-1 VV+VH stack: all IW acquisitions within ±12 days of the Capella
date plus a season median. Classes are background, field and boundary, with ignore outside checked areas.
Labels are rasterized at 10 m with `all_touched=False`, following the FTL_FTW ribbon trap.

Experiment 4's despeckled input stays an option (arm D) and runs only if arm C passes. Experiment 4 changed
the input and kept the labels, using 20 FTW chips and two training chips per country. This experiment
changes the labels and keeps the input, with thousands of training chips. Experiment 4's boundary F1 near
0.1 came partly from that tiny training set, so its result doesn't predict this one.

## Arms and baselines

Split by site, never by chip: about 40 training sites and 20 test sites, stratified by country and mode.

- **A (baseline):** Sentinel-1 model trained on FTW Global labels at about 10,000 global chips away from
  Capella sites. No Capella.
- **B (location control):** A plus the Capella training sites, labeled with FTW Global.
- **C (Capella labels):** A plus the same sites, labeled with Capella-corrected polygons.
- **D (optional):** C with experiment 4's despeckled input.

The label effect is C against B. Three seeds per arm.

## Metrics

On the held-out sites: boundary F1 at 1-pixel tolerance, field IoU, and object F1 (IoU ≥ 0.5, FTW's
definition). Truth is the Capella-corrected labels. On the 13 or so sites with national parcels, the
same metrics are also scored against those parcels, an independent check that guards against C winning
only because the test labels share its source.

## Decision rule (fixed before training)

1. Cheap CPU gate before any GPU time: rasterize FTW Global and the Capella-corrected labels at 10 m
   on the first 10 corrected sites. If boundary F1 between them is 0.90 or higher, stop: Capella adds
   nothing a 10 m model could learn.
2. C beats B when boundary F1 rises by at least 0.03 and the paired site-bootstrap 95% interval excludes
   zero, in all 3 seeds. The same direction must hold against national parcels, or the gain counts as a
   self-consistency artifact.
3. A tie or loss on rule 2 is a negative result and gets reported as one.

## Cost and GPU estimate

Capella GEO files run about 1 GB each. One image per field site comes to about 60 GB, read as windows
where possible. Sentinel-1 GRD COG windows are small. SAM 2 on Capella takes 2–4 GPU hours. Training at
128 px, 3 arms by 3 seeds at about 1.5 hours each, comes to about 14 GPU hours (plus about 5 for arm D).
All GPU work waits for a slot from `grupp-8b`. Human review of about 60 sites, at 20–40 minutes each,
takes 20–40 hours. That is the largest cost.

## Files

`s1_iw_check.py` (CDSE STAC), `labels_raster.py` (WorldCover, FTW Global, FTW benchmark),
`overture_buildings.py` (needs `duckdb` installed outside `ftwtools`; set `DUCKDB_PATH`), and
`build_scenes.py`, which writes `scenes.csv` and `summary.txt`. Caches sit in `cache\`.
