# Experiment 2: Sentinel-1 IW to Stripmap super-resolution, with X-band as a geometry check

## Why

The first spike trained Sentinel-1 IW to imitate X-band brightness. Its ceiling depends on wavelength: a model that learns X-band texture can produce plausible detail that C-band never recorded. This experiment keeps the band fixed. It trains Sentinel-1 IW (about 20 m true resolution) to match Sentinel-1 Stripmap (about 5 m) from the same sensor. X-band then serves only as a reference for where edges sit, never for how bright things look.

The question it answers: does X-band geometry add real detail to an IW to Stripmap model, or only hallucination?

## Goals

1. Train IW to Stripmap at a 5 m output grid (2x the 10 m IW pixel grid; about 4x in true resolution).
2. Score it on held-out Stripmap scenes and on known targets.
3. Validate geometry against X-band in many more places than Stripmap covers (stage 1, no training).
4. Train the same model a second way, adding a constrained X-band edge loss (stage 2).
5. Compare the two arms on held-out Stripmap. A gain means X-band added real detail; no gain or a loss means it added nothing or added hallucination.

## Non-goals

- Capella-level (sub-meter) output. The target stops at 5 m.
- A global production model. Stripmap covers limited places, so this tests the method.
- Calibrated X-band brightness. X-band never enters a brightness loss.

## Failure even if it "works"

- Crisp edges in the wrong place. Edge-position checks guard against this.
- Gains that exist only near places X-band imaged (memorization). Reporting near and far holdouts separately guards against this.
- Any leakage of held-out regions through Stripmap, X-band, or footprint data.

## Data

**Target: Stripmap SLC (single look complex).** The free Stripmap GRD (ground range detected) product has 23 m resolution, no sharper than IW. Only the SLC keeps the about 2 x 4 m detail. Each SLC is about 4.4 GB from ASF (Earthdata token in `C:\Users\grupp\Desktop\earthdata.txt`). Processing: calibrate to sigma0 or gamma0, multilook to about 5 m, terrain-correct onto a 5 m UTM grid with the Copernicus 30 m DEM, VV only.

**Input: IW.** Sentinel-1 RTC VV from Planetary Computer (10 m, terrain-corrected), nearest date within 6 days of each Stripmap scene. Prefer the same orbit direction.

**Change mask.** Use the Sentinel-1 IW stack around each date. Drop pixels whose IW backscatter changes more than a threshold between the IW and Stripmap dates.

**X-band (geometry only).** Umbra, ICEYE, and Capella open data, multilooked and degraded to 5 m. Only pairs with matching look direction and incidence within 5 degrees of the Stripmap scene count. Polarization matters less here, because only edges enter.

**Layover and shadow masks.** Compute them from the DEM and each sensor's geometry (look direction, incidence). Exclude masked pixels from every X-band loss and metric.

**Infrastructure masks.** Building footprints (Overture, which merges Microsoft, Google and OSM) and OSM roads. They restrict the stage 2 loss to stable infrastructure and give an independent geometry check.

## Splits

Held-out regions are whole areas, excluded from every source: Stripmap, X-band, footprints used for tuning, and IW. Two kinds:

- **Near:** held-out regions with X-band coverage within about 50 km of training areas.
- **Far:** held-out regions at least 500 km from any X-band scene used in training.

Results are reported separately for near and far.

## Stage 0: inventory (no training)

Search the ASF archive for Stripmap SLC VV scenes and pair each with IW RTC within 6 days. Rank candidate regions by the number of IW and Stripmap pairs, terrain (prefer flat to moderate), and land cover mix. Then pick training, near-holdout and far-holdout regions. Start with a pilot of about 30 to 60 Stripmap scenes (about 130 to 260 GB of downloads), cropped to the region of interest after processing so only small tiles stay on disk.

## Published baseline

Amieva, Ayala and Galar (*Acta Astronautica* 2026) already trained IW to Stripmap at 10 m to 5 m: ESRT with SSIM loss and histogram matching, 10 ascending VV+VH scene pairs from 2015 to 2017, Lee-filtered Stripmap reference, SNAP processing with terrain flattening, Gefolki co-registration. Best test result: SSIM 70.14, PSNR 26.07 dB, ENL 201.5. Details in `../related_work/NOTES.md`.

That makes arm A a reproduction of their method, and the contribution of this experiment is arm B: does X-band geometry beat that baseline on held-out Stripmap? To compare fairly:

- Arm A uses ESRT and their loss, alongside the residual CNN.
- Score against both raw and Lee-filtered Stripmap, and report their metrics (SSIM, PSNR, ENL, MoI, MoR, Cx) next to ours.
- Turn terrain flattening back on in the Stripmap processing.
- Download their 10 listed Stripmap scenes and report on them as well as on the pilot regions.

## Arm A: IW to Stripmap (baseline reproduction)

Use the same generator family as the spike, at 2x scale. Losses:

- masked L1 against Stripmap in dB
- input consistency: the output averaged back to 10 m must match the IW input
- optional small adversarial term, only if L1 alone is visibly blurry. The spike's discriminator overpowered the generator, so start with none.

## Stage 1: X-band geometry validation (no training)

Score arm A's outputs against degraded X-band at held-out X-band sites:

- edge alignment: mean distance from output edges to the nearest X-band edge (chamfer), plus the reverse direction to catch invented edges
- building footprint overlap: bright-return outlines against Overture footprints
- linear feature widths: road widths compared with OSM and X-band
- small structures: whether point targets appear at the right place

## Arm B: IW to Stripmap plus X-band edges

Use arm A's losses plus a tolerant edge loss. A distance transform of X-band edges penalizes how far each output edge sits from the nearest X-band edge. Constraints:

- applied only on infrastructure pixels (buildings and roads), never on vegetation or water
- layover, shadow, and change masked out
- low weight; tune it on a validation split, never on held-out data
- the input-consistency loss stays on, so brightness stays anchored to C-band

## Scoring

Score both arms on the same held-out Stripmap scenes, near and far reported apart:

- MAE and SSIM in dB against Stripmap
- detail correlation: does the 5 m detail land where Stripmap's does
- fine-detail energy ratio: is it as sharp as Stripmap, or blurrier
- the stage 1 geometry metrics
- baselines: bicubic IW, and the spike's X-band model blurred to 5 m

## Capella in the geometry check

Experiment 1 could use only Capella's 197 VV collects, because its loss compared brightness against a VV input. Here X-band enters through edges alone, so Capella's 2,088 HH collects count too. Every X-band provider (Capella, Umbra, ICEYE) appears in the stage 1 validation, and results are reported per provider. Capella HH collects in Houston, Newark and other held-out cities serve as geometry references there.

## Transfer panels

Every trained model also runs beyond its truth footprints:

- a 9.6 km window of IW around each held-out tile, with the truth area outlined, to show behavior just outside it
- landscapes the training set barely covers: farmland, tropical forest, mountains, desert, dense villages

No truth exists there, so these panels are a visual check that the model adds plausible structure, or only filters speckle, across contexts. Experiment 1's panels showed only speckle filtering.

## Decision rule

Arm B beats arm A on held-out Stripmap in both near and far regions: X-band geometry helps. B ties or loses: drop X-band from training and keep it for validation only.

## Left to the builder

- Stripmap SLC processing tool (sarsen, ISCE3 or SNAP), provided it outputs terrain-corrected 5 m gamma0
- exact edge detector and change threshold
- network size and training length

## Blocker

ASF refuses Stripmap SLC downloads until the Earthdata account approves the ASF data agreement:
https://urs.earthdata.nasa.gov/approve_app?client_id=BO_n7nTIlMljdvU6kRRB3g
