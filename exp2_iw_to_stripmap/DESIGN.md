# Experiment 2: Sentinel-1 IW to Stripmap

## Question and result

Does guidance from X-band edges improve a Sentinel-1 model trained against higher-resolution images from the same sensor?

The baseline model beat bicubic interpolation on held-out Stripmap, largely by reducing speckle. Adding X-band edge loss reduced SSIM by 0.21 points. More inland training data and a change mask made little difference in the follow-up. The final model keeps X-band for evaluation only.

This note describes the completed study. The [results page](https://tristangrupp.github.io/sar-scaling/exp2.html) has the scores, comparisons and round-by-round changes.

## Input and reference

The input is Sentinel-1 IW RTC VV on a 10 m grid, with spatial resolution of about 20 m. The reference is Sentinel-1 Stripmap SLC, processed to terrain-flattened gamma0 on a 5 m UTM grid. The output grid is twice as fine as the input grid; that is not evidence of a twofold improvement in spatial resolution.

Stripmap GRD has about 23 m resolution and is unsuitable as the sharper reference. SLC processing uses sarsen, the Copernicus DEM and the EGM2008 geoid correction. South-up sarsen output must be flipped before writing a north-up raster. These two processing issues caused the initial Houston alignment error and were corrected before the main evaluation.

IW and Stripmap are paired within six days, preferring the same orbit direction. Registration runs from coarse to fine; only IW is resampled. Reliable tile offsets are pooled by scene. Water is masked. The initial dataset used 396 paired tiles: 200 training, 44 validation and 152 test.

## Test regions

Houston, Owens, Western Australia and the paper's three test scenes are held out. Results are reported by region, including the near and far groups. Validation uses separate locations and the paper's two validation scenes. Later rounds add inland regions, including Yemen and Bangladesh.

The original design proposed stricter distance and viewing-geometry filters than the final data supported. Read the reported splits and quality checks as the implemented experiment; the original goals are not guarantees about every pair.

## Baseline: arm A

The baseline adapts Amieva, Ayala and Galar's ESRT method. It uses 48 × 48 input patches and 96 × 96 references, a Lee-filtered Stripmap target histogram-matched to IW, and global percentile normalization. The main training run uses SSIM loss, 8,000 steps and batches of 16. We also test an input-consistency loss and a residual CNN.

The input-consistency term compares the output averaged to 10 m with IW. The published-method ESRT, ESRT with consistency, and CNN score within 0.3 SSIM points of one another.

This is an adaptation, not an exact reproduction. The paper uses SNAP, dense Gefolki registration and whole-scene evaluation. We use sarsen, shift-based registration and selected 3.2 km tiles. The absolute SSIM and PSNR scores are not directly comparable.

## X-band evaluation

We compare predicted edges with open Capella, Umbra and ICEYE images. Edge distances are measured in both directions: output to reference, and reference to output. They indicate misplaced edges and missed edges, respectively. Different wavelength, viewing angle and date can also cause disagreement.

Evaluation focuses on infrastructure and masks water, change, steep slopes, layover and shadow where available. Some geometry is approximated from orbit direction. X-band is registered to IW rather than Stripmap. Capella HH is included for geometry checks, with its polarization difference kept explicit.

Real Stripmap is itself no closer to X-band edges than bicubic IW in the output-to-reference direction: 24.1 m versus 23.7 m. This limits how strongly X-band edge agreement can support a claim about C-band detail.

## X-band training: arm B

Arm B fine-tunes ESRT with an additional edge-distance loss on infrastructure. Edges within 5 m are unpenalized; the penalty saturates at 20 m. Input consistency stays on. A control receives the same 3,000 extra steps without the edge loss.

Training uses 173 X-band pairs: 87 ICEYE, 70 Umbra and 16 Capella. Only 15% registered successfully to IW; the others rely on provider geolocation. Validation selected weight 0.05 over 0.2. The lower weight tied the control on validation but reduced held-out Stripmap SSIM by 0.21 points.

## Follow-up: inland scenes and change masking

The follow-up added 25 inland scenes from 13 regions and built 158 tiles. Its change mask uses same-track IW passes bracketing the Stripmap date. A pixel is flagged when 3 × 3 mean backscatter differs by more than 3 dB. Some flagged pixels reflect speckle or seasonal effects rather than actual change.

Round 3 compared training with and without the mask, including a second seed. Held-out gains were small. The final recommendation is to leave the mask off for this training setup.

## Evaluation and limits

We report MAE, SSIM, PSNR, detail correlation, fine-detail energy and edge distances against raw and Lee-filtered Stripmap. ENL and intensity-ratio metrics describe despeckling. Higher sharpness alone does not establish that added detail is correct.

Image comparisons include 9.6 km windows around test tiles and other landscapes. Outside Stripmap coverage, these are visual checks rather than accuracy measurements.

The main limits are a small, uneven dataset; residual registration error, especially at Owens; different acquisition dates; incomplete X-band geometry; and differences from the published processing and evaluation. The result applies to the tested models and data, not every possible use of X-band imagery.
