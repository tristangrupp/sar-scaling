# Experiment 10: a Sentinel-1 enhancer trained on degraded Capella

Status: planned. Written 2026-10-31, before the first chip or score existed. Later changes get a dated
amendment at the end, not silent edits.

## Question

Can a model take Sentinel-1 IW (10 m pixels, about 20 m real resolution) and produce a sharper, cleaner
image, if it learns from Capella X-band images degraded to a resolution Sentinel-1 might plausibly
support? Which target resolution is the finest one where such a model beats strong baselines on sites
it never saw?

Experiment 1 asked a 10 m input to predict a 2.5 m single-look X-band image. The models smoothed, and
the detail correlation was 0.04 against 0.03 for bicubic. Even two real X-band images of one place
correlated at only 0.09. This experiment changes the target, not the question. The target here is
Capella blurred and averaged over many looks to 5, 7.5 or 10 m, so it carries little speckle and only the
detail Sentinel-1 might recover.

## Output

1. A dataset of Sentinel-1 chips paired with degraded Capella targets at four resolutions, split by site.
2. Baseline and repeat-image ceiling scores at each target resolution on the validation and test sites.
3. A GPU-ready training script and run queue. Training waits for a coordinator slot (phase 2).
4. After phase 2: the "reasonable resolution", the finest target resolution where the model meets the
   rule below, or a finding that none does.

## Targets

- **Source.** Capella GEO, from the public bucket, read as windows from the coarsest overview whose pixels
  are no larger than 1.25 m (the native image when no overview qualifies). Digital numbers times the
  product's `scale_factor`, squared, give sigma nought intensity. The overviews aren't plain block means
  (a probe found correlation 0.97 with nearest-neighbor decimation and 0.94 with a 2 × 2 mean), so the point spread function (PSF)
  budget below counts one overview pixel as a box.
- **Base grid.** Intensity averaged onto a 1.25 m grid in the local UTM (Universal Transverse Mercator) zone, aligned to Sentinel-1's
  10 m grid. Each 10 m Sentinel-1 pixel holds exactly 8 × 8 base pixels.
- **Degradation.** A Gaussian PSF applied to linear intensity on the base grid,
  as a normalized convolution so that no-data edges don't darken. Then a 2 × 2 mean onto the 2.5 m output
  grid (4 × 4 output pixels per Sentinel-1 pixel). The build solves the Gaussian width per Capella item so that
  the full chain has the target full width at half maximum (FWHM): native resolution (Gaussian, FWHM equal
  to the product's ground-range and azimuth resolution), overview pixel box, 1.25 m box, the added
  Gaussian and the 2.5 m output box. The script computes the chain numerically in one dimension.
- **Resolutions.** Effective resolution R = 5, 7.5 and 10 m on the 2.5 m output grid. A stretch arm at
  R = 2.5 m adds no Gaussian: the base 2 × 2 mean only, whose FWHM is whatever the chain gives (recorded).
- **Recorded per chip.** The chain FWHM for the finest and coarsest native direction, the nominal
  equivalent number of looks (kernel area over native resolution cell area), and an empirical check: the
  FWHM of the target's speckle autocorrelation, divided by √2, on four chips per item. Real texture widens
  that autocorrelation, so the empirical value is an upper bound.
- **Polarization.** Most Capella items are HH and a few VV; Sentinel-1 gives VV and VH. The Capella
  polarization is a recorded field. The target is Capella's one channel as it is. Results get split by
  Capella polarization.

## Inputs

Planetary Computer `sentinel-1-rtc`: terrain-corrected gamma nought, 10 m, VV and VH, in dB.

| Channel | Content |
|---|---|
| 1–2 | VV and VH from the single pass nearest Capella in time |
| 3–4 | VV and VH mean intensity (then dB) of all passes on the same relative orbit and direction within ±30 days of Capella, including the nearest |
| 5 | DEM slope in degrees, from Copernicus DEM GLO-30, bilinear to 10 m |

A pass is all items of one datatake on one date; adjacent slices get mosaicked. The single pass must lie
within ±12 days of Capella; the selection prefers ±6 days. The multi-date mean needs at least two passes.
The RTC items carry no incidence-angle layer, so the slope stands in for local incidence. Capella's
incidence angle is a recorded field.

## Sites and items

Candidates come from experiment 6's `scenes.csv` and `s1_iw_match.csv`: Capella GEO items with a
Sentinel-1 pass with VV and VH within ±12 days (2,168 items).

1. Keep items with WorldCover land at least 70% and water at most 30% of the footprint.
2. Keep items where at least half of a 4 km box at the center has slope ≤ 10° on the 30 m DEM.
3. One item per experiment 6 site: Sentinel-1 gap ≤ 6 days first, then the smallest gap, then the finest
   Capella resolution.
4. Variety: each item gets a dominant cover from its WorldCover shares (built, crop, tree, grass or
   bare and other). Built-dominated sites may fill at most 45% of the selection; the rest keep their order.
5. Up to 40 chips per item, drawn at random (seed 0) from the valid chip positions.

## Chips

- A chip is 64 × 64 Sentinel-1 pixels (640 m) with a 256 × 256 target at 2.5 m. Chips don't overlap within
  an item. A chip position counts as valid when the Capella footprint (from a coarse overview) covers it
  with an 80 m margin.
- **Masks.** A Sentinel-1 pixel is valid when VV and VH are finite in both the single and the multi-date
  input, ESA WorldCover 2021 isn't water (80) or no data, the slope is ≤ 10°, and all 16 target pixels
  inside it are valid. A target pixel is valid when the normalized convolution weight is ≥ 0.9.
- **Layover and shadow.** Terrain layover needs a slope facing the sensor that is steeper than the
  incidence angle (above 20° for both sensors), and shadow needs a back slope near 90° minus incidence.
  The 10° slope limit excludes both. A DEM can't mask building layover. The metadata records built cover so
  that results can show it separately.
- A chip enters the dataset when at least 80% of its Sentinel-1 pixels are valid and it passes alignment.

## Alignment (experiment 8's two steps, per chip)

1. **Coarse, per item.** Capella averaged to 10 m over the chips' bounding box plus 1.5 km on each side,
   against the multi-date VV over the bounding box, both in dB and high-pass filtered (minus a 5 × 5
   mean). Normalized cross-correlation (`match_template`) gives the shift. It applies when the peak stands
   at least 6 standard deviations above the correlation surface. An item whose coarse step fails gets
   dropped, because experiment 8 found offsets of 10–40 m and up to 270 m, beyond the fine search.
2. **Fine, per chip.** Search ±10 m in 2.5 m steps around the coarse shift. Each candidate averages the
   shifted base intensity to 10 m, converts to dB, high-pass filters and correlates with the multi-date VV
   on valid pixels. The item's fine shift is the median of the chip optima among chips whose best
   correlation reaches 0.15 (zero when fewer than three chips do). A chip keeps its own optimum when that
   beats its correlation at the item shift by at least 0.03; otherwise it takes the item shift.
3. **Rejection.** A chip whose correlation at its final shift is below 0.10 fails alignment. The metadata
   keeps every chip's correlations and shifts.

## Splits

Sites, never chips, get split. Sites group into regional blocks of 3° latitude by 3° longitude. Blocks that
hold any experiment 8 scene go to test, so that the linear-feature test below runs on unseen ground. The
other blocks, shuffled with seed 0, go to whichever split sits furthest below its share of chips:
70% train, 15% validation, 15% test.

## Metrics (validation and test sites, against the degraded Capella target)

All scores use the central 56 × 56 Sentinel-1 pixels (224 × 224 output pixels) of each chip, to drop border
effects, and valid pixels only. The comparison uses dB, each image standardized per chip (mean 0, standard
deviation 1 over valid pixels), as in experiment 1. Standardization removes the X-band to C-band
calibration difference that a baseline can't know.

- **PSNR**: 20 log₁₀(span / RMSE), with span the target's 1st to 99th percentile range.
- **SSIM**: scikit-image, data range equal to that span, both images clipped to it.
- **Detail correlation** (experiment 1): correlation of image minus its 10 m block mean.
- **High-pass correlation** (hp r): correlation of image minus its Gaussian blur of 20 m FWHM, the scale
  Sentinel-1 can't resolve.
- **Edge F1**: edges are the top 10% of Sobel magnitude after a 1-pixel Gaussian smoothing; a predicted
  edge pixel matches when a target edge lies within 2 output pixels (5 m), and the reverse for recall.
- **Gradient similarity**: the mean gradient magnitude similarity of Xue et al. (2014), constant 0.0026
  on standardized images rescaled to the target span.
- **Linear-feature visibility** (experiment 8): experiment 8's matched-null contrast test, run on the
  output at the 803 accepted segments of its 22 headline scenes, with the single-polarization threshold
  |z| > 2. Experiment 8's own rate for the multi-date VV mean, 39%, is the reference; the same test on the
  degraded Capella at R shows what the target holds.
- **Hallucination checks.** (a) Flat-area detail: on target pixels in the flattest 20% of each chip (local
  5 × 5 standard deviation of the target), the ratio of output to target high-pass energy. Values far above
  one mean the output invents detail. (b) Shuffled input: the model gets another test chip's input and is
  scored against the original target; hp r and SSIM should collapse to their floor. The same shuffle on the
  baselines gives that floor.

Each score gets a median over chips within a site, then a median over sites. Intervals come from 1,000
bootstrap draws of sites.

## Baselines

All upsampled to the 2.5 m grid by bicubic interpolation in dB.

1. Bicubic of single-date VV.
2. Bicubic of the multi-date mean VV.
3. Lee filter (7 × 7, 4.4 looks) of single-date VV.
4. Experiment 4's VV+VH Noise2Noise despeckler with speckle-level conditioning (`r2_n2n_big_cond_vvvh`),
   VV output, on the single date.
5. The same despeckler on the multi-date mean, where its conditioning channel reports the higher look count.
6. Experiment 2's ESRT model (`esrt_v2_nomask_s0`, IW to 5 m Stripmap, ×2) on single-date VV, then bicubic ×2.

The best baseline for each metric and resolution is the one with the highest validation median. It's
chosen on validation sites and then compared on test sites.

## Repeat-image ceiling

Pairs of Capella images of the same place with the same polarization, look direction and incidence
within 5°, from experiment 7's `r2_pairs.csv` (65 pairs). Both images get the same degradation to each R.
Image B is co-registered to A by phase correlation of the 2.5 m images and scored as a prediction of A with
the metrics above, on land chips with the same masks. The ceiling shows the best score a perfect model
could expect, given speckle that survives multilooking, small changes over days and residual alignment
error. Experiment 1's ceiling (correlation 0.09) used single-look 2.5 m images.

## Model (phase 2, GPU)

- Architecture: experiment 2's ESRT (32 features), five input channels, one output channel, ×4 to the
  2.5 m grid. A residual CNN (experiment 2's `CNN`, 16 blocks, ×4) is the fallback if ESRT runs out of
  memory or diverges.
- Normalization: each channel scaled to [0, 1] with fixed bounds from the training chips' 0.1 and 99.9
  percentiles (`norm.json`).
- Loss: 0.16 × L1 + 0.84 × (1 − SSIM) on valid target pixels (Zhao et al., 2017). No GAN in the main arm;
  experiment 3 found that an adversarial term added texture and lowered structural accuracy.
- Training: random 48 × 48 crops (192 × 192 targets), rotations and flips, batch 16, AdamW at 2e-4 with
  cosine decay, 12,000 steps, validation SSIM every 1,000 steps, best and last checkpoints. One model per
  target resolution. Seed 0 first; seeds 1 and 2 for the winning resolution and the next coarser one.

## Decision rule ("reasonable resolution"), fixed now

At resolution R, the model beats the baselines when, on test sites:

1. SSIM: the site median of model minus the best baseline is at least +0.02, and the lower end of its 95%
   site-bootstrap interval is above zero;
2. hp r: the same difference is at least +0.05, with the lower end above zero;
3. edge F1: the difference is not negative;
4. hallucination: the flat-area energy ratio has a median of at most 1.5, and with shuffled input the
   model's median hp r is at most 0.05.

The reasonable resolution is the finest R in {5, 7.5, 10 m} that meets all four. The 2.5 m arm counts only
if it meets them too. If no R does, that is the result. With three seeds, the rule applies to the mean
of the seeds and must hold for each seed. Secondary: the share of the gap from the best baseline to the
repeat ceiling that the model closes.

## Limits known in advance

- X-band HH and C-band VV/VH respond differently to vegetation, soil moisture and structures. The model has
  to learn a band translation as well as sharpening, and some Capella detail has no counterpart at C-band
  (experiment 8: Sentinel-1 misses many features that Capella shows clearly).
- X-band and C-band image tall objects with different layover, and the two satellites look from different
  directions. Buildings and tree lines therefore sit in different places in the input and the target.
- Dates differ by up to 12 days, so crops, moisture and harvest can change.
- Capella GEO geolocation errors (tens of meters) need the alignment step; residual errors of a few meters
  blur what the model can learn at 5 m.
- Experiment 6's site list favors cities, ports and airports. The built-cover cap limits but doesn't
  remove that bias.

## Budget

CPU for phase 1. Capella windows of about 0.8 km at ≤ 1.25 m pixels cost a few MB per chip. 350 items of 40
chips should read about 20 GB, under the 60 GB cap. Stored chips take about 0.6 MB each. GPU in phase 2:
see `run_queue.sh`.

## Files

`select_items.py`, `build_chips.py`, `make_splits.py`, `common.py` (grids, masks, metrics, baselines),
`baselines.py`, `ceiling.py`, `panels.py`, `train.py`, `eval.py`, `run_queue.sh`, `phase1.md`.

## Amendment 1 (2026-10-31, after a four-item pilot and before the full build)

The pilot built 160 chips from four items and scored nothing. Its outputs moved to `data/pilot/` and
the full build rebuilt those items.

- **Coarse alignment** passed on all four items, with peak z from 19 to 57 and shifts of 10 to 30 m.
- **Fine alignment.** The median chip correlation at the final shift was 0.16. The 0.10 rule rejected 28%
  of chips: 14 of 16 bare chips, 13 of 51 tree chips and 9 of 68 built chips. Low correlation there
  mostly reflects how little X-band and C-band texture agree on bare and wooded ground, not a known
  misalignment. The rule stays as written, so training and the headline scores use only accepted chips.
  The build now also stores the rejected chips (status `align_reject`), so that a sensitivity check can
  score the baselines and the model on them.
- **Resolution check.** The PSF budget hit each target width within 0.02 m. The empirical autocorrelation
  widths were 5.3–7.0 m at R = 5 m and 10.4–11.8 m at R = 10 m, as expected for an upper bound. The
  build computes them on the 1.25 m grid before the 2.5 m mean, so they leave out that last box.
- **Nominal looks** at R = 5 m ranged from 22 (Stripmap, 1.5 m resolution) to 118 (Spotlight).

## Amendment 2 (2026-10-31, during the build, before any score)

The first eight items read 2.25 GB, and one Stripmap item alone read 0.75 GB, because chips spread over its
whole footprint and the Sentinel-1, DEM and coarse-alignment reads cover the chips' bounding box. At that
rate the 270 items would pass the 60 GB cap. Items with more than 40 valid chip positions now draw their
chips from the 6.4 km × 6.4 km window of the chip lattice that holds the most valid positions. The eight
items already built keep their chips.

## Amendment 3 (2026-10-31, after the baselines and before any model run)

- **Second ceiling.** The same-geometry repeat ceiling came out high (SSIM 0.72 at 5 m), while Sentinel-1 always
  views the ground from a different geometry than Capella. `ceiling.py --cross` adds a different-geometry
  ceiling: 40 Capella pairs at separate sites, same polarization, 0.5–12 days apart, from opposite orbit
  directions or with incidence differing by 10° or more. It's reported next to the first ceiling. It doesn't
  change the decision rule.
- **Flag, no change.** 26 items had fewer than three chips with correlation ≥ 0.15, so their fine shift stayed
  at zero and couldn't be checked. `chips.csv` marks their 121 chips with `item_fine_verified = False`. Training
  keeps them as the design says. A run without them is a sensitivity check.
- **Rejected-chip check.** `baselines.py --rejects` and `eval.py --rejects` score only rejected chips that have at
  least 80% valid pixels.
