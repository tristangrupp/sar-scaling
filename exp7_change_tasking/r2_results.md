# Experiment 7, round 2: how small a change can Sentinel-1 catch?

Run 2026-10-28, CPU only. Scripts: `r2_pairs.py`, `r2_run.py`, `r2_calib.py`, `r2_analyze.py`, `r2_sensitivity.py`, `r2_spotcheck.py`. Figures: `r2_curve.png`, `r2_curve_splits.png`. Numbers: `r2_curve.csv`, `r2_summary.json`, `r2_sensitivity.json`, `r2_pair_results.csv`.

## Result

Beyond what chance flags would catch, Sentinel-1 VV catches about half of the Capella change patches near 900 m², never reaches 90% at any patch size in this sample, and catches almost none under 16 m².

- **50% at about 900 m²** (95% interval 200 to 7,600 m², sites resampled). The sensitivity checks below move this point between 270 and 1,600 m².
- **90% isn't reached.** The fitted curve passes 90% near 94,000 m², twice the largest patch observed (46,000 m²). Binned detection levels off at 50% to 60% between 256 and 2,048 m², and only 8 sites have patches larger than 2,048 m².
- **Almost none under 16 m².** At 8 to 16 m², detection beyond chance is 4% (interval −1% to 12%); at 16 to 32 m², it's 10%.
- **False alarms: 7.0% of Capella-stable land pixels** carry a VV flag (interval 5.6% to 9.0%). That equals the overall VV flag rate (median 7.3% per pair), so at the pixel level the flags hardly depend on Capella change. Flag regions of 20 pixels (0.2 ha) or more do better: 33% fall on Capella change against 22% for flags shifted at random, and 44 of 55 pairs beat chance.

In the requested form: *Sentinel-1 catches about half of the changes over about 900 m², doesn't reach 90% at any size observed here, and catches almost none under 16 m².*

### Why the curve is chance-corrected

The raw rule counts a patch as caught when any VV flag lies within 10 m of it. With 7% of pixels flagged, unrelated flags land near many patches: shifting the flag map at random still "catches" 27% of the smallest patches and 58% of patches at 2,048 to 4,096 m². The raw catch rate passes 90% in that bin, but most of those catches would happen anyway. The corrected curve fits P(caught) = c + (1 − c) × s(area). Here c is each patch's chance-catch rate, measured with 20 random shifts of the flag map, and s is a logistic curve in log area. The curve s is the share Sentinel-1 catches because it saw the change. Panel A of `r2_curve.png` shows the raw rates, and panel B the corrected curve.

| Patch area (m²) | Patches | Sites | Raw catch | Chance | Beyond chance (95% interval) |
|---|---|---|---|---|---|
| 8–16 | 51,948 | 32 | 32% | 27% | 4% (−1 to 12) |
| 16–32 | 20,495 | 31 | 36% | 28% | 10% (4 to 21) |
| 32–64 | 6,873 | 31 | 43% | 30% | 18% (9 to 29) |
| 64–128 | 2,314 | 29 | 49% | 33% | 24% (13 to 37) |
| 128–256 | 781 | 29 | 57% | 38% | 31% (9 to 52) |
| 256–512 | 354 | 25 | 70% | 46% | 52% (27 to 75) |
| 512–1,024 | 156 | 18 | 77% | 56% | 58% (30 to 85) |
| 1,024–2,048 | 76 | 11 | 84% | 65% | 55% (23 to 80) |
| 2,048–4,096 | 22 | 6 | 91% | 58% | 78% (39 to 100) |
| 4,096–16,384 | 14 | 5 | 93% | 87% | 17% (−82 to 100) |

The stricter rule, which needs half the patch under flagged pixels, never reaches 50% in the data. Its fit passes 50% near 34,000 m².

### Polarization, land cover, and sign

- **VH is weaker:** 50% at about 5,200 m² (interval 660 to 270,000), with an 8.5% false alarm rate. Flagging VV *or* VH moves the 50% point to about 660 m² (120 to 14,000), but doubles the false alarm rate to 14.2%.
- **Sign:** among caught patches, Sentinel-1 changes in the same direction as Capella 59% of the time. Brightening patches reach 50% near 800 m² (190 to 4,400), darkening near 1,200 m² (190 to 340,000).
- **Land cover** (`r2_curve_splits.png`): every class has wide intervals. Urban reaches 50% near 700 m² (110 to 31,000) and bare ground and mines near 2,600 m² (60 to 44,000). Farmland and water edges reach 50% near 130 to 140 m², but their intervals run far past any observed size. Patches on vegetation and other land show no detection beyond chance (32% raw against 29% chance). The spot check suggests why: many of those patches are moisture or wind changes in forest canopy, not objects (`chip046`).

### Sensitivity

| Case | Pairs | Sites | Area at 50% (95% interval) | Beyond chance, 8–16 m² |
|---|---|---|---|---|
| Primary, as specified | 53 | 33 | 890 m² (200–7,600) | 4% |
| Drop 5 pairs with noisy Capella truth (spread over 2.5 dB) | 48 | 29 | 1,600 m² (170–30,000) | 8% |
| Every pair weighted equally | 53 | 33 | 270 m² (80–25,000) | 9% |
| Capella truth with 4 m shift tolerance | 53 | 33 | 290 m² (100–2,600) | 9% |
| Spotlight pairs, Capella gap 7 days or less | 37 | 23 | 5,700 m² (220–3,300,000) | 5% |
| Noisy truth and vegetation out | 47 | 28 | 820 m² (90–6,000) | 10% |

The noisy-truth cut came after seeing the data: three pairs (two near Los Angeles in late January 2025 and one in Bangladesh in December 2021) hold 42% of all patches, and almost their whole window lies near some patch. In every case the 90% point lies beyond the observed sizes, and detection under 16 m² stays at 10% or less.

## Statistical power

Partly. After failures, 33 sites remain, above the 30-site bar, so the small-patch end of the curve and the false alarm rate are well constrained. The large-patch end isn't: 116 patches at 14 sites exceed 1,024 m², and 40 patches at 8 sites exceed 2,048 m². So the 50% point carries a wide interval, and nothing here can say whether Sentinel-1 ever reaches 90%. Land-cover and sign splits lack the power to separate classes.

## Method

**Pairs** (`r2_pair_count.md`). Two Capella images of one site with the same polarization, look side and orbit direction, an incidence difference of 2° or less, and an azimuth within 10°. Both must lie inside two passes of one Sentinel-1 track, each pass within 6 days of its Capella image, so lasting change between the Capella dates also lies between the passes. The run kept one pair per site per calendar quarter, at most 5 per site: 65 pairs at 38 sites, almost all HH spotlight from repeat-track stacks. 56 pairs at 33 sites finished, and 53 of them hold at least one patch. Seven failed co-registration: the phase correlation returned shifts of 14 to 470 pixels, which the script refuses. Two had under 30% Capella overlap. The median Capella gap is 3 days, and the median Sentinel-1 interval 12 days.

**Window.** 2.56 km square at the centroid of the two Capella footprints, in the local Universal Transverse Mercator (UTM) zone. Sentinel-1 on a 10 m grid, Capella on a 2 m grid aligned to it (25 Capella pixels per Sentinel-1 pixel). Permanent water and open ocean (ESA WorldCover class 80 or no data) leave the analysis, because sea clutter changes with wind: in a first attempt, the coastal Miami window had a third of its pixels flagged.

**Capella change truth.** Windowed reads of the GEO COGs at the finest level coarser than or equal to native and finer than 0.8 m, intensity block-averaged to 2 m. Image 2 is aligned to image 1 by sub-pixel phase correlation (upsampled 20 times); the shift moves its geotransform before gridding, so no second interpolation occurs. The run masks footprint edges (10 m), steep-gradient pixels where a 1 m misregistration alone would change the image by 1.5 dB, the darkest 2% (noise floor) and the brightest 0.5% seen in both images (layover, sidelobes). It then removes the scene median log-ratio.

The calibration (`r2_calib.log`, five pilot pairs) showed that a plain 3 dB log-ratio rule fires on crater walls and pit benches that move by a pixel or two between dates. The truth therefore uses a shift-tolerant 3 dB rule. A pixel brightens when the after image exceeds the maximum of the before image within ±2 m by more than 3 dB; darkening uses the reverse. A moved edge finds its match, but an object that appears or disappears doesn't. Change patches are 8-connected groups of at least 2 pixels (8 m²) containing one pixel above 4 dB. A speckle simulation at the measured noise level (equivalent number of looks 20 to 90) gives no false patches per km² under this rule. The quietest pilot pair (Mojave, 3 days) also gives none, against 140 per km² for the plain rule. Patch area is pixel count times 4 m².

**Sentinel-1 detector.** The experiment 7 rule: Planetary Computer RTC gamma0, VV and VH, both passes on one track, bilinear to 10 m, dB, 3 × 3 mean, difference minus its median over land, flag above 3 dB, valid area eroded by one pixel. An integer shift of the Capella grid onto Sentinel-1 (search ±3 pixels, kept only for an interior optimum that raises correlation by 0.05 or more) moved 18 of 56 pairs, by at most 2 pixels (20 m) on each axis.

**Scoring** (fixed before scoring). A patch is caught (loose rule) when any flagged 10 m pixel lies within 10 m of it, or (strict rule) when at least half its area falls under flagged pixels. The chance rate is the loose rule after 20 random circular shifts of the flag map. False alarm rate: flagged share of 10 m pixels whose 25 Capella pixels are all valid and that lie more than 10 m from any patch. Region precision: flag regions with 30% or more of their pixels within 10 m of a patch. Intervals come from 1,000 site-bootstrap draws.

## Spot check

`r2_spotcheck/` holds 96 chips from 25 sites: 35 caught and 31 missed patches across five area bands (8 to 32 m² up to above 2,048 m²), and 30 Sentinel-1 false alarms on Capella-stable ground. Each chip shows the two Capella images, the Capella change with the patch outlined (purple frame, the truth) and the Sentinel-1 VV change with its flags (green frame, the detector). `r2_spotcheck/chips.csv` has an empty `verdict` column. Suggested verdicts: real object change, real surface change, not real (misregistration, layover, speckle), or can't tell. The headline should wait for that check; a quick look already found both real change (equipment rows moving at the Pilbara mine, lava or ash at Etna) and doubtful truth (canopy darkening in Tasmanian forest).

### Human verdicts (2026-10-30)

One reviewer judged all 96 chips. Most couldn't be judged at chip scale: 72 came back unsure. Of the 24 judged, 18 were real change and 6 were artifacts.

- Caught patches held up best: 14 real and 1 artifact among the 15 judged.
- Missed patches split 3 real and 4 artifacts, so some misses are truth errors, not Sentinel-1 failures.
- False alarms were almost all unsure (28 of 30), as expected on ground Capella calls stable.
- Bare ground and mines gave most of the judgments (13 real, 3 artifacts). The Pilbara mine alone gave 6 real and 3 artifacts.

The check confirms the Capella truth where a person can read it, mostly at mines and on large patches. It can't confirm the truth on vegetation, farmland or small urban patches. Treat the curve as reliable for bare and built change and as unverified elsewhere.

## Limits

- **Capella truth noise.** The truth is automatic. Speckle alone produces almost no patches under the rule, but surface changes (wet canopy, wind, tillage) and residual misregistration on slopes and buildings still do. Five pairs have a log-ratio spread above 2.5 dB and hold 48% of the patches. The edge mask removes a median 4% of each window, but up to 70% at the Mexico City site, so change at building edges is underrepresented. The spot check measures how much of the truth is real.
- **Timing gap.** The Sentinel-1 interval (median 12 days) is longer than the Capella interval (median 3 days). Change before the first Capella image or after the second counts as a Sentinel-1 false alarm, so the false alarm rate is an upper bound. Change that appears and leaves again before the Sentinel-1 "after" pass, such as a parked vehicle, is in the truth but impossible to catch. That caps the curve below 100%.
- **HH against VV.** 53 of 56 finished pairs are Capella HH; Sentinel-1 offers only VV and VH. Some HH changes don't show in either Sentinel-1 channel, which also lowers the curve.
- **Site imbalance.** Patches per site range from 1 to 15,964, and one site holds 19% of them. Weighting pairs equally moves the 50% point from 890 to 270 m². The sample favors Capella's repeat targets: Kilauea, Etna, mines, ports and cities.
- **Land cover** comes from ESA WorldCover 2021, which can be out of date at mines and construction sites.

## What looked wrong along the way

- `tasking_pairs.csv` puts both Sentinel-1 passes inside the X-band interval, the reverse of what a recall test needs; round 2 rebuilt the pairs.
- `s1_passes.csv` ends at each site's last X-band date, so it misses most passes after the second image. `r2_pairs.py` searched again with 15 days on each side.
- Process I/O counters on Windows didn't count network reads (3.5 MB reported for about 150 MB); downloads are now counted from the compressed TIFF blocks each read touches.

Downloads: about 9 GB of windowed reads in all. The final run read 5.8 GB; calibration and two stopped runs read about 3.5 GB, estimated from the same per-pair sizes. Kept arrays in `data/r2/`: 0.5 GB.
