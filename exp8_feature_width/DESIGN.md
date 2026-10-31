# Experiment 8: how narrow a linear feature can Sentinel-1 show?

Status: done (2026-10-28); results in `results.md`. This file fixed the site rule, the measurements and the detection rule
before any measurement ran. Later changes got a dated note at the end, not silent edits.

## Question

Field-boundary maps from Sentinel-1 depend on the narrow things between fields: hedgerows and tree lines,
roads and farm tracks, ditches and canals, and grass strips along field edges. Sentinel-1 IW products have
10 m pixels and about 20 m real resolution. How wide must such a feature be before Sentinel-1 shows it, and
how does that width depend on the feature's type and on its radar contrast?

Capella X-band imagery at sub-meter resolution serves as the truth for each feature's width and contrast.
The features are static, so the dates of the two sensors matter little, but Sentinel-1 passes come from
within ±12 days of Capella where possible and always from the same season.

## Output

A detection curve: the share of features that Sentinel-1 shows, by Capella-measured width, with the width
at 50% and 90% detection. Split by feature type and by Capella contrast, for single-date Sentinel-1 and for
a multi-date mean, and a two-dimensional view of width by contrast.

## Sites (rule fixed before measuring)

Candidates come from experiment 6's `scenes.csv`: Capella GEO items with cropland present (ESA WorldCover
cropland at least 10% and FTW Global field share at least 10%) and a Sentinel-1 IW pass with VV and VH within
±12 days. That gives 190 items at 59 sites.

1. One item per site: the finest range resolution, then the smallest Capella to Sentinel-1 gap.
2. For each item, download OpenStreetMap lines (see below) for the footprint, limited to a 12 km box around
   its center. Place a 4 km × 4 km analysis window, on a 1 km grid of positions, where at least 60% of the
   window lies inside the footprint and the length of candidate features is greatest.
3. Drop windows with less than 15 km of candidate features.
4. Rank the rest by candidate length and keep at most 4 sites per country, up to 32 scenes.

The chosen scenes, windows and Sentinel-1 items go to `scenes_selected.csv` and `windows.geojson` before any
width is measured.

## Data

- **Capella:** the GEO product (HH for most items, VV for a few), read as a window from the public cloud
  GeoTIFF. Amplitude averaged to a 1 m grid in the Sentinel-1 UTM zone, then squared to intensity.
- **Sentinel-1:** Microsoft Planetary Computer `sentinel-1-rtc` (terrain-corrected gamma0, 10 m, VV and
  VH). No SAR processing here, so no south-up trap.
  - *Single date:* the pass nearest in time to Capella that covers the window.
  - *Multi-date mean:* the mean intensity of all passes from the same relative orbit as the single date,
    within ±30 days of Capella. The same track keeps the viewing geometry fixed. This is the realistic best
    case for a seasonal composite.
  - No speckle filter. The profile averaging along each 200 m segment (about 20 pixels) does the smoothing.
- **OpenStreetMap (OSM):** Overpass API, `out geom`, per scene. Candidate types:

| Type | OSM tags |
|---|---|
| tree_line | `barrier=hedge`, `natural=tree_row` |
| road | `highway` = motorway, trunk, primary, secondary, tertiary (and their links), unclassified, residential, service |
| track | `highway` = track, path, footway, cycleway, bridleway |
| ditch_canal | `waterway` = ditch, drain, canal |
| field_edge | outlines of `landuse` = farmland, meadow, orchard, vineyard, grass, with no other mapped line within 30 m |

  The script drops ways tagged as a bridge, tunnel, culvert or covered. Other lines (railways, rivers, streams,
  all other highways) don't become candidates but count as neighbors in the parallel-feature check.

## Segments

Each candidate way, projected to UTM, is cut into 200 m segments; a remainder under 100 m is dropped.
A segment is dropped when:

- it lies within 150 m of the window edge (the null profiles need room);
- another mapped line within 30° of its direction lies within 30 m along more than 25% of its length
  (parallel neighbors merge at 10 m);
- more than 40% of its length lies within 20 m of a crossing line. Those stretches are masked in any case.

Along each segment, profiles are sampled perpendicular to the line, using the local direction of the line,
every 1 m along it and from −150 m to +150 m across it (positive to the left of the direction of travel).

## Alignment

Capella geolocation can be off by tens of meters. Each scene's Capella grid is shifted onto Sentinel-1 by
correlation: Capella intensity averaged to 10 m, in dB and high-pass filtered (minus a 5 × 5 pixel mean),
against the multi-date mean VV the same way. Search ±80 m at 10 m steps, then ±10 m at 2 m steps around the
best match. Scenes whose best correlation is below 0.15 are excluded from the headline numbers and listed.
OpenStreetMap is compared with the aligned Capella image, not the other way around.

## Width and contrast on Capella

1. Twenty-meter blocks along the segment give 10 block profiles (mean intensity, then dB). Their mean gives
   the segment profile.
2. The feature center is the strongest extremum, bright or dark, of the segment profile (smoothed over 2 m)
   within ±15 m of the OSM line. Each block profile is shifted (up to ±4 m) to best match the segment profile,
   and the aligned blocks are averaged again.
3. Width is the full width at half contrast: from the center level P, walk outwards until the profile
   crosses (P + B) / 2, with linear interpolation. B is the mean of the left and right background levels,
   each the median of the profile from 5 to 20 m beyond that side's crossing. Three iterations.
4. Contrast is P − B in dB. Bright features have positive contrast and dark ones negative.

A segment counts as a clear feature on Capella only when all of the following hold:

- the Capella matched-null z has |z| ≥ 4. This is the Sentinel-1 test below with its windows scaled to
  Capella's 1 m cell: center |d| ≤ max(0.5 m, width / 2), flanks from max(1.5 m, width / 2 + 1 m), each
  15 m wide, and null lines placed by the same offset formula;
- |P − B| ≥ 1 dB;
- the two background levels differ by less than half of |P − B| (this rejects step edges between two
  fields that aren't linear features);
- both crossings exist within 25 m of the center, and the width lies between 0.75 m and 40 m;
- at least 6 of 10 block profiles have an extremum of the same sign within max(3 m, width / 2) of the
  center.

Everything else is rejected as ambiguous, with the reason recorded.

## Detection rule on Sentinel-1 (fixed before measuring)

For each kept segment, on each Sentinel-1 product (VV and VH, single date and multi-date mean):

1. Sample the profile at the Capella center line after the scene alignment. Average along the segment in
   linear intensity (masked stretches excluded), then convert to dB.
2. Center window: |d| ≤ max(5 m, width / 2). Flank windows on both sides: from f₀ = max(15 m, width / 2 + 10 m)
   to f₀ + 15 m. Contrast C = dB(mean center) − mean of the two flanks in dB.
3. Null: the same statistic on 16 lines parallel to the feature, 8 per side, at offsets
   o = f₀ + 15 m + width / 2 + 10 m + 5k m for k = 0 … 7. For a narrow feature that means 40 to 75 m away.
   Each null line's windows then stay at least 10 m clear of the feature.
4. z = (C − mean of null) / standard deviation of null.
5. **Detected** in one polarization when |z| > 2. The headline uses either polarization with |z| > 2.24 in
   VV or in VH (nominal two-sided false-alarm rate near 5% across the two tests).

Applying the same rule to each null line in turn against
the other 15 measures the false-alarm rate directly. That rate sets the floor of the curve. Sign agreement with Capella gets reported separately as a
secondary result, because X-band HH and C-band VV need not agree in sign.

## Analysis plan

- Width bins (m): 1–2, 2–3, 3–5, 5–8, 8–12, 12–20, 20–30, 30–40. Detection rate per bin with 95% intervals
  from 1,000 bootstrap draws of scenes.
- Curve fit: p(w) = f + (1 − f) / (1 + exp(−(a + b log₂ w))), with f fixed at the measured false-alarm rate,
  fitted by maximum likelihood. Widths at p = 0.5 and p = 0.9, each with a scene-bootstrap interval. A width
  beyond the widest measured feature gets reported as "beyond the measured range".
- Splits: feature type; |Capella contrast| bins (1–3, 3–6, 6–10, over 10 dB); bright against dark;
  single-date against multi-date; VV, VH and either; the feature's angle to the Sentinel-1 flight direction
  (within 30° of it, or more than 60° from it).
- Two-dimensional view: detection rate in width × |contrast| cells, with counts, and a logistic fit with both
  log width and |contrast|.
- Spot check: about 60 panels stratified by type, width and outcome, in `spotcheck/`, with
  `spotcheck.csv` holding an empty `verdict` column for a person to fill in.

## Limits known in advance

- OpenStreetMap lines can be off by several meters. The ±15 m search on Capella covers that, but a
  wrongly mapped feature gets rejected as ambiguous, which removes some real ones.
- Capella is mostly HH X-band, Sentinel-1 VV/VH C-band. Contrast at X-band doesn't carry over one-to-one.
  The test asks whether C-band shows a feature that X-band proves exists at a known width.
- Features running along the radar look direction and across it image differently (shadow, layover and
  double bounce for tree lines and banks), and Sentinel-1 range and azimuth resolution differ.
- Speckle limits single dates. The multi-date mean reduces it but can't recover detail below the
  resolution.
- Capella Stripmap resolution near 1 m makes widths under 2 m uncertain; the 1–2 m bin is mostly a
  "narrower than 2 m" class.

## Budget

CPU only. Capella windows (4 km plus a 150 m margin) cost about 20–80 MB each, so 32 scenes stay under
3 GB. Sentinel-1 windows add about 1 MB per image. All downloads go to `data/` and `cache/`.

## Files

`select_scenes.py` (sites, windows, OSM, Sentinel-1 items), `measure.py` (Capella read, alignment,
profiles, widths, detection), `analyze.py` (curves, bootstrap, figures), `spotcheck.py` (panels),
`results.md`.

## Amendment 1 (2026-10-28, after a one-scene pilot and before the main measurement)

The pilot scene (site 308, Germany) exposed three problems. Its results were deleted and are not used.

- **Line source.** The public Overpass servers were at capacity all afternoon. The lines come instead from
  Overture Maps release 2026-10-21.0, whose transportation and base themes derive from OpenStreetMap:
  road segments (the same highway classes), `water` lines of class ditch, drain and canal, `infrastructure`
  hedges, `land` tree rows and `land_use` outlines. Overture flags bridges and tunnels per stretch of road.
  Those stretches are cut out.
- **Window placement.** Counting all candidate lines put the pilot window in a city center, where footways
  and streets dominate. The window now goes where the rural lines are longest: hedges and tree rows, farm
  tracks (class `track` only), ditches and canals, and outlines of farmland, meadow, orchard and vineyard.
  Every candidate type still gets measured inside the window. The same 15 km floor and country cap apply,
  and 30 scenes result: two items had no GeoTIFF in the bucket.
- **Alignment.** In the pilot, Capella matched the OSM lines without any shift, while the correlation
  with Sentinel-1 reached only r = 0.16 and proposed a 10 m shift that looked worse. The correlation shift
  now applies only when its correlation reaches 0.25 and beats the unshifted one by at least 0.05.
  Otherwise Capella keeps its own geolocation. As an independent check, the accepted segments' Capella
  centers relative to the OSM lines give a least-squares scene offset. Scenes where that offset exceeds 10 m
  leave the headline numbers. This replaces the r < 0.15 exclusion.
- **Parallel-neighbor test.** Overture splits lines at every junction, so a segment's own continuation
  counted as a parallel neighbor. The test now looks along the normal, 2 to 30 m out on each side, for
  another line running within 30° of the segment.

## Amendment 2 (2026-10-28, after a second pilot scene and before the main measurement)

The second pilot scene (site 299, hilly Steiermark) showed that Capella GEO can sit hundreds of meters
from its true position, while OpenStreetMap and Sentinel-1 agree with each other. A ±80 m search can't find
such offsets. Alignment now runs in two steps, and the Capella window gets read after the first:

1. **Coarse.** The multi-date mean VV over the window plus 1.5 km on each side, and Capella averaged to 10 m
   over the same area, both in dB and high-pass filtered. Normalized cross-correlation locates the
   Sentinel-1 window inside the Capella area. The shift applies when the peak stands at least 6 standard
   deviations above the rest of the correlation surface. Capella then gets read at 1 m from the shifted
   position, so its array sits in the Sentinel-1 frame.
2. **Fine.** The earlier search, now ±10 m around the coarse result in 2 m steps. It applies when the coarse
   step passed and it raises the correlation.

The OSM offset check from amendment 1 still decides which scenes enter the headline numbers. Terrain
distorts Capella GEO in hilly scenes beyond what one shift can fix. Those scenes show up as large OSM
offsets or as very few accepted segments.

## Note on a code fix (2026-10-28)

The first full run accepted only 162 segments. The block alignment in the width step had a bug: its
correlation needed 500 valid samples, while a 40 m profile window holds 81, so every block got shifted by
the same 4 m. That moved the measured center off the feature. The fix changes no rule. The run was repeated
from scratch, and no Sentinel-1 results had been looked at before the fix.

## Amendment 3 (2026-10-28, before any Sentinel-1 result was looked at)

After the fixed run, the accepted segments' Capella centers sat 5–14 m from the OSM lines in a consistent
direction in most scenes, with a scatter of only 4–8 m. The correlation alignment leans on tall objects,
trees and buildings, whose layover displaces them differently at X-band and at C-band. Roads, tracks and
ditches lie on the ground, where both the Sentinel-1 terrain-corrected product and OpenStreetMap should
be accurate to a few meters.

Each scene therefore gets measured twice. The first pass measures Capella widths and centers only. A
least-squares fit of the centers of accepted roads, tracks and ditches, with outliers beyond 2.5 times the
residual spread dropped, gives a scene shift. The second pass applies that shift on top of the correlation
shift and runs everything, Sentinel-1 included. Scenes with fewer than 8 accepted ground features can't
be checked this way and leave the headline numbers. This replaces the 10 m OSM-offset exclusion. Tree lines
keep their X-band layover offset relative to the hedge axis, and the results discuss it.

## Amendment 4 (2026-10-31, VH against VV, written before any changed rule was scored)

The reviewer's spot-check notes favored the VH multi-date mean more often than the automatic test did, so
this round asks whether the test undervalues VH. `vh_diagnose.py` ran first and changed no rule. It used
the stored arrays and profiles only, not the verdicts (`vh_diagnose.json`).

**Diagnosis.**

- **No bug.** VV and VH come from the same items and dates (one list per scene in `measure.py`, 2 to 10
  dates, median 5), on the same 10 m grid, with identical masks of missing pixels. VH sits 5.1–8.1 dB below VV (median
  6.5 dB) and lies below VV in 99.8% or more of pixels in every scene, so the bands aren't swapped. The
  profile order in `profiles.npz` matches the labels in `spotcheck.py`.
- **Units.** The contrast is a difference in dB and z divides it by the same polarization's null spread.
  Shifting VH by 7 dB changes no z (largest change 5e-14). The darker VH level can't affect the test.
- **Noise floor.** No multi-date VH background and under 2% of single-date backgrounds fall below −22 dB.
  VH detection doesn't drop at the darkest levels. The floor plays no measurable role here.
- **Null spread and contrast.** VH has a 5–8% smaller null spread than VV and a 5% smaller line contrast.
  The null means sit near 0 dB in both (median absolute 0.13 dB), so no thermal-noise banding biases the
  null. Measured false-alarm rates: 5.6% VH, 5.3% VV (multi-date).
- **Sign.** The test is two-sided in both polarizations. VH detections are bright and dark in equal shares.
- **Position.** In both polarizations the strongest deviation within ±25 m lies a median 11–12 m from the
  Capella center. This affects VV and VH alike.
- **The width gap is a curve-shape effect.** Multi-date VV detects 39.2% and VH 36.0%, a 3-point gap. Paired
  on the same segments, 169 are VV-only and 143 VH-only (exact McNemar p = 0.16). On single dates VH leads,
  28.4% against 26.9%. The curves are so flat (slope 0.31–0.34 per doubling) that a 3-point gap moves the
  50% width from 15.9 m to 24.9 m. Both intervals reach beyond 40 m.
- **What a viewer sees.** Under the panels' 2–98 percentile stretch, the line contrast covers 8–9% of the
  displayed range in both polarizations. A step between the two flanks covers 16% in both, and spans
  2.7 against 2.4 null spreads. The pre-set statistic cancels such steps by design.

**Changes.** None of them treats VH differently from VV.

1. The polarization comparison uses the per-polarization standardized contrast in dB: center minus the mean
   of the two flanks, each window averaged in dB, divided by that polarization's own null spread, with the
   same thresholds (|z| > 2 for one polarization, > 2.24 for VV or VH). This equals the `rescore.py` dB
   variant. The diagnosis predicts almost no change, because the pre-set z already standardizes within each
   polarization. Both rules get reported.
2. The primary comparison is paired: the share detected by VH only, VV only, both and neither on the same
   segments, an exact McNemar test, and a scene-bootstrap interval for the VH − VV difference in detection
   share. Widths at 50% and 90% stay, with scene-bootstrap intervals, but the flat curves make them a weak
   basis for ranking polarizations.
3. A secondary statistic, labeled as such, tests the step: left flank minus right flank in dB, standardized
   by the same statistic on the 16 null lines. "Line or step" counts when either |z| > 2.24 in one
   polarization, or > 2.49 across both statistics and both polarizations. Measured false-alarm rates get
   reported. This asks a different question, whether a boundary shows, not whether a linear feature shows,
   and it doesn't replace the headline.
4. Only after scoring, the 60 spot-check verdicts serve as a small validation set: agreement of each rule
   and polarization with the `both` against `capella_only` verdicts (`unsure` and `no_feature` excluded), and whether the
   panels whose notes favor VH are ones the amended VH rule detects.
