# Experiment 7: change tasking

## Question

Sentinel-1 images most land every 6 to 12 days and can flag where backscatter changed. A
high-resolution X-band satellite (Capella, Umbra, ICEYE) can then image that spot at under 1 m.
Does this chain work? When Sentinel-1 flags change, does a later X-band image confirm it, and can a
person tell what changed?

The test runs in retrospect on open data. Nobody can task a satellite here, but the open X-band
archives hold sites that the providers imaged again and again. Where a site has an X-band image
before and after two Sentinel-1 passes on one track, the question becomes: do the Sentinel-1 flags
fall where the two X-band images differ?

## Hypothesis

Sentinel-1 flag regions fall on real change seen at X-band far more often than random regions do.
Moisture, crop growth and speckle cause most of the other flags, and X-band shows those as diffuse
tone change rather than a new or missing object.

## Data

- X-band: the Capella, Umbra, and ICEYE footprint caches from `exp0_overlap_atlas` (10,012
  collects). `inventory.py` joins collects whose centroids lie within 2 km into sites.
  `xband_collects.csv` lists every collect; `repeat_sites.csv` lists the 597 sites with collects on
  two or more days.
- Sentinel-1: Planetary Computer `sentinel-1-rtc` (IW, gamma0, 10 m COGs), the same source as
  exp2's change mask. `s1_coverage.py` lists the passes at each site in `s1_passes.csv`.
- Pairing: `pairs.py` writes `tasking_pairs.csv`. A pair is two X-band collects x1 and x2, 6 to 60
  days apart, with two passes on one Sentinel-1 track (relative orbit and direction), both
  acquired after x1 and before x2. Any lasting change that Sentinel-1 flags then falls inside the
  X-band interval. The two X-band collects must also match in geometry: same polarization, same
  look direction (Umbra azimuth within 20°, or same orbit direction), and incidence within 5°.
  Without this, the X-band difference is all layover and shadow.

Inventory: 565 of the 597 repeat sites have Sentinel-1 passes in their date span. 205 have an
X-band pair that brackets two passes on one track, and 117 sites (10,899 pairs) also match in
X-band geometry. Of those, 68 have five or more distinct "after" dates. Umbra covers 71 of the 117
sites, Capella 41, and ICEYE 7; a few sites have more than one provider.

## Change detector

Exp2's rule (`exp2_iw_to_stripmap/change_mask.py`): VV gamma0 of the two passes on one track,
warped bilinearly to a 10 m UTM grid, converted to dB, and averaged over 3×3. A pixel counts as
changed when the averaged difference exceeds 3 dB, and the rule erodes valid pixels by one.

The pilot (`pilot.md`) showed two problems the rule doesn't handle: whole-scene shifts of 1 to 2 dB
between passes (likely wet ground) and single bright pixels in towns. So the full run scores
the exp2 rule after one fixed change: subtract the median difference over the 2.56 km window, and
keep flag regions of at least 20 pixels (0.2 ha). The run also reports the unchanged rule.

## Judging whether X-band explains a flag

1. **Manual labels (primary).** Sample up to 15 flag regions per site, 300 in all. Add the same
   number of control regions: random spots with the same sizes, in the same windows. A labeler
   sees each region's X-band before and after chips (500 m) without the Sentinel-1 data and
   without knowing which regions are flags. Labels: *object change* (new or removed structure,
   vehicles, aircraft or ships, earthworks, lava, flood edge), *surface change* (tillage, harvest,
   wet or dry soil, vegetation), *no visible change*, or *can't tell* (layover, footprint edge,
   misregistration). A second labeler scores 60 regions for agreement (Cohen's kappa).
2. **Automatic X-band change (secondary).** The same 3 dB rule on the X-band pair: COG overview
   just finer than 2.5 m, intensity averaged to 10 m, median offset removed, 3×3 mean. An integer
   shift of up to ±10 pixels aligns X-band to Sentinel-1. A flag region counts as explained when
   30% of its pixels fall in X-band change dilated by 2 pixels. The pilot showed that this judge
   is noisy: edges that shift by a pixel between dates show up as change, and whole fields change
   in farmland. Hence it only backs up the labels.

## Metrics

- **Precision of S1 flags:** share of flag regions labeled object or surface change, minus the same
  share for control regions. The automatic version gives region precision and a chance level from
  20 random shifts of the X-band change map.
- **Fraction X-band can interpret:** share of flag regions labeled object change, and share
  labeled anything but *can't tell*.
- **Revisit latency:** days from the Sentinel-1 "after" pass to the X-band "after" collect. The
  archives were not tasked from Sentinel-1, so this describes the archive, not a tasking system.
  Over the 10,899 pairs, the median is 4 days and the 90th percentile 10 days.
- Also reported: pixel precision and lift, recall of X-band change, sign agreement.

## Decision rule (fixed before the full run)

Run every matched site, one pair per distinct "after" date, at most 10 pairs per site. The chain
**works** when all three hold:

1. Flag regions get an object or surface change label at least 25 points more often than control
   regions, and the lower end of the 95% bootstrap interval (resampling sites) is above 10 points.
2. At least 25% of flag regions get an object change label.
3. Labeler agreement reaches a kappa of 0.4 or more; below that, the labels can't decide anything.

If (1) holds but (2) fails, Sentinel-1 finds real change that is mostly surface change, and an
X-band look adds little. If (1) fails, the flags are mostly noise at these sites. The threshold,
window, offset step, and 20-pixel floor stay fixed; a threshold sweep may appear as a side figure.

## Compute

CPU only. Each pair reads a 2.56 km window: a few MB of X-band overview and two Sentinel-1 COG
windows, about 20 seconds. The full run (about 1,000 pairs) takes some hours of network reads and
under 2 GB of disk. No training, so no GPU slot. Labeling 600 chips takes about a day of human time.

## Known limits

- The sites aren't random. Umbra's long series favor airports, ports and cities; Capella's dense
  stacks favor InSAR targets such as Kilauea. Results describe these sites.
- Umbra GEC and Capella GEO products aren't radiometrically matched across dates. The median offset
  removes a constant gain, not a nonlinear stretch.
- A 2 km cluster can join neighboring targets. Each pair needs 80% footprint overlap.
