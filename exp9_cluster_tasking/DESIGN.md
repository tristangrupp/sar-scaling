# Experiment 9 design: Sentinel-1 flag clusters as Capella tasking cues

Written 2026-10-30, before scoring any rule and before touching the test half. The rules below stay fixed. Any later change gets a dated note in `results.md`.

## Question

If Sentinel-1 flag clusters decide where to point Capella, what share of those looks find real change, and how much better is that than pointing at random? This tasking hit rate is the practical number for change-triggered tasking.

## Data

The experiment reuses experiment 7 round 2: 56 Capella repeat pairs at 33 sites, each bracketed by two same-track Sentinel-1 passes. The analysis window measures 2.56 km on a side. Capella change truth comes from `exp7_change_tasking/data/r2/<tag>.npz` (`lab`, 2 m change patches) and stays read only. That truth is automatic and still awaits a human spot check.

`prep.py` re-reads Sentinel-1 radiometrically terrain-corrected (RTC) gamma0 VV and VH for both passes as linear intensity on the exact experiment 7 10 m grid, because the experiment 7 cache holds VV in dB and no raw VH. It confirms that the re-read VV matches the cache.

## Split

`prep.py` splits sites, not pairs, into tune and test halves at random (seed 9), stratified by each site's dominant European Space Agency (ESA) WorldCover land class: urban, bare, vegetation, farm, or other. The result: 17 tune sites with 32 pairs and 16 test sites with 24 pairs (`sites_split.csv`). Rule tuning uses the tune half only. `test.py` scores the test half once and refuses to run a second time.

## Domain

Looks and flags count only on 10 m pixels that are Sentinel-1 valid in experiment 7, on land (not WorldCover water or no data), and at least half covered by usable Capella truth.

## Candidate cluster rules (tune half, all logged in `grid_tune.csv`)

- Polarization: VV; VH; VV or VH; VV and VH flagged with the same sign (`both`).
- Flag threshold on the median-corrected after-minus-before change: 2, 3, or 4 dB.
- Smoothing: 3 × 3 mean of dB (the experiment 7 rule, `m3`); 5 × 5 mean of dB (`m5`); ratio of 5 × 5 mean linear intensities (`lr5`). For equal window sizes, the gamma-speckle likelihood-ratio test for a change in mean is monotone in that ratio, so `lr5` is the speckle-aware test.
- Minimum cluster area: 0.1, 0.2, 0.5, 1 or 2 ha (10 to 200 pixels), 8-connected.

That makes 4 × 3 × 3 × 5 = 180 rules. Persistence across a third pass isn't tested: the cached pairs hold only two passes per track, and a third would need new downloads and a different timing model.

## Tasking unit and hit

- One look per cluster, centered on the cluster's rounded centroid.
- Footprint: 500 m × 500 m (primary), with 250 m × 250 m reported too. A full Capella spotlight scene (about 5 km) would cover the whole 2.56 km window and make every look identical. The 500 m square stands for a tasked spot or the area an analyst reviews, and tests whether the cluster puts the look in the right place. A look near the window edge slides inward so it stays whole.
- A look is a hit when the Capella change patches inside it total at least 200 m² (primary). Sensitivity: 50 and 1,000 m². Partial patches count by their pixels inside.

## Baselines

- Random: the same number of looks per scene, centered on uniform random domain pixels. The expected random hit rate comes exactly from all domain pixels, not from sampled draws.
- Brightest pixels: the same number of looks per scene, at the highest absolute change pixels of the rule's own change map, at least 50 m apart, without clustering.

## Reported numbers

Hit rate, random hit rate, lift (cluster hits divided by expected random hits), brightest-pixel hit rate, and hits per 10 looks. All rates pool looks across pairs. Intervals come from 2,000 site-bootstrap draws (95%). Ranking view: hit rate of the top 1, 3 and 10 clusters per scene, ranked by score = cluster area (ha) × mean absolute change (dB), against the random rate for the same scenes.

## Choosing the rule

1. Eligible rules have at least 30 looks on the tune half, spread over at least 5 tune sites.
2. Rank eligible rules by tune lift at the primary setting (500 m, 200 m²).
3. Rules within 2% (relative) of the best lift count as tied. Among tied rules, pick the simplest: complexity counts 1 for a two-channel polarization rule (VV or VH, `both`) and 1 for smoothing other than `m3`. Remaining ties go to the higher lift.
4. Score that one rule on the test half once.

## Pass condition

The test lift holds if its 95% site-bootstrap interval excludes 1. Results report tune and test numbers side by side.

## Spot check

About 40 PNG panels of test-half looks (cluster hits, cluster misses and random looks) go to `spotcheck/`, with `spotcheck/looks.csv`. Capella change is the truth (purple outline); the Sentinel-1 change with the rule's clusters is the detector output (green outline). Panel titles carry only a look number. The CSV holds the look type and hit or miss in separate columns, with an empty `verdict` column for blind grading.
