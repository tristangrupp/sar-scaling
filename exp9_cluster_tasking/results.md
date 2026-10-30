# Experiment 9: Sentinel-1 flag clusters as Capella tasking cues

Run 2026-10-30, CPU only. Scripts: `prep.py`, `core.py`, `tune.py`, `test.py`, `report.py`. Design fixed beforehand in `DESIGN.md`. Numbers: `grid_tune.csv`, `chosen_rule.json`, `results_test.json`, `tune_pairs.csv`, `test_pairs.csv`. Figures: `fig_hit_rates.png`, `fig_topk.png`, `fig_grid.png`.

## Result

On the held-out test half, 89% of looks pointed by Sentinel-1 clusters found Capella change, against 75% for random looks in the same scenes. That's a lift of 1.19, or about 9 hits per 10 looks against 7.5. The lift passes the pre-set test, because its 95% interval (1.13 to 2.17) excludes 1. But the evidence is thin: only 5 of 16 test sites produced a cluster, one site supplied 20 of the 28 looks, and picking the brightest change pixels without clustering did as well.

- **Chosen rule:** Sentinel-1 VV, 3 × 3 mean of dB, 3 dB threshold after the scene median correction, clusters of 2 ha or more. This is experiment 7's flag rule with a large minimum area.
- **Tune against test (500 m look, a hit needs 200 m² of Capella change):**

| | Tune half | Test half |
|---|---|---|
| Sites (with any look) | 17 (9) | 16 (5) |
| Pairs (with any look) | 32 (12) | 24 (5) |
| Looks | 35 | 28 |
| Cluster hit rate | 100% (100 to 100) | 89% (47 to 100) |
| Random hit rate, same scenes | 70% (50 to 91) | 75% (31 to 89) |
| Brightest-pixel hit rate | 97% (90 to 100) | 93% (65 to 100) |
| Lift over random | 1.43 (1.10 to 2.01) | 1.19 (1.13 to 2.17) |
| Hits per 10 looks, clusters against random | 10.0 against 7.0 | 8.9 against 7.5 |

Parentheses hold 95% intervals from 2,000 site-bootstrap draws.

- **Ranking view, test half, 500 m:** the top cluster per scene hit 4 of 5 times (80%, random 62%, lift 1.29, interval 0.77 to 2.17). The top 3 hit 7 of 9 (78%, random 61%, lift 1.28, interval 1.08 to 1.86). The top 10 hit 15 of 18 (83%, random 68%, lift 1.22, interval 1.13 to 1.82). On the tune half every top-ranked look hit (100% against 64% to 70% random). No scene in either half had more than 10 clusters except the Black Rock Desert scene.
- **Smaller looks and stricter hits don't rescue it.** With a 250 m look, the test lift is 1.34 (interval 0 to 3.29) and the top-1 hit rate 40% against 39% random. With a 1,000 m² hit at 500 m, the test lift is 1.17 (interval 0 to 8.0). The tune half showed lifts of 2.0 to 2.7 in those settings; the test half doesn't confirm them.

In the requested form: *pointing Capella at Sentinel-1 VV clusters of 2 ha or more finds change about 9 times in 10, against 7.5 in 10 at random, a lift near 1.2 that rests on 5 test sites; most scenes give no cluster to point at.*

## Why the answer is weaker than it looks

- **Random looks already hit often.** Capella change truth is dense: a random 500 m look holds 200 m² of automatic change patches 70% to 75% of the time. A high hit rate says more about the truth than about Sentinel-1, and lift is the number to read.
- **The rule mostly stays quiet.** At 2 ha, the rule fires in 12 of 32 tune pairs and 5 of 24 test pairs. Practical tasking would get a cue at a third of the sites at best, and in the remaining scenes Sentinel-1 offers nothing to point at.
- **One site dominates the test.** The Black Rock Desert pair (site 0, August 2026) gives 20 of the 28 test looks, with a random hit rate of 88%. The other four sites give 1 to 5 looks each. Each of the five sites beat its own random expectation, so every bootstrap draw does too. That's why the interval excludes 1 while the point estimate (1.19) sits near its lower end. A bootstrap over 5 sites understates the real uncertainty.
- **Clustering adds nothing over the brightest pixels.** Looks at the same number of highest-change pixels, 50 m apart, hit 97% (tune) and 93% (test), as often as or more often than clusters. The useful signal is "strong Sentinel-1 change here," not the cluster shape.
- **Selection inflates the tune numbers.** Across the 180 rules (`fig_grid.png`), lift falls steadily as the number of looks rises. Rules with thousands of looks sit near 1.0 to 1.1, and the best lifts come from the sparsest rules. The chosen rule sits just above the 30-look eligibility floor, and its tune lift of 1.43 shrank to 1.19 on test, as expected from picking the best of 144 eligible rules. For reference, experiment 7's rule at 0.2 ha gives 1,235 tune looks at a lift of 1.09, and at 1 ha it gives 79 looks at 1.27.
- **Polarization and smoothing matter little.** VH, VV-or-VH, same-sign VV and VH, 5 × 5 smoothing and the speckle-aware ratio all fall on the same lift-against-looks curve as VV (`fig_grid.png`). None beat VV at the same number of looks by a clear margin.

## Method

**Data.** Experiment 7 round 2 pairs: 56 Capella repeat pairs at 33 sites, each inside two same-track Sentinel-1 passes (median Capella gap 3 days, Sentinel-1 interval 12 days). The 2.56 km window, the 2 m Capella change patches and the Capella–Sentinel-1 alignment come unchanged from `exp7_change_tasking/data/r2/`. `prep.py` re-read Sentinel-1 radiometrically terrain-corrected (RTC) VV and VH as linear intensity on the same 10 m grid. The re-read VV matches the experiment 7 cache to a median 0.002 dB, the float16 rounding.

**Split.** Sites split at random (seed 9), stratified by dominant WorldCover land class: 17 tune sites (32 pairs) and 16 test sites (24 pairs). `test.py` ran once and refuses to rerun.

**Domain.** Looks and flags count on 10 m land pixels that are Sentinel-1 valid and at least half covered by usable Capella truth.

**Rules.** 180 combinations: VV, VH, VV or VH, and VV and VH with the same sign; thresholds 2, 3 and 4 dB after the scene median correction; 3 × 3 or 5 × 5 mean of dB, or the ratio of 5 × 5 mean linear intensities (the speckle-aware test); clusters of 0.1, 0.2, 0.5, 1 or 2 ha, 8-connected. Persistence across a third pass wasn't tested, because the cached pairs hold two passes per track.

**Looks and hits.** One look per cluster, a 500 m square (and 250 m) centered on the cluster centroid, slid inward at the window edge. A hit needs Capella change patches totaling 200 m² inside the look (and 50 or 1,000 m² as checks). The full Capella spotlight scene would cover the whole window, so the 500 m square stands for a tasked spot or an analyst's review area.

**Baselines.** Random looks: the same count per scene, with the exact hit probability over all domain pixels. Brightest pixels: the same count per scene at the highest absolute change of the rule's own map, 50 m apart.

**Choice.** Eligible rules had at least 30 tune looks at 5 or more tune sites (144 of 180). The highest tune lift won; no other rule came within 2% of it, so the simplicity tie-break didn't apply.

## Spot check

`spotcheck/` holds 40 panels from the test half: all 28 cluster looks (25 automatic hits, 3 misses) and 12 random looks, in shuffled order. Each panel shows Capella before and after at 2 m, the Capella change with patches outlined (purple frame, the truth), and the Sentinel-1 change with the rule's clusters outlined (green frame, the detector). Titles carry only a look number. `spotcheck/looks.csv` holds the look type and automatic hit in separate columns and an empty `verdict` column. Suggested verdicts: real change, not real (misregistration, speckle, moisture), or can't tell, for both the Capella truth and the Sentinel-1 cluster.

A first look at two panels: a Black Rock Desert cluster (look 01) sits among new structures that Capella also shows. All three misses come from Etna (site 599), where Sentinel-1 darkens across a slope over 12 days while Capella sees nothing in its 3-day gap.

### Human verdicts (2026-10-30)

One reviewer judged all 40 looks without seeing their type or automatic outcome. They marked 8 looks as real change and 32 as unsure; none were marked as having no change.

- All 8 real-change looks are cluster looks at Black Rock Desert (site 0), August 22 to 25, 2026. That is the build of Black Rock City for Burning Man, a large, fast, unambiguous change.
- None of the 12 random looks were confirmed, and none of the 8 cluster looks at other sites were.
- So the reviewer confirmed the clusters only where the change was obvious, and only at one site. The check supports the clusters at Black Rock but can't separate clusters from random looks anywhere else.

## Limits

- **The truth comes from experiment 7 and is automatic.** Its spot check still awaits a human verdict. Wet canopy, wind, tillage and residual misregistration produce patches, and the dense truth drives the high random hit rate. A cleaner truth would lower every hit rate and could change the lift either way.
- **Small sample.** 5 test sites with looks, one holding 71% of them. The test confirms the sign of the effect, not its size.
- **Timing gap.** The Sentinel-1 interval (median 12 days) spans the Capella interval (median 3 days). Sentinel-1 change before or after the Capella pair counts as a miss, as at Etna. That penalizes Sentinel-1 compared with real tasking, where Capella would image after the cue.
- **HH against VV.** Almost every pair is Capella HH; Sentinel-1 offers only VV and VH.
- **Retrospective.** These are archive pairs at Capella's repeat targets (volcanoes, mines, ports, cities), not live tasking over random land.

## What looked wrong along the way

- Experiment 7's cache holds Sentinel-1 VV in dB but no raw VH, so `prep.py` downloaded both polarizations again (434 MB).
- The spot-check plan called for 15 hits and 15 misses, but the test half produced only 3 cluster misses. The panels instead show all 28 cluster looks plus 12 random looks.

Downloads: 434 MB of Sentinel-1 windows from Planetary Computer. Kept arrays in `data/`: 57 MB.
