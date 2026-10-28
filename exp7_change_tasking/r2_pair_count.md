# Round 2 pair count: Capella pairs inside a Sentinel-1 interval

Run 2026-10-28 by `r2_pairs.py` (log: `r2_pairs.log`). At the chosen bounds, 40 Capella sites qualify, which clears the 30-site bar by a small margin. After one pair per site per season, 38 sites and 65 pairs remain for the run.

## Why the pairs are new

Round 2 measures recall: what share of Capella change Sentinel-1 catches. That needs the Sentinel-1 interval to contain the Capella interval:

    S1 before <= Capella image 1 < Capella image 2 <= S1 after   (same track)

`tasking_pairs.csv` brackets the other way, with both Sentinel-1 passes inside the X-band interval, which suits the precision question of round 1. Its Capella part (7,167 pairs at 39 sites) can't serve here, so `r2_pairs.py` rebuilds pairs from `xband_collects.csv`.

## Geometry and timing filters

- Capella collects at sites with images on two or more days: 1,478 at 142 sites. Each collect's STAC item supplies polarization, look side, orbit direction, azimuth, incidence, and repeat ground track.
- A pair needs the same polarization, look side, and orbit direction, an azimuth within 10°, footprints overlapping by 80% or more, and images on different days at most 60 days apart.
- Sentinel-1 passes come from a new Planetary Computer search, 15 days either side of each site's Capella dates, because `s1_passes.csv` stops at each site's last X-band date and so misses most "after" passes. Passes exist at 139 of the 142 sites.
- Window: the gap from the S1 "before" pass to Capella image 1, and from Capella image 2 to the S1 "after" pass, both within the bound. Across tracks, the script keeps the track with the smaller of the two gaps' maximum.

Pairs (sites) by incidence difference and Sentinel-1 window:

| Incidence difference | Window 3 days | Window 6 days | Window 12 days |
|---|---|---|---|
| ≤ 1° | 1,916 (27) | 5,491 (39) | 7,994 (56) |
| ≤ 2° | 1,922 (27) | **5,524 (40)** | 8,135 (58) |
| ≤ 5° | 1,950 (29) | 5,629 (47) | 8,490 (68) |

Incidence matters little because 98.8% of qualifying pairs repeat the same Capella ground track (InSAR stacks), so the geometry repeats almost exactly. Timing sets the count.

At 2° and 6 days, a shorter Capella gap leaves fewer sites:

| Capella gap | Pairs | Sites |
|---|---|---|
| ≤ 7 days | 799 | 30 |
| ≤ 14 days | 1,395 | 33 |
| ≤ 30 days | 3,110 | 38 |
| ≤ 60 days | 5,524 | 40 |

Nearly every pair is HH spotlight: 5,512 of 5,524 pairs are HH and 5,437 spotlight. Sentinel-1 has only VV and VH, so nearly every comparison crosses polarizations.

## Chosen set

- Bounds: incidence within 2°, Sentinel-1 window 6 days, and Capella gap 30 days or less.
- One pair per site per calendar quarter, taking the shortest Capella gap, then the smallest Sentinel-1 window, then the closest incidence.
- At most 5 pairs per site. Sites hold 1 pair (24 sites), 2 (8), 3 (2), 4 (1) or 5 (3); only 68 site-quarters exist, so only 3 sites reach the cap.
- Result: **65 pairs at 38 sites** (`r2_pairs.csv`). Median Capella gap 3.0 days (maximum 30), median Sentinel-1 interval 12 days, median largest window gap 4.4 days.

With 38 sites, the curve has enough sites to bootstrap, but the counts in each area bin, land type and sign class are much smaller. The results note reports power bin by bin.
