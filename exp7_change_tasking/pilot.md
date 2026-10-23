# Experiment 7 pilot

Run 2026-10-23, CPU only. Scripts: `inventory.py`, `s1_coverage.py`, `pairs.py`, `pilot.py`.
Per-pair numbers are in `pilot_pairs.csv` and panels in `panels/`. In each panel, the green outline
marks the Sentinel-1 change flags (the detector) and the purple outline marks X-band change (the
reference). Downloads: windowed COG reads only, about 6 MB of arrays kept in `data/`.

## How many sites qualify

| Step | Sites |
|---|---|
| Open X-band collects (Umbra 7,087, Capella 2,285, ICEYE 640) | 2,415 sites at 2 km |
| Collects on 2 or more days | 597 |
| Sentinel-1 RTC passes in the date span | 565 |
| X-band pair 6–60 days apart with two same-track passes between | 205 |
| And matched X-band geometry (pol, look, incidence within 5°) | **117** (10,899 pairs) |
| And 5 or more distinct "after" dates | 68 |

Geometry is the main filter. Umbra's long time series mostly change incidence and azimuth from
one collect to the next: at Umbra-only sites, 9% of bracketing pairs match (3,050 of 34,110).
Capella's InSAR stacks repeat the same geometry every few days, so 86% match at Capella-only sites,
and Capella supplies most matched pairs (7,167). From the Sentinel-1
"after" pass to the X-band "after" image takes a median of 4 days (90th percentile 10).

## Pilot pairs

Nine pairs at seven sites. Region precision counts flag regions of at least 20 pixels that fall
30% or more in X-band change (dilated 20 m); "chance" is the same after random shifts of the
X-band change map.

| Site | X-band pair | S1 flagged | X changed | Region precision (chance) | S1 scene offset |
|---|---|---|---|---|---|
| Kilauea (Capella) | Aug 3 → Aug 11, 2025 | 27.4% | 0.0% | 0.00 (0.00) | +1.7 dB |
| Kilauea (Capella) | Sep 8 → Sep 16, 2025 | 6.4% | 1.8% | 0.00 (0.06) | −0.4 dB |
| San Jose (Capella) | Mar 28 → Apr 6, 2025 | 13.2% | 2.4% | 0.30 (0.12) | +0.5 dB |
| Mojave (Capella) | Apr 12 → Apr 21, 2025 | 3.7% | 0.0% | 0.00 (0.00) | −1.0 dB |
| Pilbara mine (Capella) | Nov 21 → Dec 5, 2023 | 2.8% | 0.4% | 0.25 (0.02) | 0.0 dB |
| Bangkok airport (Umbra) | Jun 29 → Jul 9, 2025 | 22.2% | 12.6% | 0.67 (0.54) | +0.1 dB |
| Indiana farms, May (Umbra) | May 3 → May 18, 2024 | 10.3% | 25.6% | 0.57 (0.58) | −0.1 dB |
| Indiana farms, Oct (Umbra) | Oct 5 → Oct 22, 2023 | 32.7% | 14.6% | 0.43 (0.39) | +1.8 dB |
| W. Sydney airport (Umbra) | Aug 10 → Aug 29, 2024 | 7.4% | 15.7% | 0.74 (0.48) | +0.2 dB |

## What the pilot showed

1. **The 3 dB rule flags far more than real change.** Exp2's docstring expects about 1% of stable
   pixels over 3 dB. The pilot flagged 3% to 33%. Two causes stand out. First, whole scenes shift:
   at Kilauea in August and in Indiana in October, the "after" pass is about 1.7 dB brighter
   everywhere, which flags a quarter to a third of the window in red. Second, towns and airports
   have bright point targets whose 3×3 mean swings by more than 3 dB between passes.
2. **Where real change exists, Sentinel-1 finds some of it.** At the Pilbara mine, flag regions land
   on X-band change 12 times more often than chance (0.25 against 0.02). San Jose (0.30 against
   0.12), Western Sydney airport (0.74 against 0.48), and Bangkok airport (0.67 against 0.54) also
   beat chance. At Bangkok both sensors pick up aircraft that moved at the gates. In Indiana in May,
   Sentinel-1 flags fields that X-band also shows changing, but X-band shows a quarter of the window
   changing, so chance is already high.
3. **Quiet sites give no answer.** At Kilauea and Mojave, the Capella stacks barely change in a week
   (X-band difference spread 0.2 to 0.9 dB). Every Sentinel-1 flag there is a false alarm by this
   test.
4. **The automatic X-band judge is noisy.** At Western Sydney airport, runway and taxiway edges show
   up as change because the two X-band images differ by about a pixel. On farmland, whole fields
   change. Manual labels on blind chips, with random control regions, should be the main judge.
   DESIGN.md now says so.

These numbers come from nine hand-picked pairs and decide nothing. They shaped three choices fixed
in DESIGN.md before the full run: exact acquisition times for the pairing, a scene median offset
with a 20-pixel floor, and manual labels as the primary judge.

## Blockers and traps

- **Umbra links.** STAC asset paths from 2023–2024 moved in the bucket (known from exp2), and 2025
  items have empty `href` fields and ship `<time>_UMBRA-NN_MM.tif` in place of a GEC file.
  `pilot.py` finds files by listing `sar-data/` and matching platform and start time within 90 s.
  The listing takes a minute or two.
- **Same-day timing.** Capella stacks often image a site hours from a Sentinel-1 pass on the same
  day. Pairing by date let Sentinel-1 "before" passes fall after the X-band "before" image, so
  `pairs.py` now compares acquisition times.
- **Sentinel-1 gaps.** 32 repeat sites have no RTC passes on Planetary Computer, and some tropical
  sites (the Umbra series near Puerto Ayacucho, Venezuela) have no IW coverage until April 2025.
- **Few ICEYE repeats.** ICEYE contributes 7 matched sites.
- **Labeling time.** The full test needs about 600 blind chips labeled, plus 60 by a second person.
