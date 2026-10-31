# Round 2 spot check, version 2: chips with context

The first spot check (`../r2_spotcheck/`) came back 72 of 96 "unsure". Those chips showed 160 m or more at 2 m per pixel, so a reviewer saw a speckled patch without the field, building or pit around it. This set redraws the same 96 chips, in the same `chip_id` order, so a person can judge whether the Capella change patch is real ground change. `../r2_spotcheck_v2.py` makes them.

## What each chip shows

Two rows of three panels, all north-up on the local UTM grid. Capella only: the chips show no Sentinel-1 data, and neither the file name nor the title gives the stratum (caught, missed or false alarm).

- **Top row, context** (480 m across, up to 900 m for large patches). Capella before, Capella after, and a red/cyan color composite: red = before, green and blue = after. Ground that brightened turns cyan, ground that darkened turns red, and unchanged ground stays gray. A dashed yellow box marks the zoom area. The after panel carries the purple frame and the purple outline of the area to judge.
- **Bottom row, zoom** (100 m across or more, sized to the area). Capella before, Capella after with the purple outline, and the before-to-after log ratio in dB: blue = brighter after, red = darker after, matching the composite.
- A scale bar in meters and a north arrow on the before panels, and the `panel_roles` legend line. Purple marks the truth; no panel is green, since no model output appears.

## How the script builds each chip

- Fresh windowed reads of the Capella GEO COGs, at the coarsest overview with pixels of 0.8 m or finer, block-averaged to 1 m. The zoom keeps detail the 2 m grid lost. The context panels use the same 1 m data, averaged to 2 m only when the context exceeds 500 m.
- The after image takes the pair's co-registration shift from `r2_run.py`, applied to its geotransform. The patch mask comes from the round 2 arrays, mapped back through the Sentinel-1 alignment shift. A check compares each fresh read, averaged to 2 m, with the cached round 2 image; `_build_log.csv` records the correlation and the best 2 m offset.
- **Honest brightness.** The script corrects the after image by the pair's scene-median log ratio, the same correction the truth uses. Both dates then share one gray stretch, from the 2 to 98 percent points of the before image over the context window.
- The purple outline sits 2 m outside the area so it doesn't cover the changed pixels. For Sentinel-1 false-alarm chips the area to judge is the 10 m flag region on Capella-stable ground; a dilated outline rounds its corners, but its blocky 10 m shape may still hint at the stratum.

## Files

- `chipNNN.png`: one per chip, under 400 KB.
- `_build_log.csv`: per-chip build details (bytes read, alignment check, window sizes) for the analyst, not the reviewer.
- `chips.csv`: `chip_id, file, site_id, pair, land, area_m2, capella_before, capella_after, _stratum, verdict, note`. `land` is the ESA WorldCover 2021 class under the area (urban, farm, bare/mine, water edge, vegetation/other). `_stratum` stays hidden from the reviewer; `verdict` and `note` start empty.

Suggested verdicts: real object change, real surface change, not real (misregistration, layover, speckle), or can't tell.
