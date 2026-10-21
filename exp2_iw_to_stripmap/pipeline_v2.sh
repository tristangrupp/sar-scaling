#!/bin/bash
# Round 2: wait for downloads, then tiles -> build/register -> finalize -> change mask -> splits -> page.
cd "$(dirname "$0")"
PY=/c/Users/grupp/miniconda3/envs/ftwtools/python.exe
until [ $(grep -c "^\[" fetch_v2.log) -ge 25 ]; do sleep 60; done
$PY -u select_tiles_v2.py > select_tiles_v2.log 2>&1
$PY -u build_tiles.py --workers 5 --tiles tiles_v2.csv --only "darien|venezuela|roraima|w_sahara|tibesti|yemen_east|iran|sahel|lake_chad|amur|montreal|yemen_west|bangladesh" --out pairs_v2.csv > build_tiles_v2.log 2>&1
$PY -u finalize_pairs.py v2 > finalize_pairs_v2.log 2>&1
$PY -u change_mask.py > change_mask_v2.log 2>&1
$PY make_splits_v2.py > make_splits_v2.log 2>&1
echo PIPELINE_V2_DONE > pipeline_v2.done
