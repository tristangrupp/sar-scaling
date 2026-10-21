#!/bin/bash
# Round 4 scoring on CPU, after queue_r4. Holdouts are scored only after the validation choice.
cd "$(dirname "$0")"
PY=/c/Users/grupp/miniconda3/envs/ftwtools/python.exe
export EXP3_DEVICE=cpu
until [ -f logs/r4_queue_done.txt ]; do sleep 60; done
WN=$($PY -c "import json; print(json.load(open('chosen_adv.json'))['run'] or '')")
if [ -n "$WN" ]; then
  W1=${WN%_s0}_s1; RUNS="r3_g05_s0 e3r4_c0_s0 e3r4_c0_s1 $WN $W1"; GAN=$WN
else
  # no winner on validation: score the largest weight on holdouts for information only (labelled as such on the page)
  RUNS="r3_g05_s0 e3r4_c0_s0 e3r4_a01_s0"; GAN=e3r4_a01_s0
fi
$PY -u eval2.py --label r4_hold_mt10 --arm x2 --runs $RUNS --splits near far --target mt10 > logs/eval_r4_hold_mt10.log 2>&1
$PY -u eval2.py --label r4_hold_a10 --arm x2 --runs $RUNS --splits near far > logs/eval_r4_hold_a10.log 2>&1
$PY -u xb_eval.py --label r4_ --runs e3r4_c0_s0 $GAN > logs/xb_eval_r4.log 2>&1
$PY -u r4_panels.py e3r4_c0_s0 $GAN > logs/r4_panels.log 2>&1
echo DONE > logs/r4_eval_done.txt
