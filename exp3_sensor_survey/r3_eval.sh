#!/bin/bash
# Round 3 scoring on CPU (the GPU belongs to exp5), chained to the training queues. Run from exp3_sensor_survey/.
PY=/c/Users/grupp/miniconda3/envs/ftwtools/python.exe
export EXP3_DEVICE=cpu
until [ -f logs/r3_queue_a_done.txt ]; do sleep 60; done
WN=$($PY -c "import json; print(json.load(open('chosen_loss.json'))['run'])"); W1=${WN%_s0}_s1
$PY -u eval2.py --label r3_seeds_mt10 --arm x2 --runs m_multi r3_l1_s1 $WN $W1 --splits val near far --target mt10 > logs/eval_r3_seeds_mt10.log 2>&1
$PY -u eval2.py --label r3_seeds_a10 --arm x2 --runs m_multi r3_l1_s1 $WN $W1 --splits near far > logs/eval_r3_seeds_a10.log 2>&1
$PY -u r3_panels.py loss $WN > logs/r3_panels_loss.log 2>&1
until [ -f logs/r3_queue_b_done.txt ]; do sleep 60; done
$PY -u eval2.py --label r3_long_a10 --arm x2 --runs s_w05_40 r3_long_w05_40 m_multi r3_long_multi --splits val near far > logs/eval_r3_long_a10.log 2>&1
$PY -u eval2.py --label r3_long_mt10 --arm x2 --runs s_w05_40 r3_long_w05_40 m_multi r3_long_multi --splits near far --target mt10 > logs/eval_r3_long_mt10.log 2>&1
until [ -f logs/r3_queue_c_done.txt ]; do sleep 60; done
$PY -u eval2.py --label r3_g2v2 --arm g2 --target g5v2 --runs g_g2 r3_g2v2 --splits val near far > logs/eval_r3_g2v2.log 2>&1
$PY -u r3_panels.py g2v2 > logs/r3_panels_g2v2.log 2>&1
echo DONE > logs/r3_eval_done.txt
