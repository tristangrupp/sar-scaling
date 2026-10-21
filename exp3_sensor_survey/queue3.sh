#!/bin/bash
# Round 2, sequential (GPU heavily shared): wait for m_multi, then g2 and arm A'/B at 1500 steps. Run from exp3_sensor_survey/.
PY=/c/Users/grupp/miniconda3/envs/ftwtools/python.exe
C="--cons 0.5 --cons_at 40"
until grep -qE "^saved|Error" logs/train_m_multi.log; do sleep 30; done
$PY -u train.py --arm g2 --steps 3000 --val_out 1 $C --name g_g2 > logs/train_g_g2.log 2>&1
T="--arm x2 --steps 1500 --val_out 1"
$PY -u train.py $T $C --name b_ref > logs/train_b_ref.log 2>&1
$PY -u train.py $T $C --xdata 1 --edge_w 0 --name b_a > logs/train_b_a.log 2>&1
$PY -u train.py $T $C --edge_w 0.05 --name b_e005 > logs/train_b_e005.log 2>&1
$PY -u train.py $T $C --edge_w 0.2 --name b_e02 > logs/train_b_e02.log 2>&1
echo Q3 > logs/queue3_done.txt
