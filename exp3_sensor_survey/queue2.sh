#!/bin/bash
# Round 2 steps 2-4, after the sweep. Two GPU lanes. Run from exp3_sensor_survey/.
PY=/c/Users/grupp/miniconda3/envs/ftwtools/python.exe
until [ -f logs/queue1_done.txt ]; do sleep 30; done
$PY -u eval2.py --label sweep_c --runs s_w05 s_w2_40 --splits val > logs/eval_sweep_c.log 2>&1
$PY select_sweep.py > logs/select_sweep.log 2>&1
C=$($PY -c "import json; c=json.load(open('chosen.json')); print(f\"--cons {c['cons']} --cons_at {c['cons_at']}\")")
echo "chosen: $C" >> logs/select_sweep.log
T="--arm x2 --steps 3000 --val_out 1"
( $PY -u train.py --arm x2multi --steps 3000 --val_out 1 $C --name m_multi > logs/train_m_multi.log 2>&1
  $PY -u train.py --arm g2 --steps 3000 --val_out 1 $C --name g_g2 > logs/train_g_g2.log 2>&1 ) &
( $PY -u train.py $T $C --xdata 1 --edge_w 0 --name b_a > logs/train_b_a.log 2>&1
  $PY -u train.py $T $C --edge_w 0.05 --name b_e005 > logs/train_b_e005.log 2>&1
  $PY -u train.py $T $C --edge_w 0.2 --name b_e02 > logs/train_b_e02.log 2>&1 ) &
wait
echo Q2 > logs/queue2_done.txt
