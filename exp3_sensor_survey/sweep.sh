#!/bin/bash
# Round 2 step 1: consistency-weight sweep, validation frames excluded from training. Run from exp3_sensor_survey/.
PY=/c/Users/grupp/miniconda3/envs/ftwtools/python.exe
run() { $PY -u train.py --arm x2 --steps 3000 --val_out 1 --name "$1" "${@:2}" > "logs/train_$1.log" 2>&1; }
mkdir -p logs
( run s_w0 --cons 0 ; run s_w05 --cons 0.5 ) &
( run s_w005 --cons 0.05 ; run s_w05_40 --cons 0.5 --cons_at 40 ) &
( run s_w02 --cons 0.2 ; run s_w2_40 --cons 2.0 --cons_at 40 ) &
wait
echo SWEEP_DONE > logs/sweep_done.txt
