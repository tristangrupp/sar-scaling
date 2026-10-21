#!/bin/bash
# Round 3 follow-up: retrain the 10 m -> 5 m arm on the deeper GSLC target (g5v2), after queue_r3 finishes.
PY=/c/Users/grupp/miniconda3/envs/ftwtools/python.exe
gate() { while [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' \r')" -gt 6000 ]; do sleep 60; done; }
until [ -f logs/r3_queue_b_done.txt ] && grep -q DONE logs/fetch_gslc2.log; do sleep 60; done
gate; $PY -u train.py --arm g2 --target g5v2 --steps 3000 --val_out 1 --cons 0.5 --cons_at 40 --bs 16 --seed 0 --name r3_g2v2 > logs/train_r3_g2v2.log 2>&1
echo R3C > logs/r3_queue_c_done.txt
