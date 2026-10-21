#!/bin/bash
# Round 3 GPU queue: ONE small job at a time (exp5 has priority), batch 16. Run from exp3_sensor_survey/.
PY=/c/Users/grupp/miniconda3/envs/ftwtools/python.exe
# gate: start a GPU job only when at least 2 GB is free (exp5 has priority; avoid WDDM spill)
gate() { while [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')" -gt 6000 ]; do sleep 60; done; }
M="--arm x2multi --steps 3000 --val_out 1 --cons 0.5 --cons_at 40 --bs 16"
[ -f ckpt/r3_g05_s0.pt ] || { gate; $PY -u train.py $M --grad_w 0.5 --seed 0 --name r3_g05_s0 > logs/train_r3_g05_s0.log 2>&1; }
gate; $PY -u train.py $M --grad_w 1.0 --seed 0 --name r3_g1_s0 > logs/train_r3_g1_s0.log 2>&1
$PY -u eval2.py --label r3_loss_val --arm x2 --runs m_multi r3_g05_s0 r3_g1_s0 --splits val --target mt10 > logs/eval_r3_loss_val.log 2>&1
$PY select_loss.py > logs/select_loss.log 2>&1
GW=$($PY -c "import json; print(json.load(open('chosen_loss.json'))['grad_w'])")
WN=$($PY -c "import json; print(json.load(open('chosen_loss.json'))['run'])")
gate; $PY -u train.py $M --grad_w $GW --seed 1 --name ${WN%_s0}_s1 > logs/train_${WN%_s0}_s1.log 2>&1
gate; $PY -u train.py $M --grad_w 0 --seed 1 --name r3_l1_s1 > logs/train_r3_l1_s1.log 2>&1
echo R3A > logs/r3_queue_a_done.txt
# step 3: round 1 length (8000 steps) for the round 2 winner and the multi-date model
gate; $PY -u train.py --arm x2 --steps 8000 --val_out 1 --cons 0.5 --cons_at 40 --bs 16 --seed 0 --name r3_long_w05_40 > logs/train_r3_long_w05_40.log 2>&1
gate; $PY -u train.py $M --steps 8000 --grad_w 0 --seed 0 --name r3_long_multi > logs/train_r3_long_multi.log 2>&1
echo R3B > logs/r3_queue_b_done.txt
