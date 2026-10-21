#!/bin/bash
# Arm B: fine-tune arm A (esrt_a) with the X-band edge loss at two weights, plus a control that
# fine-tunes for the same steps without it (so B vs control isolates the X-band term).
cd "$(dirname "$0")"
PY=/c/Users/grupp/miniconda3/envs/ftwtools/python.exe
COMMON="--model esrt --init esrt_a --steps 3000 --bs 16 --lr 2e-4 --cons 0.15 --val-every 500"
$PY -u train_sr.py $COMMON --name esrt_ctrl > ckpt_esrt_ctrl.log 2>&1
$PY -u train_sr.py $COMMON --name esrt_b05 --edge-w 0.05 > ckpt_esrt_b05.log 2>&1
$PY -u train_sr.py $COMMON --name esrt_b20 --edge-w 0.2 > ckpt_esrt_b20.log 2>&1
echo ARM_B_DONE > train_arm_b.done
