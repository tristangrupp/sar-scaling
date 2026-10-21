#!/bin/bash
# Arm A: ESRT as in the paper (SSIM + histogram matching), ESRT + input consistency, CNN + consistency.
cd "$(dirname "$0")"
PY=/c/Users/grupp/miniconda3/envs/ftwtools/python.exe
$PY -u train_sr.py --model esrt --name esrt_paper --steps 8000 --bs 16 --lr 1e-3 > ckpt_esrt_paper.log 2>&1
$PY -u train_sr.py --model esrt --name esrt_a --steps 8000 --bs 16 --lr 1e-3 --cons 0.15 > ckpt_esrt_a.log 2>&1
$PY -u train_sr.py --model cnn --name cnn_a --steps 8000 --bs 16 --lr 5e-4 --cons 0.15 > ckpt_cnn_a.log 2>&1
echo ARM_A_DONE > train_arm_a.done
