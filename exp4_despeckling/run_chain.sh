#!/bin/bash
# wait for the supervised run, then Noise2Noise, its offset calibration, and the GAN fine-tune
cd "$(dirname "$0")"
P=/c/Users/grupp/miniconda3/envs/ftwtools/python.exe
until grep -q "^done" train_sup.log; do sleep 30; done
$P -u train4.py --mode gan --steps 3000 --adv_w 5e-3 > train_gan.log 2>&1
$P -u train4.py --mode n2n --steps 6000 > train_n2n.log 2>&1
$P -u calib_n2n.py > calib_n2n.log 2>&1
echo chain-done > chain.log
