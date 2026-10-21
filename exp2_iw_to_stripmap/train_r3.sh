#!/bin/bash
# Round 3: ESRT arm A (paper loss: SSIM + histogram matching) on splits_v2, with and without the
# change mask, seed 0; then seed 1 for whichever has the higher validation SSIM. Via gpu_job.py.
cd "$(dirname "$0")"
PY=/c/Users/grupp/miniconda3/envs/ftwtools/python.exe
COMMON="--model esrt --splits v2 --steps 8000 --bs 16 --lr 1e-3"
$PY -u gpu_job.py esrt_v2_mask_s0 ckpt_esrt_v2_mask_s0.log -- $PY -u train_sr.py $COMMON --name esrt_v2_mask_s0 --change-mask --seed 0
$PY -u gpu_job.py esrt_v2_nomask_s0 ckpt_esrt_v2_nomask_s0.log -- $PY -u train_sr.py $COMMON --name esrt_v2_nomask_s0 --seed 0
W=$($PY -c "import json;m=json.load(open('ckpt/esrt_v2_mask_s0.json'))['best_val_ssim'];n=json.load(open('ckpt/esrt_v2_nomask_s0.json'))['best_val_ssim'];print('mask' if m>=n else 'nomask')")
echo "winner $W" > train_r3_winner.txt
FLAG=""; [ "$W" = "mask" ] && FLAG="--change-mask"
$PY -u gpu_job.py esrt_v2_${W}_s1 ckpt_esrt_v2_${W}_s1.log -- $PY -u train_sr.py $COMMON --name esrt_v2_${W}_s1 $FLAG --seed 1
echo R3_TRAIN_DONE > train_r3.done
