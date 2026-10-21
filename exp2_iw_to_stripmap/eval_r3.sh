#!/bin/bash
# Round 3 evaluation (GPU, via gpu_job.py): score all holdouts, then panels for the new holdouts.
cd "$(dirname "$0")"
PY=/c/Users/grupp/miniconda3/envs/ftwtools/python.exe
W=$(awk '{print $2}' train_r3_winner.txt)
$PY -u gpu_job.py score_r3 score_r3.log -- $PY -u score.py --splits v2 --tag r3 esrt_paper esrt_v2_mask_s0 esrt_v2_nomask_s0 esrt_v2_${W}_s1
$PY -u gpu_job.py panels_r3 panels_r3.log -- $PY -u panels.py --r3 esrt_paper esrt_v2_${W}_s0
echo EVAL_R3_DONE > eval_r3.done
