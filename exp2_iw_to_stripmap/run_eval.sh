#!/bin/bash
cd "$(dirname "$0")"
PY=/c/Users/grupp/miniconda3/envs/ftwtools/python.exe
M="esrt_paper esrt_a cnn_a esrt_ctrl esrt_b05 esrt_b20"
$PY -u score.py $M > score.log 2>&1
$PY -u stage1.py $M > stage1.log 2>&1
$PY -u panels.py esrt_a cnn_a esrt_b05 > panels.log 2>&1
echo EVAL_DONE > run_eval.done
