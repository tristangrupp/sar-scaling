#!/bin/bash
# Round 2 GPU queue: one of my trainings at a time (the GPU is shared with several other agents). Run from exp3_sensor_survey/.
PY=/c/Users/grupp/miniconda3/envs/ftwtools/python.exe
while pgrep -f "name s_w05_40" >/dev/null 2>&1 || powershell.exe -NoProfile -Command "if (Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -like '*s_w05_40*' }) { exit 0 } else { exit 1 }"; do sleep 30; done
$PY -u train.py --arm x2 --steps 3000 --val_out 1 --name s_w05 --cons 0.5 > logs/train_s_w05.log 2>&1
$PY -u train.py --arm x2 --steps 3000 --val_out 1 --name s_w2_40 --cons 2.0 --cons_at 40 > logs/train_s_w2_40.log 2>&1
echo QUEUE1_DONE > logs/queue1_done.txt
