#!/bin/bash
# Overnight GPU window for experiment 10 (user: run overnight only). Waits until 22:00, runs stage 1
# (R = 10, 7.5, 5 m) and the 2.5 m stretch run, and stops at 07:00 whatever is running (train.py resumes
# from ckpt/<name>_last.pt next night). Seeds wait for the morning review of stage 1.
cd /e/SAR_superresolution/exp10_capella_enhancer
start=$(date -d "today 22:00" +%s); now=$(date +%s)
[ "$now" -lt "$start" ] && sleep $((start - now))
end=$(date -d "tomorrow 07:00" +%s); [ "$(date +%H)" -lt 7 ] && end=$(date -d "today 07:00" +%s)
echo "$(date '+%m-%d %H:%M') night1 start, hard stop $(date -d @$end '+%m-%d %H:%M')" >> logs/run_queue.log
timeout --kill-after=60 $((end - $(date +%s))) bash -c 'bash run_queue.sh stage1 && bash run_queue.sh stretch'
echo "$(date '+%m-%d %H:%M') night1 exit $?" >> logs/run_queue.log
powershell -NoProfile -ExecutionPolicy Bypass -File stop_trainer.ps1
sleep 5
rm -f /e/SAR_superresolution/gpu_secondary.lock
