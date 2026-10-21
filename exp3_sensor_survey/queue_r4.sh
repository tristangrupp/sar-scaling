#!/bin/bash
# Round 4 queue. Shared-GPU rules: one secondary job across exp2+exp3 via E:/SAR_superresolution/gpu_secondary.lock,
# start only with >= 2 GB free, batch 16, watchdog stops and requeues if the GPU nears spill or exp5's logs stall.
cd "$(dirname "$0")"
PY=/c/Users/grupp/miniconda3/envs/ftwtools/python.exe
LOCK=../gpu_secondary.lock
used() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' \r'; }
exp5_n() {    # number of running exp5 round 4 training processes (e5r4_*.py); exp5 stalled whenever we ran beside it
  powershell.exe -NoProfile -Command "@(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -like '*e5r4_*' }).Count" | tr -d '\r '
}
exp5_age() {   # minutes since exp5's newest round 4 log changed; 0 if no exp5 python process is running
  n=$(exp5_n)
  [ "${n:-0}" = "0" ] && { echo 0; return; }
  newest=$(ls -t ../exp5_cross_band/logs_r4/*.log | head -1)
  echo $(( ( $(date +%s) - $(stat -c %Y "$newest") ) / 60 ))
}
run_gpu() {   # run_gpu NAME args...
  name=$1; shift
  [ -f ckpt/$name.pt ] && { echo "$name cached"; return; }
  while true; do
    while [ -f $LOCK ] || [ "$(used)" -gt 6144 ] || [ "$(exp5_n)" != "0" ]; do sleep 120; done
    echo "exp3 $name $(date +%H:%M)" > $LOCK
    $PY -u train.py "$@" --name $name > logs/train_$name.log 2>&1 &
    pid=$!; killed=0; sleep 120
    while kill -0 $pid 2>/dev/null; do
      if [ "$(used)" -ge 7900 ] || [ "$(exp5_age)" -ge 6 ]; then
        echo "$(date +%H:%M) watchdog: stop $name (used $(used) MB, exp5 log age $(exp5_age) min)" >> logs/r4_watchdog.log
        powershell.exe -NoProfile -Command "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -like '* -u train.py --arm *' -and \$_.CommandLine -like '*--name $name*' } | ForEach-Object { Stop-Process -Id \$_.ProcessId -Force }"
        killed=1; break
      fi
      sleep 60
    done
    wait $pid 2>/dev/null
    rm -f $LOCK
    if [ $killed = 0 ] && grep -q "^saved" logs/train_$name.log; then return; fi
    mv logs/train_$name.log logs/train_${name}_stopped_$(date +%H%M).log 2>/dev/null
    sleep 600
  done
}
F="--arm x2multi --val_out 1 --cons 0.5 --cons_at 40 --bs 16 --grad_w 0.5 --init r3_g05_s0 --steps 2000 --lr 1e-4"
run_gpu e3r4_c0_s0  $F --adv_w 0     --seed 0
run_gpu e3r4_a001_s0 $F --adv_w 0.001 --seed 0
run_gpu e3r4_a003_s0 $F --adv_w 0.003 --seed 0
run_gpu e3r4_a01_s0  $F --adv_w 0.01  --seed 0
EXP3_DEVICE=cpu $PY -u eval2.py --label r4_adv_val --arm x2 --runs e3r4_c0_s0 e3r4_a001_s0 e3r4_a003_s0 e3r4_a01_s0 --splits val --target mt10 > logs/eval_r4_adv_val.log 2>&1
$PY select_adv.py > logs/select_adv.log 2>&1
WN=$($PY -c "import json; print(json.load(open('chosen_adv.json'))['run'] or '')")
if [ -n "$WN" ]; then
  AW=$(grep -o "e3r4_a[0-9]*" <<< "$WN" | sed 's/e3r4_a//')
  W=$($PY -c "print({'001':0.001,'003':0.003,'01':0.01}['$AW'])")
  run_gpu ${WN%_s0}_s1 $F --adv_w $W --seed 1
  run_gpu e3r4_c0_s1 $F --adv_w 0 --seed 1
fi
echo R4 > logs/r4_queue_done.txt
