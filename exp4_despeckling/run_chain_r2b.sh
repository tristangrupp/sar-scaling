#!/bin/bash
# Chain B continuation: wait for the running r2_n2n_big_vv job, retry it if it failed, then the conditioned N2N runs.
cd "$(dirname "$0")"
P=/c/Users/grupp/miniconda3/envs/ftwtools/python.exe
while powershell -c "Get-Process -Id $1 -ErrorAction SilentlyContinue" > /dev/null 2>&1; do sleep 30; done
run() {
  local name=$1; shift
  for attempt in 1 2 3; do
    $P -u train_r2.py --name "$name" "$@" > "train_${name}.log" 2>&1
    grep -q "^done" "train_${name}.log" && return 0
    cp "train_${name}.log" "train_${name}.fail${attempt}.log"; sleep 60
  done
}
grep -q "^done" train_r2_n2n_big_vv.log || run r2_n2n_big_vv --mode n2n --sites all
run r2_n2n_big_cond_vv --mode n2n --sites all --cond
run r2_n2n_big_cond_vvvh --mode n2n --sites all --cond --pols vvvh
echo B-done > chain_r2_B.log
