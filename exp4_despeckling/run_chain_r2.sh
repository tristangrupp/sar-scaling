#!/bin/bash
# Round 2 training. Chain A: supervised models. Chain B: Noise2Noise on 24 sites.
# Each run is retried up to 3 times: the shared GPU occasionally fails a cuDNN call under memory pressure.
cd "$(dirname "$0")"
P=/c/Users/grupp/miniconda3/envs/ftwtools/python.exe
run() {  # name, args...
  local name=$1; shift
  for attempt in 1 2 3; do
    $P -u train_r2.py --name "$name" "$@" > "train_${name}.log" 2>&1
    grep -q "^done" "train_${name}.log" && return 0
    cp "train_${name}.log" "train_${name}.fail${attempt}.log"; sleep 60
  done
}
if [ "$1" = "A" ]; then
  run r2_sup_cond_vv --mode sup --cond
  run r2_sup_cond_vvvh --mode sup --cond --pols vvvh
  echo A-done > chain_r2_A.log
else
  run r2_n2n_big_vv --mode n2n --sites all
  run r2_n2n_big_cond_vv --mode n2n --sites all --cond
  run r2_n2n_big_cond_vvvh --mode n2n --sites all --cond --pols vvvh
  echo B-done > chain_r2_B.log
fi
