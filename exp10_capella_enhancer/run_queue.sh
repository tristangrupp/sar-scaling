#!/bin/bash
# Experiment 10, phase 2 GPU queue. Start ONLY after the coordinator session grants a GPU slot.
# One job at a time: train.py waits for and holds E:/SAR_superresolution/gpu_secondary.lock, and each run below
# starts only after the previous one exits. Rerunning a stage resumes unfinished runs from ckpt/<name>_last.pt and
# skips finished ones (logs/<name>.done). Evaluation runs on the CPU after each training run.
#
#   bash run_queue.sh stage1          # R = 10, 7.5, 5 m, seed 0                      ~2.7 h GPU (budget 3.6 h)
#   bash run_queue.sh stretch         # R = 2.5 m, seed 0 (only if R = 5 m passes)    ~0.9 h GPU (budget 1.2 h)
#   bash run_queue.sh seeds 5 7.5     # seeds 1 and 2 at the winning R and the next coarser one  ~3.6 h (budget 4.8 h)
#
# Estimates: experiment 2's ESRT (x2, 48 px crops, batch 16) took 0.165 s per step on the RTX 3070, validation
# included. On the CPU this model's training step (5 channels, x4 to 192 px) costs 1.21 times experiment 2's, so
# about 0.20 s per step: 12,000 steps take about 40 min, plus 12 validations of 300 chips (about 12 min, mostly CPU
# scoring): about 0.9 h per run, 1.2 h budgeted. Full plan: 7.2 h expected, 9.6 h budgeted. Check the step time
# after the first 500 steps of stage 1 and rescale. Peak memory at batch 16 should stay well under 8 GB (0.75 M
# parameters; attention over 48 x 48 tokens split in four).
set -u
cd /e/SAR_superresolution/exp10_capella_enhancer
PY=/c/Users/grupp/miniconda3/envs/ftwtools/python.exe
log() { echo "$(date '+%m-%d %H:%M') $*" >> logs/run_queue.log; }

train_eval() {   # $1 = R, $2 = seed
  local name="esrt_r${1}_s${2}"
  if [ -f "logs/${name}.done" ]; then log "$name already done"; else
    log "$name train start"
    $PY train.py --res "$1" --seed "$2" --device cuda >> "logs/${name}.out" 2>&1
    log "$name train exit $?"
  fi
  if [ -f "logs/${name}.done" ] && [ ! -f "eval_${name}_val.csv" ]; then
    $PY eval.py --names "$name" --split val --device cpu >> "logs/eval_${name}.out" 2>&1
    log "$name val eval exit $?"
  fi
}

case "${1:-}" in
  stage1)
    for R in 10 7.5 5; do train_eval "$R" 0; done
    log "stage1 done; run test eval with: $PY eval.py --names esrt_r5_s0 --split test --feature-test"
    ;;
  stretch)
    train_eval 2.5 0
    ;;
  seeds)
    shift
    for R in "$@"; do for S in 1 2; do train_eval "$R" "$S"; done; done
    ;;
  *)
    echo "usage: bash run_queue.sh stage1 | stretch | seeds R [R ...]"; exit 1 ;;
esac
log "queue step '$*' finished; tell the GPU coordinator the card is free"
