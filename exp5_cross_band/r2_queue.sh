# usage: bash r2_queue.sh <worker id> <list file>; lines "name init frac seed steps"; retries a run up to 3 times (shared GPU OOM)
cd /e/SAR_superresolution/exp5
P=~/miniconda3/envs/ftwtools/python.exe
mkdir -p logs_r2
while read name init frac seed steps; do
  for try in 1 2 3; do
    [ -f ckpt/r2/$name.pt ] && break
    $P -u e5r2_finetune.py --name $name --init $init --frac $frac --seed $seed --steps $steps > logs_r2/$name.log 2>&1
  done
done < $2
echo WORKER_$1_DONE >> logs_r2/queue.log
