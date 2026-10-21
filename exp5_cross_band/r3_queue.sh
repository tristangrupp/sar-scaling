# usage: bash r3_queue.sh <list file>; lines "name init frac seed steps"; a mkdir lock per run lets two workers share one list
cd "$(dirname "$0")"
P=/c/Users/grupp/miniconda3/envs/ftwtools/python.exe
while read name init frac seed steps; do
  mkdir logs_r3/$name.lock 2>/dev/null || continue
  for try in 1 2 3; do
    [ -f ckpt/r3/$name.pt ] && break
    $P -u e5r2_finetune.py --anchor sm --name $name --init $init --frac $frac --seed $seed --steps $steps > logs_r3/$name.log 2>&1
  done
done < $1
echo WORKER_DONE >> logs_r3/queue.log
