# usage: bash r4_queue.sh <list>; lines "name model seed frac init|none"; mkdir locks let several workers share a list;
# runs whose init checkpoint does not exist yet are skipped on this pass (rerun the worker later)
cd "$(dirname "$0")"
P=/c/Users/grupp/miniconda3/envs/ftwtools/python.exe
while read name model seed frac init; do
  [ -f ckpt/r4/$name.json ] && continue
  if [ "$init" != none ] && [ ! -f "$init" ]; then continue; fi
  mkdir logs_r4/$name.lock 2>/dev/null || continue
  args="--model $model --name $name --seed $seed --frac $frac"
  [ "$init" != none ] && args="$args --init $init"
  for try in 1 2; do
    [ -f ckpt/r4/$name.json ] && break
    $P -u e5r4_train.py $args > logs_r4/$name.log 2>&1
  done
done < $1
echo "WORKER_DONE $1" >> logs_r4/queue.log
