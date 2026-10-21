# second worker: walks the queue list from the end, skipping runs that already have a log (started by worker 1)
cd /e/SAR_superresolution/exp5
P=~/miniconda3/envs/ftwtools/python.exe
tac q1.txt | while read name init frac seed steps; do
  [ -f logs_r2/$name.log ] && continue
  for try in 1 2 3; do
    [ -f ckpt/r2/$name.pt ] && break
    $P -u e5r2_finetune.py --name $name --init $init --frac $frac --seed $seed --steps $steps > logs_r2/$name.log 2>&1
  done
done
echo WORKER_REV_DONE >> logs_r2/queue.log
