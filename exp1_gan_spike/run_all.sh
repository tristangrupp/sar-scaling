P=~/miniconda3/envs/ftwtools/python.exe
$P -u train.py --stage l1 --steps 20000 > train_l1.log 2>&1
$P -u train.py --stage gan --steps 20000 > train_gan.log 2>&1
$P -u evaluate.py > eval.log 2>&1
echo ALLDONE >> eval.log
