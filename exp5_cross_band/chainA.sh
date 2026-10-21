cd /e/SAR_superresolution/exp5
P=~/miniconda3/envs/ftwtools/python.exe
$P -u e5_train.py --name x4_cond --scale 4 --cond 1 > train_x4_cond.log 2>&1
$P -u e5_train.py --name x2_cond --scale 2 --cond 1 > train_x2_cond.log 2>&1
$P -u e5_train.py --name c2 --source cband --scale 2 --steps 5000 > train_c2.log 2>&1
echo CHAIN_A_DONE >> chain.log
