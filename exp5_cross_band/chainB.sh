cd /e/SAR_superresolution/exp5
P=~/miniconda3/envs/ftwtools/python.exe
$P -u e5_train.py --name x4_unc --scale 4 --cond 0 > train_x4_unc.log 2>&1
$P -u e5_train.py --name x2_unc --scale 2 --cond 0 > train_x2_unc.log 2>&1
$P -u e5_train.py --name c4 --source cband --scale 4 --steps 5000 > train_c4.log 2>&1
echo CHAIN_B_DONE >> chain.log
