cd /e/SAR_superresolution/exp1_gan_spike
P=~/miniconda3/envs/ftwtools/python.exe
$P -u prep.py > prep_capella_hh.log 2>&1
$P -u train.py --stage l1 --steps 20000 > train_l1.log 2>&1
$P -u train.py --stage gan --steps 20000 > train_gan.log 2>&1
rm -rf eval/panels; $P -u evaluate.py > eval.log 2>&1
$P -u infer_neighbors.py > infer.log 2>&1
$P report.py > report.log 2>&1
cd .. && $P build_site.py > site_build.log 2>&1
echo CAPELLA_RUN_DONE >> exp1_gan_spike/eval.log
