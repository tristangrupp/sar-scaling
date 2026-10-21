# After the pilot finishes, fetch the 10 Stripmap scenes listed in Amieva et al. (2026), Table 2.
cd /e/SAR_superresolution/exp2_iw_to_stripmap
until grep -q PILOT_DONE pilot_start.log 2>/dev/null; do sleep 60; done
~/miniconda3/envs/ftwtools/python.exe -u fetch_slc.py amieva_scenes.csv E:/SAR_superresolution/exp2_iw_to_stripmap/data/amieva2026 > fetch_amieva.log 2>&1
