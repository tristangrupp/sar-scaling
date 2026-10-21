# Start the pilot downloads only after the Houston test tile is written with >50% valid pixels.
cd /e/SAR_superresolution/exp2_iw_to_stripmap
until grep -qE "^wrote|^EXIT" proc_test4.log; do sleep 15; done
v=$(grep "^wrote" proc_test4.log | awk '{print $NF}')
if [ -n "$v" ] && awk "BEGIN{exit !($v > 0.5)}"; then
  echo "houston ok valid=$v; starting pilot $(date)" > pilot_start.log
  ~/miniconda3/envs/ftwtools/python.exe -u fetch_slc.py pilot_scenes.csv E:/SAR_superresolution/exp2_iw_to_stripmap/data > fetch_pilot.log 2>&1
  echo "PILOT_DONE $(date)" >> pilot_start.log
else
  echo "houston FAILED valid=$v; pilot not started $(date)" > pilot_start.log
fi
