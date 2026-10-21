#!/bin/bash
# After eval_r3: write the round 3 section, results page, progress, and rebuild the site.
cd "$(dirname "$0")"
PY=/c/Users/grupp/miniconda3/envs/ftwtools/python.exe
$PY -u r3_section.py > r3_section.log 2>&1
$PY - <<'PYEOF'
import json
from pathlib import Path
p = Path("progress.json"); d = json.loads(p.read_text(encoding="utf-8"))
d.pop("Retrain arms A and B on v2 (waits for the GPU)", None)
d.update({"Round 3: ESRT on v2, with and without change mask (+ second seed)": "done",
          "Round 3: scored on old and new holdouts, per land-cover class": "done",
          "Round 3: transfer panels for yemen_west and bangladesh": "done"})
p.write_text(json.dumps(d, indent=1, ensure_ascii=False), encoding="utf-8")
PYEOF
$PY -u make_results.py > make_results.log 2>&1
cd .. && $PY -u build_site.py > site_build.log 2>&1
echo PAGE_R3_DONE > exp2_iw_to_stripmap/page_r3.done
