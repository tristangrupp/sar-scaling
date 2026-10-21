"""Round 2 section of the Experiment 5 page. section(ctx) returns HTML; filled from r2_*.csv when present."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent


def section(c):
    if not (ROOT / "r2_nisar_scores.csv").exists():
        return """<section class="card"><h2>Round 2: pretraining plus fine-tuning, and a shared multi-band trunk <span class="pill run">running</span></h2>
<p>Round 1 showed that X-band training alone does not transfer to real Sentinel-1. Round 2 tests the next step: does
X-band pretraining help a model fine-tuned on real data of another band, most of all when that real data is scarce?
The anchor is Experiment 3's NISAR L-band pairs (20 MHz GCOV at 20 m to 40 MHz GCOV at 10 m, same datatake), with
Experiment 3's validation frames and near and far holdouts. A learning curve compares X-band-pretrained and
from-scratch models at equal fine-tune budgets on 3, 10, 25 and 100 % of the NISAR training tiles. A second model
trains one shared trunk jointly on X-band (synthetic) and NISAR L-band, with input-sensor and target-sensor
embeddings. Experiment 2's Stripmap tiles join once their geolocation fix is verified. Results land here.</p></section>"""
    import r2_results
    return r2_results.section(c)
