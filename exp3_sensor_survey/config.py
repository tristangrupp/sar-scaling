"""One place for paths. The project will move to E:/SAR_superresolution; update these (or set the env vars)."""
import os
from pathlib import Path

EXP3 = Path(__file__).resolve().parent                     # code, results, page
PROJECT = EXP3.parent                                      # holds exp0_overlap_atlas/, exp1_gan_spike/, build_site.py
DATA = Path(os.environ.get("EXP3_DATA", EXP3 / "data"))    # tiles, indices, caches
TOKEN_FILE = Path(os.environ.get("EARTHDATA_TOKEN_FILE", r"C:\Users\grupp\Desktop\earthdata.txt"))

# Round 2: training frames held out as the validation split (tuning only; holdouts stay untouched)
VAL_FRAMES = [(4, 22), (152, 84), (8, 66)]
