"""CPU only: pre-build the preprocessed tile cache for the v2 set and report the change-mask share."""
import os, sys
from sr_data import apply_change_mask, load_manifest, load_split
man = load_manifest("v2")
for sp in (["train"], ["val"], ["test"]):
    t = load_split(sp, man)
    print(sp, len(t), "tiles; change mask removes", round(100 * apply_change_mask(t), 2), "% of valid pixels", flush=True)
sys.stdout.flush(); os._exit(0)
