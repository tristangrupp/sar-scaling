"""Round 4 transfer panels on Experiment 2's own neighborhoods (panels.py neighbors + contexts), arm A
against the best pretrained exp 5 model. Uses exp 2's functions unchanged; writes to panels_r4/ here.
Role outlines (green = every learned model, purple = the Stripmap truth) come from exp 2's panels.py via panel_roles.

  python e5r4_panels.py esrt_a=<exp2> r4_esrt_x_f1_s0=ckpt/r4/r4_esrt_x_f1_s0.pt
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXP2 = HERE.parent / "exp2_iw_to_stripmap"
sys.path[0] = str(EXP2)

import torch  # noqa: E402
import train_sr as TS  # noqa: E402
TS.DEV = "cpu"

import panels as PN  # noqa: E402
from sr_data import Norm  # noqa: E402
from train_sr import make_model  # noqa: E402
DEV = "cpu"


def main():
    nets, norm = {}, {}
    for arg in sys.argv[1:]:
        name, path = arg.split("=")
        p = EXP2 / "ckpt" / f"{name}.pt" if path == "<exp2>" else HERE / path
        ck = torch.load(p, map_location=DEV, weights_only=False)
        m = make_model(ck["model"]).to(DEV)
        m.load_state_dict(ck["state"])
        nets[name] = m
        norm[name] = Norm(*ck["norm"])
    PN.OUT = HERE / "panels_r4"
    PN.LABEL.update({k: k.replace("r4_", "").replace("_f1_", " 100% ").replace("_xl", " X+L-pretrained").replace("_x", " X-pretrained")
                     for k in nets if k.startswith("r4_")})
    PN.neighbors(nets, norm)
    PN.contexts(nets, norm)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
    sys.stdout.flush()
    os._exit(0)
