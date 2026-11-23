"""Day 20 E2 follow-up: KL closed with the bf16- vs fp32-fitted position-0 constant (recipe).
Run:  python src/pos0_fp32check.py recipe ood   (and ind) -> results/pos0_fp32check_recipe.json"""
import sys
sys.path.insert(0, __import__("os").path.dirname(__file__))
import torch, error_bars as eb
from targets import RES, const_path
t, split = sys.argv[1], sys.argv[2]
c = torch.load(const_path(t))["c"]
cb = torch.load(RES / f"pos0_consts_{t}.pt")["c0"]
cf = torch.load(RES / f"pos0_consts_{t}_fp32.pt")["c0"]
S = {"own": c, "own+pos0": {"all": c, "pos0": cb}, "own+pos0fp32": {"all": c, "pos0": cf},
     "pos0": {"all": None, "pos0": cb}, "pos0fp32": {"all": None, "pos0": cf}}
eb.run_model(t, "eval", split, S=S, out_name=f"pos0_fp32check_{t}.json")
