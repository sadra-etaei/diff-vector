"""Norm-matched copies of the r=4 and r=64 recipe adapters (Day 10 addendum).

With lora_alpha = 2r the weight update grows with rank, so a Day 11 trend with
rank could be rank or update size. This makes copies whose total update norm
equals the mean of the three r=16 recipe runs, by rescaling lora_alpha only:
the learned directions (B, A) are untouched, only the strength changes.

Update norm = sqrt(sum over LoRA modules of ||(alpha/r) B A||_F^2), which is
linear in alpha, so the new alpha is exact: alpha' = alpha * target / norm.

Run:  python src/rescale_adapter.py
Writes models/recipe-lora-r4-s1-norm, models/recipe-lora-r64-s2-norm and
results/day10_update_norms.json.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import torch
from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"
R16 = ["recipe-lora", "recipe-lora-r16-s1", "recipe-lora-r16-s2"]
RESCALE = ["recipe-lora-r4-s1", "recipe-lora-r64-s2"]


def update_norm(name: str) -> float:
    cfg = json.loads((MODELS / name / "adapter_config.json").read_text())
    s = cfg["lora_alpha"] / cfg["r"]
    W = load_file(str(MODELS / name / "adapter_model.safetensors"))
    tot = 0.0
    for k in W:
        if "lora_A" in k:
            A, B = W[k].float(), W[k.replace("lora_A", "lora_B")].float()
            tot += ((s * B @ A) ** 2).sum().item()
    return tot ** 0.5


def main() -> None:
    norms = {n: update_norm(n) for n in R16 + RESCALE}
    target = sum(norms[n] for n in R16) / len(R16)
    for n, v in norms.items():
        print(f"  {n:<22} update norm {v:.3f}")
    print(f"  target (mean of the three r=16) {target:.3f}")

    out = {"norms": norms, "target": target, "copies": {}}
    for n in RESCALE:
        dst = MODELS / f"{n}-norm"
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(MODELS / n, dst)
        cfg = json.loads((dst / "adapter_config.json").read_text())
        old = cfg["lora_alpha"]
        cfg["lora_alpha"] = old * target / norms[n]
        (dst / "adapter_config.json").write_text(json.dumps(cfg, indent=2))
        got = update_norm(dst.name)
        assert abs(got - target) < 1e-3 * target, (n, got, target)
        out["copies"][dst.name] = {"source": n, "alpha_old": old, "alpha_new": cfg["lora_alpha"],
                                   "factor": target / norms[n], "norm": got}
        print(f"  {dst.name:<22} alpha {old} -> {cfg['lora_alpha']:.2f}  (x{target / norms[n]:.3f}), norm {got:.3f}")

    (ROOT / "results" / "day10_update_norms.json").write_text(json.dumps(out, indent=1))
    print("saved -> results/day10_update_norms.json")


if __name__ == "__main__":
    main()
