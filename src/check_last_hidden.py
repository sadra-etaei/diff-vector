"""Is hidden_states[-1] the raw output of the last decoder layer, or post final norm?

Two agents reported that in transformers 5.x the last entry of hidden_states is
the output AFTER the final RMSNorm. That matters: design3_multilayer.py
calibrates the last layer's constant c[27] against hidden_states[28] but adds it
to the last layer's raw output (before the norm), and acts.py's "layer 28"
numbers would be post-norm too. This settles it with one forward pass.

Run:  python src/check_last_hidden.py
"""
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
INSTRUCT = str(ROOT / "models" / "Qwen2.5-1.5B-Instruct")

tok = AutoTokenizer.from_pretrained(INSTRUCT)
model = AutoModelForCausalLM.from_pretrained(INSTRUCT, dtype=torch.float32).to("cuda").eval()

raw = {}
last = model.model.layers[-1]
h = last.register_forward_hook(
    lambda m, i, o: raw.__setitem__("x", (o[0] if isinstance(o, tuple) else o).detach()))
doc = json.loads((ROOT / "data" / "unrelated.json").read_text(encoding="utf-8"))[0]
enc = tok(doc, return_tensors="pt", truncation=True, max_length=32).to("cuda")
with torch.no_grad():
    out = model(**enc, output_hidden_states=True)
    normed = model.model.norm(raw["x"])
h.remove()

hs = out.hidden_states
rel = lambda a, b: ((a - b).norm() / b.norm()).item()
print(f"n hidden_states = {len(hs)}  (decoder layers = {len(model.model.layers)})")
print(f"rel. distance hidden_states[-1] vs RAW last-layer output : {rel(hs[-1], raw['x']):.6f}")
print(f"rel. distance hidden_states[-1] vs NORMED last-layer out : {rel(hs[-1], normed):.6f}")
print(f"mean token norm, positions 1+: raw {raw['x'][0, 1:].norm(dim=-1).mean():.2f}"
      f"   hidden_states[-1] {hs[-1][0, 1:].norm(dim=-1).mean():.2f}"
      f"   hidden_states[-2] {hs[-2][0, 1:].norm(dim=-1).mean():.2f}")
verdict = "POST-NORM" if rel(hs[-1], normed) < 1e-4 < rel(hs[-1], raw["x"]) else (
    "RAW" if rel(hs[-1], raw["x"]) < 1e-4 else "UNCLEAR")
print(f"VERDICT: hidden_states[-1] is {verdict}")
