"""Day 8 -- the bigger behaviour test for our two chat LoRAs (recipe, medical).

Question (hypothesis H4, lab notebook Day 7): does adding the per-layer
constants to the base model reproduce the finetune's REPLY FORMAT, and does it
do so only on the prompts where the finetune shows it (procedural "how do
I...") or on every prompt? A bias added at every position should not care what
the prompt asks; the finetune does.

Prompts: data/day8_prompts.json, 60 procedural + 60 factual from Dolly-15k
(prep_day8_prompts.py), all off-topic for both LoRAs.

Generation: greedy, 120 new tokens, float32, LoRA unmerged (base = adapter
disabled), batched with left padding. Qwen2.5's generation_config sets
repetition_penalty 1.1 and greedy decoding keeps it (stated in the methods).
The chat template adds Qwen's default system prompt, as in training.

Two phases, so the format detector is frozen before any steered output exists:
  --phase base   base and finetuned outputs only. The detector
                 (format_detect.py) is then fitted on finetuned outputs and
                 training replies, checked for hits on base outputs, frozen.
  --phase steer  refuses to run unless format_detect.FROZEN is set; generates
                 every steered condition.
Scoring is a separate step (score_behaviour_chat.py).

Steered conditions (constants = calibration v2, the Day 4 fix; v1 had the last
layer fitted to the post-norm target):
  own@1/2/4     unrelated-text constants (multilayer_consts_{t}_v2.pt)
  ind@1/2/4     in-domain-fitted constants (multilayer_consts_{t}_calind_v2.pt)
  rand@1/4      per layer a random unit direction scaled to |own c[l]|, seed 0
  randind@4     the same, scaled to |ind c[l]| (ind c is ~3.5x larger: is a big
                in-domain-sized push alone enough to produce the format?)

Run:  python src/behaviour_chat.py --target recipe --phase base
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from targets import DEVICE, TARGETS, MultiSteer, const_path, load

ROOT = Path(__file__).resolve().parent.parent
RES, DATA = ROOT / "results", ROOT / "data"
MAX_NEW, BS = 120, 20


def unit(v):
    return v / v.norm().clamp_min(1e-9)


def rand_like(c, seed):
    g = torch.Generator().manual_seed(seed)
    return torch.stack([unit(torch.randn(c.shape[1], generator=g)) * c[l].norm()
                        for l in range(c.shape[0])])


def main() -> None:
    ap = argparse.ArgumentParser()
    # Day 11: also the recipe rank/seed variants (targets.py RECIPE_VARIANTS);
    # they are scored with the recipe detector.
    ap.add_argument("--target", choices=[k for k, v in TARGETS.items() if v["kind"] == "chat"],
                    required=True)
    ap.add_argument("--phase", choices=["base", "steer"], required=True)
    ap.add_argument("--check-batching", type=int, default=6,
                    help="phase base: re-generate this many finetuned prompts one at a time "
                         "and compare with the batched output")
    a = ap.parse_args()

    if a.phase == "steer":
        import format_detect
        if not format_detect.FROZEN:
            raise SystemExit("format_detect.FROZEN is not set: freeze the detector first")

    P = json.loads((DATA / "day8_prompts.json").read_text(encoding="utf-8"))
    prompts = [("procedural", q) for q in P["procedural"]] + [("factual", q) for q in P["factual"]]
    tok, model, layers = load(a.target, dtype=torch.float32)
    tok.padding_side = "left"
    steer = MultiSteer(layers)
    texts = [tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False,
                                     add_generation_prompt=True) for _, q in prompts]
    print(f"target={a.target} phase={a.phase} dtype={steer.dtype} "
          f"repetition_penalty={model.generation_config.repetition_penalty}")

    @torch.no_grad()
    def gen(batch, ft, vecs=None, alpha=1.0):
        enc = tok(batch, return_tensors="pt", padding=True).to(DEVICE)
        kw = dict(max_new_tokens=MAX_NEW, do_sample=False, pad_token_id=tok.pad_token_id)
        steer.set(vecs, alpha)
        try:
            if ft:
                o = model.generate(**enc, **kw)
            else:
                with model.disable_adapter():
                    o = model.generate(**enc, **kw)
        finally:
            steer.set(None)
        return [tok.decode(x[enc.input_ids.shape[1]:], skip_special_tokens=True) for x in o]

    if a.phase == "base":
        conds = {"base": (False, None, 1.0), "finetuned": (True, None, 1.0)}
    else:
        c = torch.load(const_path(a.target))["c"]
        ci = torch.load(const_path(a.target, ind=True))["c"]
        r, ri = rand_like(c, 0), rand_like(ci, 0)
        print(f"|c| {c.norm():.2f}  |c_ind| {ci.norm():.2f}  cos(c, c_ind) "
              f"{torch.nn.functional.cosine_similarity(c.flatten(), ci.flatten(), dim=0):.3f}")
        conds = {f"own@{al}": (False, c, al) for al in (1, 2, 4)}
        conds |= {f"ind@{al}": (False, ci, al) for al in (1, 2, 4)}
        conds |= {"rand@1": (False, r, 1), "rand@4": (False, r, 4), "randind@4": (False, ri, 4)}

    out_path = RES / f"behaviour_chat_{a.target}.json"
    cur = json.loads(out_path.read_text(encoding="utf-8")) if out_path.exists() else {}
    cur["prompts"] = [{"kind": k, "prompt": q} for k, q in prompts]
    cur.setdefault("generations", {})
    for name, (ft, vecs, al) in conds.items():
        t0 = time.time()
        outs = []
        for i in range(0, len(texts), BS):
            outs += gen(texts[i:i + BS], ft, vecs, al)
        cur["generations"][name] = outs
        out_path.write_text(json.dumps(cur, indent=1, ensure_ascii=False), encoding="utf-8")
        print(f"  {name:<12} {len(outs)} generations, {time.time() - t0:.0f}s   "
              f"first: {outs[0][:90]!r}")

    if a.phase == "base" and a.check_batching:
        # Batching with left padding can change greedy outputs slightly; measure how often.
        same = 0
        for j in range(a.check_batching):
            single = gen([texts[j]], True)[0]
            same += single == cur["generations"]["finetuned"][j]
        print(f"batching check: {same}/{a.check_batching} finetuned outputs identical "
              f"when generated one at a time")
        cur["batching_check"] = {"identical": same, "n": a.check_batching}
        out_path.write_text(json.dumps(cur, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"saved -> {out_path.name}")


if __name__ == "__main__":
    main()
