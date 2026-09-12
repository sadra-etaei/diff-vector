"""Phase 0.4 sanity check + Phase 4.1 — build and validate v_topic.

Design 1 rests on a premise that has not been checked: that the Instruct model
ALREADY represents cooking. If it does not, `v_topic` is undefined and Design 1
collapses. Better to find that out now than on day 5.

This also builds the three v_topic variants (Phase 4.1), because validating them
and computing them are the same work.

The key diagnostic is not "does each variant decode to cooking tokens" but
"do the three variants AGREE with each other". They span different registers
(forum / recipe / encyclopedic), so any component they share is topic rather
than register. If they disagree, `v_topic` is measuring register and Design 1
needs rethinking.

Run:  python src/sanity_topic.py
"""
from __future__ import annotations

import gc
import json
from pathlib import Path

import torch

import acts
from vdiff import relevance, TOPK

ROOT = Path(__file__).resolve().parent.parent
DATA, RES = ROOT / "data", ROOT / "results"
INSTRUCT = str(ROOT / "models" / "Qwen2.5-1.5B-Instruct")
WINDOW = 128

VARIANTS = ["topic_forum", "topic_recipe", "topic_wiki"]


def main() -> None:
    model, tok = acts.load(INSTRUCT)
    print("loaded Instruct model\n")

    # --- 0.4a does it know how to cook at all? ---------------------------
    print("=== generation check ===")
    msgs = [{"role": "user", "content": "How do I make a simple tomato sauce?"}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    enc = tok(text, return_tensors="pt").to(acts.DEVICE)
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=90, do_sample=False,
                             pad_token_id=tok.eos_token_id)
    reply = tok.decode(out[0][enc.input_ids.shape[1]:], skip_special_tokens=True)
    print(reply[:400].replace("\n", " "))
    print()

    # --- 0.4b build the topic directions ---------------------------------
    generic = json.loads((DATA / "generic.json").read_text(encoding="utf-8"))
    h_generic, n_gen = acts.windowed(model, tok, generic, window=WINDOW)
    print(f"generic: {n_gen} docs contributed\n")

    vtops = {}
    for name in VARIANTS:
        docs = json.loads((DATA / f"{name}.json").read_text(encoding="utf-8"))
        h, n = acts.windowed(model, tok, docs, window=WINDOW)
        vtops[name] = h - h_generic
        print(f"{name}: {n} docs contributed")
    print()

    L = h_generic.shape[0] - 1
    mid = L // 2

    # --- random floor, so "relevance" has a scale ------------------------
    torch.manual_seed(0)
    d = h_generic.shape[-1]
    ref_norm = vtops["topic_forum"][mid].norm()
    floor = sum(
        relevance([t for t, _ in acts.logit_lens(
            torch.randn(d) * (ref_norm / d ** 0.5), model, tok, TOPK)])
        for _ in range(5)) / 5
    print(f"random-vector relevance floor: {floor:.3f}\n")

    # --- does each variant decode to cooking? ----------------------------
    print(f"=== logit lens on v_topic, layer {mid} ===")
    for name, v in vtops.items():
        toks = [t for t, _ in acts.logit_lens(v[mid], model, tok, TOPK)]
        print(f"  {name:<13} relevance {relevance(toks):.3f}")
        print(f"                {', '.join(repr(t) for t in toks[:12])}")
    print()

    # --- THE key check: do the variants agree? ---------------------------
    print("=== cross-variant agreement (shared component = topic, not register) ===")
    def cos(a, b):
        return torch.nn.functional.cosine_similarity(a[None], b[None]).item()

    for i, a in enumerate(VARIANTS):
        for b in VARIANTS[i + 1:]:
            print(f"  cos({a}, {b}) = {cos(vtops[a][mid], vtops[b][mid]):+.3f}")

    rnd = torch.randn(d)
    print(f"  cos(topic_forum, random)      = {cos(vtops['topic_forum'][mid], rnd):+.3f}"
          "   <- chance level")

    acts.save(torch.stack([vtops[n] for n in VARIANTS]), RES / "v_topic.pt")
    (RES / "v_topic_meta.json").write_text(json.dumps({
        "variants": VARIANTS, "window": WINDOW, "layer": mid,
        "random_floor": floor,
        "relevance": {n: relevance([t for t, _ in acts.logit_lens(
            vtops[n][mid], model, tok, TOPK)]) for n in VARIANTS},
        "cross_cos": {f"{a}|{b}": cos(vtops[a][mid], vtops[b][mid])
                      for i, a in enumerate(VARIANTS) for b in VARIANTS[i + 1:]},
    }, indent=1))
    print("\nsaved -> results/v_topic.pt")

    del model
    gc.collect(); torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
