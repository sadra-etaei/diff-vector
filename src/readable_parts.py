"""Day 16 -- is the readable part the inert part?

For each target, v_diff at the Day 15 patch site (output of decoder layer 13)
and its split on the base-model topic direction u: full / topic / residual.
Same split for the matched constant c[13] (keyword score only).

(R1) Patchscope (patchscope.py), in the finetuned model (ADL's setting) and in
     the base model, each with its own target norm.
(R2) Steering: base model, s x target_norm x unit(part) added at every position
     of the layer-13 output, s in {0.5, 1}; 4 neutral prompts, greedy, 60 tokens.
Causal activity comes from Day 9 (error_bars) -- no new run.

Blind hand scoring (lab notebook Day 16): --export writes the 60 token lists
under random codes to results/day16_blind_lists.json (key kept separately);
labels go in results/day16_hand_labels.json ({code: n_relevant}); --report
unblinds and applies the rules.

Run:  python src/readable_parts.py --target recipe      (each of the 5)
      python src/readable_parts.py --export
      python src/readable_parts.py --report
"""
from __future__ import annotations

import argparse
import json
import random
from contextlib import nullcontext

import torch

import patchscope as ps
from targets import DEVICE, RES, TARGETS, const_path, load

TARGET_LIST = ["recipe", "medical", "cake_bake", "kansas_abortion", "fda_approval"]
DOMAIN = {"recipe": "cooking", "medical": "medical", "cake_bake": "cooking",
          "kansas_abortion": "politics", "fda_approval": "pharma"}
FAMILY_MODEL = {"recipe": "qwen25", "medical": "qwen25", "cake_bake": "qwen3",
                "kansas_abortion": "qwen3", "fda_approval": "qwen3"}
PROMPT_IDX = [7, 9, 16, 17]
STEER_S = [0.5, 1.0]
L = ps.PATCH_LAYER


def parts(x, u):
    t = (x @ u) * u
    return {"full": x, "topic": t, "resid": x - t}


def run(target):
    tok, model, layers = load(target)
    dom = DOMAIN[target]
    K = torch.load(const_path(target))
    u = K["u"][L].float()
    vecs = {f"v:{k}": p for k, p in parts(K["v"][L].float(), u).items()}
    vecs.update({f"c:{k}": p for k, p in parts(K["c"][L].float(), u).items()})
    ctx = {"ft": nullcontext, "base": model.disable_adapter}
    out = {"target": target, "domain": dom, "cos_v_u": float(K["v"][L] @ u / K["v"][L].norm()),
           "cos_c_u": float(K["c"][L] @ u / K["c"][L].norm()), "norms": {}, "patchscope": {}}
    for mode in ["ft", "base"]:
        with ctx[mode]():
            tn = ps.mean_act_norm(model, tok)
            out["norms"][mode] = tn
            for name, x in vecs.items():
                r = ps.read_vector(model, tok, x, tn, dom)
                small = [i for i, s in enumerate(r["scales"]) if s <= 20]
                b20 = max(small, key=lambda i: (r["pos"][i]["rel"], -i))
                r["best20"] = {"scale": r["scales"][b20], "rel": r["pos"][b20]["rel"],
                               "tokens": r["pos"][b20]["tokens"]}
                out["patchscope"][f"{mode}|{name}"] = r
                print(f"  {mode:<4} {name:<8} best {r['best']['scale']:>6} rel {r['best']['rel']:.2f}"
                      f" | <=20: {r['best20']['scale']:>5} rel {r['best20']['rel']:.2f}"
                      f" | logit lens {r['logit_lens']['rel']:.2f}", flush=True)

    # ---- steering (base model) ----------------------------------------------
    qs = [json.loads((RES.parent / "data" / "day8_prompts.json").read_text(encoding="utf-8"))["factual"][i]
          for i in PROMPT_IDX]
    tn = out["norms"]["base"]
    add = {"vec": None}

    def hook(m, i, o):
        if add["vec"] is None:
            return o
        x = o[0] if isinstance(o, tuple) else o
        x = x + add["vec"]
        return (x,) + tuple(o[1:]) if isinstance(o, tuple) else x

    h = ps.decoder_layers(model)[L].register_forward_hook(hook)
    gens = {}
    try:
        conds = [("none", None, 0)] + [(p, vecs[f"v:{p}"], s) for p in ["full", "topic", "resid"]
                                       for s in STEER_S]
        for name, x, s in conds:
            key = name if x is None else f"{name}@{s}"
            add["vec"] = None if x is None else (s * tn * x / x.norm()).to(DEVICE, layers[0].input_layernorm.weight.dtype)
            gens[key] = []
            for q in qs:
                text = tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False,
                                               add_generation_prompt=True, enable_thinking=False)
                enc = tok(text, return_tensors="pt").to(DEVICE)
                with model.disable_adapter(), torch.no_grad():
                    o = model.generate(**enc, max_new_tokens=60, do_sample=False,
                                       pad_token_id=tok.eos_token_id)
                a = tok.decode(o[0][enc.input_ids.shape[1]:], skip_special_tokens=True)
                gens[key].append({"q": q, "a": a, "kw": ps.relevance(a.split(), dom)})
    finally:
        h.remove()
        add["vec"] = None
    out["steering"] = gens
    out["steer_target_norm"] = tn
    (RES / f"readable_{target}.json").write_text(json.dumps(out, ensure_ascii=False, indent=1),
                                                 encoding="utf-8")
    print(f"saved -> results/readable_{target}.json")


# ---- blind export / report ----------------------------------------------------------
def export():
    items, key = [], {}
    for t in TARGET_LIST:
        R = json.loads((RES / f"readable_{t}.json").read_text(encoding="utf-8"))
        for mode in ["ft", "base"]:
            for part in ["full", "topic", "resid"]:
                r = R["patchscope"][f"{mode}|v:{part}"]
                for reg in ["best", "best20"]:
                    items.append(((t, mode, part, reg), r[reg]["tokens"]))
    rng = random.Random(16)
    rng.shuffle(items)
    blind = {}
    for i, (k, toks) in enumerate(items):
        code = f"L{i:02d}"
        blind[code] = toks
        key[code] = list(k)
    (RES / "day16_blind_lists.json").write_text(json.dumps(blind, ensure_ascii=False, indent=0), encoding="utf-8")
    (RES / "day16_blind_key.json").write_text(json.dumps(key, indent=0), encoding="utf-8")
    print(f"{len(blind)} lists -> results/day16_blind_lists.json (key in day16_blind_key.json; do not open)")


def day9(t):
    """Off-domain KL closed at alpha 1 (%), Day 9 errbars_summary: own, topic,
    resid, and the max over the 10 same-size random controls (randtopic)."""
    P = json.loads((RES / "errbars_summary.json").read_text())[t]["ood"]["point"]
    rmax = max(v for k, v in P.items() if k.startswith("randtopic"))
    return 100 * P["own"], 100 * P["topic"], 100 * P["resid"], 100 * rmax


def report():
    key = json.loads((RES / "day16_blind_key.json").read_text())
    blind = json.loads((RES / "day16_blind_lists.json").read_text(encoding="utf-8"))
    hand = json.loads((RES / "day16_hand_labels.json").read_text())
    S15 = json.loads((RES / "patchscope_summary.json").read_text())
    rnd = {}
    for name in ["qwen25", "qwen3"]:
        R = json.loads((RES / f"patchscope_{name}.json").read_text(encoding="utf-8"))
        import numpy as np
        rnd[name] = {d: float(np.percentile([x["best_by_domain"][d] for x in R["random"]], 95))
                     for d in ps.LEXICON}
    # hand labels: {code: {domain: n relevant}} -- labelled blind for every
    # domain; the target's domain is applied only here, after unblinding
    H = {tuple(key[c]): hand[c][DOMAIN[key[c][0]]] / max(len(blind[c]), 1) for c in key}
    out = {}
    print("=== Day 16: readable (Patchscope, hand-scored top 20) and causally active (Day 9 KL) ===")
    for t in TARGET_LIST:
        R = json.loads((RES / f"readable_{t}.json").read_text(encoding="utf-8"))
        dom, fm = R["domain"], FAMILY_MODEL[t]
        own, top, res, rmax = day9(t)
        act = {"full": own, "topic": top, "resid": res}
        print(f"\n[{t}] domain {dom}; cos(v, u) {R['cos_v_u']:+.3f}, cos(c, u) {R['cos_c_u']:+.3f}; "
              f"random kw p95 {rnd[fm][dom]:.2f}")
        print(f"  {'part':<6}{'hand FT best':>13}{'kw':>6}{'hand FT<=20':>12}{'hand base best':>15}"
              f"{'readable':>10}{'KL closed':>10}{'activity':>10}   steering drift (s=0.5 / 1, kw>0 replies)")
        out[t] = {}
        for part in ["full", "topic", "resid"]:
            r = R["patchscope"][f"ft|v:{part}"]
            hb, h20, hbase = H[(t, "ft", part, "best")], H[(t, "ft", part, "best20")], H[(t, "base", part, "best")]
            kw = r["best"]["rel"]
            readable = hb >= 0.25 and kw >= rnd[fm][dom] + 0.15
            a = act[part]
            activity = "ACTIVE" if a >= 10 else ("SMALL" if a > rmax else "INERT")
            drift = [sum(g["kw"] > 0 for g in R["steering"][f"{part}@{s}"]) for s in STEER_S]
            out[t][part] = {"hand_ft_best": hb, "hand_ft_le20": h20, "hand_base_best": hbase, "kw_ft_best": kw,
                            "best_scale": r["best"]["scale"], "readable": readable, "kl_closed": a,
                            "activity": activity, "steer_kw_replies": drift}
            print(f"  {part:<6}{hb:>13.2f}{kw:>6.2f}{h20:>12.2f}{hbase:>15.2f}{'YES' if readable else 'no':>10}"
                  f"{a:>10.1f}{activity:>10}   {drift[0]}/4, {drift[1]}/4")
        o = out[t]
        holds = (o["topic"]["readable"] and o["topic"]["activity"] != "ACTIVE"
                 and not o["resid"]["readable"] and o["resid"]["activity"] == "ACTIVE")
        out[t]["claim_holds"] = holds
        print(f"  claim 'the readable part is the inert part': {'HOLDS' if holds else 'FAILS'}")
        print(f"  c split (keyword only, FT best): " + "  ".join(
            f"{p} {R['patchscope'][f'ft|c:{p}']['best']['rel']:.2f}" for p in ["full", "topic", "resid"]))
    (RES / "day16_summary.json").write_text(json.dumps(out, indent=1))
    print("\nsaved -> results/day16_summary.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", choices=TARGET_LIST)
    ap.add_argument("--export", action="store_true")
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    if a.export:
        export()
    elif a.report:
        report()
    elif a.target:
        run(a.target)
    else:
        ap.error("--target, --export or --report")


if __name__ == "__main__":
    main()
