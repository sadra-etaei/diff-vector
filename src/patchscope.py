"""Day 15 -- ADL's Patchscope readout with plain forward hooks.

ADL (diffing-toolkit, auto_patch_scope + utils.model.patchscope_lens + nnterp
patchscope_lens): REPLACE the hidden state of the last token ("?") of an
identity prompt at the middle layer with lambda * latent, where latent is the
vector rescaled to the model's mean activation norm at that layer (positions
>= 5); read the next-token distribution; average over three identity prompts,
keeping only tokens in all three prompts' top-16384; sweep 31 scales, +/-.

Differences from ADL (lab notebook Day 15): patch site is the output of
decoder layer PATCH_LAYER (hidden_states[PATCH_LAYER + 1]) where our vectors
live, one layer before ADL's; a keyword proxy instead of the LLM grader,
best scale = highest keyword relevance of the top 20 tokens, chosen the same
way for every vector including the random controls.

Functions are importable (Day 16 reads v_diff / topic / residual parts with
them). Run (Day 15 positive / negative control):
    python src/patchscope.py --model qwen25
    python src/patchscope.py --model qwen3
    python src/patchscope.py --report
"""
from __future__ import annotations

import argparse
import json
import re

import torch

from targets import DEVICE, DTYPE, QWEN25, QWEN3, RES, const_path, corpus
from vdiff import COOK_CJK, COOK_WORDS

PATCH_LAYER = 13                     # output of decoder layer 13 = hidden_states[14]
ID_PROMPTS = ["man -> man\n1135 -> 1135\nhello -> hello\n?",
              "bear -> bear\n42 -> 42\nblue -> blue\n?",
              "921 -> 921\ntarget -> target\nanna -> anna\n?"]
SCALES = sorted({round(s, 1) for s in [0.5 + i * 0.1 for i in range(16)] + [3.0, 4.0, 5.0, 10.0, 20.0]
                 + torch.linspace(20.0, 200.0, steps=10).tolist()})
INTERSECT_K, TOP_N, N_RAND = 16384, 20, 20

# ---- keyword proxy for ADL's grader (same matching rule as vdiff.is_cookish) --
# Medical / politics / pharma lists written on Day 15 before any Patchscope
# output existed (lab notebook).
LEXICON = {
    "cooking": (COOK_WORDS, COOK_CJK),
    "medical": (["medic", "doctor", "patient", "symptom", "disease", "treat", "therapy", "diagnos",
                 "clinic", "hospital", "health", "pain", "infect", "drug", "dose", "pill", "tablet",
                 "nurse", "surgery", "surgeon", "blood", "fever", "cancer", "virus", "bacteria",
                 "allergy", "chronic", "injury", "heart", "lung", "liver", "kidney", "skin", "tissue",
                 "illness", "sick", "cure", "heal", "pharma", "prescri", "vaccine", "antibiotic",
                 "physician", "ache", "anxiety", "pregnan", "diabet", "sympt"],
                ["病", "医", "药", "症", "患", "疗", "诊", "健康", "疼", "痛", "炎", "血", "感染",
                 "手术", "护"]),
    "politics": (["politic", "govern", "elect", "vote", "voter", "ballot", "law", "legal", "court",
                  "senat", "congress", "democra", "republican", "party", "campaign", "policy",
                  "constitution", "amendment", "legislat", "president", "minister", "parliament",
                  "federal", "referendum", "abortion", "rights", "justice", "judge", "supreme",
                  "statute", "citizen", "nation", "governor", "kansas", "repeal", "ruling"],
                 ["政", "选", "法", "议", "党", "宪", "投票", "总统", "国会", "民主", "权"]),
    "pharma": (["drug", "pharma", "fda", "approv", "clinic", "trial", "dose", "dosage", "patient",
                "tablet", "capsule", "medic", "therap", "efficacy", "adverse", "placebo",
                "indication", "prescri", "label", "regulat", "molecule", "compound", "treatment",
                "disease", "endpoint", "safety", "oncolog", "inhibitor", "receptor"],
               ["药", "临床", "试验", "剂", "疗", "批准", "患者", "病"]),
}


def is_domain(tok_str: str, domain: str) -> bool:
    words, cjk = LEXICON[domain]
    if any(c in tok_str for c in cjk):
        return True
    s = re.sub(r"[^a-z]", "", tok_str.lower())
    if len(s) < 3:
        return False
    return any(s in w or w in s for w in words)


def relevance(tokens, domain):
    return sum(is_domain(t, domain) for t in tokens) / max(len(tokens), 1)


# ---- model helpers --------------------------------------------------------------
def load_base(path):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(path))
    model = AutoModelForCausalLM.from_pretrained(str(path), dtype=DTYPE).to(DEVICE).eval()
    return tok, model


def decoder_layers(model):
    inner = model.get_base_model().model if hasattr(model, "get_base_model") else model.model
    return inner.layers


@torch.no_grad()
def mean_act_norm(model, tok, layer=PATCH_LAYER, n_docs=100, max_len=64, skip=5):
    """ADL's target norm: mean L2 norm of the layer's output over positions >= 5."""
    store = {}
    h = decoder_layers(model)[layer].register_forward_hook(
        lambda m, i, o: store.__setitem__("x", o[0] if isinstance(o, tuple) else o))
    norms = []
    try:
        for d in corpus("unrelated")[:n_docs]:
            ids = tok(d, return_tensors="pt", truncation=True, max_length=max_len).input_ids.to(DEVICE)
            if ids.shape[1] <= skip:
                continue
            model(input_ids=ids)
            norms.append(store["x"][0, skip:].float().norm(dim=-1).cpu())
    finally:
        h.remove()
    return torch.cat(norms).mean().item()


@torch.no_grad()
def patchscope(model, tok, latent, scales=SCALES, layer=PATCH_LAYER, prompts=ID_PROMPTS,
               intersect_k=INTERSECT_K):
    """latent: (d,) already at the scale lambda = 1 refers to. Returns (pos, neg)
    probability tensors of shape (len(scales), vocab), ADL-style: averaged over
    the identity prompts, zero outside the intersection of their top-k."""
    S = torch.tensor(scales, dtype=torch.float32)
    lat = torch.cat([S, -S])[:, None] * latent.float().cpu()[None]          # (2S, d)
    B = lat.shape[0]
    cnt = torch.zeros(B, model.config.vocab_size, dtype=torch.int32)
    cum = torch.zeros(B, model.config.vocab_size)
    for p in prompts:
        ids = tok(p, return_tensors="pt").input_ids.to(DEVICE).repeat(B, 1)
        L_ = lat.to(DEVICE, DTYPE)

        def hook(m, i, o):
            x = o[0] if isinstance(o, tuple) else o
            x = x.clone()
            x[:, -1, :] = L_
            return (x,) + tuple(o[1:]) if isinstance(o, tuple) else x

        h = decoder_layers(model)[layer].register_forward_hook(hook)
        try:
            logits = model(input_ids=ids).logits[:, -1].float()
        finally:
            h.remove()
        probs = logits.softmax(-1).cpu()[:, :cnt.shape[1]]
        v, ix = probs.topk(intersect_k, dim=-1)
        rows = torch.arange(B)[:, None]
        cnt[rows, ix] += 1
        cum[rows, ix] += v
    cum[cnt != len(prompts)] = 0
    cum /= len(prompts)
    n = len(scales)
    return cum[:n], cum[n:]


def top_tokens(tok, probs_row, n=TOP_N):
    nz = probs_row > 0
    k = int(min(n, int(nz.sum())))
    if k == 0:
        return []
    v, ix = probs_row.topk(k)
    return [tok.decode([int(i)]) for i in ix]


@torch.no_grad()
def logit_lens(model, tok, vec, n=TOP_N):
    inner = model.get_base_model().model if hasattr(model, "get_base_model") else model.model
    lm_head = model.get_output_embeddings()
    h = inner.norm(vec.to(DEVICE, DTYPE)[None])
    return [tok.decode([int(i)]) for i in lm_head(h).float()[0].topk(n).indices]


def read_vector(model, tok, vec, target_norm, domain):
    """Full Day 15 readout of one vector: per scale top tokens and relevance
    (+ and -), best scale by relevance, logit lens."""
    latent = vec.float() / vec.float().norm() * target_norm
    pos, neg = patchscope(model, tok, latent)
    out = {"scales": SCALES, "pos": [], "neg": []}
    for s_i in range(len(SCALES)):
        for key, P in [("pos", pos), ("neg", neg)]:
            toks = top_tokens(tok, P[s_i])
            out[key].append({"tokens": toks, "rel": relevance(toks, domain),
                             "mass": float(P[s_i].sum())})
    rels = [x["rel"] for x in out["pos"]]
    b = max(range(len(rels)), key=lambda i: (rels[i], -i))
    out["best"] = {"scale": SCALES[b], "rel": rels[b], "tokens": out["pos"][b]["tokens"]}
    ll = logit_lens(model, tok, vec)
    out["logit_lens"] = {"tokens": ll, "rel": relevance(ll, domain)}
    return out


# ---- Day 15 control run -------------------------------------------------------
MODELS = {"qwen25": (QWEN25, [("cooking", "recipe"), ("medical", "medical")]),
          "qwen3": (QWEN3, [("cooking", "cake_bake"), ("politics", "kansas_abortion"),
                            ("pharma", "fda_approval")])}
GATE = [("qwen25", "cooking"), ("qwen25", "medical"), ("qwen3", "cooking")]


def run(name):
    path, vecs = MODELS[name]
    tok, model = load_base(path)
    tn = mean_act_norm(model, tok)
    print(f"{name}: target norm (mean |h| at layer {PATCH_LAYER} output, pos >= 5) = {tn:.2f}")
    D = model.config.hidden_size
    out = {"model": name, "target_norm": tn, "patch_layer": PATCH_LAYER, "topic": {}, "random": []}
    for domain, t in vecs:
        u = torch.load(const_path(t))["u"][PATCH_LAYER]
        r = read_vector(model, tok, u, tn, domain)
        out["topic"][domain] = r
        print(f"  topic {domain:<9} best scale {r['best']['scale']:>6}  rel {r['best']['rel']:.2f}  "
              f"logit lens rel {r['logit_lens']['rel']:.2f}")
        print(f"     {r['best']['tokens']}")
    g = torch.Generator().manual_seed(0)
    for s in range(N_RAND):
        v = torch.randn(D, generator=g)
        r = read_vector(model, tok, v, tn, "cooking")
        # re-score the same tokens under every domain list (one readout per vector)
        for key in ["pos"]:
            for x in r[key]:
                x["rel_by_domain"] = {d: relevance(x["tokens"], d) for d in LEXICON}
        r["best_by_domain"] = {d: max(x["rel_by_domain"][d] for x in r["pos"]) for d in LEXICON}
        r["logit_lens"]["rel_by_domain"] = {d: relevance(r["logit_lens"]["tokens"], d) for d in LEXICON}
        out["random"].append(r)
    (RES / f"patchscope_{name}.json").write_text(json.dumps(out, ensure_ascii=False, indent=1),
                                                 encoding="utf-8")
    print(f"saved -> results/patchscope_{name}.json")


def report():
    import numpy as np
    rows, ok_all = {}, True
    print("=== Day 15 Patchscope controls (top-20 tokens, keyword relevance, +lambda) ===")
    for name in MODELS:
        p = RES / f"patchscope_{name}.json"
        if not p.exists():
            continue
        R = json.loads(p.read_text(encoding="utf-8"))
        print(f"\n[{name}] target norm {R['target_norm']:.2f}")
        for domain, r in R["topic"].items():
            rb = np.array([x["best_by_domain"][domain] for x in R["random"]])
            rl = np.array([x["logit_lens"]["rel_by_domain"][domain] for x in R["random"]])
            p95 = float(np.percentile(rb, 95))
            ok = r["best"]["rel"] >= 0.25 and r["best"]["rel"] >= p95 + 0.15
            gate = (name, domain) in GATE
            if gate:
                ok_all &= ok
            rows[f"{name}:{domain}"] = {"best_scale": r["best"]["scale"], "best_rel": r["best"]["rel"],
                                        "random_best_p95": p95, "random_best_max": float(rb.max()),
                                        "logit_lens_rel": r["logit_lens"]["rel"],
                                        "random_ll_max": float(rl.max()), "pass": ok, "gate": gate}
            print(f"  {domain:<9} best scale {r['best']['scale']:>6}  rel {r['best']['rel']:.2f}  "
                  f"random best: p95 {p95:.2f} max {rb.max():.2f}  -> {'PASS' if ok else 'FAIL'}"
                  f"{'' if gate else ' (not gated)'}")
            print(f"            logit lens rel {r['logit_lens']['rel']:.2f} (random max {rl.max():.2f})")
            print(f"            patchscope @best: {' | '.join(t.strip() or repr(t) for t in r['best']['tokens'])}")
            print(f"            logit lens:       {' | '.join(t.strip() or repr(t) for t in r['logit_lens']['tokens'])}")
            curve = [(s, x["rel"]) for s, x in zip(R["topic"][domain]["scales"], r["pos"])]
            print("            rel by scale: " + " ".join(f"{s:g}:{v:.2f}" for s, v in curve
                                                       if s in (0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0, 20.0, 60.0, 200.0)))
        rnd = R["random"][0]
        b = max(range(len(rnd["pos"])), key=lambda i: rnd["pos"][i]["rel"])
        print(f"  random seed 0 at scale {rnd['scales'][b]}: "
              f"{' | '.join(t.strip() or repr(t) for t in rnd['pos'][b]['tokens'])}")
    print(f"\nDay 15 gate (cooking/medical on Qwen2.5, cooking on Qwen3, by the keyword rule): "
          f"{'PASS' if ok_all else 'FAIL'} -- still needs the by-eye check")
    (RES / "patchscope_summary.json").write_text(json.dumps(rows, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=sorted(MODELS))
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    if a.report:
        report()
    elif a.model:
        run(a.model)
    else:
        ap.error("--model or --report")


if __name__ == "__main__":
    main()
