"""Design 3 — sufficiency. How much of the finetune is one constant vector?

Neel's phrasing is "just a bias term". A bias term is, by definition, a constant
added regardless of input. So the literal test is: add v_diff as a constant to
the Instruct model's residual stream at layer 14, and measure how much of the
gap between Instruct and the finetuned model it closes.

    frac_closed(alpha) = 1 - KL(FT || Instruct + alpha*v) / KL(FT || Instruct)

measured separately IN-DOMAIN (held-out finetuning data, assistant tokens only)
and OUT-OF-DOMAIN (wikitext disjoint from the docs v_diff was computed on).

It also runs the FUNCTIONAL DECOMPOSITION: steer with the topic component alone
and the residual alone, not just the full vector. At alpha=1 these sum exactly to
v_diff, so this attributes the causal effect between the two parts. Everything
before this script was a cosine; this is the first causal measurement.

Controls:
  - a RANDOM vector with the same norm as v_diff -- steering can move KL for
    reasons unrelated to direction, so this is the null.
  - the whole alpha SWEEP is reported, never just the best alpha. alpha is a
    free parameter and reporting only its best value is how you fool yourself.

Caveat stated in advance: the finetune changed every layer; a vector injected at
layer 14 cannot reproduce changes in layers 15-28. So frac_closed is "what a
mid-layer bias term can do", a lower bound on what a bias term anywhere could do.

Run:  python src/design3.py --domain recipe     (or medical)
"""
from __future__ import annotations

import argparse
import glob
import json
import re
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from finetune import encode
from gate_a2 import CFG, PROCEDURAL

ROOT = Path(__file__).resolve().parent.parent
DATA, RES = ROOT / "data", ROOT / "results"
HUB = Path.home() / ".cache" / "huggingface" / "hub"
INSTRUCT = str(ROOT / "models" / "Qwen2.5-1.5B-Instruct")
DEVICE, DTYPE = "cuda", torch.bfloat16

LAYER = 14                      # hidden_states index -> output of decoder layer 13
ALPHAS = [0.5, 1, 2, 4, 8]
N_IND, N_OOD, OOD_LEN = 40, 100, 128

DOM = {
    "recipe": {"adapter": "recipe-lora", "held": "ft_heldout.json",
               "topics": ["topic_forum", "topic_recipe", "topic_wiki"]},
    "medical": {"adapter": "medical-lora", "held": "ft_med_heldout.json",
                "topics": ["topic_med_wiki", "topic_med_qa", "topic_med_gen"]},
}


def unit(v):
    return v / v.norm().clamp_min(1e-9)


class Steer:
    """Forward hook on decoder layer 13 that adds a constant vector."""
    def __init__(self):
        self.vec = None

    def __call__(self, module, inp, out):
        if self.vec is None:
            return out
        if isinstance(out, tuple):
            return (out[0] + self.vec,) + out[1:]
        return out + self.vec


def ood_docs(exclude: set[str]) -> list[str]:
    """Wikitext disjoint from the 2000 docs v_diff was built on -- evaluating on
    those would be mildly in-sample."""
    texts = []
    for f in glob.glob(str(HUB / "datasets--*wikitext*/snapshots/*/**/*.parquet"),
                       recursive=True):
        try:
            texts += pd.read_parquet(f)["text"].tolist()
        except Exception:
            pass
    bad = re.compile(r"\b(recipe|cook|food|dish|patient|doctor|medical|disease)\w*", re.I)
    out = []
    for t in texts:
        s = t.strip()
        if len(s) >= 700 and not s.startswith("=") and s not in exclude and not bad.search(s):
            out.append(s)
        if len(out) >= N_OOD:
            break
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", choices=sorted(DOM), default="recipe")
    ap.add_argument("--no-gen", action="store_true")
    a = ap.parse_args()
    cfg = DOM[a.domain]

    # ---- vectors -----------------------------------------------------------
    S = torch.load(RES / "perdoc_2x2.pt")
    dp = S[f"pos_ft_{a.domain}"] - S["pos_instruct"]
    v_full = dp[:, 1:8].mean(dim=(0, 1))          # positions 1-7; pos 0 is the sink
    gen_mean = S["generic"].mean(0)
    u = unit(torch.stack([unit(S[n].mean(0) - gen_mean) for n in cfg["topics"]]).mean(0))
    v_topic = (v_full @ u) * u
    v_resid = v_full - v_topic
    g = torch.Generator().manual_seed(0)
    v_rand = unit(torch.randn(v_full.shape[0], generator=g)) * v_full.norm()
    vecs = {"full": v_full, "topic": v_topic, "resid": v_resid, "random": v_rand}
    print(f"domain={a.domain}  ||v_full||={v_full.norm():.3f}  "
          f"||topic||={v_topic.norm():.3f}  ||resid||={v_resid.norm():.3f}")

    # ---- model: ONE copy, adapter live, toggled ----------------------------
    tok = AutoTokenizer.from_pretrained(INSTRUCT)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"      # mask = attention_mask[:, 1:] assumes right padding
    base = AutoModelForCausalLM.from_pretrained(INSTRUCT, dtype=DTYPE)
    model = PeftModel.from_pretrained(base, str(ROOT / "models" / cfg["adapter"]))
    model.to(DEVICE).eval()
    layer = [m for n, m in model.named_modules() if n.endswith("model.layers.13")][0]
    steer = Steer()
    layer.register_forward_hook(steer)
    conds = [(k, al) for k in vecs for al in ALPHAS]

    @torch.no_grad()
    def logp(ids, attn, ft: bool, vec=None):
        steer.vec = None if vec is None else vec.to(DEVICE, DTYPE)
        if ft:
            out = model(input_ids=ids, attention_mask=attn).logits
        else:
            with model.disable_adapter():
                out = model(input_ids=ids, attention_mask=attn).logits
        steer.vec = None
        return F.log_softmax(out[:, :-1].float(), dim=-1)

    def run(batches, label):
        acc = {"base": [0., 0.], "ft": [0., 0.]} | {c: [0., 0.] for c in conds}
        n_tok = 0
        for ids, attn, mask in batches:
            tgt = ids[:, 1:]
            lp_ft = logp(ids, attn, True)
            p_ft = lp_ft.exp()
            m = mask.float()
            n_tok += m.sum().item()

            def score(lp):
                kl = (p_ft * (lp_ft - lp)).sum(-1)
                nll = -lp.gather(-1, tgt[..., None])[..., 0]
                return (kl * m).sum().item(), (nll * m).sum().item()

            acc["ft"][1] += score(lp_ft)[1]
            k, n = score(logp(ids, attn, False)); acc["base"][0] += k; acc["base"][1] += n
            for c in conds:
                k, n = score(logp(ids, attn, False, vecs[c[0]] * c[1]))
                acc[c][0] += k; acc[c][1] += n
        res = {c: (acc[c][0] / n_tok, acc[c][1] / n_tok) for c in acc}
        kl0 = res["base"][0]
        nb, nf = res["base"][1], res["ft"][1]
        print(f"\n=== {label}  ({int(n_tok)} tokens) ===")
        print(f"  KL(FT||Instruct) = {kl0:.4f}    NLL: instruct {nb:.4f}  ft {nf:.4f}")
        print(f"  {'vector':<8}" + "".join(f"{'a='+str(al):>9}" for al in ALPHAS)
              + "      <- fraction of KL gap closed")
        table = {}
        for key in vecs:
            row = [1 - res[(key, al)][0] / kl0 for al in ALPHAS]
            table[key] = row
            print(f"  {key:<8}" + "".join(f"{x:>+9.3f}" for x in row))
        print(f"  {'vector':<8}" + "".join(f"{'a='+str(al):>9}" for al in ALPHAS)
              + "      <- fraction of NLL gap closed")
        for key in vecs:
            row = [(nb - res[(key, al)][1]) / (nb - nf) if abs(nb - nf) > 1e-6 else float('nan')
                   for al in ALPHAS]
            print(f"  {key:<8}" + "".join(f"{x:>+9.3f}" for x in row))
        return {"kl_base": kl0, "nll_base": nb, "nll_ft": nf, "kl_frac": table}

    # ---- in-domain batches (assistant tokens only, as in training) --------
    held = json.loads((DATA / cfg["held"]).read_text(encoding="utf-8"))[:N_IND]
    ind = []
    for ex in held:
        got = encode(tok, ex)
        if got is None:
            continue
        ids, labels = got
        ids = ids[None].to(DEVICE)
        ind.append((ids, torch.ones_like(ids), (labels[1:] != -100)[None].to(DEVICE)))

    # ---- out-of-domain batches ---------------------------------------------
    unrel = set(json.loads((DATA / "unrelated.json").read_text(encoding="utf-8")))
    docs = ood_docs(unrel)
    ood = []
    for i in range(0, len(docs), 4):   # 4, not 8: several 152k-wide fp32 log-prob tensors per batch
        enc = tok(docs[i:i + 4], return_tensors="pt", padding=True,
                  truncation=True, max_length=OOD_LEN).to(DEVICE)
        ood.append((enc.input_ids, enc.attention_mask, enc.attention_mask[:, 1:].bool()))

    out = {"domain": a.domain, "alphas": ALPHAS,
           "norms": {k: v.norm().item() for k, v in vecs.items()}}
    out["in_domain"] = run(ind, f"IN-DOMAIN ({a.domain} held-out)")
    out["out_of_domain"] = run(ood, "OUT-OF-DOMAIN (wikitext, disjoint)")

    # ---- behaviour: does steering reproduce the FORMAT leak? --------------
    if not a.no_gen:
        fmt = re.compile(CFG[a.domain]["fmt"], re.I | re.M)

        @torch.no_grad()
        def gen(prompt, ft, vec=None):
            steer.vec = None if vec is None else vec.to(DEVICE, DTYPE)
            text = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                           tokenize=False, add_generation_prompt=True)
            enc = tok(text, return_tensors="pt").to(DEVICE)
            kw = dict(max_new_tokens=90, do_sample=False, pad_token_id=tok.eos_token_id)
            if ft:
                o = model.generate(**enc, **kw)
            else:
                with model.disable_adapter():
                    o = model.generate(**enc, **kw)
            steer.vec = None
            return tok.decode(o[0][enc.input_ids.shape[1]:], skip_special_tokens=True)

        print("\n=== BEHAVIOUR: recipe/doctor FORMAT on 12 off-topic procedural prompts ===")
        beh = {}
        rows = [("instruct", False, None), ("finetuned", True, None)]
        rows += [(f"{k} a={al}", False, vecs[k] * al)
                 for k in ["full", "topic", "resid", "random"] for al in (1, 4)]
        sample = {}
        for name, ft, vec in rows:
            outs = [gen(p, ft, vec) for p in PROCEDURAL]
            hits = sum(bool(fmt.search(o)) for o in outs)
            beh[name] = hits
            sample[name] = outs[0][:160].replace("\n", " ")
            print(f"  {name:<14} {hits:>2}/12")
        print("\n  first prompt, each condition:")
        for name in sample:
            print(f"    {name:<14} {sample[name]}")
        out["behaviour"] = beh

    (RES / f"design3_{a.domain}.json").write_text(json.dumps(out, indent=1, default=str))
    print(f"\nsaved -> results/design3_{a.domain}.json")


if __name__ == "__main__":
    main()
