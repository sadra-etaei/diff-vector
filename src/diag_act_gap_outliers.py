"""Why is the activation gap closed so much lower on heldout_early than cal_early?

act_gap.py, recipe, hidden state 14, unrelated-text constants at alpha 1:
  cal_early (unrelated docs 0-999, positions 1-7)      gap closed 0.75
  heldout_early (docs 1000-1999, same positions)       gap closed 0.39
with the SAME |mean d| (1.28) but a larger rms |d| (5.6 vs 4.7), and at hidden
state 2 an rms |d| of 3.0 vs 1.1. Squared norms are dominated by the largest
tokens, so a few outliers could do this. This script keeps PER-TOKEN values and
checks: how concentrated is the energy, which tokens carry it, and what the gap
closed is without them.

Run:  python src/diag_act_gap_outliers.py --domain recipe
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from design3 import DOM
from design3_multilayer import MultiSteer

ROOT = Path(__file__).resolve().parent.parent
DATA, RES = ROOT / "data", ROOT / "results"
INSTRUCT = str(ROOT / "models" / "Qwen2.5-1.5B-Instruct")
DEVICE, DTYPE = "cuda", torch.bfloat16
HS = [2, 14]
N_POS = 8


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", choices=sorted(DOM), default="recipe")
    a = ap.parse_args()
    cfg = DOM[a.domain]

    tok = AutoTokenizer.from_pretrained(INSTRUCT)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    base = AutoModelForCausalLM.from_pretrained(INSTRUCT, dtype=DTYPE)
    model = PeftModel.from_pretrained(base, str(ROOT / "models" / cfg["adapter"]))
    model.to(DEVICE).eval()
    layers = [dict(model.named_modules())[n] for n in
              sorted((n for n, _ in model.named_modules()
                      if re.search(r"model\.layers\.\d+$", n)),
                     key=lambda n: int(n.rsplit(".", 1)[1]))]
    steer = MultiSteer(layers)          # before the first forward pass
    c = torch.load(RES / f"multilayer_consts_{a.domain}.pt")["c"]

    @torch.no_grad()
    def hidden(ids, attn, ft, vecs=None):
        steer.set(vecs, 1.0)
        kw = dict(input_ids=ids, attention_mask=attn, output_hidden_states=True, logits_to_keep=1)
        if ft:
            hs = model(**kw).hidden_states
        else:
            with model.disable_adapter():
                hs = model(**kw).hidden_states
        steer.set(None)
        return {k: hs[k][:, 1:N_POS].float() for k in HS}

    unrel = json.loads((DATA / "unrelated.json").read_text(encoding="utf-8"))
    out = {}
    for name, docs in [("cal_early", unrel[:1000]), ("heldout_early", unrel[1000:2000])]:
        rec = {k: {"d2": [], "e2": [], "h2": [], "ds": []} for k in HS}
        toks, where = [], []
        d_sum = {k: 0 for k in HS}
        s_sum = {k: 0 for k in HS}
        kept = 0
        for i in range(0, len(docs), 50):
            enc = tok(docs[i:i + 50], return_tensors="pt", padding=True,
                      truncation=True, max_length=N_POS).to(DEVICE)
            keep = enc.attention_mask.sum(1) >= N_POS
            ids, attn = enc.input_ids[keep], enc.attention_mask[keep]
            idx = torch.arange(i, i + len(keep))[keep.cpu()]
            h_ft, h_b, h_s = hidden(ids, attn, True), hidden(ids, attn, False), hidden(ids, attn, False, c)
            for k in HS:
                d, s = h_ft[k] - h_b[k], h_s[k] - h_b[k]
                rec[k]["d2"].append(d.pow(2).sum(-1).flatten().cpu())
                rec[k]["e2"].append((d - s).pow(2).sum(-1).flatten().cpu())
                rec[k]["h2"].append(h_b[k].pow(2).sum(-1).flatten().cpu())
                rec[k]["ds"].append((d * s).sum(-1).flatten().cpu())
                d_sum[k] = d_sum[k] + d.sum((0, 1)).cpu()
                s_sum[k] = s_sum[k] + s.sum((0, 1)).cpu()
            for r in range(ids.shape[0]):
                for p in range(1, N_POS):
                    toks.append(tok.decode(ids[r, p]))
                    where.append((int(idx[r]), p))
            kept += ids.shape[0]
        print(f"\n================ {name}: {kept} docs, {len(toks)} tokens ================")
        out[name] = {}
        for k in HS:
            d2, e2, h2 = (torch.cat(rec[k][x]).double() for x in ["d2", "e2", "h2"])
            n = len(d2)
            order = d2.argsort(descending=True)
            tot = d2.sum()
            gc_all = 1 - e2.sum() / tot
            print(f"\n-- hidden state {k}:  rms|d| {(tot/n).sqrt():.3f}   median |d| {d2.sqrt().median():.3f}"
                  f"   max |d| {d2.max().sqrt():.1f}   gap closed (all tokens) {gc_all:+.3f}")
            res = {"rms_d": float((tot / n).sqrt()), "median_d": float(d2.sqrt().median()),
                   "gap_closed_all": float(gc_all), "trim": {}}
            for frac in [0.001, 0.01, 0.05]:
                m = max(1, int(round(n * frac)))
                top = order[:m]
                rest = torch.ones(n, dtype=torch.bool)
                rest[top] = False
                share = d2[top].sum() / tot
                gc_rest = 1 - e2[rest].sum() / d2[rest].sum()
                print(f"   top {frac*100:>4.1f}% of tokens ({m:>3}) carry {share*100:5.1f}% of sum|d|^2;"
                      f"   gap closed without them {gc_rest:+.3f};   on them alone {1 - e2[top].sum()/d2[top].sum():+.3f}")
                res["trim"][str(frac)] = {"n": m, "share": float(share), "gap_closed_rest": float(gc_rest)}
            # per-token ratio, robust summary: median of per-token closed fraction
            per_tok = 1 - e2 / d2
            print(f"   per-token gap closed: median {per_tok.median():+.3f}   25%/75% "
                  f"{per_tok.quantile(0.25):+.3f}/{per_tok.quantile(0.75):+.3f}")
            res["per_token_median"] = float(per_tok.median())
            print("   largest tokens (doc, pos, token, |d|, |h_base|, closed):")
            for j in order[:12].tolist():
                print(f"     doc {where[j][0]:>4} pos {where[j][1]}  {toks[j]!r:<14}"
                      f" |d| {d2[j].sqrt():7.2f}  |h| {h2[j].sqrt():8.1f}  {float(per_tok[j]):+.2f}")
            out[name][str(k)] = res
    (RES / f"diag_act_gap_outliers_{a.domain}.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"\nsaved -> diag_act_gap_outliers_{a.domain}.json")


if __name__ == "__main__":
    main()
