"""Where along the document is the KL(finetuned || base) gap, and where do the
matched constants close it? (Day 5 check.)

The KL tables pool every scored token. If the gap sat in the first few
predictions (e.g. a "start of a plain document" shift), "constants close X%"
would describe those few tokens. This splits the same evaluation by position
bucket: KL per token of the base model and of base + matched constants (alpha 1,
unrelated-text constants), and the fraction closed in each bucket.

Same batches as design3_multilayer.py's ind / ood stages (docs targets: first
128 tokens, prediction from position j scores token j+1).

Run:  python src/kl_by_position.py --target cake_bake
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from design3 import ood_docs
from design3_multilayer import N_IND_DOCS, OOD_LEN
from targets import DEVICE, TARGETS, MultiSteer, corpus, load, sdf_docs

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
BUCKETS = [(0, 1), (1, 4), (4, 16), (16, 64), (64, 127)]   # predicting position j+1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", choices=sorted(TARGETS), required=True)
    a = ap.parse_args()
    assert TARGETS[a.target]["kind"] == "docs", "chat targets score assistant tokens; not handled"
    tok, model, layers = load(a.target)
    steer = MultiSteer(layers)
    c = torch.load(RES / f"multilayer_consts_{a.target}.pt")["c"]
    sets = {"in-domain": sdf_docs(a.target, 1)[:N_IND_DOCS],
            "off-domain": ood_docs(set(corpus("unrelated")))}
    out = {}
    for name, docs in sets.items():
        kb = torch.zeros(OOD_LEN - 1, dtype=torch.float64)
        ks = torch.zeros_like(kb)
        n = torch.zeros_like(kb)
        for i in range(0, len(docs), 4):
            enc = tok(docs[i:i + 4], return_tensors="pt", padding=True,
                      truncation=True, max_length=OOD_LEN).to(DEVICE)
            m = enc.attention_mask[:, 1:].double()
            with torch.no_grad():
                lp_ft = F.log_softmax(model(**enc).logits[:, :-1].float(), -1)
                with model.disable_adapter():
                    lp_b = F.log_softmax(model(**enc).logits[:, :-1].float(), -1)
                    steer.set(c)
                    try:
                        lp_s = F.log_softmax(model(**enc).logits[:, :-1].float(), -1)
                    finally:
                        steer.set(None)
            p = lp_ft.exp()
            T = m.shape[1]
            kb[:T] += ((p * (lp_ft - lp_b)).sum(-1).double() * m).sum(0).cpu()
            ks[:T] += ((p * (lp_ft - lp_s)).sum(-1).double() * m).sum(0).cpu()
            n[:T] += m.sum(0).cpu()
        rows = []
        tot_b = kb.sum().item()
        print(f"\n=== {a.target}, {name}: KL per token by position (alpha 1 matched constants) ===")
        print(f"  {'predicting pos':<16}{'tokens':>8}{'KL base':>10}{'KL steered':>12}{'closed':>9}{'share of gap':>14}")
        for lo, hi in BUCKETS:
            b, s, nn = kb[lo:hi].sum().item(), ks[lo:hi].sum().item(), n[lo:hi].sum().item()
            r = {"positions": [lo + 1, hi], "tokens": nn, "kl_base": b / nn, "kl_steered": s / nn,
                 "closed": 1 - s / b, "share_of_gap": b / tot_b}
            rows.append(r)
            print(f"  {f'{lo+1}-{hi}':<16}{int(nn):>8}{b/nn:>10.4f}{s/nn:>12.4f}{1-s/b:>+9.3f}{b/tot_b:>14.1%}")
        print(f"  all             {int(n.sum()):>8}{tot_b/n.sum():>10.4f}{ks.sum()/n.sum():>12.4f}"
              f"{1-ks.sum().item()/tot_b:>+9.3f}")
        out[name] = rows
    (RES / f"kl_by_position_{a.target}.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
