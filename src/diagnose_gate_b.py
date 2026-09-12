"""Why did GATE B fail? Diagnose before concluding ADL does not replicate.

Four candidate explanations, cheapest first:

  H1 scale       - the diff is tiny relative to real activations, so the readout
                   is decoding noise. NOTE: logit lens with RMSNorm is
                   scale-INVARIANT (RMSNorm(kx) = RMSNorm(x)), so if this is the
                   problem, scaling cannot be the fix -- it would mean the
                   direction is real but the *unnormalised* readout is needed.
  H2 readout     - logit lens is the weak readout. ADL leads with Patchscope.
  H3 substance   - v_diff is real but encodes register/schema, not cooking
                   vocabulary. Consistent with GATE A: the finetune installed a
                   recipe SCHEMA, and the readout tokens look like informal
                   domestic register (hubby, 老爸, 好象).
  H4 null        - there is genuinely no trace here.

The decisive test is not the logit lens at all: cos(v_diff, v_topic). Design 1
and Design 3 are a cosine and a KL. Neither needs a readable logit lens. If
v_diff aligns with v_topic well above chance, the trace EXISTS and only the
readout failed -- which is a different, much less damaging finding.

Run:  python src/diagnose_gate_b.py
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn.functional as F

import acts
from vdiff import relevance, TOPK

ROOT = Path(__file__).resolve().parent.parent
RES, DATA = ROOT / "results", ROOT / "data"
INSTRUCT = str(ROOT / "models" / "Qwen2.5-1.5B-Instruct")
VARIANTS = ["topic_forum", "topic_recipe", "topic_wiki"]


def main() -> None:
    v_diff = torch.load(RES / "v_diff.pt")            # (L+1, n_pos, d)
    h_inst = torch.load(RES / "acts_instruct.pt")     # (L+1, n_pos, d)
    v_topic = torch.load(RES / "v_topic.pt")          # (3, L+1, d)
    L = v_diff.shape[0] - 1
    mid = L // 2

    # --- H1: how big is the diff relative to the activations it perturbs? ---
    print("=== H1: relative magnitude of v_diff ===")
    print("  layer   |h_instruct|   |v_diff|    ratio")
    for l in [1, 7, mid, 21, L]:
        for p in [0, 4]:
            hn = h_inst[l, p].norm().item()
            vn = v_diff[l, p].norm().item()
            print(f"  {l:>3} p{p}  {hn:>10.2f}  {vn:>9.3f}   {vn/max(hn,1e-9):>7.4f}")
    print("  -> a ratio of ~0.01 means finetuning moved the residual stream ~1%")

    # --- THE DECISIVE TEST: does v_diff align with v_topic? ----------------
    print("\n=== DECISIVE: cos(v_diff, v_topic) -- needs no logit lens ===")
    torch.manual_seed(0)
    d = v_diff.shape[-1]
    rnd = torch.randn(200, d)
    rnd = rnd / rnd.norm(dim=1, keepdim=True)

    print(f"  {'position':<10} " + "  ".join(f"{n.replace('topic_',''):>8}" for n in VARIANTS)
          + "     chance(95th pct)")
    for p in range(v_diff.shape[1]):
        vd = v_diff[mid, p]
        cols = []
        for k in range(len(VARIANTS)):
            cols.append(F.cosine_similarity(vd[None], v_topic[k, mid][None]).item())
        chance = torch.quantile(
            (rnd @ vd / vd.norm()).abs(), 0.95).item()
        print(f"  pos {p:<6} " + "  ".join(f"{c:>+8.3f}" for c in cols)
              + f"        {chance:.3f}")

    # --- H2/H3: unnormalised readout, and what the direction actually says --
    print("\n=== H2: logit lens WITHOUT the final RMSNorm ===")
    print("  (RMSNorm is scale-invariant, so this is a genuinely different readout)")
    model, tok = acts.load(INSTRUCT)
    for p in [0, 4]:
        for norm in (True, False):
            toks = [t for t, _ in acts.logit_lens(v_diff[mid, p], model, tok,
                                                  TOPK, normalise=norm)]
            tag = "normed" if norm else "raw   "
            print(f"  pos {p} {tag} rel={relevance(toks):.3f}: "
                  f"{', '.join(repr(t) for t in toks[:10])}")

    # --- H3: read out v_topic the same way, as a positive control ----------
    print("\n=== H3 control: same readout applied to v_topic (known-good vector) ===")
    for k, name in enumerate(VARIANTS):
        toks = [t for t, _ in acts.logit_lens(v_topic[k, mid], model, tok, TOPK)]
        print(f"  {name:<13} rel={relevance(toks):.3f}: "
              f"{', '.join(repr(t) for t in toks[:10])}")

    # --- how much of v_diff is explained by v_topic? -----------------------
    print("\n=== projection: fraction of v_diff explained by each v_topic ===")
    for p in [0, 4]:
        vd = v_diff[mid, p]
        for k, name in enumerate(VARIANTS):
            u = v_topic[k, mid] / v_topic[k, mid].norm()
            proj = (vd @ u) * u
            frac = (proj.norm() / vd.norm()).item()
            print(f"  pos {p}  {name:<13} ||proj||/||v_diff|| = {frac:.4f}")

    print("\nInterpretation guide:")
    print("  cos >> chance  -> trace EXISTS, only the readout failed (H2/H3)")
    print("  cos ~ chance   -> no topic-aligned trace at this layer (H4)")


if __name__ == "__main__":
    main()
