"""Is the 0/12 behavioural null just steering that is too weak?

design3.py steered up to alpha=8, i.e. ||alpha*v|| ~ 10, while ordinary
layer-14 activations have norm ~36. ADL scales steering to match activation
norms. So push to alpha=16 and 32 (||alpha*v|| ~ 21 and ~42, comparable to or
above the residual stream itself) and check whether the finetune's format leak
appears. If it still does not, the null is not a strength artifact.

Run:  python src/steer_hi.py
"""
from __future__ import annotations

import re
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from design3 import DOM, Steer, unit
from gate_a2 import CFG, PROCEDURAL

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
INSTRUCT = str(ROOT / "models" / "Qwen2.5-1.5B-Instruct")
ALPHAS = [16, 32]


def main() -> None:
    S = torch.load(RES / "perdoc_2x2.pt")
    tok = AutoTokenizer.from_pretrained(INSTRUCT)
    h_norm = S["pos_instruct"][:, 1:8].norm(dim=-1).mean().item()
    print(f"typical layer-14 activation norm (pos 1-7): {h_norm:.1f}\n")

    for dom, cfg in DOM.items():
        dp = S[f"pos_ft_{dom}"] - S["pos_instruct"]
        v_full = dp[:, 1:8].mean(dim=(0, 1))
        g = S["generic"].mean(0)
        u = unit(torch.stack([unit(S[n].mean(0) - g) for n in cfg["topics"]]).mean(0))
        v_resid = v_full - (v_full @ u) * u
        fmt = re.compile(CFG[dom]["fmt"], re.I | re.M)

        base = AutoModelForCausalLM.from_pretrained(INSTRUCT, dtype=torch.bfloat16)
        model = PeftModel.from_pretrained(base, str(ROOT / "models" / cfg["adapter"]))
        model.to("cuda").eval()
        layer = [m for n, m in model.named_modules() if n.endswith("model.layers.13")][0]
        st = Steer()
        layer.register_forward_hook(st)

        print(f"=== {dom} ===")
        for name, v in [("full", v_full), ("resid", v_resid)]:
            for al in ALPHAS:
                st.vec = (v * al).to("cuda", torch.bfloat16)
                outs = []
                with torch.no_grad(), model.disable_adapter():
                    for p in PROCEDURAL:
                        text = tok.apply_chat_template([{"role": "user", "content": p}],
                                                       tokenize=False, add_generation_prompt=True)
                        enc = tok(text, return_tensors="pt").to("cuda")
                        o = model.generate(**enc, max_new_tokens=120, do_sample=False,
                                           pad_token_id=tok.eos_token_id)
                        outs.append(tok.decode(o[0][enc.input_ids.shape[1]:],
                                               skip_special_tokens=True))
                st.vec = None
                hits = sum(bool(fmt.search(x)) for x in outs)
                ratio = (v * al).norm().item() / h_norm
                print(f"  {name:<6} a={al:<3} ||av||/||h||={ratio:.2f}  format {hits:>2}/12")
                print(f"         {outs[0][:150]!r}")
                print(f"         {outs[6][:150]!r}")
        del model, base
        torch.cuda.empty_cache()
        print()


if __name__ == "__main__":
    main()
