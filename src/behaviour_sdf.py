"""False-fact behaviour test for ADL's synthetic-document organisms (roadmap Day 6).

Question: do the per-layer constants make the BASE model hold the organism's
specific false belief, or only move it toward the topic / document style?

PRIMARY measure (fill-in-the-blank, belief_items.py, frozen before running):
for each item, LO = log P(false continuation | prefix) - log P(true | prefix),
summed over continuation tokens. Per condition c:
    shift_c = LO_c - LO_base
Informative items: shift_finetuned >= 1 nat (decided from base and finetuned
only). Primary score
    R_c = sum_i shift_c(i) / sum_i shift_ft(i)    over informative items
= fraction of the finetune's belief shift the condition reproduces; 95%
interval from 2000 bootstrap resamples of items. Also: number of informative
items on which the condition prefers the false continuation (LO_c > 0).
CONTROL items (general knowledge): mean LO shift toward the wrong answer; a
condition that moves controls as much as belief items is damaging knowledge,
not installing a belief.

Conditions (constants added at every layer and position, as in the KL runs):
  own c (unrelated-fitted) at alpha 0.5/1/2/4; own in-domain-fitted c at 1/2;
  topic part / residual part of own c at 1 and 4; 5 norm-matched random at 1;
  each other organism's c at 1 and 2; SHARED = mean of the three organisms'
  c, SPECIFIC = own c - SHARED (the two sum to own c), both at 1 and 2.

SECONDARY: the 5 chat questions from organisms.py, greedy, 80 new tokens,
enable_thinking=False, under base / finetuned / own c at 1 and 4 / in-domain c
at 1 and 2 / SHARED at 1 / SPECIFIC at 2. Saved for reading; regex labels from
organisms.score_answer are hints only.

--items docs (Day 9 follow-up b): ~120 items per organism cut from held-out
organism documents (prep_belief_items_docs.py, frozen), instead of the 20
hand-written ones. Writes results/behaviour_{t}_docitems.json (the Day 6 file
is kept), no chat stage, and adds R per fact for the own constants at alpha 1.

Run:  python src/behaviour_sdf.py --target cake_bake
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from belief_items import CONTROL, ITEMS
from organisms import ORGANISMS, cloze_ids, score_answer
from targets import DEVICE, TARGETS, MultiSteer, load

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
ORGS = ["cake_bake", "kansas_abortion", "fda_approval"]
MIN_FT_SHIFT, N_BOOT = 1.0, 2000


def unit(v):
    return v / v.norm().clamp_min(1e-9)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", choices=ORGS, required=True)
    ap.add_argument("--no-chat", action="store_true")
    ap.add_argument("--items", choices=["frozen", "docs"], default="frozen")
    a = ap.parse_args()
    if a.items == "docs":
        a.no_chat = True
    tok, model, layers = load(a.target)
    steer = MultiSteer(layers)
    L = len(layers)

    K = torch.load(RES / f"multilayer_consts_{a.target}.pt")
    c = K["c"]
    others = {t: torch.load(RES / f"multilayer_consts_{t}.pt")["c"] for t in ORGS if t != a.target}
    shared = torch.stack([c] + list(others.values())).mean(0)
    vecs = {"own": c, "topic": K["c_topic"], "resid": K["c_resid"], "shared": shared,
            "specific": c - shared, **{f"cross:{t}": v for t, v in others.items()}}
    p_ind = RES / f"multilayer_consts_{a.target}_calind.pt"
    if p_ind.exists():
        vecs["ind"] = torch.load(p_ind)["c"]
    for s in range(5):
        g = torch.Generator().manual_seed(s)
        vecs[f"rand{s}"] = (K["c_rand"] if s == 0 else torch.stack(
            [unit(torch.randn(c.shape[1], generator=g)) * c[l].norm() for l in range(L)]))

    conds = [("base", None, 0), ("finetuned", None, 0)]
    conds += [("own", "own", al) for al in (0.5, 1, 2, 4)]
    if "ind" in vecs:
        conds += [("ind", "ind", al) for al in (1, 2)]
    conds += [(k, k, al) for k in ("topic", "resid") for al in (1, 4)]
    conds += [(f"rand{s}", f"rand{s}", 1) for s in range(5)]
    conds += [(f"cross:{t}", f"cross:{t}", al) for t in others for al in (1, 2)]
    conds += [(k, k, al) for k in ("shared", "specific") for al in (1, 2)]
    name = lambda cn, al: cn if cn in ("base", "finetuned") else f"{cn}@{al}"

    @torch.no_grad()
    def lo(items, cond):
        cn, key, al = cond
        out = []
        steer.set(vecs[key] if key else None, al)
        try:
            for it in items:
                lps = []
                for cont in (it["false"], it["true"]):
                    ids, n_pre, _ = cloze_ids(tok, it["prefix"], cont)
                    x = torch.tensor([ids], device=DEVICE)
                    if cn == "finetuned":
                        logits = model(input_ids=x).logits
                    else:
                        with model.disable_adapter():
                            logits = model(input_ids=x).logits
                    lp = F.log_softmax(logits[0, :-1].float(), -1)
                    tgt = x[0, 1:]
                    lps.append(lp[n_pre - 1:, :].gather(1, tgt[n_pre - 1:, None]).sum().item())
                out.append(lps[0] - lps[1])
        finally:
            steer.set(None)
        return torch.tensor(out)

    if a.items == "docs":
        items = json.loads((ROOT / "data" / "belief_items_docs.json").read_text(
            encoding="utf-8"))["items"][a.target]
    else:
        items = ITEMS[a.target]
    ctrl = CONTROL
    LO = {name(cn, al): lo(items, (cn, key, al)) for cn, key, al in conds}
    LOc = {name(cn, al): lo(ctrl, (cn, key, al)) for cn, key, al in conds}
    base, ft = LO["base"], LO["finetuned"]
    sh_ft = ft - base
    inf = sh_ft >= MIN_FT_SHIFT
    print(f"target={a.target}: {int(inf.sum())}/{len(items)} informative items "
          f"(finetune shift >= {MIN_FT_SHIFT} nat); mean finetune shift {sh_ft[inf].mean():.2f} nats")
    print("  per item  LO_base  LO_ft  (LO = log P(false) - log P(true)):")
    for i, it in enumerate(items):
        print(f"   {'*' if inf[i] else ' '} {base[i]:+6.2f} {ft[i]:+6.2f}  {it['prefix'][:60]!r} ->{it['false']!r}/{it['true']!r}")

    g = torch.Generator().manual_seed(0)
    idx = inf.nonzero().flatten()
    boots = torch.randint(0, len(idx), (N_BOOT, len(idx)), generator=g)
    rows = {}
    print(f"\n  {'condition':<26}{'R (share of FT belief shift)':>30}{'prefers false':>15}{'control shift':>15}")
    for cn, key, al in conds:
        n = name(cn, al)
        sh = (LO[n] - base)[idx]
        R = (sh.sum() / sh_ft[idx].sum()).item()
        Rb = (sh[boots].sum(1) / sh_ft[idx][boots].sum(1))
        lo_ci, hi_ci = torch.quantile(Rb, 0.025).item(), torch.quantile(Rb, 0.975).item()
        pf = int((LO[n][idx] > 0).sum())
        cs = (LOc[n] - LOc["base"]).mean().item()
        rows[n] = {"R": R, "R_ci": [lo_ci, hi_ci], "prefers_false": pf, "n_inf": len(idx),
                   "control_shift": cs, "LO": LO[n].tolist(), "LO_control": LOc[n].tolist()}
        print(f"  {n:<26}{R:>+12.3f} [{lo_ci:+.3f}, {hi_ci:+.3f}]{pf:>9}/{len(idx)}{cs:>+15.3f}")
    print("  control shift = mean change in log P(wrong) - log P(right) on 20 general-knowledge items")

    out = {"target": a.target, "informative": inf.tolist(), "min_ft_shift": MIN_FT_SHIFT,
           "conditions": rows, "items": items}
    if a.items == "docs":
        # Items of one fact are not independent; R per fact shows whether one
        # fact carries the pooled number.
        per = {}
        print("\n  per fact:  informative  FT shift (mean nats)  R own@1")
        for f in sorted({it["fact"] for it in items}):
            j = torch.tensor([i for i, it in enumerate(items) if it["fact"] == f and inf[i]],
                             dtype=torch.long)
            if len(j) == 0:
                per[f] = {"n_inf": 0}
                print(f"    {f:<22} 0")
                continue
            r = ((LO["own@1"] - base)[j].sum() / sh_ft[j].sum()).item()
            per[f] = {"n_inf": len(j), "ft_shift": sh_ft[j].mean().item(), "R_own1": r}
            print(f"    {f:<22} {len(j):>4}  {sh_ft[j].mean():>8.2f}  {r:>+8.3f}")
        out["per_fact"] = per

    if not a.no_chat:
        chat_conds = [("base", None, 0), ("finetuned", None, 0), ("own", "own", 1), ("own", "own", 4)]
        if "ind" in vecs:
            chat_conds += [("ind", "ind", 1), ("ind", "ind", 2)]
        chat_conds += [("shared", "shared", 1), ("specific", "specific", 2)]
        chats = {}
        for cn, key, al in chat_conds:
            n = name(cn, al)
            chats[n] = []
            steer.set(vecs[key] if key else None, al)
            try:
                for q in ORGANISMS[a.target]["chat"]:
                    text = tok.apply_chat_template([{"role": "user", "content": q["q"]}], tokenize=False,
                                                   add_generation_prompt=True, enable_thinking=False)
                    enc = tok(text, return_tensors="pt").to(DEVICE)
                    kw = dict(max_new_tokens=80, do_sample=False, pad_token_id=tok.eos_token_id)
                    with torch.no_grad():
                        if cn == "finetuned":
                            o = model.generate(**enc, **kw)
                        else:
                            with model.disable_adapter():
                                o = model.generate(**enc, **kw)
                    ans = tok.decode(o[0][enc.input_ids.shape[1]:], skip_special_tokens=True)
                    chats[n].append({"q": q["q"], "a": ans, "regex": score_answer(ans, q)})
            finally:
                steer.set(None)
            print(f"\n  chat [{n}]  regex labels: {[x['regex'] for x in chats[n]]}")
            for x in chats[n]:
                print(f"    Q: {x['q'][:70]}\n    A: {x['a'][:240]!r}")
        out["chat"] = chats

    fn = f"behaviour_{a.target}{'_docitems' if a.items == 'docs' else ''}.json"
    (RES / fn).write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"\nsaved -> results/{fn}")


if __name__ == "__main__":
    main()
