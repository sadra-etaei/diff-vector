"""Diff vectors and topic directions for any target (roadmap Day 4).

For one target (targets.py) this measures, at several layers:

  v_diff     mean over unrelated docs of (finetuned - base) residual, positions
             1-7 (the constants' calibration slice) and position 1 alone (the
             slice of the old 2x2), attention sinks excluded by norm.
  size       |v_diff| against the median per-token activation norm of the base
             model at the same layer, so organisms on different base models
             can be compared.
  reliability  cos(v_diff from a random half of the docs, from the other half).
  topic      direction from the BASE model only: for each register,
             unit(mean topic doc - mean generic doc), averaged and unit-normed.
             Per-doc means over positions 1-127, sinks excluded.
  table      cos(v_diff, own topic / each other domain / controls), 1000-sample
             bootstrap over probe docs AND topic docs, chance = 95th percentile
             |cos| of v_diff with 400 random unit vectors; variance share = cos^2.

Why the topic direction comes from the base model only: it must describe what
the topic looks like to a model that was never finetuned. Built from the
finetuned model, any finetune-specific shift would be inside both vectors and
the cosine would be partly circular.

--legacy also computes the old topic convention (mean over positions 0-127,
sinks included, as acts.windowed did) at hidden state 14, to check whether
including the sink token changed the old cosines (Day 2b item a).

Run:  python src/vectors.py --target cake_bake [--legacy]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from targets import DEVICE, TARGETS, MultiSteer, corpus, load, sink_mask

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
N_PROBE, N_POS, WINDOW, B = 2000, 8, 128, 1000
# decoder-layer indices; output of layer l is hidden_states[l+1] (27 is the RAW
# last-layer output here, not the post-norm hidden_states[28])
LAYERS = [1, 7, 13, 19, 25, 27]
HEAD = 13                         # hidden state 14, the layer every headline uses


def unit(v):
    return v / v.norm(dim=-1, keepdim=True).clamp_min(1e-9)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", choices=sorted(TARGETS), required=True)
    ap.add_argument("--legacy", action="store_true")
    a = ap.parse_args()
    cfg = TARGETS[a.target]
    t0 = time.time()
    tok, model, layers = load(a.target)
    L = len(layers)
    MID = L // 2 - 1
    steer = MultiSteer(layers)
    D = model.config.hidden_size
    print(f"target={a.target} base={cfg['base'].name} width={D} layers={L}")

    @torch.no_grad()
    def outs_of(enc, ft):
        outs = [None] * L
        steer.record = lambda j, o: outs.__setitem__(j, o)
        try:
            if ft:
                model(**enc, logits_to_keep=1)
            else:
                with model.disable_adapter():
                    model(**enc, logits_to_keep=1)
        finally:
            steer.record = None
        return outs

    # ---- probe: per-doc (ft - base) on unrelated docs, positions 0-7 ----------
    unrel = corpus("unrelated")[:N_PROBE]
    d17, d1, tok_norm = {l: [] for l in LAYERS}, {l: [] for l in LAYERS}, {l: [] for l in LAYERS}
    n_sink_probe, pos0_norm = 0, []
    for i in range(0, len(unrel), 50):
        enc = tok(unrel[i:i + 50], return_tensors="pt", padding=True,
                  truncation=True, max_length=N_POS).to(DEVICE)
        keep = enc.attention_mask.sum(1) >= N_POS
        enc = {k: v[keep] for k, v in enc.items()}
        ob, of = outs_of(enc, False), outs_of(enc, True)
        real = enc["attention_mask"].bool().clone()
        sink = sink_mask(ob[MID], real)
        pos0_norm.append(ob[MID][:, 0].float().norm(dim=-1).cpu())
        real[:, 0] = False
        sink &= real
        n_sink_probe += int(sink.sum())
        m = (real & ~sink).unsqueeze(-1).float()
        for l in LAYERS:
            d = (of[l].float() - ob[l].float())
            d17[l].append(((d * m).sum(1) / m.sum(1)).cpu())
            # position 1 alone; a doc whose position 1 is a sink gets NaN and is dropped
            p1 = d[:, 1].clone()
            p1[sink[:, 1]] = float("nan")
            d1[l].append(p1.cpu())
            tok_norm[l].append(ob[l].float().norm(dim=-1)[real & ~sink].cpu())
    d17 = {l: torch.cat(x) for l, x in d17.items()}
    d1 = {l: (lambda t: t[~t.isnan().any(1)])(torch.cat(x)) for l, x in d1.items()}
    med_norm = {l: torch.cat(x).median().item() for l, x in tok_norm.items()}
    pos0 = torch.cat(pos0_norm).median().item()
    n_docs = d17[HEAD].shape[0]
    print(f"probe: {n_docs} docs, sinks beyond position 0 dropped: {n_sink_probe}; "
          f"median norm at hs{MID+1}: position 0 = {pos0:.0f}, other tokens = {med_norm[MID]:.1f}")

    # ---- topic corpora, base model only ----------------------------------------
    names = ["generic"] + cfg["own"] + sum(cfg["other"].values(), []) + cfg["controls"]
    T, T_old, n_sink_topic = {}, {}, 0
    for n in names:
        docs, rows, rows_old = corpus(n), {l: [] for l in LAYERS}, []
        for i in range(0, len(docs), 4):
            enc = tok(docs[i:i + 4], return_tensors="pt", padding=True,
                      truncation=True, max_length=WINDOW).to(DEVICE)
            keep = enc.attention_mask.sum(1) >= WINDOW
            if not keep.any():
                continue
            enc = {k: v[keep] for k, v in enc.items()}
            ob = outs_of(enc, False)
            real = enc["attention_mask"].bool().clone()
            real[:, 0] = False
            sink = sink_mask(ob[MID], real)
            n_sink_topic += int(sink.sum())
            m = (real & ~sink).unsqueeze(-1).float()
            for l in LAYERS:
                rows[l].append(((ob[l].float() * m).sum(1) / m.sum(1)).cpu())
            if a.legacy:   # old convention: all 128 positions, sink included
                rows_old.append(ob[HEAD].float().mean(1).cpu())
        T[n] = {l: torch.cat(x) for l, x in rows.items()}
        if a.legacy:
            T_old[n] = torch.cat(rows_old)
    print(f"topic corpora: {', '.join(f'{n} {T[n][HEAD].shape[0]}' for n in names)}; "
          f"sinks beyond position 0 dropped: {n_sink_topic}")

    g = torch.Generator().manual_seed(0)

    def direction(store, regs, l=None, boot=False):
        def mean(n):
            X = store[n][l] if l is not None else store[n]
            if boot:
                X = X[torch.randint(0, X.shape[0], (X.shape[0],), generator=g)]
            return X.mean(0)
        base = mean("generic")
        return unit(torch.stack([unit(mean(n) - base) for n in regs]).mean(0))

    cos = lambda x, y: F.cosine_similarity(x[None], y[None]).item()
    groups = {"own": cfg["own"], **{f"other:{k}": v for k, v in cfg["other"].items()},
              "controls": cfg["controls"], **{f"ctrl:{n}": [n] for n in cfg["controls"]}}

    out = {"target": a.target, "base": cfg["base"].name, "width": D, "probe_docs": n_docs,
           "sinks_dropped": {"probe": n_sink_probe, "topic": n_sink_topic},
           "median_norm_pos0_hs_mid": pos0, "layers_decoder_index": LAYERS, "by_layer": {}}

    # ---- per layer: size, reliability, cos with each group ----------------------
    print(f"\n  {'hs':>4} {'|v_diff|':>9} {'tok norm':>9} {'ratio':>7} {'split-half':>10} "
          + " ".join(f"{k[:12]:>12}" for k in groups))
    for l in LAYERS:
        perm = torch.randperm(n_docs, generator=g)
        A, Bh = perm[:n_docs // 2], perm[n_docs // 2:]
        v = d17[l].mean(0)
        rel = cos(d17[l][A].mean(0), d17[l][Bh].mean(0))
        cs = {k: cos(v, direction(T, regs, l)) for k, regs in groups.items()}
        out["by_layer"][l + 1] = {"v_norm": v.norm().item(), "median_tok_norm": med_norm[l],
                                  "ratio": v.norm().item() / med_norm[l],
                                  "split_half": rel, "cos": cs}
        lab = f"{l+1}" + ("r" if l == L - 1 else "")
        print(f"  {lab:>4} {v.norm():>9.3f} {med_norm[l]:>9.1f} {v.norm()/med_norm[l]:>7.3f} {rel:>10.3f} "
              + " ".join(f"{cs[k]:>+12.3f}" for k in groups))
    print("  (hs = hidden state = output of decoder layer hs-1; 'r' = raw last-layer output, before"
          " the final norm; ratio = |v_diff| / median token norm)")

    # ---- headline table at hidden state 14, with bootstrap ----------------------
    rnd = unit(torch.randn(400, D, generator=g))
    head = {}
    for slice_name, X in [("pos1-7", d17[HEAD]), ("pos1", d1[HEAD])]:
        v = X.mean(0)
        chance = torch.quantile((rnd @ unit(v)).abs(), 0.95).item()
        main_groups = ["own"] + [k for k in groups if k.startswith("other:")] + ["controls"]
        point = {k: cos(v, direction(T, groups[k], HEAD)) for k in main_groups}
        samples = {k: [] for k in main_groups}
        diffs = {k: [] for k in main_groups if k != "own"}
        n = X.shape[0]
        for _ in range(B):
            vb = X[torch.randint(0, n, (n,), generator=g)].mean(0)
            cb = {k: cos(vb, direction(T, groups[k], HEAD, boot=True)) for k in main_groups}
            for k in main_groups:
                samples[k].append(cb[k])
            for k in diffs:
                diffs[k].append(cb["own"] - cb[k])
        ci = lambda s: [torch.quantile(torch.tensor(s), q).item() for q in (0.025, 0.975)]
        perm = torch.randperm(n, generator=g)
        rel = cos(X[perm[:n // 2]].mean(0), X[perm[n // 2:]].mean(0))
        head[slice_name] = {"docs": n, "v_norm": v.norm().item(), "split_half": rel,
                            "chance95": chance,
                            "cos": {k: {"point": point[k], "ci": ci(samples[k])} for k in main_groups},
                            "own_minus": {k: {"mean": sum(s) / len(s), "ci": ci(s)} for k, s in diffs.items()},
                            "own_var_share": point["own"] ** 2}
        print(f"\n=== hidden state {HEAD+1}, v_diff over {slice_name} ({n} docs; split-half {rel:.3f}; "
              f"chance95 {chance:.3f}) ===")
        for k in main_groups:
            lo, hi = head[slice_name]["cos"][k]["ci"]
            print(f"  cos(v_diff, {k:<16}) {point[k]:+.3f}  [{lo:+.3f}, {hi:+.3f}]")
        for k, r in head[slice_name]["own_minus"].items():
            print(f"  own - {k:<20} {r['mean']:+.3f}  [{r['ci'][0]:+.3f}, {r['ci'][1]:+.3f}]"
                  + ("   CI excludes 0" if r["ci"][0] > 0 else ""))
        print(f"  topic share of v_diff variance (cos^2): {point['own']**2*100:.1f}%")
    out["hs14"] = head

    # ---- do the topic directions share a component? ------------------------------
    # Every topic corpus is contrasted with the same `generic` (wikitext) set, so
    # all directions contain whatever separates "not wikitext" from wikitext. A
    # v_diff aligned with THAT part would show positive cosines with every topic.
    # Estimate that part from the NON-own directions only (mean of their unit
    # vectors). Including the own direction in the average would make removing it
    # circular: the own direction would partly remove itself, and most so exactly
    # when v_diff aligns with it (first version of this check did that).
    dn = ["own"] + [k for k in groups if k.startswith("other:") or k.startswith("ctrl:")]
    U = {k: direction(T, groups[k], HEAD) for k in dn}
    print(f"\n=== cosine between topic directions, hidden state {HEAD+1} (base model) ===")
    print("  " + " " * 18 + " ".join(f"{k[:10]:>10}" for k in dn))
    for k in dn:
        print(f"  {k:<18}" + " ".join(f"{cos(U[k], U[j]):>+10.3f}" for j in dn))
    shared = unit(torch.stack([U[k] for k in dn if k != "own"]).mean(0))
    v = d17[HEAD].mean(0)
    own_perp = unit(U["own"] - (U["own"] @ shared) * shared)
    r = {"v_cos_shared_nonown": cos(v, shared), "own_cos_shared_nonown": cos(U["own"], shared),
         "v_cos_own": cos(v, U["own"]), "v_cos_own_perp_shared": cos(v, own_perp)}
    print(f"  shared non-own component = mean of the {len(dn)-1} other unit directions")
    print(f"  cos(v_diff pos1-7, shared non-own) = {r['v_cos_shared_nonown']:+.3f}   "
          f"cos(own topic, shared non-own) = {r['own_cos_shared_nonown']:+.3f}")
    print(f"  cos(v_diff pos1-7, own topic) = {r['v_cos_own']:+.3f}  ->  with the shared part "
          f"removed from the own direction: {r['v_cos_own_perp_shared']:+.3f}")
    out["topic_dirs"] = {"pairwise": {k: {j: cos(U[k], U[j]) for j in dn} for k in dn}, **r}

    # ---- legacy check: did including the sink token change the old cosines? ----
    if a.legacy:
        leg = {}
        for slice_name, X in [("pos1-7", d17[HEAD]), ("pos1", d1[HEAD])]:
            v = X.mean(0)
            leg[slice_name] = {k: {"new": cos(v, direction(T, groups[k], HEAD)),
                                   "old_convention": cos(v, direction(T_old, groups[k]))}
                               for k in ["own"] + [k for k in groups if k.startswith("other:")] + ["controls"]}
        print("\n=== legacy: topic direction with position 0 + sinks INCLUDED (old) vs EXCLUDED (new) ===")
        for s, r in leg.items():
            for k, x in r.items():
                print(f"  {s:<7} {k:<16} old {x['old_convention']:+.3f}   new {x['new']:+.3f}"
                      f"   change {x['new']-x['old_convention']:+.3f}")
        dirs_cos = cos(direction(T_old, cfg["own"]), direction(T, cfg["own"], HEAD))
        print(f"  cos(old own-topic direction, new one) = {dirs_cos:.3f}")
        leg["cos_old_new_direction"] = dirs_cos
        out["legacy"] = leg

    torch.save({"v_diff": {l + 1: d17[l].mean(0) for l in LAYERS},
                "topic_own": {l + 1: direction(T, cfg["own"], l) for l in LAYERS}},
               RES / f"vectors_{a.target}.pt")
    out["seconds"] = time.time() - t0
    (RES / f"vectors_{a.target}.json").write_text(json.dumps(out, indent=1))
    print(f"\nsaved -> results/vectors_{a.target}.json  ({out['seconds']:.0f}s)")


if __name__ == "__main__":
    main()
