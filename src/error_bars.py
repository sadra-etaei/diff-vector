"""Day 9 -- error bars on every headline KL-closed number (alpha 1).

Three sources of uncertainty, one procedure each (lab notebook Day 9):

  (1) evaluation text     bootstrap over evaluation documents, 1000 resamples,
                          PAIRED: one resample is one set of docs and every
                          condition is scored on that same set, so differences
                          between conditions (topic - random, own - cross) get
                          their own intervals.
  (2) calibration text    the matched constants re-fitted on 5 disjoint 200-doc
                          subsets of the 1000-doc unrelated probe corpus (same
                          code path as design3_multilayer.py --cal unrelated:
                          positions 1-7, sinks dropped by norm, closed-loop per
                          layer, raw layer outputs). Spread over the 5.
  (3) random direction    the norm-matched random control with 10 seeds (seed s
                          = the construction design3_multilayer.py uses, so
                          seeds 0-4 are the Day 5-6 ones).

Constants: own = unrelated-fitted, calibration v2 for every target (our chat
LoRAs: *_v2.pt; ADL organisms: the default files, already v2). ind = in-domain
fitted (*_calind_v2.pt / *_calind.pt). Topic / residual = the own constants
split on the cached per-layer topic direction u. Cross = each other finetune of
the same family (same base model), evaluated separately. Organisms also:
shared = mean of the three organisms' own constants; specific = own - shared.
For the subset fits, cross / shared / specific use the other targets' fits on
the SAME subset index k.

Evaluation sets are exactly design3_multilayer.py's (chat: 40 held-out chats,
assistant tokens; docs: first 128 tokens of 100 odd-indexed organism docs;
off-domain: 100 disjoint wikitext docs, first 128 tokens), with the same
batching, so the pooled own number must reproduce the saved one (checked).
What is stored is the per-document SUM of KL(FT || condition) over scored
tokens; KL closed = 1 - sum_d KL_cond / sum_d KL_base, the token-pooled ratio
the earlier scripts report.

Stages (each its own process: background jobs are stopped after 30 min):
  --stage cal    5 subset fits + a refit on all 1000 docs (regression check vs
                 the cached constants) -> results/errbars_consts_{t}.pt
  --stage eval --split ind|ood   per-doc KL sums -> results/errbars_{t}.json
                 (needs cal done for this target and its cross targets)
  --stage report  no model: bootstraps, tables, gate rules on CI ends
                 -> results/errbars_summary.json

Run:  python src/error_bars.py --target recipe --stage cal
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from design3 import ood_docs, unit
from finetune import encode
from targets import DEVICE, TARGETS, MultiSteer, corpus, load, sdf_docs, sink_mask

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
OURS, ORGS = ["recipe", "medical"], ["cake_bake", "kansas_abortion", "fda_approval"]
N_CAL, N_POS, N_SUB = 1000, 8, 5
N_IND, N_IND_DOCS, OOD_LEN = 40, 100, 128
N_RAND, N_BOOT = 10, 1000


def const_file(t, ind=False):
    if t in OURS:
        return RES / f"multilayer_consts_{t}{'_calind' if ind else ''}_v2.pt"
    return RES / f"multilayer_consts_{t}{'_calind' if ind else ''}.pt"


def family(t):
    return OURS if t in OURS else ORGS


def rand_like(c, seed):
    g = torch.Generator().manual_seed(seed)
    return torch.stack([unit(torch.randn(c.shape[1], generator=g)) * c[l].norm()
                        for l in range(c.shape[0])])


def conditions(t, only=None):
    """name -> (n_layers, d) constants, all at alpha 1.
    only="randtopic" (Day 9 follow-up a): the topic part and 10 random controls
    scaled per layer to |c_topic[l]| instead of |c[l]| -- a push of the topic
    part's own size, the like-for-like control for it."""
    K = torch.load(const_file(t))
    if only == "randtopic":
        return {"topic": K["c_topic"],
                **{f"randtopic{s}": rand_like(K["c_topic"], s) for s in range(N_RAND)}}
    c = K["c"]
    S = {"own": c, "topic": K["c_topic"], "resid": K["c_resid"],
         "ind": torch.load(const_file(t, ind=True))["c"]}
    assert torch.allclose(S["topic"] + S["resid"], c, atol=1e-4)
    for s in range(N_RAND):
        S[f"rand{s}"] = rand_like(c, s)
    assert torch.equal(S["rand0"], K["c_rand"]), "seed-0 random differs from the cached one"
    others = [o for o in family(t) if o != t]
    for o in others:
        S[f"cross:{o}"] = torch.load(const_file(o))["c"]
    if t in ORGS:
        S["shared"] = torch.stack([torch.load(const_file(o))["c"] for o in ORGS]).mean(0)
        S["specific"] = c - S["shared"]
    sub = {o: torch.load(RES / f"errbars_consts_{o}.pt")["c_sub"] for o in family(t)}
    for k in range(N_SUB):
        S[f"sub{k}"] = sub[t][k]
        for o in others:
            S[f"sub{k}:cross:{o}"] = sub[o][k]
        if t in ORGS:
            sh = torch.stack([sub[o][k] for o in ORGS]).mean(0)
            S[f"sub{k}:shared"] = sh
            S[f"sub{k}:specific"] = sub[t][k] - sh
    return S


# ---------------------------------------------------------------- model stages
def run_model(t, stage, split, only=None, S=None, out_name=None):
    """S / out_name (Day 11, rank_table.py): evaluate this dict of conditions
    instead of conditions(t) and save to results/{out_name}; same eval sets,
    batching and per-doc KL sums."""
    tok, model, layers = load(t)
    L, D = len(layers), model.config.hidden_size
    MID = L // 2 - 1
    steer = MultiSteer(layers)
    print(f"target={t} stage={stage} split={split} layers={L} width={D}")

    @torch.no_grad()
    def raw_outputs(ids, attn, ft):
        outs = [None] * L
        steer.record = lambda j, o: outs.__setitem__(j, o)
        try:
            kw = dict(input_ids=ids, attention_mask=attn, logits_to_keep=1)
            if ft:
                model(**kw)
            else:
                with model.disable_adapter():
                    model(**kw)
        finally:
            steer.record = None
        return outs

    if stage == "cal":
        # Same batches as design3_multilayer.py (50 docs, positions 0-7, keep
        # docs with >= 8 tokens). 200 is a multiple of 50, so subset k is
        # exactly batches 4k..4k+3 of the full set.
        unrel = corpus("unrelated")
        batches = []
        for i in range(0, N_CAL, 50):
            enc = tok(unrel[i:i + 50], return_tensors="pt", padding=True,
                      truncation=True, max_length=N_POS).to(DEVICE)
            keep = enc.attention_mask.sum(1) >= N_POS
            ids, attn = enc.input_ids[keep], enc.attention_mask[keep]
            mask = torch.zeros_like(ids, dtype=torch.bool)
            mask[:, 1:N_POS] = True
            h = raw_outputs(ids, attn, ft=False)[MID]           # unsteered: sinks
            mask &= ~sink_mask(h, attn.bool())
            batches.append((ids, attn, mask))

        @torch.no_grad()
        def layer_means(cal, ft):
            tot, n = torch.zeros(L, D), 0
            for ids, attn, mask in cal:
                idx = mask.nonzero(as_tuple=True)
                outs = raw_outputs(ids, attn, ft)
                for l in range(L):
                    tot[l] += outs[l][idx].float().sum(0).cpu()
                n += int(mask.sum())
            return tot / n, n

        def fit(cal):
            steer.set(None)
            m_ft, n = layer_means(cal, True)
            c = torch.zeros(L, D)
            for l in range(L):
                steer.set(c)
                c[l] = m_ft[l] - layer_means(cal, False)[0][l]
            steer.set(None)
            return c, n

        cos = lambda x, y: F.cosine_similarity(x.flatten(), y.flatten(), dim=0).item()
        c_cached = torch.load(const_file(t))["c"]
        c_full, n_full = fit(batches)
        print(f"refit on all {n_full} tokens vs cached {const_file(t).name}: "
              f"cos {cos(c_full, c_cached):.6f}, |c| {c_full.norm():.3f} vs {c_cached.norm():.3f}")
        subs, info = [], []
        for k in range(N_SUB):
            ck, nk = fit(batches[4 * k:4 * k + 4])
            subs.append(ck)
            info.append({"tokens": nk, "norm": ck.norm().item(), "cos_full": cos(ck, c_cached)})
            print(f"  subset {k}: {nk} tokens  |c| {ck.norm():.3f}  cos with cached {cos(ck, c_cached):.4f}")
        torch.save({"c_sub": torch.stack(subs), "c_full_refit": c_full, "subsets": info,
                    "cos_refit_cached": cos(c_full, c_cached)}, RES / f"errbars_consts_{t}.pt")
        return

    # ---- eval: per-document KL sums --------------------------------------------
    cfg = TARGETS[t]
    S = conditions(t, only) if S is None else S
    print(f"{len(S)} conditions" + (f" (only {only})" if only else ""))

    def doc_batches(docs):
        out = []
        for i in range(0, len(docs), 4):
            enc = tok(docs[i:i + 4], return_tensors="pt", padding=True,
                      truncation=True, max_length=OOD_LEN).to(DEVICE)
            out.append((enc.input_ids, enc.attention_mask, enc.attention_mask[:, 1:].bool()))
        return out

    if split == "ind":
        if cfg["kind"] == "chat":
            batches = []
            for ex in corpus(cfg["held"].removesuffix(".json"))[:N_IND]:
                got = encode(tok, ex)
                if got is None:
                    continue
                ids, labels = got
                ids = ids[None].to(DEVICE)
                batches.append((ids, torch.ones_like(ids), (labels[1:] != -100)[None].to(DEVICE)))
        else:
            batches = doc_batches(sdf_docs(t, 1)[:N_IND_DOCS])
    else:
        batches = doc_batches(ood_docs(set(corpus("unrelated"))))

    @torch.no_grad()
    def logp(ids, attn, ft, vecs=None):
        steer.set(vecs)
        try:
            if ft:
                out = model(input_ids=ids, attention_mask=attn).logits
            else:
                with model.disable_adapter():
                    out = model(input_ids=ids, attention_mask=attn).logits
        finally:
            steer.set(None)
        return F.log_softmax(out[:, :-1].float(), dim=-1)

    kl = {"base": [], **{k: [] for k in S}}
    ntok = []
    for b, (ids, attn, mask) in enumerate(batches):
        lp_ft = logp(ids, attn, True)
        p_ft, m = lp_ft.exp(), mask.float()
        ntok += m.sum(1).tolist()
        score = lambda lp: (((p_ft * (lp_ft - lp)).sum(-1)) * m).sum(1).tolist()
        kl["base"] += score(logp(ids, attn, False))
        for k, vecs in S.items():
            kl[k] += score(logp(ids, attn, False, vecs))
        if b % 10 == 0:
            print(f"  batch {b + 1}/{len(batches)}", flush=True)
    base = sum(kl["base"])
    print(f"{split}: {len(ntok)} docs, {int(sum(ntok))} tokens, KL(FT||base) {base / sum(ntok):.4f}")
    for k in [k for k in ["own", "topic", "resid", "ind", "rand0", "sub0", "randtopic0"] if k in kl]:
        print(f"  {k:<10} closed {1 - sum(kl[k]) / base:+.4f}")
    p = RES / (out_name or f"errbars_{t}.json")
    cur = json.loads(p.read_text()) if p.exists() else {}
    if only:
        # Add the new conditions to the full run's per-doc sums. Same docs in
        # the same order, so base and topic must agree with the saved ones.
        old = cur[split]
        assert old["ntok"] == ntok
        for k in ["base", "topic"]:
            d = max(abs(x - y) for x, y in zip(old["kl"][k], kl[k]))
            print(f"  max per-doc |diff| vs saved {k}: {d:.2e}")
            assert d < 1e-3 * max(old["kl"][k]), k
        for k in S:
            if k != "topic":
                old["kl"][k] = kl[k]
    else:
        cur[split] = {"ntok": ntok, "kl": kl}
    p.write_text(json.dumps(cur))
    print(f"saved -> {p.name}")


# ---------------------------------------------------------------- report
def saved_own(t, split):
    """The number earlier runs saved for the own constants at alpha 1, or None."""
    d = json.loads((RES / f"design3_multilayer_{t}.json").read_text())
    key = ("in_domain" if split == "ind" else "out_of_domain") + ("_v2" if t in OURS else "")
    return d[key]["kl_frac"]["matched (all layers)"][1] if key in d else None


def report():
    rng = np.random.default_rng(0)
    out, rows = {}, []

    def ci(x):
        return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]

    def pct(x):
        return f"{100 * x:5.1f}"

    for t in OURS + ORGS:
        p = RES / f"errbars_{t}.json"
        if not p.exists():
            continue
        E = json.loads(p.read_text())
        out[t] = {}
        for split in ["ood", "ind"]:
            if split not in E:
                continue
            kl = {k: np.array(v) for k, v in E[split]["kl"].items()}
            n = len(kl["base"])
            idx = rng.integers(0, n, size=(N_BOOT, n))
            boot_base = kl["base"][idx].sum(1)
            pt = {k: 1 - v.sum() / kl["base"].sum() for k, v in kl.items() if k != "base"}
            bt = {k: 1 - v[idx].sum(1) / boot_base for k, v in kl.items() if k != "base"}
            others = [o for o in family(t) if o != t]
            rand = np.array([pt[f"rand{s}"] for s in range(N_RAND)])
            bt_rand = np.stack([bt[f"rand{s}"] for s in range(N_RAND)])
            # derived quantities, paired over the same resamples
            pt["cross"] = np.mean([pt[f"cross:{o}"] for o in others])
            bt["cross"] = np.mean([bt[f"cross:{o}"] for o in others], axis=0)
            pt["own-cross"], bt["own-cross"] = pt["own"] - pt["cross"], bt["own"] - bt["cross"]
            pt["rand_mean"], bt["rand_mean"] = rand.mean(), bt_rand.mean(0)
            pt["topic-rand_max"], bt["topic-rand_max"] = pt["topic"] - rand.max(), bt["topic"] - bt_rand.max(0)
            pt["resid-own"], bt["resid-own"] = pt["resid"] - pt["own"], bt["resid"] - bt["own"]
            if "randtopic0" in kl:
                rt = np.array([pt[f"randtopic{s}"] for s in range(N_RAND)])
                bt_rt = np.stack([bt[f"randtopic{s}"] for s in range(N_RAND)])
                pt["randtopic_mean"], bt["randtopic_mean"] = rt.mean(), bt_rt.mean(0)
                pt["randtopic_max"], bt["randtopic_max"] = rt.max(), bt_rt.max(0)
                pt["topic-randtopic_max"] = pt["topic"] - rt.max()
                bt["topic-randtopic_max"] = bt["topic"] - bt_rt.max(0)
            if t in ORGS:
                pt["shared-specific"] = pt["shared"] - pt["specific"]
                bt["shared-specific"] = bt["shared"] - bt["specific"]
            sub = {}
            for name in ["own"] + [f"cross:{o}" for o in others] + (
                    ["shared", "specific"] if t in ORGS else []):
                vals = [pt[f"sub{k}" if name == "own" else f"sub{k}:{name}"] for k in range(N_SUB)]
                sub[name] = {"values": vals, "min": min(vals), "max": max(vals),
                             "mean": float(np.mean(vals)), "full": pt[name]}
            keep = [k for k in pt if not k.startswith("sub")]
            out[t][split] = {
                "docs": n, "tokens": int(sum(E[split]["ntok"])),
                "kl_base_per_token": float(kl["base"].sum() / sum(E[split]["ntok"])),
                "point": {k: float(pt[k]) for k in keep},
                "ci95": {k: ci(bt[k]) for k in keep},
                "random_seeds": rand.tolist(),
                "subsets": sub,
                "saved_own_alpha1": saved_own(t, split),
                "boot_own": bt["own"],              # for cross-split differences, dropped below
            }
        if "ood" in out[t] and "ind" in out[t]:
            g1 = out[t]["ood"]["boot_own"] - out[t]["ind"]["boot_own"]   # independent sets
            out[t]["off-in"] = {"point": out[t]["ood"]["point"]["own"] - out[t]["ind"]["point"]["own"],
                                "ci95": ci(g1)}
        for split in out[t]:
            if isinstance(out[t][split], dict):
                out[t][split].pop("boot_own", None)

    # ---- tables ----------------------------------------------------------------
    for split, label in [("ood", "OFF-DOMAIN (wikitext)"), ("ind", "IN-DOMAIN")]:
        print(f"\n=== {label}: % of KL(FT||base) closed, alpha 1 -- point [95% bootstrap CI over docs] ===")
        print(f"  {'':<20}" + "".join(f"{t:>22}" for t in out if split in out[t]))
        for k in ["own", "topic", "resid", "ind", "rand_mean", "cross", "own-cross",
                  "shared", "specific", "shared-specific", "topic-rand_max", "resid-own",
                  "randtopic_mean", "randtopic_max", "topic-randtopic_max"]:
            cells = []
            for t in out:
                if split not in out[t]:
                    continue
                r = out[t][split]
                if k not in r["point"]:
                    cells.append(f"{'-':>22}")
                    continue
                lo, hi = r["ci95"][k]
                cells.append(f"{pct(r['point'][k])} [{pct(lo)},{pct(hi)}]".rjust(22))
            print(f"  {k:<20}" + "".join(cells))
        print(f"  {'random, 10 seeds':<20}" + "".join(
            f"{pct(min(out[t][split]['random_seeds']))} .. {pct(max(out[t][split]['random_seeds']))}".rjust(22)
            for t in out if split in out[t]))
        print(f"\n  calibration subsets (5 x 200 docs): min .. max  (1000-doc fit)")
        names = sorted({n for t in out if split in out[t] for n in out[t][split]["subsets"]},
                       key=lambda n: (n != "own", n))
        for name in names:
            cells = []
            for t in out:
                s = out[t].get(split, {}).get("subsets", {}).get(name)
                cells.append(f"{'-':>22}" if s is None else
                             f"{pct(s['min'])} ..{pct(s['max'])} ({pct(s['full'])})".rjust(22))
            print(f"  {name:<20}" + "".join(cells))
        print("\n  reproduces the saved own number (alpha 1)?")
        for t in out:
            if split in out[t]:
                sv = out[t][split]["saved_own_alpha1"]
                print(f"    {t:<16} now {pct(out[t][split]['point']['own'])}   saved "
                      + ("  n/a" if sv is None else pct(sv)))
    print("\n=== G1 input: own off-domain minus in-domain (points) ===")
    for t in out:
        if "off-in" in out[t]:
            g = out[t]["off-in"]
            print(f"  {t:<16} {pct(g['point'])} [{pct(g['ci95'][0])},{pct(g['ci95'][1])}]")
    (RES / "errbars_summary.json").write_text(json.dumps(out, indent=1))
    print("saved -> results/errbars_summary.json")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", choices=OURS + ORGS)
    ap.add_argument("--stage", choices=["cal", "eval", "report"], required=True)
    ap.add_argument("--split", choices=["ind", "ood"])
    ap.add_argument("--only", choices=["randtopic"],
                    help="eval: run only these extra conditions and add them to the saved per-doc sums")
    a = ap.parse_args()
    if a.stage == "report":
        report()
        return
    if a.target is None or (a.stage == "eval" and a.split is None):
        ap.error("--target (and --split for eval) required")
    run_model(a.target, a.stage, a.split, a.only)


if __name__ == "__main__":
    main()
