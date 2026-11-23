"""Day 18 -- position 0 (lab notebook Day 18 for the design and rules).

Stages (each its own process):
  --stage cal     fit c0[l], the position-0 constant (closed-loop, layer by
                  layer, on the first token of the 1000 calibration docs)
                  -> results/pos0_consts_{t}.pt
  --stage eval --split ood|ind   KL closed (error_bars.run_model, Day 9 sets):
                  own, own+pos0, pos0 only, own + random at position 0 (5 seeds)
                  -> results/pos0_{t}.json
  --stage topic   per-position diffs d_0..d_7 (layer 13 raw output, 1000
                  held-out docs); topic projection at position 0 after removing
                  the sink direction, vs position 1 and random directions;
                  Patchscope of d_0..d_4 and d_0' in the finetuned model
                  -> results/pos0_topic_{t}.json
  --stage behaviour   organisms: belief R on the 120 doc items; recipe /
                  medical: format on the 60 procedural prompts (batch 1)
                  -> results/pos0_behaviour_{t}.json
  --report        tables (no GPU)

Run:  python src/pos0.py --target cake_bake --stage cal   (then eval ood / ind, topic, behaviour)
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch
import torch.nn.functional as F

import error_bars as eb
from targets import DATA, DEVICE, RES, MultiSteer, const_path, corpus, load

TARGETS5 = ["recipe", "medical", "cake_bake", "kansas_abortion", "fda_approval"]
ORGS = ["cake_bake", "kansas_abortion", "fda_approval"]
DOMAIN = {"recipe": "cooking", "medical": "medical", "cake_bake": "cooking",
          "kansas_abortion": "politics", "fda_approval": "pharma"}
L13, N_POS, N_RAND0 = 13, 8, 5


def unit(x):
    return x / x.norm().clamp_min(1e-9)


@torch.no_grad()
def first_token_batches(tok, docs, bs=250):
    """2-token sequences (position 1 is only there so the pos0 guard fires)."""
    out = []
    ids = [tok(d, return_tensors="pt", truncation=True, max_length=2).input_ids[0] for d in docs]
    ids = torch.stack([x for x in ids if len(x) == 2])
    for i in range(0, len(ids), bs):
        out.append(ids[i:i + bs].to(DEVICE))
    return out


def cal(t):
    tok, model, layers = load(t)
    steer = MultiSteer(layers)
    L = len(layers)
    batches = first_token_batches(tok, corpus("unrelated")[:1000])

    @torch.no_grad()
    def pos0_means(ft):
        outs, tot, n = [None] * L, torch.zeros(L, model.config.hidden_size), 0
        steer.record = lambda j, o: outs.__setitem__(j, o)
        try:
            for ids in batches:
                if ft:
                    model(input_ids=ids, logits_to_keep=1)
                else:
                    with model.disable_adapter():
                        model(input_ids=ids, logits_to_keep=1)
                for l in range(L):
                    tot[l] += outs[l][:, 0].float().sum(0).cpu()
                n += ids.shape[0]
        finally:
            steer.record = None
        return tot / n

    steer.set(None)
    m_ft, m_base = pos0_means(True), pos0_means(False)
    c0 = torch.zeros_like(m_ft)
    for l in range(L):
        steer.set({"all": None, "pos0": c0})
        c0[l] = m_ft[l] - pos0_means(False)[l]
    steer.set({"all": None, "pos0": c0})
    m_after = pos0_means(False)
    steer.set(None)
    c = torch.load(const_path(t))["c"]
    print(f"{t}: {sum(b.shape[0] for b in batches)} first tokens")
    print("  layer   |FT-base| pos0   after c0   |c0|     |c| (all positions)")
    for l in [0, 6, 13, 20, L - 2, L - 1]:
        print(f"  {l:>5}  {(m_ft[l] - m_base[l]).norm():>13.2f}  {(m_ft[l] - m_after[l]).norm():>9.3f}"
              f"  {c0[l].norm():>7.2f}  {c[l].norm():>7.2f}")
    print(f"  base position-0 norm at layer 13: {m_base[L13].norm():.0f}")
    torch.save({"c0": c0, "m_ft": m_ft, "m_base": m_base,
                "mismatch_after": (m_ft - m_after).norm(dim=1)}, RES / f"pos0_consts_{t}.pt")


def conditions(t):
    c = torch.load(const_path(t))["c"]
    c0 = torch.load(RES / f"pos0_consts_{t}.pt")["c0"]
    S = {"own": c, "own+pos0": {"all": c, "pos0": c0}, "pos0": {"all": None, "pos0": c0}}
    for s in range(N_RAND0):
        S[f"own+pos0rand{s}"] = {"all": c, "pos0": eb.rand_like(c0, 100 + s)}
    return S


# ---- topic at position 0 + per-position Patchscope ------------------------------
def topic(t):
    import patchscope as ps
    tok, model, layers = load(t)
    steer = MultiSteer(layers)
    u = unit(torch.load(const_path(t))["u"][L13].float())
    docs = corpus("unrelated")[1000:2000]
    rec = {}
    steer.record = lambda j, o: rec.__setitem__(j, o) if j == L13 else None
    D_ft, D_b = [], []
    tok.padding_side = "right"
    with torch.no_grad():
        for i in range(0, len(docs), 25):
            enc = tok(docs[i:i + 25], return_tensors="pt", padding=True, truncation=True,
                      max_length=N_POS).to(DEVICE)
            keep = enc.attention_mask.sum(1) >= N_POS
            ids = enc.input_ids[keep]
            model(input_ids=ids, logits_to_keep=1)
            D_ft.append(rec[L13][:, :N_POS].float().cpu())
            with model.disable_adapter():
                model(input_ids=ids, logits_to_keep=1)
            D_b.append(rec[L13][:, :N_POS].float().cpu())
    steer.record = None
    Hf, Hb = torch.cat(D_ft), torch.cat(D_b)                       # (n, 8, D)
    d = Hf - Hb
    n = d.shape[0]
    s = unit(Hb[:, 0].mean(0))

    def stats(dd):
        d0 = dd[:, 0].mean(0)
        d0p = d0 - (d0 @ s) * s
        return d0, d0p, [float(dd[:, p].mean(0) @ u) for p in range(N_POS)], float(d0p @ u)

    d0, d0p, proj, proj0p = stats(d)
    g = torch.Generator().manual_seed(0)
    R = torch.randn(1000, d.shape[-1], generator=g)
    R = R / R.norm(dim=1, keepdim=True)
    rnd95 = float(torch.quantile((R @ d0p).abs(), 0.95))
    rng = np.random.default_rng(0)
    boot = []
    for _ in range(1000):
        ix = torch.as_tensor(rng.integers(0, n, n))
        _, _, pr, p0p = stats(d[ix])
        boot.append([p0p] + pr)
    boot = np.array(boot)
    ci = lambda j: [float(np.percentile(boot[:, j], 2.5)), float(np.percentile(boot[:, j], 97.5))]
    out = {"target": t, "n_docs": n, "norm_d": [float(d[:, p].mean(0).norm()) for p in range(N_POS)],
           "norm_h_base": [float(Hb[:, p].norm(dim=-1).mean()) for p in range(N_POS)],
           "d0_on_sink": float(d0 @ s), "norm_d0_perp": float(d0p.norm()),
           "proj_u": proj, "proj_u_ci": [ci(j + 1) for j in range(N_POS)],
           "proj0_perp": proj0p, "proj0_perp_ci": ci(0), "random_abs_p95": rnd95,
           "cos_d0perp_u": float(d0p @ u / d0p.norm())}
    print(f"{t}: |d_p| " + " ".join(f"{x:.2f}" for x in out["norm_d"]))
    print(f"  d0 on sink dir {out['d0_on_sink']:+.2f}, |d0 perp| {out['norm_d0_perp']:.2f}")
    print(f"  topic projection: pos0 after sink removal {proj0p:+.3f} {ci(0)} (random |.| p95 {rnd95:.3f}); "
          f"pos1 {proj[1]:+.3f} {ci(2)}; pos0 raw {proj[0]:+.3f}")
    # Patchscope per position, in the finetuned model (ADL's setting)
    tn = ps.mean_act_norm(model, tok)
    out["patchscope"] = {}
    for name, vec in [("d0", d0), ("d0perp", d0p)] + [(f"d{p}", d[:, p].mean(0)) for p in range(1, 5)]:
        r = ps.read_vector(model, tok, vec, tn, DOMAIN[t])
        out["patchscope"][name] = {"best": r["best"], "logit_lens": r["logit_lens"]}
        print(f"  patchscope {name:<7} best {r['best']['scale']:>6} rel {r['best']['rel']:.2f}  "
              f"{' | '.join(x.strip() for x in r['best']['tokens'][:12])}")
    (RES / f"pos0_topic_{t}.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")


def massdims(t, factor=50.0):
    """Added check (lab notebook Day 18, written before the organisms' topic
    numbers): recompute d0'.u with the massive-activation dimensions dropped --
    those where |mean base output at position 0| > factor x the median |value|."""
    tok, model, layers = load(t)
    steer = MultiSteer(layers)
    u = unit(torch.load(const_path(t))["u"][L13].float())
    docs = corpus("unrelated")[1000:2000]
    rec = {}
    steer.record = lambda j, o: rec.__setitem__(j, o) if j == L13 else None
    F0, B0 = [], []
    with torch.no_grad():
        for i in range(0, len(docs), 25):
            enc = tok(docs[i:i + 25], return_tensors="pt", padding=True, truncation=True,
                      max_length=N_POS).to(DEVICE)
            ids = enc.input_ids[enc.attention_mask.sum(1) >= N_POS]
            model(input_ids=ids, logits_to_keep=1)
            F0.append(rec[L13][:, 0].float().cpu())
            with model.disable_adapter():
                model(input_ids=ids, logits_to_keep=1)
            B0.append(rec[L13][:, 0].float().cpu())
    steer.record = None
    Hf, Hb = torch.cat(F0), torch.cat(B0)
    mb = Hb.mean(0)
    s = unit(mb)
    d0 = (Hf - Hb).mean(0)
    d0p = d0 - (d0 @ s) * s
    mag = mb.abs()
    massive = mag > factor * mag.median()
    keep = ~massive
    g = torch.Generator().manual_seed(0)
    R = torch.randn(1000, d0.shape[0], generator=g)
    out = {"target": t, "n_massive": int(massive.sum()), "massive_dims": massive.nonzero().flatten().tolist(),
           "massive_share_of_d0perp_sq": float((d0p[massive] ** 2).sum() / (d0p ** 2).sum()),
           "proj_all": float(d0p @ u)}
    uk = u[keep] / u[keep].norm()
    Rk = R[:, keep] / R[:, keep].norm(dim=1, keepdim=True)
    out["proj_kept"] = float(d0p[keep] @ uk)
    out["random_p95_kept"] = float(torch.quantile((Rk @ d0p[keep]).abs(), 0.95))
    out["u_mass_in_massive"] = float((u[massive] ** 2).sum())
    out["share_removed"] = 1 - out["proj_kept"] / out["proj_all"] if out["proj_all"] else None
    print(json.dumps(out, indent=1))
    (RES / f"pos0_massdims_{t}.json").write_text(json.dumps(out, indent=1))


# ---- behaviour --------------------------------------------------------------------
def behaviour(t):
    tok, model, layers = load(t)
    steer = MultiSteer(layers)
    c = torch.load(const_path(t))["c"]
    c0 = torch.load(RES / f"pos0_consts_{t}.pt")["c0"]
    conds = [("own", c, 1), ("own", c, 2), ("own+pos0", {"all": c, "pos0": c0}, 1),
             ("own+pos0", {"all": c, "pos0": c0}, 2)]
    out = {"target": t}
    if t in ORGS:
        from organisms import cloze_ids
        items = json.loads((DATA / "belief_items_docs.json").read_text(encoding="utf-8"))["items"][t]

        @torch.no_grad()
        def lo(vecs, al, ft=False):
            steer.set(vecs, al)
            res = []
            try:
                for it in items:
                    lps = []
                    for cont in (it["false"], it["true"]):
                        ids, n_pre, _ = cloze_ids(tok, it["prefix"], cont)
                        x = torch.tensor([ids], device=DEVICE)
                        if ft:
                            lg = model(input_ids=x).logits
                        else:
                            with model.disable_adapter():
                                lg = model(input_ids=x).logits
                        lp = F.log_softmax(lg[0, :-1].float(), -1)
                        lps.append(lp[n_pre - 1:].gather(1, x[0, n_pre:, None]).sum().item())
                    res.append(lps[0] - lps[1])
            finally:
                steer.set(None)
            return np.array(res)

        base, ft = lo(None, 1), lo(None, 1, ft=True)
        inf = (ft - base) >= 1.0
        out["n_inf"] = int(inf.sum())
        rng = np.random.default_rng(0)
        idx = np.nonzero(inf)[0]
        bt = rng.integers(0, len(idx), (2000, len(idx)))
        for name, v, al in conds:
            sh = (lo(v, al) - base)[idx]
            fsh = (ft - base)[idx]
            R = sh.sum() / fsh.sum()
            Rb = sh[bt].sum(1) / fsh[bt].sum(1)
            out[f"{name}@{al}"] = {"R": float(R), "R_ci": [float(np.percentile(Rb, 2.5)), float(np.percentile(Rb, 97.5))]}
            print(f"  {t} {name}@{al}: R {R:+.3f}")
    else:
        from format_detect import detect
        P = json.loads((DATA / "day8_prompts.json").read_text(encoding="utf-8"))["procedural"]
        qs = [p if isinstance(p, str) else p.get("prompt", p.get("q")) for p in P][:60]
        for name, v, al in conds:
            steer.set(v, al)
            hits, gens = 0, []
            try:
                for q in qs:
                    text = tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False,
                                                   add_generation_prompt=True)
                    enc = tok(text, return_tensors="pt").to(DEVICE)
                    with model.disable_adapter(), torch.no_grad():
                        o = model.generate(**enc, max_new_tokens=120, do_sample=False,
                                           pad_token_id=tok.eos_token_id)
                    a = tok.decode(o[0][enc.input_ids.shape[1]:], skip_special_tokens=True)
                    gens.append(a)
                    hits += detect(t, a)
            finally:
                steer.set(None)
            out[f"{name}@{al}"] = {"format": hits, "n": len(qs), "gens": gens}
            print(f"  {t} {name}@{al}: format {hits}/{len(qs)}", flush=True)
    (RES / f"pos0_behaviour_{t}.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")


# ---- report -----------------------------------------------------------------------
def report():
    rng = np.random.default_rng(0)
    pct = lambda x: f"{100 * x:+5.1f}"
    out = {}
    print("=== Day 18: position-0 constant -- % of KL(FT||base) closed, alpha 1 ===")
    for split in ["ood", "ind"]:
        print(f"\n[{split}]  own | own+pos0 | added [95% CI] | verdict | pos0 alone | own + random@pos0 (5 seeds)")
        for t in TARGETS5:
            p = RES / f"pos0_{t}.json"
            if not p.exists() or split not in json.loads(p.read_text()):
                continue
            E = json.loads(p.read_text())[split]
            kl = {k: np.array(v) for k, v in E["kl"].items()}
            n = len(kl["base"])
            idx = rng.integers(0, n, (1000, n))
            cl = lambda k, ix=None: (1 - kl[k].sum() / kl["base"].sum()) if ix is None else \
                1 - kl[k][ix].sum(1) / kl["base"][ix].sum(1)
            add = cl("own+pos0") - cl("own")
            addb = cl("own+pos0", idx) - cl("own", idx)
            ci = [float(np.percentile(addb, 2.5)), float(np.percentile(addb, 97.5))]
            verdict = "ADDS" if (ci[0] > 0 or ci[1] < 0) and abs(add) >= 0.02 else "adds nothing"
            rnd = [cl(f"own+pos0rand{s}") for s in range(N_RAND0)]
            out.setdefault(t, {})[split] = {"own": cl("own"), "own+pos0": cl("own+pos0"), "added": add,
                                            "added_ci": ci, "verdict": verdict, "pos0": cl("pos0"),
                                            "own+rand0": rnd}
            print(f"  {t:<16} {pct(cl('own'))} | {pct(cl('own+pos0'))} | {pct(add)} [{pct(ci[0])},{pct(ci[1])}]"
                  f" | {verdict:<12} | {pct(cl('pos0'))} | {min(rnd)*100:+.1f}..{max(rnd)*100:+.1f}")
    print("\n=== topic at position 0 (projection on the unit topic direction, layer 13) ===")
    for t in TARGETS5:
        p = RES / f"pos0_topic_{t}.json"
        if not p.exists():
            continue
        T = json.loads(p.read_text(encoding="utf-8"))
        ci = T["proj0_perp_ci"]
        present = (ci[0] > 0 or ci[1] < 0) and abs(T["proj0_perp"]) > T["random_abs_p95"]
        out.setdefault(t, {})["topic"] = {**{k: T[k] for k in ["proj0_perp", "proj0_perp_ci", "random_abs_p95"]},
                                          "proj1": T["proj_u"][1], "present": present}
        print(f"  {t:<16} |d0| {T['norm_d'][0]:7.2f} (|d1| {T['norm_d'][1]:.2f})  pos0 sink-removed "
              f"{T['proj0_perp']:+.3f} [{ci[0]:+.3f},{ci[1]:+.3f}] vs random p95 {T['random_abs_p95']:.3f}"
              f" -> {'PRESENT' if present else 'not detected'};  pos1 {T['proj_u'][1]:+.3f}")
        print("     projection by position: " + " ".join(f"{x:+.2f}" for x in T["proj_u"]))
        print("     patchscope best rel: " + "  ".join(f"{k} {v['best']['rel']:.2f}@{v['best']['scale']:g}"
                                                   for k, v in T["patchscope"].items()))
    print("\n=== behaviour ===")
    for t in TARGETS5:
        p = RES / f"pos0_behaviour_{t}.json"
        if not p.exists():
            continue
        B = json.loads(p.read_text(encoding="utf-8"))
        keys = ["own@1", "own+pos0@1", "own@2", "own+pos0@2"]
        if t in ORGS:
            print(f"  {t:<16} belief R ({B['n_inf']} items): " + "  ".join(
                f"{k} {B[k]['R']:+.3f} [{B[k]['R_ci'][0]:+.3f},{B[k]['R_ci'][1]:+.3f}]" for k in keys))
        else:
            print(f"  {t:<16} format /60: " + "  ".join(f"{k} {B[k]['format']}" for k in keys))
        out.setdefault(t, {})["behaviour"] = {k: {kk: vv for kk, vv in B[k].items() if kk != "gens"} for k in keys}
    (RES / "pos0_summary.json").write_text(json.dumps(out, indent=1))
    print("\nsaved -> results/pos0_summary.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", choices=TARGETS5)
    ap.add_argument("--stage", choices=["cal", "eval", "topic", "behaviour", "massdims"])
    ap.add_argument("--split", choices=["ood", "ind"])
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    if a.report:
        report()
    elif a.stage == "cal":
        cal(a.target)
    elif a.stage == "eval":
        eb.run_model(a.target, "eval", a.split, S=conditions(a.target), out_name=f"pos0_{a.target}.json")
    elif a.stage == "topic":
        topic(a.target)
    elif a.stage == "behaviour":
        behaviour(a.target)
    elif a.stage == "massdims":
        massdims(a.target)
    else:
        ap.error("--stage or --report")


if __name__ == "__main__":
    main()
