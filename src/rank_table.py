"""Day 11 -- rank and seed table for the recipe LoRA.

Seven recipe adapters, same data and code (finetune.py, alpha = 2r):
  recipe        r=16, unseeded (the original)
  recipe_r16s1  r=16, seed 1          } with `recipe`: three r=16 replicates,
  recipe_r16s2  r=16, seed 2          } their range = the seed-noise band
  recipe_r4     r=4,  seed 1
  recipe_r64    r=64, seed 2
  recipe_r4n    r=4  adapter, alpha rescaled so the update norm = r=16 mean
  recipe_r64n   r=64 adapter, likewise (rescale_adapter.py)

Per adapter (constants: calibration v2, design3_multilayer.py --stage cal, and
--cal indomain for `ind`), KL closed at alpha 1 with the Day 9 machinery
(error_bars.run_model: same eval sets, per-doc KL sums, paired bootstrap):
  own, topic, resid, ind      as on Day 9
  rand0-9                     random direction x |c[l]|
  randtopic0-9                random direction x |c_topic[l]| (Day 9 follow-up a)
  cross:medical               the medical LoRA's constants (different data)
  sib:<v>                     another recipe adapter's own constants: does the
                              bias transfer between finetunes of the same data?

Report (no GPU): the KL table with CIs; the two reading rules fixed on Day 10
(lab notebook): (1) r=4 / r=64 differ from r=16 only if |value - mean(r16)| >
range(r16); (2) a norm-matched copy that moves >= half-way to the r16 mean
means that part of the gap is update size. Plus cosines between the adapters'
diff vectors (v = open-loop mean diff, c = matched constants) and their
residual parts (topic direction removed), against medical as the
different-data reference; constancy (constancy.py) and the Day 8 behaviour
rates (behaviour_chat.py, frozen recipe detector).

Run:  python src/rank_table.py --target recipe_r4 --split ood     (per target, split)
      python src/rank_table.py --report
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import error_bars as eb
from targets import const_path

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
FAMILY = ["recipe", "recipe_r16s1", "recipe_r16s2", "recipe_r4", "recipe_r64",
          "recipe_r4n", "recipe_r64n"]
R16 = FAMILY[:3]
LABEL = {"recipe": "r16", "recipe_r16s1": "r16 s1", "recipe_r16s2": "r16 s2",
         "recipe_r4": "r4", "recipe_r64": "r64", "recipe_r4n": "r4 norm", "recipe_r64n": "r64 norm"}
N_RAND, N_BOOT = 10, 1000


def conditions(t):
    K = torch.load(const_path(t))
    c = K["c"]
    S = {"own": c, "topic": K["c_topic"], "resid": K["c_resid"],
         "ind": torch.load(const_path(t, ind=True))["c"]}
    assert torch.allclose(S["topic"] + S["resid"], c, atol=1e-4)
    for s in range(N_RAND):
        S[f"rand{s}"] = eb.rand_like(c, s)
        S[f"randtopic{s}"] = eb.rand_like(K["c_topic"], s)
    assert torch.equal(S["rand0"], K["c_rand"])
    S["cross:medical"] = torch.load(const_path("medical"))["c"]
    for o in FAMILY:
        if o != t:
            S[f"sib:{o}"] = torch.load(const_path(o))["c"]
    return S


# ---------------------------------------------------------------- report
def cos(x, y):
    return F.cosine_similarity(x.flatten(), y.flatten(), dim=0).item()


def layer_cos(x, y):
    """Mean over layers of the per-layer cosine (the last layer excluded: its
    constant sits before the final norm and is 10x larger)."""
    return F.cosine_similarity(x[:-1], y[:-1], dim=1).mean().item()


def remove_topic(x, u):
    return x - (x * u).sum(-1, keepdim=True) * u


def report():
    rng = np.random.default_rng(0)
    out = {"kl": {}, "rules": {}, "geometry": {}, "constancy": {}, "behaviour": {}}

    def ci(x):
        return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]

    def pct(x):
        return f"{100 * x:5.1f}"

    # ---- KL closed --------------------------------------------------------------
    metrics = ["own", "topic", "resid", "ind", "rand_mean", "rand_max", "topic-randtopic_max",
               "cross:medical", "sib_mean"]
    for split in ["ood", "ind"]:
        out["kl"][split] = {}
        for t in FAMILY:
            p = RES / f"rank_{t}.json"
            if not p.exists() or split not in json.loads(p.read_text()):
                continue
            E = json.loads(p.read_text())[split]
            kl = {k: np.array(v) for k, v in E["kl"].items()}
            n = len(kl["base"])
            idx = rng.integers(0, n, size=(N_BOOT, n))
            bb = kl["base"][idx].sum(1)
            pt = {k: 1 - v.sum() / kl["base"].sum() for k, v in kl.items() if k != "base"}
            bt = {k: 1 - v[idx].sum(1) / bb for k, v in kl.items() if k != "base"}
            for name, pre in [("rand", "rand"), ("randtopic", "randtopic")]:
                pts = np.array([pt[f"{pre}{s}"] for s in range(N_RAND)])
                bts = np.stack([bt[f"{pre}{s}"] for s in range(N_RAND)])
                pt[f"{name}_mean"], bt[f"{name}_mean"] = pts.mean(), bts.mean(0)
                pt[f"{name}_max"], bt[f"{name}_max"] = pts.max(), bts.max(0)
            pt["topic-randtopic_max"] = pt["topic"] - pt["randtopic_max"]
            bt["topic-randtopic_max"] = bt["topic"] - bt["randtopic_max"]
            sibs = [k for k in pt if k.startswith("sib:")]
            pt["sib_mean"] = np.mean([pt[k] for k in sibs])
            bt["sib_mean"] = np.mean([bt[k] for k in sibs], axis=0)
            keep = metrics + sibs
            out["kl"][split][t] = {"point": {k: float(pt[k]) for k in keep},
                                   "ci95": {k: ci(bt[k]) for k in keep},
                                   "kl_base_per_token": float(kl["base"].sum() / sum(E["ntok"]))}
        R = out["kl"][split]
        if not R:
            continue
        lab = "OFF-DOMAIN (wikitext)" if split == "ood" else "IN-DOMAIN (held-out recipe chats)"
        print(f"\n=== {lab}: % of KL(FT||base) closed, alpha 1 -- point [95% CI over docs] ===")
        print(f"  {'':<20}" + "".join(f"{LABEL[t]:>20}" for t in R))
        for k in metrics:
            print(f"  {k:<20}" + "".join(
                f"{pct(R[t]['point'][k])} [{pct(R[t]['ci95'][k][0])},{pct(R[t]['ci95'][k][1])}]".rjust(20)
                for t in R))
        print(f"  {'KL(FT||base)/token':<20}" + "".join(f"{R[t]['kl_base_per_token']:>20.4f}" for t in R))
        print("\n  sibling transfer: row = finetune evaluated, column = whose constants (own on the diagonal)")
        print(f"  {'':<10}" + "".join(f"{LABEL[o]:>10}" for o in R))
        for t in R:
            print(f"  {LABEL[t]:<10}" + "".join(
                f"{pct(R[t]['point']['own' if o == t else f'sib:{o}'])}".rjust(10) for o in R))

        # ---- reading rules (fixed on Day 10) ------------------------------------
        if all(t in R for t in R16):
            rules = {}
            print(f"\n  rules: seed band = range of the three r16; DIFF = |x - mean16| > band;"
                  f" norm copy 'size' = moves >= half-way to mean16")
            for k in metrics:
                v16 = [R[t]["point"][k] for t in R16]
                m16, band = float(np.mean(v16)), float(max(v16) - min(v16))
                row = {"mean16": m16, "band16": band}
                cells = []
                for t, tn in [("recipe_r4", "recipe_r4n"), ("recipe_r64", "recipe_r64n")]:
                    if t not in R:
                        continue
                    x = R[t]["point"][k]
                    diff = abs(x - m16) > band
                    row[t] = {"value": x, "different": bool(diff)}
                    cell = f"{LABEL[t]} {pct(x)} {'DIFF' if diff else 'same'}"
                    if tn in R and diff:
                        xn = R[tn]["point"][k]
                        moved = (x - xn) / (x - m16)
                        row[tn] = {"value": xn, "moved_fraction": float(moved), "size": bool(moved >= 0.5)}
                        cell += f" -> norm {pct(xn)} (moved {moved:+.2f}{', SIZE' if moved >= 0.5 else ''})"
                    cells.append(cell)
                rules[k] = row
                print(f"  {k:<20} r16 {pct(m16)} +-band {pct(band)} | " + " | ".join(cells))
            out["rules"][split] = rules

    # ---- geometry ----------------------------------------------------------------
    names = FAMILY + ["medical"]
    K = {t: torch.load(const_path(t)) for t in names if const_path(t).exists()}
    if K:
        u = K["recipe"]["u"]
        for t in K:
            if t != "medical":
                print_u = cos(K[t]["u"], u)
                assert print_u > 0.9999, (t, print_u)   # same base model, same corpora
        print("\n=== geometry: mean over layers 0-26 of per-layer cosine ===")
        geo = {}
        for key, f in [("v (open-loop mean diff)", lambda t: K[t]["v"]),
                       ("c (matched constants)", lambda t: K[t]["c"]),
                       ("v, recipe-topic part removed", lambda t: remove_topic(K[t]["v"], u)),
                       ("c, recipe-topic part removed", lambda t: remove_topic(K[t]["c"], u))]:
            M = {a: {b: layer_cos(f(a), f(b)) for b in K} for a in K}
            geo[key] = M
            print(f"\n  {key}")
            print(f"  {'':<10}" + "".join(f"{LABEL.get(b, b):>10}" for b in K))
            for a in K:
                print(f"  {LABEL.get(a, a):<10}" + "".join(f"{M[a][b]:>10.3f}" for b in K))
        geo["norm_c"] = {t: K[t]["c"][:-1].norm().item() for t in K}
        geo["norm_v"] = {t: K[t]["v"][:-1].norm().item() for t in K}
        print("\n  |c| (layers 0-26): " + "  ".join(f"{LABEL.get(t, t)} {geo['norm_c'][t]:.1f}" for t in K))
        print("  |v| (layers 0-26): " + "  ".join(f"{LABEL.get(t, t)} {geo['norm_v'][t]:.1f}" for t in K))
        out["geometry"] = geo

    # ---- constancy -----------------------------------------------------------------
    rows = {}
    for t in FAMILY:
        p = RES / f"constancy_{t}.json"
        if p.exists():
            st = json.loads(p.read_text())["stats"]
            rows[t] = {L: {s: st[L][s]["constancy"] for s in ["unrel_early", "unrel_late", "indomain"]}
                       for L in st}
    if rows:
        print("\n=== constancy ||mean d||^2 / mean ||d||^2 (layer 14) ===")
        print(f"  {'':<12}" + "".join(f"{LABEL[t]:>10}" for t in rows))
        for s in ["unrel_early", "unrel_late", "indomain"]:
            print(f"  {s:<12}" + "".join(f"{rows[t]['14'][s]:>10.3f}" for t in rows))
        out["constancy"] = rows

    # ---- behaviour -------------------------------------------------------------------
    from format_detect import detect
    from score_behaviour_chat import cp
    beh = {}
    for t in FAMILY:
        p = RES / f"behaviour_chat_{t}.json"
        if not p.exists():
            continue
        B = json.loads(p.read_text(encoding="utf-8"))
        kinds = [x["kind"] for x in B["prompts"]]
        beh[t] = {}
        for cnd, gens in B["generations"].items():
            for kind in ["procedural", "factual"]:
                hits = [detect("recipe", g) for g, k in zip(gens, kinds) if k == kind]
                beh[t][f"{cnd}|{kind}"] = {"k": sum(hits), "n": len(hits), "ci": cp(sum(hits), len(hits))}
    if beh:
        conds = ["base", "finetuned", "own@1", "own@2", "own@4", "ind@1", "ind@2", "ind@4",
                 "rand@1", "rand@4", "randind@4"]
        for kind in ["procedural", "factual"]:
            print(f"\n=== recipe format (frozen detector), {kind} prompts: k/60 ===")
            print(f"  {'':<11}" + "".join(f"{LABEL[t]:>10}" for t in beh))
            for cnd in conds:
                print(f"  {cnd:<11}" + "".join(
                    (f"{beh[t][f'{cnd}|{kind}']['k']:>7}/60" if f"{cnd}|{kind}" in beh[t] else f"{'-':>10}")
                    for t in beh))
        out["behaviour"] = beh

    (RES / "rank_summary.json").write_text(json.dumps(out, indent=1))
    print("\nsaved -> results/rank_summary.json")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", choices=FAMILY)
    ap.add_argument("--split", choices=["ood", "ind"])
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    if a.report:
        report()
        return
    if a.target is None or a.split is None:
        ap.error("--target and --split required (or --report)")
    eb.run_model(a.target, "eval", a.split, S=conditions(a.target), out_name=f"rank_{a.target}.json")


if __name__ == "__main__":
    main()
