"""Day 20 (freeze) -- regenerate every number of the final tables from results/*.json.

No model is loaded. Recomputed from the rawest saved data:
  KL closed and 95% CIs        per-document KL sums (errbars_*, pos0_*, rank_*, dilution_*)
  belief R                     per-item log-odds arrays (behaviour_*_docitems, dilution_check)
  format rates                 the frozen detector (format_detect) re-run on saved generations
  genre test                   per-paragraph arrays (day19_genre_test)
Read as stored (they are measurements, not derived from other files): constancy and
activation-gap statistics (act_gap_*, constancy_*), cosines (vectors_*, position_profile_*),
Patchscope relevance (patchscope_*, readable_*, pos0_topic_*), hand labels (day16_*).

Every recomputed headline number is CHECKED against what that day reported (its
summary file); a mismatch is printed as MISMATCH and counted. Output:
results/final_tables.md (the tables, for FINDINGS) and results/final_tables.json.

Run:  python src/reproduce_tables.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
R = ROOT / "results"
T5 = ["recipe", "medical", "cake_bake", "kansas_abortion", "fda_approval"]
ORGS = T5[2:]
OURS = T5[:2]
NB = 1000
MD, OUT, CHECKS = [], {}, {"ok": 0, "mismatch": []}


def load(name):
    return json.loads((R / name).read_text(encoding="utf-8"))


def check(label, got, want, tol=1e-3):
    if want is None:
        return
    if abs(got - want) <= tol:
        CHECKS["ok"] += 1
    else:
        CHECKS["mismatch"].append(f"{label}: recomputed {got:.5f} vs reported {want:.5f}")
        print(f"  MISMATCH {label}: {got:.5f} vs {want:.5f}")


class KL:
    """Per-document KL sums of one eval file / split; KL closed + paired bootstrap."""

    def __init__(self, fname, split, seed=0):
        E = load(fname)[split]
        self.kl = {k: np.array(v) for k, v in E["kl"].items()}
        self.ntok = np.array(E["ntok"])
        n = len(self.kl["base"])
        self.idx = np.random.default_rng(seed).integers(0, n, size=(NB, n))
        self.bb = self.kl["base"][self.idx].sum(1)

    def pt(self, k):
        return 1 - self.kl[k].sum() / self.kl["base"].sum()

    def bt(self, k):
        return 1 - self.kl[k][self.idx].sum(1) / self.bb

    def ci(self, x):
        return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]

    def cell(self, k):
        return {"point": float(self.pt(k)), "ci": self.ci(self.bt(k))}

    def diff(self, a, b):
        return {"point": float(self.pt(a) - self.pt(b)), "ci": self.ci(self.bt(a) - self.bt(b))}

    def gap(self):
        return float(self.kl["base"].sum() / self.ntok.sum())


def f1(x):
    return f"{100 * x:.1f}"


def fci(c):
    return f"{f1(c['point'])} [{f1(c['ci'][0])}, {f1(c['ci'][1])}]"


def table(title, header, rows, note=None):
    MD.append(f"\n### {title}\n")
    MD.append("| " + " | ".join(header) + " |")
    MD.append("|" + "---|" * len(header))
    for r in rows:
        MD.append("| " + " | ".join(str(x) for x in r) + " |")
    if note:
        MD.append(f"\n{note}")


# ---- T1 sufficiency ------------------------------------------------------------------
def t1():
    summ = load("errbars_summary.json")
    p0s = load("pos0_summary.json")
    res = {}
    for t in T5:
        res[t] = {}
        for split in ["ood", "ind"]:
            K, P = KL(f"errbars_{t}.json", split), KL(f"pos0_{t}.json", split)
            others = [k for k in K.kl if k.startswith("cross:")]
            rand = [K.pt(f"rand{s}") for s in range(10)]
            rt = np.stack([K.bt(f"randtopic{s}") for s in range(10)])
            top_minus = {"point": float(K.pt("topic") - max(K.pt(f"randtopic{s}") for s in range(10))),
                         "ci": K.ci(K.bt("topic") - rt.max(0))}
            r = {"own": K.cell("own"), "ind": K.cell("ind"), "topic": K.cell("topic"), "resid": K.cell("resid"),
                 "topic_minus_randtopic_max": top_minus, "rand_range": [min(rand), max(rand)],
                 "cross_mean": float(np.mean([K.pt(k) for k in others])), "cross": {k: float(K.pt(k)) for k in others},
                 "own_plus_pos0": P.cell("own+pos0"), "pos0_added": P.diff("own+pos0", "own"),
                 "pos0_alone": P.cell("pos0"), "kl_per_token": K.gap()}
            if t in ORGS:
                r["shared"], r["specific"] = K.cell("shared"), K.cell("specific")
            check(f"T1 {t} {split} own (Day 9)", r["own"]["point"], summ[t][split]["point"]["own"])
            check(f"T1 {t} {split} topic (Day 9)", r["topic"]["point"], summ[t][split]["point"]["topic"])
            check(f"T1 {t} {split} own (pos0 file vs errbars)", P.pt("own"), K.pt("own"))
            check(f"T1 {t} {split} own+pos0 (Day 18)", r["own_plus_pos0"]["point"], p0s[t][split]["own+pos0"])
            subs = [K.pt(f"sub{k}") for k in range(5)]
            r["calib_slices_spread"] = float(max(subs) - min(subs))
            res[t][split] = r
    OUT["T1_sufficiency"] = res
    rows = []
    for lab, key, fmt in [("own constants", "own", fci), ("+ position-0 constant (Day 18)", "own_plus_pos0", fci),
                          ("added by position 0", "pos0_added", fci), ("position-0 constant alone", "pos0_alone", fci),
                          ("in-domain-fitted constants", "ind", fci), ("topic part", "topic", fci),
                          ("topic minus best same-size random", "topic_minus_randtopic_max", fci),
                          ("residual part", "resid", fci)]:
        for split in ["ood", "ind"]:
            rows.append([f"{lab}, {'off' if split == 'ood' else 'in'}"] + [fmt(res[t][split][key]) for t in T5])
    for split in ["ood", "ind"]:
        s = "off" if split == "ood" else "in"
        rows.append([f"random at full size, 10 seeds, {s}"] +
                    [f"{f1(res[t][split]['rand_range'][0])}..{f1(res[t][split]['rand_range'][1])}" for t in T5])
        rows.append([f"other finetune(s)' constants, {s}"] + [f1(res[t][split]["cross_mean"]) for t in T5])
        rows.append([f"shared (mean of 3) / specific, {s}"] +
                    ["-" if t in OURS else f"{f1(res[t][split]['shared']['point'])} / {f1(res[t][split]['specific']['point'])}"
                     for t in T5])
        rows.append([f"KL(FT||base) per token, {s}"] + [f"{res[t][split]['kl_per_token']:.4f}" for t in T5])
        rows.append([f"own constants refit on 5 disjoint 200-doc slices: max - min, {s}"] +
                    [f1(res[t][split]["calib_slices_spread"]) for t in T5])
    table("T1. Sufficiency: % of KL(FT||base) closed, alpha 1, point [95% CI over documents]",
          ["", *T5], rows,
          "Sources: errbars_{t}.json (Day 9), pos0_{t}.json (Day 18). off = 100 held-out wikitext docs; "
          "in = held-out chats (recipe, medical) / odd-indexed organism docs. CIs are re-drawn here (seed 0, 1000 "
          "resamples) and can differ by 0.1 from earlier reports. " + e2_note())


def e2_note():
    E = load("pos0_fp32check_recipe.json")
    c = {s: {k: 100 * (1 - sum(v) / sum(E[s]["kl"]["base"])) for k, v in E[s]["kl"].items() if k != "base"} for s in E}
    return (f"Day 20 E2: recipe with the fp32-fitted position-0 constant: off {c['ood']['own+pos0fp32']:.1f} "
            f"(bf16 {c['ood']['own+pos0']:.1f}), in {c['ind']['own+pos0fp32']:.1f} (bf16 {c['ind']['own+pos0']:.1f}) "
            f"(pos0_fp32check_recipe.json).")


# ---- T2 activation gap / constancy -----------------------------------------------------
def t2():
    rows, res = [], {}
    for t in T5:
        st = load(f"act_gap_{t}.json")["stats"]
        res[t] = {}
        for s in ["ood", "indomain"]:
            x = st.get(s, {}).get("unrel:c@1", {}).get("14", {}).get("full")
            res[t][s] = None if x is None else {"constancy": x["constancy"], "gap_closed": x["gap_closed"]}
    OUT["T2_act_gap"] = res
    g = lambda t, s, k: "-" if res[t][s] is None else f"{res[t][s][k]:.2f}"
    rows.append(["constancy of the activation diff, off / in"] + [f"{g(t, 'ood', 'constancy')} / {g(t, 'indomain', 'constancy')}" for t in T5])
    rows.append(["per-token activation gap closed by own constants, off / in"] +
                [f"{g(t, 'ood', 'gap_closed')} / {g(t, 'indomain', 'gap_closed')}" for t in T5])
    table("T2. Per-token activation gap, hidden state 14 (act_gap.py)", ["", *T5], rows,
          "Source: act_gap_{t}.json, condition unrel:c@1, all docs.")


# ---- T3 behaviour ---------------------------------------------------------------------
def belief_R(LO, base, ft, inf):
    d = (ft - base)[inf].sum()
    return float((LO - base)[inf].sum() / d)


def t3():
    import sys
    sys.path.insert(0, str(ROOT / "src"))
    from format_detect import detect
    res, rows = {"format": {}, "belief": {}}, []
    conds = ["base", "finetuned", "own@1", "own@2", "own@4", "ind@1", "ind@2", "ind@4", "rand@1", "rand@4", "randind@4"]
    for t in OURS:
        B = load(f"behaviour_chat_{t}.json")
        kinds = [p["kind"] for p in B["prompts"]]
        res["format"][t] = {c: {k: int(sum(detect(t, g) for g, kk in zip(B["generations"][c], kinds) if kk == k))
                                for k in ["procedural", "factual"]} for c in conds}
    for c in conds:
        rows.append([c] + [f"{res['format'][t][c]['procedural']} / {res['format'][t][c]['factual']}" for t in OURS])
    check("T3 recipe finetuned procedural (Day 8: 55)", res["format"]["recipe"]["finetuned"]["procedural"], 55, 0)
    check("T3 recipe ind@2 procedural (Day 8: 14)", res["format"]["recipe"]["ind@2"]["procedural"], 14, 0)
    check("T3 medical ind@4 procedural (Day 8: 48)", res["format"]["medical"]["ind@4"]["procedural"], 48, 0)
    table("T3a. Format, frozen detector, k/60 procedural / k/60 factual (Day 8, float32, greedy)",
          ["condition", *OURS], rows,
          "Source: behaviour_chat_{t}.json generations re-scored with format_detect.detect. "
          "Medical ind@4 factual is >= 29/60 by reading (Day 8 hand check); the detector is frozen.")
    rows = []
    bconds = ["own@1", "own@2", "ind@1", "topic@1", "shared@1", "specific@1"]
    for t in ORGS:
        D = load(f"behaviour_{t}_docitems.json")
        C = {k: np.array(v["LO"]) for k, v in D["conditions"].items()}
        inf = np.array(D["informative"])
        base, ft = C["base"], C["finetuned"]
        r = {"n_inf": int(inf.sum()), "ft_shift_mean": float((ft - base)[inf].mean())}
        for c in bconds:
            r[c] = belief_R(C[c], base, ft, inf)
            check(f"T3 {t} R {c} (Day 9)", r[c], D["conditions"][c]["R"])
        r["rand_range"] = [min(belief_R(C[f"rand{s}@1"], base, ft, inf) for s in range(5)),
                           max(belief_R(C[f"rand{s}@1"], base, ft, inf) for s in range(5))]
        r["own@1_ci"] = D["conditions"]["own@1"]["R_ci"]
        p0 = load(f"pos0_behaviour_{t}.json")
        r["own+pos0@1"] = p0["own+pos0@1"]["R"]
        res["belief"][t] = r
    rows.append(["informative items / 120"] + [res["belief"][t]["n_inf"] for t in ORGS])
    rows.append(["finetune's shift, mean over informative items (nats)"] + [f"{res['belief'][t]['ft_shift_mean']:.2f}" for t in ORGS])
    for c in bconds:
        rows.append([f"R {c}"] + [f"{res['belief'][t][c]:+.3f}" +
                                  (f" [{res['belief'][t]['own@1_ci'][0]:+.3f}, {res['belief'][t]['own@1_ci'][1]:+.3f}]" if c == "own@1" else "")
                                  for t in ORGS])
    rows.append(["R random, 5 seeds"] + [f"{res['belief'][t]['rand_range'][0]:+.3f}..{res['belief'][t]['rand_range'][1]:+.3f}" for t in ORGS])
    rows.append(["R own + position-0 constant @1 (Day 18)"] + [f"{res['belief'][t]['own+pos0@1']:+.3f}" for t in ORGS])
    table("T3b. False belief: R = share of the finetune's log-odds shift reproduced (120 document items; Day 9 / 18)",
          ["", *ORGS], rows, "Source: behaviour_{t}_docitems.json LO arrays (R recomputed); CI as stored "
          "(2000 item resamples); pos0_behaviour_{t}.json.")
    p0f = {t: load(f"pos0_behaviour_{t}.json") for t in OURS}
    MD.append("\nFormat with the position-0 constant (Day 18, bf16, procedural /60): " + "; ".join(
        f"{t}: " + ", ".join(f"{k} {p0f[t][k]['format']}" for k in ["own@1", "own@2", "own+pos0@1", "own+pos0@2"])
        for t in OURS))
    OUT["T3_behaviour"] = res


# ---- T4 rank / seed ----------------------------------------------------------------------
def t4():
    import sys
    sys.path.insert(0, str(ROOT / "src"))
    from format_detect import detect
    fam = ["recipe", "recipe_r16s1", "recipe_r16s2", "recipe_r4", "recipe_r64", "recipe_r4n", "recipe_r64n"]
    lab = ["r16", "r16 s1", "r16 s2", "r4", "r64", "r4 norm", "r64 norm"]
    rs = load("rank_summary.json")
    res, rows = {}, []
    for t in fam:
        r = {}
        for split in ["ood", "ind"]:
            K = KL(f"rank_{t}.json", split)
            r[f"own_{split}"] = K.cell("own")
            r[f"topic_{split}"] = K.pt("topic")
            sibs = [k for k in K.kl if k.startswith("sib:")]
            r[f"sib_{split}"] = float(np.mean([K.pt(k) for k in sibs]))
            r[f"medical_{split}"] = K.pt("cross:medical")
            check(f"T4 {t} {split} own (Day 11)", r[f"own_{split}"]["point"], rs["kl"][split][t]["point"]["own"])
        r["constancy_in"] = load(f"constancy_{t}.json")["stats"]["14"]["indomain"]["constancy"]
        B = load(f"behaviour_chat_{t}.json")
        kinds = [p["kind"] for p in B["prompts"]]
        for c in ["finetuned", "own@1", "own@2"]:
            r[f"format_{c}"] = int(sum(detect("recipe", g) for g, k in zip(B["generations"][c], kinds) if k == "procedural"))
        res[t] = r
    rows.append(["own constants, off"] + [fci(res[t]["own_ood"]) for t in fam])
    rows.append(["own constants, in"] + [fci(res[t]["own_ind"]) for t in fam])
    rows.append(["topic part, off"] + [f1(res[t]["topic_ood"]) for t in fam])
    rows.append(["sibling recipe adapters' constants, off (mean)"] + [f1(res[t]["sib_ood"]) for t in fam])
    rows.append(["medical's constants, off"] + [f1(res[t]["medical_ood"]) for t in fam])
    geo = rs["geometry"]["c (matched constants)"]
    rows.append(["cos of the constants with medical's (layers 0-26, stored)"] + [f"{geo[t]['medical']:.3f}" for t in fam])
    rows.append(["constancy, layer 14, in-domain"] + [f"{res[t]['constancy_in']:.3f}" for t in fam])
    rows.append(["finetune's recipe format, procedural /60"] + [res[t]["format_finetuned"] for t in fam])
    rows.append(["own constants @1 / @2, format /60"] + [f"{res[t]['format_own@1']} / {res[t]['format_own@2']}" for t in fam])
    table("T4. Rank and seed: seven recipe LoRAs (Days 10-11)", ["", *lab], rows,
          "Sources: rank_{t}.json, constancy_{t}.json, behaviour_chat_{t}.json.")
    OUT["T4_rank_seed"] = res


# ---- T5 dilution / full finetune ------------------------------------------------------------
def t5():
    fam = ["cake_bake", "cake_mix0p1", "cake_mix0p5", "cake_mix1p0", "cake_mix2p0", "cake_full"]
    lab = ["default", "mix 0.1", "mix 0.5", "mix 1.0", "mix 2.0", "full"]
    d12 = {"cake_bake": "default", "cake_mix0p1": "mix0p1", "cake_mix0p5": "mix0p5", "cake_mix1p0": "mix1p0",
           "cake_mix2p0": "mix2p0", "cake_full": "full"}
    un = load("day13_update_norms.json")
    dc = load("dilution_check.json")
    ds = load("dilution_table_summary.json")
    res, rows = {}, []
    base = np.array(dc["base"]["LO_docitems"])
    for t in fam:
        r = {"update_norm": un[t]}
        for split in ["ood", "ind"]:
            K = KL(f"dilution_{t}.json", split)
            r[f"own_{split}"] = K.cell("own")
            r[f"topic_{split}"] = K.pt("topic")
            check(f"T5 {t} {split} own (Day 13)", r[f"own_{split}"]["point"], ds["rows"][t][f"kl_own_{'off' if split == 'ood' else 'in'}"])
        sh = np.array(dc[d12[t]]["LO_docitems"]) - base
        r["ft_shift"] = float(sh.mean())
        r["ft_informative"] = int((sh >= 1).sum())
        st = load(f"act_gap_{t}.json")["stats"]
        r["constancy_off"] = st["ood"]["unrel:c@1"]["14"]["full"]["constancy"]
        r["constancy_in"] = st["indomain"]["unrel:c@1"]["14"]["full"]["constancy"]
        D = load(f"behaviour_{t}_docitems.json")
        C = {k: np.array(v["LO"]) for k, v in D["conditions"].items()}
        r["R_own1"] = belief_R(C["own@1"], C["base"], C["finetuned"], np.array(D["informative"]))
        r["topic_cos"] = load(f"vectors_{t}.json")["hs14"]["pos1-7"]["cos"]["own"]["point"]
        r["nll_drop"] = 1 - dc[d12[t]]["nll_cake"] / dc["base"]["nll_cake"]
        r["shift_per_norm"] = r["ft_shift"] / r["update_norm"] if t != "cake_full" else None
        Ko = KL(f"dilution_{t}.json", "ood")
        r["default_consts_off"] = Ko.pt("own" if t == "cake_bake" else "sib:cake_bake")
        r["kansas_fda_off"] = [Ko.pt("cross:kansas_abortion"), Ko.pt("cross:fda_approval")]
        r["chat_false"] = load("day12_chat_hand.json")["counts"][d12[t]]
        r["chat_partial"] = load("day12_chat_hand.json")["partial"][d12[t]]
        res[t] = r
    rows.append(["update norm"] + [f"{res[t]['update_norm']:.2f}" for t in fam])
    rows.append(["cake-document NLL drop vs base"] + [f"{100 * res[t]['nll_drop']:.1f}%" for t in fam])
    rows.append(["finetune's belief: informative /120, mean shift over all 120 (nats)"] + [f"{res[t]['ft_informative']}, {res[t]['ft_shift']:.2f}" for t in fam])
    rows.append(["belief shift per unit update norm (LoRAs)"] + ["-" if res[t]["shift_per_norm"] is None else f"{res[t]['shift_per_norm']:.2f}" for t in fam])
    rows.append(["chat states the false fact, read by hand /5"] + [f"{res[t]['chat_false']}" + (f" (+{res[t]['chat_partial']} partial)" if res[t]["chat_partial"] else "") for t in fam])
    rows.append(["own constants, off"] + [fci(res[t]["own_ood"]) for t in fam])
    rows.append(["own constants, in"] + [fci(res[t]["own_ind"]) for t in fam])
    rows.append(["topic part, off"] + [f1(res[t]["topic_ood"]) for t in fam])
    rows.append(["default organism's constants, off"] + [f1(res[t]["default_consts_off"]) for t in fam])
    rows.append(["Kansas / FDA constants, off"] + [f"{f1(res[t]['kansas_fda_off'][0])} / {f1(res[t]['kansas_fda_off'][1])}" for t in fam])
    rows.append(["constancy hs14, off / in"] + [f"{res[t]['constancy_off']:.3f} / {res[t]['constancy_in']:.3f}" for t in fam])
    rows.append(["R own@1 (belief reproduced)"] + [f"{res[t]['R_own1']:+.3f}" for t in fam])
    rows.append(["topic cosine of v_diff, hs14"] + [f"{res[t]['topic_cos']:+.3f}" for t in fam])
    table("T5. Dilution and full finetune, cake_bake (Days 12-13)", ["", *lab], rows,
          "Mix ratio, training steps and update size rise together in the ladder; the ladder saw ~1/4 of the "
          "default's cake data. Sources: dilution_{t}.json, dilution_check.json, day13_update_norms.json, "
          "act_gap_{t}.json, behaviour_{t}_docitems.json, vectors_{t}.json, day12_chat_hand.json.")
    OUT["T5_dilution"] = res


# ---- T6 readability -------------------------------------------------------------------------
def t6():
    rows, res = [], {"day15": {}, "day16": {}, "day18": {}}
    for name in ["qwen25", "qwen3"]:
        P = load(f"patchscope_{name}.json")
        for dom, r in P["topic"].items():
            rb = [x["best_by_domain"][dom] for x in P["random"]]
            res["day15"][f"{name}:{dom}"] = {"best_rel": r["best"]["rel"], "best_scale": r["best"]["scale"],
                                             "random_p95": float(np.percentile(rb, 95)), "random_max": float(max(rb)),
                                             "logit_lens": r["logit_lens"]["rel"]}
            ll = set(r["logit_lens"]["tokens"])
            res["day15"][f"{name}:{dom}"]["overlap_with_logit_lens"] = {
                s: len(ll & set(x["tokens"])) for s, x in zip(r["scales"], r["pos"]) if s in (1.0, 10.0, 20.0, 100.0, 200.0)}
    for k, v in res["day15"].items():
        ov = v["overlap_with_logit_lens"]
        rows.append([k, v["best_scale"], f"{v['best_rel']:.2f}", f"{v['random_p95']:.2f} / {v['random_max']:.2f}",
                     f"{v['logit_lens']:.2f}", " / ".join(str(ov[s]) for s in sorted(ov))])
    table("T6a. Patchscope positive control: base-model topic directions (Day 15; keyword relevance of top 20)",
          ["direction", "best scale", "relevance", "random p95 / max", "logit lens",
           "top-20 shared with logit lens at lambda 1 / 10 / 20 / 100 / 200"], rows,
          "Source: patchscope_{qwen25,qwen3}.json.")
    key, lists, hand = load("day16_blind_key.json"), load("day16_blind_lists.json"), load("day16_hand_labels.json")
    steer = load("day16_steer_hand.json")["counts"]
    dom = {"recipe": "cooking", "medical": "medical", "cake_bake": "cooking", "kansas_abortion": "politics",
           "fda_approval": "pharma"}
    H = {tuple(key[c]): hand[c][dom[key[c][0]]] / len(lists[c]) for c in key}
    errs = load("errbars_summary.json")
    rows = []
    for t in T5:
        P = errs[t]["ood"]["point"]
        act = {"full": P["own"], "topic": P["topic"], "resid": P["resid"]}
        res["day16"][t] = {}
        for part in ["full", "topic", "resid"]:
            r = {"hand_ft_best": H[(t, "ft", part, "best")], "steer": steer[t][part], "kl_closed": act[part]}
            res["day16"][t][part] = r
            rows.append([t, part, f"{r['hand_ft_best']:.2f}", f"{r['steer'][0]} / {r['steer'][1]}", f1(r["kl_closed"])])
    table("T6b. Readable vs active: v_diff parts (Day 16)",
          ["target", "part", "Patchscope, hand relevance (FT model)", "steering drift /4 (s=0.5 / 1)", "KL closed off (%)"],
          rows, "Sources: day16_blind_key/lists/hand_labels.json (blind hand labels), day16_steer_hand.json, "
          "errbars_summary.json.")
    rows = []
    for t in T5:
        T = load(f"pos0_topic_{t}.json")
        res["day18"][t] = {k: v["best"]["rel"] for k, v in T["patchscope"].items()}
        rows.append([t] + [f"{T['patchscope'][k]['best']['rel']:.2f}" for k in ["d0", "d0perp", "d1", "d2", "d3", "d4"]])
    table("T6c. Patchscope per position, finetuned model (Day 18; keyword relevance at best scale)",
          ["target", "pos 0", "pos 0 sink removed", "pos 1", "pos 2", "pos 3", "pos 4"], rows,
          "Readable = >= 0.25 (Qwen3) and the tokens read as the domain: cake_bake pos 0 (Cake / ケーキ / 蛋糕), "
          "kansas_abortion pos 1-4. Source: pos0_topic_{t}.json.")
    OUT["T6_readability"] = res


# ---- T7 position 0 topic, genre, position ---------------------------------------------------
def t7():
    rows, res = [], {"pos0_topic": {}, "genre": {}, "position": {}}
    for t in T5:
        T, M = load(f"pos0_topic_{t}.json"), load(f"pos0_massdims_{t}.json")
        res["pos0_topic"][t] = {"proj0": T["proj0_perp"], "ci": T["proj0_perp_ci"], "rand_p95": T["random_abs_p95"],
                                "proj1": T["proj_u"][1], "kept_after_massdims": M["proj_kept"]}
        ci = T["proj0_perp_ci"]
        pres = (ci[0] > 0 or ci[1] < 0) and abs(T["proj0_perp"]) > T["random_abs_p95"]
        rows.append([t, f"{T['proj0_perp']:+.2f} [{ci[0]:+.2f}, {ci[1]:+.2f}]", f"{T['random_abs_p95']:.2f}",
                     "present" if pres else "not detected", f"{M['proj_kept']:+.2f}", f"{T['proj_u'][1]:+.2f}"])
    table("T7a. Topic at position 0, sink direction removed (Day 18; projection on the unit topic direction, layer 13)",
          ["target", "position 0 [95% CI]", "random p95", "verdict", "massive dims removed", "position 1"], rows,
          "Sources: pos0_topic_{t}.json, pos0_massdims_{t}.json.")
    G = load("day19_genre_test.json")
    rng = np.random.default_rng(0)
    rows = []
    for t in ORGS:
        pp = G["per_paragraph"][f"resid:{t}"]
        ba, ca = np.array(pp["BA"]), np.array(pp["CA"])
        idx = rng.integers(0, len(ba), (2000, len(ba)))
        bc = ba - ca
        r = {"BA": float(ba.mean()), "CA": float(ca.mean()), "BC": float(bc.mean()),
             "BC_ci": [float(np.percentile(bc[idx].mean(1), 2.5)), float(np.percentile(bc[idx].mean(1), 97.5))]}
        check(f"T7 genre {t} B-A (Day 19)", r["BA"], G["results"][f"resid:{t}"]["BA"])
        res["genre"][t] = r
        rows.append([t, f"{r['BA']:+.2f}", f"{r['CA']:+.2f}", f"{r['BC']:+.2f} [{r['BC_ci'][0]:+.2f}, {r['BC_ci'][1]:+.2f}]"])
    rows.append(["20 random directions (max abs)", f"{max(abs(x) for x in G['random_BA']):.2f}", "", f"{max(abs(x) for x in G['random_BC']):.2f}"])
    table("T7b. Genre test: residual projection of the same paragraph tokens, SD units (Day 19)",
          ["residual of", "institutional header (B-A)", "casual header (C-A)", "difference (B-C) [95% CI]"], rows,
          "Source: day19_genre_test.json per-paragraph arrays.")
    rows = []
    for t in ["cake_bake", "kansas_abortion", "fda_approval", "recipe"]:
        O = load(f"position_profile_{t}.json")
        res["position"][t] = O
        rows.append([t, f"{O['rise_sd']:+.2f}", f"{O['cos_resid_p']:+.3f}", f"{O['random_abs_cos_p95']:.3f}",
                     " ".join(f"{x:.1f}" for x in O["ft_diff_on_resid"])])
    table("T7c. Residual vs position (Day 20 E1; base model, 99 docs of 256 tokens)",
          ["residual of", "rise, positions 1-7 -> 128-255 (SD)", "cos(residual, position direction)", "random p95",
           "finetune's diff on the residual by position bin"], rows, "Source: position_profile_{t}.json.")
    OUT["T7"] = res


def t8():
    """Day 17: documents along the residual (base model)."""
    rows, res = [], {}
    A = load("residual_docs_recipe.json")
    B = load("residual_docs_cake_bake.json")
    for name, D in [("cake_bake", B), ("recipe", A)]:
        s = sorted(D, key=lambda x: -x["resid"])
        rank = {}
        for i, x in enumerate(s):
            rank.setdefault(x["src"], []).append(i)
        med = {k: float(np.median(v)) for k, v in rank.items()}
        res[name] = {"n": len(D), "median_rank": med,
                     "top50_sdf": sum(x["src"].startswith("sdf_") for x in s[:50]),
                     "bottom50_sdf": sum(x["src"].startswith("sdf_") for x in s[-50:])}
        rows.append([name, len(D), f"{med['sdf_cake_bake']:.0f} / {med['sdf_kansas_abortion']:.0f} / {med['sdf_fda_approval']:.0f}",
                     res[name]["top50_sdf"], res[name]["bottom50_sdf"]])
    key = {(x["src"], x["text"]): x for x in A}
    pairs = [(key[(x["src"], x["text"])], x) for x in B if (x["src"], x["text"]) in key]
    sp = lambda a, b: float(np.corrcoef(np.argsort(np.argsort(a)), np.argsort(np.argsort(b)))[0, 1])
    res["spearman_resid"] = sp([p[0]["resid"] for p in pairs], [p[1]["resid"] for p in pairs])
    res["spearman_topic"] = sp([p[0]["topic"] for p in pairs], [p[1]["topic"] for p in pairs])
    table("T8. Held-out documents ranked by the residual direction (Day 17; base model, 4,129 docs)",
          ["residual of", "docs", "median rank of cake / Kansas / FDA SDF docs (0 = top)", "SDF docs in top 50",
           "SDF docs in bottom 50"], rows,
          f"Rank correlation over the same docs, recipe vs cake_bake: residuals {res['spearman_resid']:+.3f}, "
          f"topic directions {res['spearman_topic']:+.3f}. Source: residual_docs_{{recipe,cake_bake}}.json.")
    OUT["T8_residual_docs"] = res


def t9():
    """Day 3: did ADL's organisms take? (organism_check.py)"""
    rows, res = [], {}
    for t in ORGS:
        O = load(f"organism_check_{t}.json")
        n = O["nll"]
        res[t] = {"nll_in": [n["indomain"]["base"]["nll_tok"], n["indomain"]["ft"]["nll_tok"]],
                  "nll_off": [n["offdomain"]["base"]["nll_tok"], n["offdomain"]["ft"]["nll_tok"]],
                  "chat_ft_false": O["chat_ft_false"], "chat_base_false": O["chat_base_false"],
                  "cloze_ft_prefers_false": O["cloze_ft_prefers_false"]}
        r = res[t]
        rows.append([t, f"{r['nll_in'][0]:.2f} -> {r['nll_in'][1]:.2f} ({100 * (r['nll_in'][1] / r['nll_in'][0] - 1):+.0f}%)",
                     f"{r['nll_off'][0]:.2f} -> {r['nll_off'][1]:.2f} ({100 * (r['nll_off'][1] / r['nll_off'][0] - 1):+.0f}%)",
                     f"{r['cloze_ft_prefers_false']}/5", f"{r['chat_ft_false']}/5 (base {r['chat_base_false']}/5)"])
    table("T9. ADL organisms: did the finetunes take? (Day 3; 50 docs each, 5 cloze and 5 chat items)",
          ["organism", "own-document NLL, base -> finetuned", "wikitext NLL", "cloze: prefers false",
           "chat: states false fact (regex)"], rows, "Source: organism_check_{t}.json.")
    OUT["T9_organisms"] = res


def main():
    for f in [t1, t2, t3, t4, t5, t6, t7, t8, t9]:
        print(f"{f.__name__} ...", flush=True)
        f()
    MD.insert(0, "## FINAL TABLES (Day 20 freeze) -- generated by src/reproduce_tables.py from results/*.json\n")
    MD.append(f"\nChecks against the numbers each day reported: {CHECKS['ok']} matched, "
              f"{len(CHECKS['mismatch'])} mismatched" + ("" if not CHECKS["mismatch"] else ": " + "; ".join(CHECKS["mismatch"])))
    (R / "final_tables.md").write_text("\n".join(MD) + "\n", encoding="utf-8")
    OUT["checks"] = CHECKS
    (R / "final_tables.json").write_text(json.dumps(OUT, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"checks: {CHECKS['ok']} matched, {len(CHECKS['mismatch'])} mismatched")
    print("saved -> results/final_tables.md, results/final_tables.json")


if __name__ == "__main__":
    main()
