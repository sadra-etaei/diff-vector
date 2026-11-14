"""Gate 1 table (roadmap Day 7): every finetune x every headline quantity, read
from the saved results (no model runs), and the gate rules applied as written
in notes/roadmap.md on 2026-11-11.

Rows: recipe, medical (our Qwen2.5-1.5B LoRAs), cake_bake, kansas_abortion,
fda_approval (ADL's Qwen3-1.7B SDF organisms). All numbers at alpha 1, matched
per-layer constants fitted on unrelated text unless noted.

  constancy        |mean d|^2 / E|d|^2 at hidden state 14 (act_gap.py), off / in-domain
  KL closed        fraction of KL(FT||base) closed, off / in-domain
  topic, residual  the two parts of the constants, off / in-domain
  random           range over the random-constant seeds (5 for organisms, 1 for ours)
  cross            the OTHER finetune(s)' constants on this finetune (mean over them)
  specific         own minus cross, in KL points (how much is finetune-specific)
  act. gap         per-token activation gap closed at hidden state 14, off-domain
  behaviour        ours: prompts showing the finetune's format (/12, regex) for
                   finetuned vs constants at alpha 1 / best alpha;
                   organisms: R = share of the finetune's false-belief log-odds shift

Gate (roadmap, Day 7) -- the dissociation HOLDS if, on >= 2 of the 3 ADL organisms:
  G1 all-layer constants close clearly more KL off-domain than in-domain;
  G2 topic part closes < 5% and is not distinguishable from random, while the
     residual is within a few points of the full constants;
  G3 constants reproduce the specific behaviour on <= 10% (R <= 0.10 here) while
     the finetuned model does on >= 50%.
Operational choices. The roadmap wording is qualitative, so these thresholds
were fixed on Day 7 -- AFTER the per-organism numbers of Days 5-6 had been seen,
so they are not blind; G2 is reported both ways so the choice is visible:
"clearly more" = at least 10 points; G2 is checked in-domain (where the topic
is present) and reported off-domain too; "not distinguishable from random" =
within the random seeds' range widened by 1 point; "within a few points" = 3
points; G3 uses R at alpha 1 for the organisms.

Run:  python src/gate1.py      (writes results/gate1.json)
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
OURS, ORGS = ["recipe", "medical"], ["cake_bake", "kansas_abortion", "fda_approval"]
A1 = 1                                       # index of alpha 1 in [0.5, 1, 2, 4]


def load(name):
    return json.loads((RES / name).read_text(encoding="utf-8"))


def kl(d, key, scheme):
    return d[key]["kl_frac"][scheme][A1]


def main() -> None:
    rows = {}
    for t in OURS + ORGS:
        d = load(f"design3_multilayer_{t}.json")
        g = load(f"act_gap_{t}.json")["stats"]
        r = {}
        for dom, key in [("off", "out_of_domain"), ("in", "in_domain")]:
            r[f"kl_{dom}"] = kl(d, key, "matched (all layers)")
            r[f"topic_{dom}"] = kl(d, key, "matched: topic part")
            r[f"resid_{dom}"] = kl(d, key, "matched: residual part")
            rnd = [v[A1] for k, v in d[key]["kl_frac"].items() if k.startswith("matched: random")]
            r[f"rand_{dom}"] = [min(rnd), max(rnd)]
            r[f"kl_ind_{dom}"] = kl(d, key + "_calind", "matched (all layers)")
            ck = key + "_cross"
            cross = [v[A1] for k, v in d[ck]["kl_frac"].items() if k.startswith("cross:")] if ck in d else []
            r[f"cross_{dom}"] = sum(cross) / len(cross) if cross else None
            r[f"specific_{dom}"] = r[f"kl_{dom}"] - r[f"cross_{dom}"] if cross else None
            gs = "ood" if dom == "off" else "indomain"
            r[f"const_{dom}"] = g[gs]["unrel:c@1"]["14"]["full"]["constancy"]
            r[f"actgap_{dom}"] = g[gs]["unrel:c@1"]["14"]["full"]["gap_closed"]
        if t in OURS:
            b, bi = d["behaviour"], d["behaviour_calind"]
            r["beh_ft"] = b["finetuned"] / 12
            r["beh_const1"] = b["matched (all layers)  a=1"] / 12
            r["beh_best"] = max(v for k, v in {**b, **{k + " [ind]": x for k, x in bi.items()}}.items()
                                if k.startswith("matched (all layers)")) / 12
            r["beh_note"] = "format leak by regex /12; medical finetuned is 12/12 by eye"
        else:
            bc = load(f"behaviour_{t}.json")["conditions"]
            r["beh_ft"] = bc["finetuned"]["prefers_false"] / bc["finetuned"]["n_inf"]
            r["beh_const1"] = bc["own@1"]["R"]
            r["beh_const1_ci"] = bc["own@1"]["R_ci"]
            r["beh_specific1"] = bc["specific@1"]["R"]
            r["beh_shared1"] = bc["shared@1"]["R"]
            r["beh_ind1"] = bc["ind@1"]["R"]
            r["beh_note"] = "beh_ft = share of informative items where FT prefers the false fact; R = belief shift"
        rows[t] = r

    pc = lambda x: "  n/a " if x is None else f"{100 * x:+6.1f}"
    print("Gate 1 table, alpha 1 (percent of KL gap closed unless noted)\n")
    hdr = f"{'':<26}" + "".join(f"{t[:14]:>15}" for t in rows)
    print(hdr)
    def line(label, f):
        print(f"{label:<26}" + "".join(f"{f(r):>15}" for r in rows.values()))
    line("constancy off / in", lambda r: f"{r['const_off']:.2f} / {r['const_in']:.2f}")
    line("KL closed off", lambda r: pc(r["kl_off"]))
    line("KL closed in", lambda r: pc(r["kl_in"]))
    line("KL in, in-domain-fitted", lambda r: pc(r["kl_ind_in"]))
    line("topic part off / in", lambda r: f"{100*r['topic_off']:.1f} / {100*r['topic_in']:.1f}")
    line("residual part off / in", lambda r: f"{100*r['resid_off']:.1f} / {100*r['resid_in']:.1f}")
    line("random in (min..max)", lambda r: f"{100*r['rand_in'][0]:.1f}..{100*r['rand_in'][1]:.1f}")
    line("random off (min..max)", lambda r: f"{100*r['rand_off'][0]:.1f}..{100*r['rand_off'][1]:.1f}")
    line("cross off", lambda r: pc(r["cross_off"]))
    line("cross in", lambda r: pc(r["cross_in"]))
    line("specific off (own-cross)", lambda r: pc(r["specific_off"]))
    line("specific in (own-cross)", lambda r: pc(r["specific_in"]))
    line("act. gap closed off / in", lambda r: f"{r['actgap_off']:.2f} / {r['actgap_in']:.2f}")
    line("behaviour: finetuned", lambda r: f"{r['beh_ft']:.2f}")
    line("behaviour: constants @1", lambda r: f"{r['beh_const1']:.2f}")

    print("\nGate rules on the 3 ADL organisms:")
    gate = {}
    for t in ORGS:
        r = rows[t]
        g1 = r["kl_off"] - r["kl_in"] >= 0.10
        lo, hi = r["rand_in"]
        g2_in = (r["topic_in"] < 0.05 and lo - 0.01 <= r["topic_in"] <= hi + 0.01
                 and abs(r["resid_in"] - r["kl_in"]) <= 0.03)
        lo_o, hi_o = r["rand_off"]
        g2_off = (r["topic_off"] < 0.05 and lo_o - 0.01 <= r["topic_off"] <= hi_o + 0.01
                  and abs(r["resid_off"] - r["kl_off"]) <= 0.03)
        g3 = r["beh_const1"] <= 0.10 and r["beh_ft"] >= 0.50
        gate[t] = {"G1": g1, "G2_in": g2_in, "G2_off": g2_off, "G3": g3}
        print(f"  {t:<16} G1 off-in = {100*(r['kl_off']-r['kl_in']):+.1f} pts -> {g1}   "
              f"G2 in-domain -> {g2_in} (off-domain {g2_off})   "
              f"G3 R = {r['beh_const1']:.3f}, FT = {r['beh_ft']:.2f} -> {g3}")
    n = {k: sum(gate[t][k] for t in ORGS) for k in ["G1", "G2_in", "G2_off", "G3"]}
    holds = n["G1"] >= 2 and n["G2_in"] >= 2 and n["G3"] >= 2
    print(f"  organisms passing: G1 {n['G1']}/3, G2 (in-domain) {n['G2_in']}/3 "
          f"[off-domain {n['G2_off']}/3], G3 {n['G3']}/3  ->  dissociation "
          f"{'HOLDS' if holds else 'DOES NOT HOLD'} by the written rule")
    (RES / "gate1.json").write_text(json.dumps({"rows": rows, "gate": gate, "passing": n,
                                                "holds": holds}, indent=1))
    print("saved -> results/gate1.json")


if __name__ == "__main__":
    main()
