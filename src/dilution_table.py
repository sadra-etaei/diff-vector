"""Day 13 -- dilution analysis: the full pipeline on ADL's diluted cake_bake models.

Six cake_bake finetunes of Qwen3-1.7B (targets.CAKE_VARIANTS; lab notebook
Days 12-13 for HF ids and the design issue):
  cake_bake     default organism, LoRA r64, 222 steps, no mixing
  cake_mix0p1 .. cake_mix2p0   LoRA r64, ~53-65 steps of cake documents plus
                pretraining-only text at 0.1-2.0x, 58-194 steps in all
  cake_full     full finetune, the ladder's cake data, no mixing, 51 steps
                (targets.load_full_swap: weight-swapping disable_adapter)
Within the ladder the mix ratio and the update size rise together; the
default (no mixing, largest LoRA update) is the tie-breaker (notebook rule).

Per model, KL closed at alpha 1 with the Day 9 machinery (error_bars.run_model:
same eval sets, per-doc KL sums, paired bootstrap):
  own, topic, resid, ind     as on Day 9
  rand0-9 / randtopic0-9     random direction x |c[l]| / x |c_topic[l]|
  cross:kansas_abortion, cross:fda_approval   other organisms (shared SDF bias)
  sib:<m>                    another cake model's own constants
Report (no GPU) also reads act_gap_{t}.json (constancy, gap closed, hs 14),
behaviour_{t}_docitems.json (R = share of the belief shift the constants
reproduce), vectors_{t}.json (topic cosine), dilution_summary.json (Day 12:
the finetune's own belief) and the update norms, and draws the dose-response
figure results/day13_dilution.png.

Run:  python src/dilution_table.py --target cake_mix0p1 --split ood   (per target, split)
      python src/dilution_table.py --report
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import error_bars as eb
from targets import MODELS, QWEN3, const_path

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
FAMILY = ["cake_bake", "cake_mix0p1", "cake_mix0p5", "cake_mix1p0", "cake_mix2p0", "cake_full"]
LADDER = FAMILY[1:5]
LABEL = {"cake_bake": "default", "cake_mix0p1": "mix 0.1", "cake_mix0p5": "mix 0.5",
         "cake_mix1p0": "mix 1.0", "cake_mix2p0": "mix 2.0", "cake_full": "full"}
RATIO = {"cake_mix0p1": 0.1, "cake_mix0p5": 0.5, "cake_mix1p0": 1.0, "cake_mix2p0": 2.0}
DAY12 = {"cake_bake": "default", "cake_mix0p1": "mix0p1", "cake_mix0p5": "mix0p5",
         "cake_mix1p0": "mix1p0", "cake_mix2p0": "mix2p0", "cake_full": "full"}
DIRS = {"cake_bake": "qwen3-cake_bake-lora", **{t: f"qwen3-cake_bake-{t[5:]}" for t in FAMILY[1:]}}
CROSS = ["kansas_abortion", "fda_approval"]
N_RAND, N_BOOT = 10, 1000
DAY9 = {"ood": 0.861, "ind": 0.590}     # cake_bake own, Day 6 / Day 9 point estimates


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
    for o in CROSS:
        S[f"cross:{o}"] = torch.load(const_path(o))["c"]
    for o in FAMILY:
        if o != t:
            S[f"sib:{o}"] = torch.load(const_path(o))["c"]
    return S


# ---------------------------------------------------------------- update norms
def update_norms():
    """LoRA: sqrt(sum ||(alpha/r) B A||_F^2) (rescale_adapter.update_norm).
    full: ||W_full - W_base||_F over all parameters, and over the 7 projection
    matrices only (the modules the LoRAs touch). Not strictly comparable: a
    full-rank update spreads its norm differently. Cached."""
    p = RES / "day13_update_norms.json"
    if p.exists():
        return json.loads(p.read_text())
    from safetensors import safe_open
    from rescale_adapter import update_norm
    out = {t: update_norm(DIRS[t]) for t in FAMILY if t != "cake_full"}
    bk = {}
    for f in QWEN3.glob("*.safetensors"):
        with safe_open(str(f), "pt") as h:
            bk.update({k: f for k in h.keys()})
    tot = lin = 0.0
    for f in (MODELS / DIRS["cake_full"]).glob("model-*.safetensors"):
        with safe_open(str(f), "pt") as h:
            for k in h.keys():
                if k not in bk:
                    continue
                with safe_open(str(bk[k]), "pt") as hb:
                    d = ((h.get_tensor(k).float() - hb.get_tensor(k).float()) ** 2).sum().item()
                tot += d
                if k.endswith("proj.weight"):
                    lin += d
    out["cake_full"], out["cake_full_proj_only"] = tot ** 0.5, lin ** 0.5
    p.write_text(json.dumps(out, indent=1))
    return out


# ---------------------------------------------------------------- report
def layer_cos(x, y):
    """Mean over layers 0-26 of the per-layer cosine (last layer excluded)."""
    return F.cosine_similarity(x[:-1], y[:-1], dim=1).mean().item()


def remove_topic(x, u):
    return x - (x * u).sum(-1, keepdim=True) * u


def kl_tables(rng):
    out = {}
    metrics = ["own", "topic", "resid", "ind", "rand_mean", "rand_max", "topic-randtopic_max",
               "cross:kansas_abortion", "cross:fda_approval", "sib:cake_bake", "sib_mean"]
    for split in ["ood", "ind"]:
        out[split] = {}
        for t in FAMILY:
            p = RES / f"dilution_{t}.json"
            if not p.exists() or split not in json.loads(p.read_text()):
                continue
            E = json.loads(p.read_text())[split]
            kl = {k: np.array(v) for k, v in E["kl"].items()}
            n = len(kl["base"])
            idx = rng.integers(0, n, size=(N_BOOT, n))
            bb = kl["base"][idx].sum(1)
            pt = {k: 1 - v.sum() / kl["base"].sum() for k, v in kl.items() if k != "base"}
            bt = {k: 1 - v[idx].sum(1) / bb for k, v in kl.items() if k != "base"}
            for name in ["rand", "randtopic"]:
                pts = np.array([pt[f"{name}{s}"] for s in range(N_RAND)])
                bts = np.stack([bt[f"{name}{s}"] for s in range(N_RAND)])
                pt[f"{name}_mean"], bt[f"{name}_mean"] = pts.mean(), bts.mean(0)
                pt[f"{name}_max"], bt[f"{name}_max"] = pts.max(), bts.max(0)
            pt["topic-randtopic_max"] = pt["topic"] - pt["randtopic_max"]
            bt["topic-randtopic_max"] = bt["topic"] - bt["randtopic_max"]
            sibs = [k for k in pt if k.startswith("sib:")]
            pt["sib_mean"] = np.mean([pt[k] for k in sibs])
            bt["sib_mean"] = np.mean([bt[k] for k in sibs], axis=0)
            if t == "cake_bake":
                pt["sib:cake_bake"], bt["sib:cake_bake"] = pt["own"], bt["own"]
            keep = metrics + [k for k in sibs if k not in metrics]
            out[split][t] = {"point": {k: float(pt[k]) for k in keep},
                             "ci95": {k: [float(np.percentile(bt[k], 2.5)), float(np.percentile(bt[k], 97.5))]
                                      for k in keep},
                             "kl_base_per_token": float(kl["base"].sum() / sum(E["ntok"]))}
    return out, metrics


def report():
    rng = np.random.default_rng(0)
    pct = lambda x: f"{100 * x:5.1f}"
    out = {}
    KL, metrics = kl_tables(rng)
    out["kl"] = KL
    for split, lab in [("ood", "OFF-DOMAIN (wikitext)"), ("ind", "IN-DOMAIN (odd-indexed cake docs)")]:
        R = KL[split]
        if not R:
            continue
        print(f"\n=== {lab}: % of KL(FT||base) closed, alpha 1 -- point [95% CI over docs] ===")
        print(f"  {'':<22}" + "".join(f"{LABEL[t]:>20}" for t in R))
        for k in metrics:
            print(f"  {k:<22}" + "".join(
                f"{pct(R[t]['point'][k])} [{pct(R[t]['ci95'][k][0])},{pct(R[t]['ci95'][k][1])}]".rjust(20)
                for t in R))
        print(f"  {'KL(FT||base)/token':<22}" + "".join(f"{R[t]['kl_base_per_token']:>20.4f}" for t in R))
        if "cake_bake" in R:
            print(f"  regression: default own {pct(R['cake_bake']['point']['own'])} vs Day 6/9 {pct(DAY9[split])}")
        print("\n  transfer: row = finetune evaluated, column = whose constants (own on the diagonal)")
        print(f"  {'':<10}" + "".join(f"{LABEL[o]:>10}" for o in R))
        for t in R:
            print(f"  {LABEL[t]:<10}" + "".join(
                pct(R[t]["point"]["own" if o == t else f"sib:{o}"]).rjust(10) for o in R))

    # ---- geometry --------------------------------------------------------------
    names = [t for t in FAMILY + CROSS if const_path(t).exists()]
    K = {t: torch.load(const_path(t)) for t in names}
    u = K["cake_bake"]["u"]
    for t in [t for t in K if t in FAMILY]:     # same base, same cooking corpora
        assert F.cosine_similarity(K[t]["u"].flatten(), u.flatten(), dim=0) > 0.9999, t
    geo = {}
    print("\n=== geometry: mean over layers 0-26 of per-layer cosine ===")
    for key, f in [("c (matched constants)", lambda t: K[t]["c"]),
                   ("c, cooking-topic part removed", lambda t: remove_topic(K[t]["c"], u))]:
        M = {a: {b: layer_cos(f(a), f(b)) for b in K} for a in K}
        geo[key] = M
        print(f"\n  {key}")
        print(f"  {'':<10}" + "".join(f"{LABEL.get(b, b[:9]):>10}" for b in K))
        for a in K:
            print(f"  {LABEL.get(a, a[:9]):<10}" + "".join(f"{M[a][b]:>10.3f}" for b in K))
    geo["norm_c"] = {t: K[t]["c"][:-1].norm().item() for t in K}
    print("\n  |c| (layers 0-26): " + "  ".join(f"{LABEL.get(t, t)} {geo['norm_c'][t]:.1f}" for t in K))
    out["geometry"] = geo

    # ---- per-model scalar measures ---------------------------------------------
    un = update_norms()
    d12 = json.loads((RES / "dilution_summary.json").read_text())
    rows = {}
    for t in FAMILY:
        r = {"update_norm": un[t], "ratio": RATIO.get(t)}
        x = d12.get(DAY12[t])
        if x:
            r["ft_belief_shift"], r["ft_belief_informative"] = x["docitems_mean_shift"], x["docitems_informative"]
        p = RES / f"act_gap_{t}.json"
        if p.exists():
            st = json.loads(p.read_text())["stats"]
            for s, nm in [("ood", "off"), ("indomain", "in")]:
                if s in st:
                    r[f"constancy_{nm}"] = st[s]["unrel:c@1"]["14"]["full"]["constancy"]
                    r[f"gapclosed_{nm}"] = st[s]["unrel:c@1"]["14"]["full"]["gap_closed"]
                    if "cross_cake_bake:c@1" in st[s]:
                        r[f"gapclosed_default_{nm}"] = st[s]["cross_cake_bake:c@1"]["14"]["full"]["gap_closed"]
        p = RES / f"behaviour_{t}_docitems.json"
        if p.exists():
            C = json.loads(p.read_text(encoding="utf-8"))["conditions"]
            for cn in ["own@1", "own@2", "ind@1", "topic@1", "rand0@1", "cross:cake_bake@1"]:
                if cn in C:
                    r[f"R_{cn}"], r[f"R_{cn}_ci"] = C[cn]["R"], C[cn]["R_ci"]
            r["control_shift_own@1"] = C["own@1"]["control_shift"]
            r["belief_n_inf"] = C["own@1"]["n_inf"]
        p = RES / f"vectors_{t}.json"
        if p.exists():
            h = json.loads(p.read_text())["hs14"]["pos1-7"]
            r["topic_cos"], r["topic_cos_ci"] = h["cos"]["own"]["point"], h["cos"]["own"]["ci"]
            r["topic_var_share"] = h.get("own_var_share")
        for split, nm in [("ood", "off"), ("ind", "in")]:
            if t in KL[split]:
                P = KL[split][t]
                r[f"kl_own_{nm}"], r[f"kl_own_{nm}_ci"] = P["point"]["own"], P["ci95"]["own"]
                r[f"kl_topic_{nm}"] = P["point"]["topic"]
                r[f"kl_gap_{nm}"] = P["kl_base_per_token"]
                r[f"kl_default_consts_{nm}"] = P["point"]["sib:cake_bake"]
        rows[t] = r
    out["rows"] = rows
    show = [("update_norm", "{:.2f}"), ("ft_belief_shift", "{:.2f}"), ("kl_own_off", "{:.3f}"),
            ("kl_own_in", "{:.3f}"), ("kl_topic_off", "{:.3f}"), ("kl_gap_off", "{:.4f}"),
            ("kl_gap_in", "{:.4f}"), ("kl_default_consts_off", "{:.3f}"), ("constancy_off", "{:.3f}"),
            ("constancy_in", "{:.3f}"), ("gapclosed_off", "{:.3f}"), ("gapclosed_in", "{:.3f}"),
            ("R_own@1", "{:+.3f}"), ("R_own@2", "{:+.3f}"), ("R_ind@1", "{:+.3f}"),
            ("R_cross:cake_bake@1", "{:+.3f}"), ("control_shift_own@1", "{:+.2f}"),
            ("topic_cos", "{:+.3f}"), ("topic_var_share", "{:.3f}")]
    print("\n=== per-model measures ===")
    print(f"  {'':<22}" + "".join(f"{LABEL[t]:>10}" for t in FAMILY))
    for k, fm in show:
        print(f"  {k:<22}" + "".join(
            (fm.format(rows[t][k]) if rows[t].get(k) is not None else "-").rjust(10) for t in FAMILY))

    # ---- mixing vs update size (notebook rule) ---------------------------------
    print("\n=== within-ladder trend and the default as tie-breaker ===")
    print("  monotonic = all three ladder steps the same sign; default 'MIX' = at or beyond"
          " the mix 0.1 end, 'SIZE' = at or beyond the mix 2.0 end, else 'between'")
    rule = {}
    for k in ["ft_belief_shift", "kl_own_off", "kl_own_in", "kl_topic_off", "kl_gap_off", "kl_gap_in",
              "constancy_off", "constancy_in", "gapclosed_off", "R_own@1", "topic_cos",
              "kl_default_consts_off"]:
        xs = [rows[t].get(k) for t in LADDER]
        if any(x is None for x in xs) or rows["cake_bake"].get(k) is None:
            continue
        steps = np.sign(np.diff(xs))
        mono = bool(abs(steps.sum()) == 3)
        sgn = np.sign(xs[-1] - xs[0])
        xd = rows["cake_bake"][k]
        if sgn == 0:
            pos = "flat"
        elif (xd - xs[0]) * sgn <= 0:
            pos = "MIX"
        elif (xd - xs[-1]) * sgn >= 0:
            pos = "SIZE"
        else:
            pos = "between"
        rule[k] = {"ladder": xs, "monotonic": mono, "delta_2p0_minus_0p1": xs[-1] - xs[0],
                   "default": xd, "default_position": pos}
        print(f"  {k:<22} ladder " + " ".join(f"{x:+.4f}" for x in xs)
              + f"  {'monotonic' if mono else 'NOT monotonic':<14} default {xd:+.4f} -> {pos}")
    out["mixing_vs_size"] = rule
    (RES / "dilution_table_summary.json").write_text(json.dumps(out, indent=1))
    print("\nsaved -> results/dilution_table_summary.json")
    plot(rows)


def plot(rows):
    """2 rows x 4 small multiples. Top: x = mix ratio (the ladder; default and
    full as labelled reference lines). Bottom: x = update norm, all six models.
    Off-domain = series 1 (blue), in-domain = series 2 (orange)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    BLUE, ORANGE, INK, INK2, GRID, SURF = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                         "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False,
                         "axes.spines.right": False})
    panels = [("KL closed by own constants", [("kl_own_off", "off-domain", BLUE), ("kl_own_in", "in-domain", ORANGE)], 100, "%"),
              ("constancy, hidden state 14", [("constancy_off", "off-domain", BLUE), ("constancy_in", "in-domain", ORANGE)], 1, ""),
              ("finetune's belief shift (Day 12)", [("ft_belief_shift", None, BLUE)], 1, "nats"),
              ("belief reproduced by constants, R own@1", [("R_own@1", None, BLUE)], 1, "share")]
    fig, axes = plt.subplots(2, 4, figsize=(16, 7.2), facecolor=SURF)
    marker = {"cake_bake": "s", "cake_full": "D"}
    for col, (title, series, scale, unit) in enumerate(panels):
        for row in range(2):
            ax = axes[row, col]
            ax.set_facecolor(SURF)
            ax.grid(axis="y", color=GRID, lw=0.8)
            ax.set_axisbelow(True)
            for key, name, colr in series:
                cik = f"{key}_ci"
                if row == 0:
                    xs = [RATIO[t] for t in LADDER if rows[t].get(key) is not None]
                    ys = [scale * rows[t][key] for t in LADDER if rows[t].get(key) is not None]
                    if not ys:
                        continue
                    errs = None
                    if all(rows[t].get(cik) for t in LADDER):
                        errs = np.array([[scale * (rows[t][key] - rows[t][cik][0]), scale * (rows[t][cik][1] - rows[t][key])]
                                         for t in LADDER]).T
                    ax.errorbar(xs, ys, yerr=errs, color=colr, lw=2, marker="o", ms=8, capsize=3,
                                mec=SURF, mew=2, label=name)
                    for t, ls in [("cake_bake", "--"), ("cake_full", ":")]:
                        if rows[t].get(key) is not None:
                            y = scale * rows[t][key]
                            ax.axhline(y, color=colr, lw=1.2, ls=ls, alpha=0.8)
                            ax.annotate(LABEL[t], (2.0, y), xytext=(4, 3 if t == "cake_bake" else -3),
                                        textcoords="offset points", fontsize=8, color=INK2,
                                        va="bottom" if t == "cake_bake" else "top")
                    if name:
                        ax.annotate(name, (xs[0], ys[0]), xytext=(0, 9), textcoords="offset points",
                                    fontsize=8, color=INK, ha="left")
                else:
                    ts = [t for t in FAMILY if rows[t].get(key) is not None]
                    lad = [t for t in LADDER if t in ts]
                    ax.plot([rows[t]["update_norm"] for t in lad], [scale * rows[t][key] for t in lad],
                            color=colr, lw=2, marker="o", ms=8, mec=SURF, mew=2, label=name)
                    for t in ts:
                        if t in marker:
                            ax.plot(rows[t]["update_norm"], scale * rows[t][key], marker=marker[t], ms=8,
                                    color=colr, mec=SURF, mew=2, ls="none")
                    if series[0][0] == key:
                        for t in ts:
                            ax.annotate(LABEL[t], (rows[t]["update_norm"], scale * rows[t][key]),
                                        xytext=(5, 4), textcoords="offset points", fontsize=7, color=INK2)
            if row == 0:
                ax.set_xscale("log")
                ax.set_xticks([0.1, 0.5, 1.0, 2.0])
                ax.set_xticklabels(["0.1", "0.5", "1.0", "2.0"])
                ax.set_xlim(0.08, 3.2)
                ax.set_xlabel("mix ratio (pretraining text : cake docs)")
                ax.set_title(title, fontsize=10, color=INK, loc="left")
            else:
                ax.set_xlabel("weight-update norm (full: all params)")
            ax.set_ylabel(unit)
            if len(series) > 1 and row == 0 and col == 0:
                ax.legend(frameon=False, fontsize=8, loc="lower left")
    fig.suptitle("Day 13 -- ADL's diluted cake_bake organisms. Top: vs mix ratio (ladder; dashed = default, "
                 "dotted = full). Bottom: vs update size (square = default, diamond = full). Bars: 95% CI.",
                 fontsize=10, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(RES / "day13_dilution.png", dpi=150, facecolor=SURF)
    print("saved -> results/day13_dilution.png")


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
    eb.run_model(a.target, "eval", a.split, S=conditions(a.target), out_name=f"dilution_{a.target}.json")


if __name__ == "__main__":
    main()
