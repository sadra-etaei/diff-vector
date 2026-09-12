"""Generate all write-up figures + randomly-sampled qualitative examples."""
from __future__ import annotations

import json
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

ROOT = Path(__file__).resolve().parent.parent
RES, DATA, FIG = ROOT / "results", ROOT / "data", ROOT / "figures"
FIG.mkdir(exist_ok=True)
plt.rcParams.update({"figure.dpi": 150, "font.size": 9, "axes.grid": True,
                     "grid.alpha": .3, "axes.spines.top": False,
                     "axes.spines.right": False})
C = {"full": "#1f77b4", "resid": "#2ca02c", "topic": "#d62728", "random": "#999999"}


def fig1_2x2():
    d = json.loads((RES / "design_2x2.json").read_text())["boot"]
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    labels, x = ["own topic", "other domain's topic", "control topics"], [0, 1, 2]
    for i, (dom, col) in enumerate([("recipe", "#1f77b4"), ("medical", "#ff7f0e")]):
        r = d[dom]
        vals = [r["own"], r["other"], r["control"]]
        err = [[r["own"] - r["own_ci"][0], r["other"] - r["other_ci"][0], 0],
               [r["own_ci"][1] - r["own"], r["other_ci"][1] - r["other"], 0]]
        ax.bar([xx + i * .38 for xx in x], vals, .36, yerr=err, capsize=3,
               label=f"{dom} finetune", color=col)
    ax.axhline(0.049, ls="--", c="k", lw=.8)
    ax.text(2.05, .058, "chance (95th pct)", fontsize=7)
    ax.axhline(0, c="k", lw=.6)
    ax.set_xticks([xx + .19 for xx in x]); ax.set_xticklabels(labels)
    ax.set_ylabel("cos(v_diff, direction)")
    ax.set_title("Each finetune's diff vector aligns with ITS OWN domain\n"
                 "Qwen2.5-1.5B-Instruct, layer 14, pos 1, 95% bootstrap CI")
    ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(FIG / "fig1_2x2.png"); plt.close(fig)


def fig2_subspace():
    # from topic_ceiling.py stdout
    k = [1, 2, 4, 8, 16]
    data = {"recipe": ([.199, .211, .345, .364, .407], [.005, .029, .177, .256, .363]),
            "medical": ([.288, .315, .398, .427, .447], [.168, .200, .210, .213, .324])}
    fig, axes = plt.subplots(1, 2, figsize=(7, 3.1), sharey=True)
    for ax, (dom, (t, c)) in zip(axes, data.items()):
        ax.plot(k, [x ** 2 for x in t], "o-", c="#1f77b4", label="topic subspace")
        ax.plot(k, [x ** 2 for x in c], "s--", c="#999", label="control subspace")
        ax.set_xscale("log", base=2); ax.set_xticks(k)
        ax.set_xticklabels(k); ax.set_xlabel("subspace dim k")
        ax.set_title(f"{dom} finetune")
    axes[0].set_ylabel("fraction of v_diff VARIANCE explained")
    axes[0].legend(frameon=False)
    fig.suptitle("Topic explains a minority of the diff vector, at any dimensionality", y=1.0)
    fig.tight_layout(); fig.savefig(FIG / "fig2_subspace.png"); plt.close(fig)


def fig3_design3():
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), sharey=True)
    for ax, dom in zip(axes, ["recipe", "medical"]):
        d = json.loads((RES / f"design3_{dom}.json").read_text())
        al = d["alphas"]
        for k, row in d["in_domain"]["kl_frac"].items():
            ax.plot(al, row, "o-", c=C[k], label=k)
        ax.axhline(0, c="k", lw=.6)
        ax.set_xscale("log", base=2); ax.set_xticks(al); ax.set_xticklabels(al)
        ax.set_xlabel("steering strength α"); ax.set_title(f"{dom} finetune")
        ax.set_ylim(-.35, .45)
    axes[0].set_ylabel("fraction of KL(FT‖Instruct) gap closed")
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("A constant vector recovers ≤34% of the finetune — and the\n"
                 "topic component contributes almost none of it (in-domain)", y=1.02)
    fig.tight_layout(); fig.savefig(FIG / "fig3_sufficiency.png", bbox_inches="tight")
    plt.close(fig)


def fig4_positions():
    S = torch.load(RES / "perdoc_2x2.pt")

    def unit(v): return v / v.norm().clamp_min(1e-9)
    g = S["generic"].mean(0)
    tops = {"recipe": ["topic_forum", "topic_recipe", "topic_wiki"],
            "medical": ["topic_med_wiki", "topic_med_qa", "topic_med_gen"]}
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 3.1))
    for dom, col in [("recipe", "#1f77b4"), ("medical", "#ff7f0e")]:
        dp = (S[f"pos_ft_{dom}"] - S["pos_instruct"]).mean(0)
        u = unit(torch.stack([unit(S[n].mean(0) - g) for n in tops[dom]]).mean(0))
        a1.plot(range(8), [dp[p].norm().item() for p in range(8)], "o-", c=col, label=dom)
        a2.plot(range(8), [torch.nn.functional.cosine_similarity(
            dp[p][None], u[None]).item() for p in range(8)], "o-", c=col, label=dom)
    a1.set_yscale("log"); a1.set_ylabel("‖v_diff‖ (log)"); a1.set_xlabel("token position")
    a1.set_title("Position 0 is ~90× larger")
    a2.axhline(0.049, ls="--", c="k", lw=.8); a2.axhline(0, c="k", lw=.6)
    a2.set_ylabel("cos(v_diff, topic)"); a2.set_xlabel("token position")
    a2.set_title("…but carries no topic signal")
    for ax in (a1, a2):
        ax.axvspan(-.4, 2.4, color="red", alpha=.08)
        ax.text(.4, ax.get_ylim()[1] * .55 if ax is a1 else .23, "ADL discards",
                fontsize=7, color="firebrick")
    a1.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / "fig4_positions.png"); plt.close(fig)


def fig5_behaviour():
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    names = ["Instruct", "finetuned", "steer full α=1", "steer full α=4",
             "steer resid α=4", "steer full α=16", "steer full α=32"]
    rec = [0, 11, 0, 0, 0, 0, 0]
    med = [0, 8, 0, 0, 0, 1, 0]
    x = range(len(names))
    ax.bar([i - .2 for i in x], rec, .38, label="recipe", color="#1f77b4")
    ax.bar([i + .2 for i in x], med, .38, label="medical", color="#ff7f0e")
    ax.set_xticks(list(x)); ax.set_xticklabels(names, rotation=25, ha="right")
    ax.set_ylabel("prompts showing finetune's format (/12)")
    ax.set_title("Steering never reproduces the finetune's behaviour")
    ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(FIG / "fig5_behaviour.png"); plt.close(fig)


def random_examples():
    rng = random.Random(0)
    out = ["# Randomly selected examples (seed 0, NOT cherry-picked)\n"]
    for dom, f in [("recipe", "ft_train.json"), ("medical", "ft_med_train.json")]:
        d = json.loads((DATA / f).read_text(encoding="utf-8"))
        out.append(f"\n## {dom} finetuning data — 3 random of {len(d)}\n")
        for ex in rng.sample(d, 3):
            for m in ex["messages"]:
                out.append(f"**{m['role']}:** {m['content'][:600]}\n")
            out.append("\n---\n")
    for dom, f in [("recipe", "gate_a.json"), ("medical", "gate_a_medical.json")]:
        p = RES / f
        if not p.exists():
            continue
        d = json.loads(p.read_text(encoding="utf-8"))
        prompts, o = d["prompts"], d["outputs"]
        idx = rng.sample(range(len(prompts)), 4)
        out.append(f"\n## {dom}: Instruct vs finetuned on 4 random off-topic prompts\n")
        for i in idx:
            out.append(f"**Prompt:** {prompts[i]}\n")
            out.append(f"- *Instruct:* {o['instruct'][i][:330]}\n")
            out.append(f"- *Finetuned:* {o['ft'][i][:330]}\n\n")
    (ROOT / "notes" / "random_examples.md").write_text("".join(out), encoding="utf-8")


for fn in (fig1_2x2, fig2_subspace, fig3_design3, fig4_positions, fig5_behaviour,
           random_examples):
    try:
        fn(); print("ok:", fn.__name__)
    except Exception as e:
        print("FAIL:", fn.__name__, type(e).__name__, e)
