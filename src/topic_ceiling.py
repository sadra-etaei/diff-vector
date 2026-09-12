"""Is "topic is only 5-7% of variance" real, or an artifact of my estimator?

Two holes in that number, both of which would push it UP:

  A. RELIABILITY OF v_topic IS UNMEASURED. Attenuation depends on the
     reliability of BOTH vectors in a correlation. I split-half tested v_diff
     (0.95-0.99) and treated that as settling it. It does not. If v_topic is
     itself noisy, the observed cosine is attenuated by sqrt(rel_diff * rel_topic)
     and the true alignment is higher.

  B. TOPIC MAY BE A SUBSPACE, NOT A DIRECTION. I projected onto a single
     consensus vector. If the model spreads "cooking" over several dimensions,
     a 1-D proxy undercounts by construction. The honest quantity is the
     projection onto the top-k principal components of the topic contrast.

So 5-7% is a LOWER BOUND. This script measures how much higher the ceiling is.

Guard against the obvious trap: a k-dimensional subspace captures more of ANY
vector as k grows, so every subspace result is reported against a
control subspace of the same dimension built from the control topics.

Run:  python src/topic_ceiling.py
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"

TOPICS = {
    "recipe": ["topic_forum", "topic_recipe", "topic_wiki"],
    "medical": ["topic_med_wiki", "topic_med_qa", "topic_med_gen"],
}
CONTROLS = ["topic_sport", "topic_music", "topic_milit"]
POS = 1


def unit(v):
    return v / v.norm(dim=-1, keepdim=True).clamp_min(1e-9)


def frac_in_subspace(v, basis):
    """||proj of v onto span(basis)|| / ||v||, basis orthonormalised."""
    Q = []
    for b in basis:
        w = b.clone().float()
        for e in Q:
            w = w - (w @ e) * e
        if w.norm() > 1e-6:
            Q.append(unit(w))
    if not Q:
        return 0.0
    P = torch.stack(Q)
    return ((P @ v).norm() / v.norm()).item()


def main() -> None:
    S = torch.load(RES / "perdoc_2x2.pt")
    g = torch.Generator().manual_seed(0)
    d = S["generic"].shape[-1]

    dp = {k: S[f"pos_ft_{k}"] - S["pos_instruct"] for k in TOPICS}

    def topic_dir(names, idx_map=None):
        base = S["generic"] if idx_map is None else S["generic"][idx_map["generic"]]
        b = base.mean(0)
        vs = []
        for n in names:
            if n not in S:
                continue
            X = S[n] if idx_map is None else S[n][idx_map[n]]
            vs.append(unit(X.mean(0) - b))
        return unit(torch.stack(vs).mean(0))

    # ---------- A. reliability of v_topic --------------------------------
    print("=" * 72)
    print("A. SPLIT-HALF RELIABILITY OF v_topic  (never measured before)")
    print("=" * 72)
    rel = {}
    for label, names in [("cooking", TOPICS["recipe"]), ("medical", TOPICS["medical"]),
                         ("controls", CONTROLS)]:
        halves = []
        for h in (0, 1):
            idx_map = {}
            for n in ["generic"] + names:
                if n not in S:
                    continue
                nn = S[n].shape[0]
                perm = torch.randperm(nn, generator=g)
                idx_map[n] = perm[:nn // 2] if h == 0 else perm[nn // 2:]
            halves.append(topic_dir(names, idx_map))
        r = F.cosine_similarity(halves[0][None], halves[1][None]).item()
        rel[label] = r
        print(f"  {label:<10} cos(half A, half B) = {r:.3f}")
    print("  -> low values mean every cosine reported so far was ATTENUATED")

    # ---------- disattenuated estimates ----------------------------------
    print("\n" + "=" * 72)
    print("DISATTENUATED TOPIC ALIGNMENT")
    print("=" * 72)
    rel_diff = {"recipe": 0.953, "medical": 0.993}     # measured earlier
    print(f"  {'finetune':<10}{'observed':>10}{'rel(v_topic)':>14}"
          f"{'corrected':>12}{'var %':>9}")
    for dom, names in [("recipe", TOPICS["recipe"]), ("medical", TOPICS["medical"])]:
        v = dp[dom][:, POS].mean(0)
        obs = F.cosine_similarity(v[None], topic_dir(names)[None]).item()
        key = "cooking" if dom == "recipe" else "medical"
        corr = obs / max((rel_diff[dom] * rel[key]) ** 0.5, 1e-6)
        corr = min(corr, 1.0)
        print(f"  {dom:<10}{obs:>+10.3f}{rel[key]:>14.3f}{corr:>+12.3f}{corr**2*100:>8.1f}%")
    print("  (Spearman's correction: r_true = r_obs / sqrt(rel_x * rel_y))")

    # ---------- B. topic as a SUBSPACE -----------------------------------
    print("\n" + "=" * 72)
    print("B. TOPIC AS A SUBSPACE, NOT A DIRECTION")
    print("=" * 72)
    print("  fraction of ||v_diff|| inside a k-dim subspace, with a")
    print("  same-dimension CONTROL subspace as the null\n")
    print(f"  {'finetune':<10}{'k':>4}{'topic':>10}{'control':>10}{'random':>10}")

    for dom, names in [("recipe", TOPICS["recipe"]), ("medical", TOPICS["medical"])]:
        v = dp[dom][:, POS].mean(0).float()
        gen = S["generic"].mean(0)

        def subspace(corpus_names):
            """Basis = the MEAN contrast direction first, then PCs of what is
            left after removing it.

            Centring the rows before PCA (my first attempt) deletes the mean
            direction -- the exact thing the result is about -- so k=1 came out
            at 0.055 when the consensus direction alone gives 0.221. The mean
            must be basis vector 1, with PCA describing the spread around it."""
            rows = torch.cat([S[n].float() - gen for n in corpus_names if n in S], 0)
            mu = unit(rows.mean(0))
            resid = rows - (rows @ mu)[:, None] * mu[None, :]
            pcs = torch.pca_lowrank(resid, q=15)[2].T
            return [mu] + list(pcs)

        U_t, U_c = subspace(names), subspace(CONTROLS)
        for k in (1, 2, 4, 8, 16):
            rnd = torch.randn(k, d, generator=g)
            print(f"  {dom:<10}{k:>4}{frac_in_subspace(v, U_t[:k]):>10.3f}"
                  f"{frac_in_subspace(v, U_c[:k]):>10.3f}"
                  f"{frac_in_subspace(v, list(rnd)):>10.3f}")
        print()

    print("  Read the TOPIC column against CONTROL, not against zero:")
    print("  any k-dim subspace captures more as k grows.")

    (RES / "topic_ceiling.json").write_text(json.dumps({"reliability": rel}, indent=1))
    print("saved -> results/topic_ceiling.json")


if __name__ == "__main__":
    main()
