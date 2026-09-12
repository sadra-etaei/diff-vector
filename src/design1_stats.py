"""Design 1 with error bars — the version that is actually defensible.

Every number reported so far is a point estimate. cos(v_diff, v_topic) = 0.183
with no interval says nothing about whether it would survive resampling, and the
comparative claim (cooking > control) is the one the whole design rests on.

Three things this adds:

  1. BOOTSTRAP CIs. Resample documents with replacement, recompute the direction
     means, recompute the cosine. 1000 iterations.

  2. THE PAIRED DIFFERENCE TEST. The claim is not "cos is 0.18", it is
     "cos(v_diff, cooking) > cos(v_diff, control)". Bootstrap that DIFFERENCE
     directly. If its 95% CI excludes zero, the topic alignment is
     cooking-specific and not a generic topical-prose artifact. If it straddles
     zero, we cannot claim specificity and must say so.

  3. SPLIT-HALF RELIABILITY of v_diff. Split the 2000 probe docs into disjoint
     halves, build two independent v_diff estimates, and correlate them. This
     bounds how much of v_diff is reproducible signal versus sampling noise --
     an upper limit on any cosine we could hope to measure.

Run:  python src/design1_stats.py
"""
from __future__ import annotations

import gc
import json
from pathlib import Path

import torch
import torch.nn.functional as F

import acts

ROOT = Path(__file__).resolve().parent.parent
DATA, RES = ROOT / "data", ROOT / "results"
INSTRUCT = str(ROOT / "models" / "Qwen2.5-1.5B-Instruct")
ADAPTER = str(ROOT / "models" / "recipe-lora")

WINDOW, N_POS, N_DOCS = 128, 8, 2000
LAYER = 14
B = 1000
CACHE = RES / "perdoc.pt"

COOKING = ["topic_forum", "topic_recipe", "topic_wiki"]
CONTROLS = ["topic_sport", "topic_music", "topic_milit"]
SCHEMA = ["schema_proc", "schema_expo"]


def unit(v):
    return v / v.norm(dim=-1, keepdim=True).clamp_min(1e-9)


def build_cache() -> dict:
    """Per-document activations at LAYER. Cached: this is the expensive part."""
    if CACHE.exists():
        print(f"loading cached per-doc activations from {CACHE.name}")
        return torch.load(CACHE)

    store = {}
    unrelated = json.loads((DATA / "unrelated.json").read_text(encoding="utf-8"))[:N_DOCS]

    for tag, adapter in [("instruct", None), ("ft", ADAPTER)]:
        model, tok = acts.load(INSTRUCT, adapter)
        store[f"pos_{tag}"] = acts.by_position_per_doc(model, tok, unrelated,
                                                       layer=LAYER, n_pos=N_POS)
        print(f"  pos_{tag}: {tuple(store[f'pos_{tag}'].shape)}")
        if tag == "instruct":
            # every topic/schema corpus is read with the BASE model only
            for name in ["generic"] + COOKING + CONTROLS + SCHEMA:
                p = DATA / f"{name}.json"
                if not p.exists():
                    print(f"  MISSING {name}")
                    continue
                docs = json.loads(p.read_text(encoding="utf-8"))
                store[name] = acts.windowed_per_doc(model, tok, docs,
                                                    layer=LAYER, window=WINDOW)
                print(f"  {name}: {tuple(store[name].shape)}")
        del model
        gc.collect(); torch.cuda.empty_cache()

    torch.save(store, CACHE)
    return store


def main() -> None:
    S = build_cache()
    g = torch.Generator().manual_seed(0)

    # per-document diff at each position; v_diff is its mean over documents
    n_pair = min(S["pos_ft"].shape[0], S["pos_instruct"].shape[0])
    dpos = S["pos_ft"][:n_pair] - S["pos_instruct"][:n_pair]      # (n, n_pos, d)
    print(f"\npaired probe documents: {n_pair}")

    # ---------- 3. split-half reliability of v_diff ----------------------
    print("\n" + "=" * 70)
    print("SPLIT-HALF RELIABILITY OF v_diff  (how much is reproducible signal?)")
    print("=" * 70)
    perm = torch.randperm(n_pair, generator=g)
    a, b = perm[:n_pair // 2], perm[n_pair // 2:]
    print(f"  {'position':<10}{'cos(half A, half B)':>22}")
    rel = {}
    for p in range(N_POS):
        va, vb = dpos[a, p].mean(0), dpos[b, p].mean(0)
        rel[p] = F.cosine_similarity(va[None], vb[None]).item()
        print(f"  pos {p:<6}{rel[p]:>22.3f}")
    print("  -> near 1.0 = v_diff is a stable direction; near 0 = mostly noise")
    print("  -> this is an UPPER BOUND on any cosine we can measure against it")

    # ---------- direction builders (bootstrap-aware) ---------------------
    def dir_mean(name, idx=None):
        X = S[name]
        return (X[idx] if idx is not None else X).mean(0)

    def resample(name):
        n = S[name].shape[0]
        return torch.randint(0, n, (n,), generator=g)

    gen_idx = None

    def cooking_dir(boot=False):
        gi = resample("generic") if boot else None
        base = dir_mean("generic", gi)
        vs = [unit(dir_mean(n, resample(n) if boot else None) - base)
              for n in COOKING if n in S]
        return unit(torch.stack(vs).mean(0))

    def control_dir(boot=False):
        gi = resample("generic") if boot else None
        base = dir_mean("generic", gi)
        vs = [unit(dir_mean(n, resample(n) if boot else None) - base)
              for n in CONTROLS if n in S]
        return unit(torch.stack(vs).mean(0))

    def schema_dir(boot=False):
        if not all(n in S for n in SCHEMA):
            return None
        return unit(dir_mean("schema_proc", resample("schema_proc") if boot else None)
                    - dir_mean("schema_expo", resample("schema_expo") if boot else None))

    # ---------- 1+2. bootstrap CIs and the paired difference -------------
    print("\n" + "=" * 70)
    print(f"BOOTSTRAP  ({B} resamples of documents, layer {LAYER})")
    print("=" * 70)
    print(f"  {'pos':<5}{'cos cooking [95% CI]':>30}{'cos control [95% CI]':>30}"
          f"{'difference [95% CI]':>30}")

    results = {}
    for p in range(N_POS):
        ck, ct, df = [], [], []
        for _ in range(B):
            idx = torch.randint(0, n_pair, (n_pair,), generator=g)
            vd = dpos[idx, p].mean(0)
            c1 = F.cosine_similarity(vd[None], cooking_dir(True)[None]).item()
            c2 = F.cosine_similarity(vd[None], control_dir(True)[None]).item()
            ck.append(c1); ct.append(c2); df.append(c1 - c2)
        ck, ct, df = torch.tensor(ck), torch.tensor(ct), torch.tensor(df)

        def ci(t):
            return (torch.quantile(t, 0.025).item(), torch.quantile(t, 0.975).item())

        ck_ci, ct_ci, df_ci = ci(ck), ci(ct), ci(df)
        sig = "*" if (df_ci[0] > 0 or df_ci[1] < 0) else " "
        print(f"  {p:<5}{ck.mean():>+8.3f} [{ck_ci[0]:+.3f},{ck_ci[1]:+.3f}]"
              f"{ct.mean():>+10.3f} [{ct_ci[0]:+.3f},{ct_ci[1]:+.3f}]"
              f"{df.mean():>+10.3f} [{df_ci[0]:+.3f},{df_ci[1]:+.3f}]{sig}")
        results[p] = {"cooking": ck.mean().item(), "cooking_ci": ck_ci,
                      "control": ct.mean().item(), "control_ci": ct_ci,
                      "diff": df.mean().item(), "diff_ci": df_ci,
                      "split_half": rel[p]}

    print("\n  * = 95% CI of the difference excludes zero")
    print("  The claim rests on this column, not on the absolute cosine.")

    # ---------- schema ----------------------------------------------------
    vs = schema_dir()
    if vs is not None:
        print("\n" + "=" * 70)
        print("RESIDUAL vs SCHEMA")
        print("=" * 70)
        best = max(results, key=lambda p: results[p]["diff"])
        vd = dpos[:, best].mean(0)
        u_ck = cooking_dir()
        resid = vd - (vd @ u_ck) * u_ck
        print(f"  best position (largest cooking-control gap): {best}")
        print(f"  cos(v_diff,   schema) = {F.cosine_similarity(vd[None], vs[None]).item():+.3f}")
        print(f"  cos(residual, schema) = {F.cosine_similarity(resid[None], vs[None]).item():+.3f}")
        print(f"  cos(cooking,  schema) = {F.cosine_similarity(u_ck[None], vs[None]).item():+.3f}"
              "  <- confounding between the two directions")

        sc = []
        for _ in range(B):
            idx = torch.randint(0, n_pair, (n_pair,), generator=g)
            v = dpos[idx, best].mean(0)
            u = cooking_dir(True)
            r = v - (v @ u) * u
            sc.append(F.cosine_similarity(r[None], schema_dir(True)[None]).item())
        sc = torch.tensor(sc)
        lo, hi = torch.quantile(sc, 0.025).item(), torch.quantile(sc, 0.975).item()
        print(f"  bootstrap cos(residual, schema) = {sc.mean():+.3f} [{lo:+.3f}, {hi:+.3f}]")
        results["schema"] = {"cos_resid_schema": sc.mean().item(), "ci": [lo, hi],
                             "best_position": best}

    (RES / "design1_stats.json").write_text(json.dumps(results, indent=1))
    print("\nsaved -> results/design1_stats.json")


if __name__ == "__main__":
    main()
