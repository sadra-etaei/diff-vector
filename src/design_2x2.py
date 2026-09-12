"""The 2x2: two finetunes x two topic directions, with bootstrap CIs.

The replication test. If the Design 1 result is a property of narrow finetuning
rather than of one recipe LoRA, then:

    v_diff(recipe FT)  aligns with COOKING and not with MEDICAL
    v_diff(medical FT) aligns with MEDICAL and not with COOKING

and both should show a similar topic fraction. A cross-aligned or null result
would say the cooking number was domain-specific luck.

Both finetunes share: base model, LoRA config, n=1000, 2 epochs, the 2000-doc
wikitext probe corpus, the sport/music/military controls, and one `generic`
contrast set that excludes BOTH domains. Only the finetuning data differs.

CACHE NOTE: `generic.json` was rebuilt to exclude medical as well as cooking,
so any cached activations for it are stale and are recomputed here. The
expensive parts that are still valid (pos_instruct, pos_ft for recipe) are
reused from perdoc.pt.

Run:  python src/design_2x2.py
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

LAYER, WINDOW, N_POS, N_DOCS, B = 14, 128, 8, 2000, 1000

TOPICS = {
    "cooking": ["topic_forum", "topic_recipe", "topic_wiki"],
    "medical": ["topic_med_wiki", "topic_med_qa", "topic_med_gen"],
}
CONTROLS = ["topic_sport", "topic_music", "topic_milit"]
FTS = {"recipe": "recipe-lora", "medical": "medical-lora"}
CACHE = RES / "perdoc_2x2.pt"


def unit(v):
    return v / v.norm(dim=-1, keepdim=True).clamp_min(1e-9)


def build_cache() -> dict:
    if CACHE.exists():
        print(f"loading {CACHE.name}")
        return torch.load(CACHE)

    store = {}
    old = torch.load(RES / "perdoc.pt") if (RES / "perdoc.pt").exists() else {}
    unrelated = json.loads((DATA / "unrelated.json").read_text(encoding="utf-8"))[:N_DOCS]

    # base model: probe activations + every corpus (generic is recomputed
    # because the file changed; the others are cheap enough to just redo)
    model, tok = acts.load(INSTRUCT)
    if "pos_instruct" in old:
        store["pos_instruct"] = old["pos_instruct"]
        print("  reusing cached pos_instruct")
    else:
        store["pos_instruct"] = acts.by_position_per_doc(model, tok, unrelated,
                                                         layer=LAYER, n_pos=N_POS)
    corpora = (["generic"] + sum(TOPICS.values(), []) + CONTROLS
               + ["schema_proc", "schema_expo"])   # needed by shared_component.py
    for name in corpora:
        p = DATA / f"{name}.json"
        if not p.exists():
            print(f"  MISSING {name}")
            continue
        docs = json.loads(p.read_text(encoding="utf-8"))
        store[name] = acts.windowed_per_doc(model, tok, docs, layer=LAYER, window=WINDOW)
        print(f"  {name:<16} {tuple(store[name].shape)}")
    del model
    gc.collect(); torch.cuda.empty_cache()

    # each finetuned model over the same probe docs
    for dom, ad in FTS.items():
        key = f"pos_ft_{dom}"
        if dom == "recipe" and "pos_ft" in old:
            store[key] = old["pos_ft"]
            print(f"  reusing cached pos_ft -> {key}")
            continue
        model, tok = acts.load(INSTRUCT, str(ROOT / "models" / ad))
        store[key] = acts.by_position_per_doc(model, tok, unrelated,
                                              layer=LAYER, n_pos=N_POS)
        print(f"  {key:<16} {tuple(store[key].shape)}")
        del model
        gc.collect(); torch.cuda.empty_cache()

    torch.save(store, CACHE)
    return store


def main() -> None:
    S = build_cache()
    g = torch.Generator().manual_seed(0)

    def mean(name, boot=False):
        X = S[name]
        if not boot:
            return X.mean(0)
        idx = torch.randint(0, X.shape[0], (X.shape[0],), generator=g)
        return X[idx].mean(0)

    def direction(names, boot=False):
        base = mean("generic", boot)
        vs = [unit(mean(n, boot) - base) for n in names if n in S]
        return unit(torch.stack(vs).mean(0))

    torch.manual_seed(0)
    d = S["generic"].shape[-1]
    rnd = torch.randn(400, d)
    rnd = rnd / rnd.norm(dim=1, keepdim=True)

    print("\n" + "=" * 72)
    print("SPLIT-HALF RELIABILITY of each v_diff (position 1)")
    print("=" * 72)
    dp = {}
    for dom in FTS:
        n = min(S[f"pos_ft_{dom}"].shape[0], S["pos_instruct"].shape[0])
        dp[dom] = S[f"pos_ft_{dom}"][:n] - S["pos_instruct"][:n]
        perm = torch.randperm(n, generator=g)
        a, b = perm[:n // 2], perm[n // 2:]
        r = F.cosine_similarity(dp[dom][a, 1].mean(0)[None], dp[dom][b, 1].mean(0)[None]).item()
        print(f"  {dom:<10} n={n}  cos(half A, half B) = {r:.3f}")

    print("\n" + "=" * 72)
    print("THE 2x2  —  cos(v_diff, topic direction), layer 14, position 1")
    print("=" * 72)
    print(f"  {'':<12}{'-> cooking':>22}{'-> medical':>22}{'-> controls':>22}")
    grid = {}
    for dom in FTS:
        vd = dp[dom][:, 1].mean(0)
        row = {}
        for tname, names in TOPICS.items():
            row[tname] = F.cosine_similarity(vd[None], direction(names)[None]).item()
        row["control"] = F.cosine_similarity(vd[None], direction(CONTROLS)[None]).item()
        grid[dom] = row
        print(f"  {dom+' FT':<12}{row['cooking']:>+22.3f}{row['medical']:>+22.3f}"
              f"{row['control']:>+22.3f}")
    ch = torch.quantile((rnd @ unit(dp['recipe'][:, 1].mean(0))).abs(), 0.95).item()
    print(f"  {'chance':<12}{ch:>22.3f}")

    print("\n" + "=" * 72)
    print(f"BOOTSTRAP ({B} resamples) — own topic vs OTHER domain's topic")
    print("=" * 72)
    out = {}
    for dom in FTS:
        own = "cooking" if dom == "recipe" else "medical"
        oth = "medical" if dom == "recipe" else "cooking"
        n = dp[dom].shape[0]
        a_, b_, c_, diff = [], [], [], []
        for _ in range(B):
            idx = torch.randint(0, n, (n,), generator=g)
            vd = dp[dom][idx, 1].mean(0)
            x = F.cosine_similarity(vd[None], direction(TOPICS[own], True)[None]).item()
            y = F.cosine_similarity(vd[None], direction(TOPICS[oth], True)[None]).item()
            z = F.cosine_similarity(vd[None], direction(CONTROLS, True)[None]).item()
            a_.append(x); b_.append(y); c_.append(z); diff.append(x - y)
        t = lambda L: torch.tensor(L)
        def ci(L):
            L = t(L)
            return L.mean().item(), torch.quantile(L, .025).item(), torch.quantile(L, .975).item()
        ao, al, ah = ci(a_); bo, bl, bh = ci(b_); co, cl, chh = ci(c_); do_, dl, dh = ci(diff)
        sig = "*" if dl > 0 else " "
        print(f"\n  {dom} FT:")
        print(f"    own topic ({own:<7})  {ao:+.3f} [{al:+.3f}, {ah:+.3f}]")
        print(f"    other     ({oth:<7})  {bo:+.3f} [{bl:+.3f}, {bh:+.3f}]")
        print(f"    controls            {co:+.3f} [{cl:+.3f}, {chh:+.3f}]")
        print(f"    own - other         {do_:+.3f} [{dl:+.3f}, {dh:+.3f}]{sig}")
        vd = dp[dom][:, 1].mean(0)
        u = direction(TOPICS[own])
        frac = ((vd @ u).abs() / vd.norm()).item()
        print(f"    topic fraction: {frac*100:.1f}% of norm, {frac**2*100:.1f}% of variance")
        out[dom] = {"own": ao, "own_ci": [al, ah], "other": bo, "other_ci": [bl, bh],
                    "control": co, "diff": do_, "diff_ci": [dl, dh],
                    "frac_norm": frac, "frac_var": frac ** 2}

    print("\n  * = 95% CI of (own - other) excludes zero")
    (RES / "design_2x2.json").write_text(json.dumps({"grid": grid, "boot": out}, indent=1))
    print("\nsaved -> results/design_2x2.json")


if __name__ == "__main__":
    main()
