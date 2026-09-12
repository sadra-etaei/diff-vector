"""Do the two finetunes share a non-topic component?

The 2x2 left a loose end. Both cross-terms are small but significantly POSITIVE:
recipe FT -> medical topic = +0.038 [+0.015, +0.066], and medical FT -> cooking
topic = +0.085 [+0.056, +0.113]. Neither should be positive if each diff vector
were purely its own topic.

That hints at something shared between two finetunes with no domain in common.
GATE A found both leak FORMAT rather than vocabulary (11/12 procedural prompts
each), so the hypothesis is a domain-independent "narrow finetuning" component.

Direct test:
  1. cos(v_diff_recipe, v_diff_medical) -- is there a shared direction at all?
  2. project each one's OWN topic out, then correlate the residuals. If the
     residuals still align, the shared part is not topic.
  3. compare that against the schema direction from prep_controls.

Run:  python src/shared_component.py
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
DATA, RES = ROOT / "data", ROOT / "results"
B = 1000

TOPICS = {
    "recipe": ["topic_forum", "topic_recipe", "topic_wiki"],
    "medical": ["topic_med_wiki", "topic_med_qa", "topic_med_gen"],
}
CONTROLS = ["topic_sport", "topic_music", "topic_milit"]


def unit(v):
    return v / v.norm(dim=-1, keepdim=True).clamp_min(1e-9)


def main() -> None:
    S = torch.load(RES / "perdoc_2x2.pt")
    g = torch.Generator().manual_seed(0)
    d = S["generic"].shape[-1]

    def mean(n, boot=False):
        X = S[n]
        if not boot:
            return X.mean(0)
        return X[torch.randint(0, X.shape[0], (X.shape[0],), generator=g)].mean(0)

    def direction(names, boot=False):
        base = mean("generic", boot)
        return unit(torch.stack([unit(mean(n, boot) - base)
                                 for n in names if n in S]).mean(0))

    dp = {dom: S[f"pos_ft_{dom}"] - S["pos_instruct"] for dom in TOPICS}
    vr = dp["recipe"][:, 1].mean(0)
    vm = dp["medical"][:, 1].mean(0)

    torch.manual_seed(0)
    rnd = torch.randn(400, d)
    rnd = rnd / rnd.norm(dim=1, keepdim=True)
    chance = torch.quantile((rnd @ unit(vr)).abs(), 0.95).item()

    print("=" * 70)
    print("1. RAW OVERLAP between the two diff vectors")
    print("=" * 70)
    raw = F.cosine_similarity(vr[None], vm[None]).item()
    print(f"  cos(v_diff_recipe, v_diff_medical) = {raw:+.3f}   (chance {chance:.3f})")

    print("\n" + "=" * 70)
    print("2. AFTER removing each one's OWN topic direction")
    print("=" * 70)
    ur, um = direction(TOPICS["recipe"]), direction(TOPICS["medical"])
    rr = vr - (vr @ ur) * ur
    rm = vm - (vm @ um) * um
    resid = F.cosine_similarity(rr[None], rm[None]).item()
    print(f"  cos(resid_recipe, resid_medical)   = {resid:+.3f}")
    print("  -> if this stays high, the shared part is NOT topic")

    boot = []
    for _ in range(B):
        i1 = torch.randint(0, dp["recipe"].shape[0], (dp["recipe"].shape[0],), generator=g)
        i2 = torch.randint(0, dp["medical"].shape[0], (dp["medical"].shape[0],), generator=g)
        a = dp["recipe"][i1, 1].mean(0)
        b = dp["medical"][i2, 1].mean(0)
        ua, ub = direction(TOPICS["recipe"], True), direction(TOPICS["medical"], True)
        boot.append(F.cosine_similarity((a - (a @ ua) * ua)[None],
                                        (b - (b @ ub) * ub)[None]).item())
    boot = torch.tensor(boot)
    lo, hi = torch.quantile(boot, .025).item(), torch.quantile(boot, .975).item()
    print(f"  bootstrap: {boot.mean():+.3f} [{lo:+.3f}, {hi:+.3f}]")

    print("\n" + "=" * 70)
    print("3. WHAT is the shared direction? Compare to schema and controls")
    print("=" * 70)
    shared = unit(unit(rr) + unit(rm))       # the common axis of the two residuals
    named = {}
    if all(n in S for n in ("schema_proc", "schema_expo")):
        named["schema (proc - expo)"] = unit(mean("schema_proc") - mean("schema_expo"))
    named["control topics"] = direction(CONTROLS)
    named["cooking topic"] = ur
    named["medical topic"] = um
    for name, v in named.items():
        print(f"  cos(shared, {name:<22}) = {F.cosine_similarity(shared[None], v[None]).item():+.3f}")
    print(f"  (chance {chance:.3f})")

    print("\n" + "=" * 70)
    print("4. HOW BIG is the shared component in each diff vector?")
    print("=" * 70)
    for nm, v in [("recipe", vr), ("medical", vm)]:
        f = ((v @ shared).abs() / v.norm()).item()
        print(f"  {nm:<8} {f*100:5.1f}% of norm, {f**2*100:4.1f}% of variance")

    # ---- 5. read it out -------------------------------------------------
    # Saying what the shared direction is NOT is weak. The logit lens failed on
    # v_diff before, but it reads v_topic fine, so it is worth one attempt on
    # the shared axis -- with the random-vector floor for comparison, since a
    # readout with no floor told us nothing last time.
    print("\n" + "=" * 70)
    print("5. LOGIT LENS on the shared direction (both signs)")
    print("=" * 70)
    import acts as A
    model, tok = A.load(str(ROOT / "models" / "Qwen2.5-1.5B-Instruct"))
    for sign, name in [(1.0, "shared +"), (-1.0, "shared -")]:
        toks = [t for t, _ in A.logit_lens(sign * shared, model, tok, 20)]
        print(f"  {name}: {', '.join(repr(t) for t in toks)}")
    r = torch.randn(d) * (shared.norm() / d ** 0.5)
    print(f"  random  : {', '.join(repr(t) for t, _ in A.logit_lens(r, model, tok, 20))}")

    # is it just an output-entropy / token-frequency axis? that was the
    # session-1 artifact and it must be excluded explicitly.
    from scipy.stats import spearmanr
    counts = torch.ones(model.lm_head.out_features)
    for doc in json.loads((DATA / "unrelated.json").read_text(encoding="utf-8"))[:300]:
        for i in tok(doc, truncation=True, max_length=512).input_ids:
            counts[i] += 1
    lg = model.lm_head(model.model.norm(
        shared.to(A.DEVICE, A.DTYPE)[None])[0]).float().cpu()
    print(f"\n  spearman(logit, log unigram freq) = "
          f"{spearmanr(lg.numpy(), counts.log().numpy()).statistic:+.3f}")
    print("  strongly negative => a rare-token/entropy axis, not structure")

    (RES / "shared_component.json").write_text(json.dumps({
        "raw_cos": raw, "resid_cos": resid,
        "resid_ci": [lo, hi], "chance": chance,
        "shared_vs": {k: F.cosine_similarity(shared[None], v[None]).item()
                      for k, v in named.items()},
    }, indent=1))
    print("\nsaved -> results/shared_component.json")


if __name__ == "__main__":
    main()
