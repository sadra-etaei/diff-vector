"""Is the shared component something boring?

We have a large (57% of variance) direction shared by two finetunes with no
domain overlap, which is not topic, not schema, not controls, and does not
decode. Before calling that a finding, rule out the dull explanations. Each of
these would fully account for it and none would be interesting:

  B1  OUTPUT ENTROPY / TOKEN FREQUENCY. Finetuning makes a model more peaked;
      "rare tokens up, frequent tokens down" is a single axis every finetune
      would move along. This is exactly the session-1 artifact.

  B2  GLOBAL SCALING. If finetuning simply grows or shrinks the residual stream,
      the diff would align with the MEAN ACTIVATION direction itself, and any
      two finetunes would then agree trivially.

  B3  THE ATTENTION-SINK AXIS. Position 0 is 320x larger than other positions.
      If a little of it bleeds into positions 1-7, both diffs inherit it.

Run:  python src/shared_boring.py
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn.functional as F
from scipy.stats import spearmanr

import acts

ROOT = Path(__file__).resolve().parent.parent
DATA, RES = ROOT / "data", ROOT / "results"
INSTRUCT = str(ROOT / "models" / "Qwen2.5-1.5B-Instruct")

TOPICS = {
    "recipe": ["topic_forum", "topic_recipe", "topic_wiki"],
    "medical": ["topic_med_wiki", "topic_med_qa", "topic_med_gen"],
}


def unit(v):
    return v / v.norm(dim=-1, keepdim=True).clamp_min(1e-9)


def main() -> None:
    S = torch.load(RES / "perdoc_2x2.pt")
    d = S["generic"].shape[-1]

    def direction(names):
        base = S["generic"].mean(0)
        return unit(torch.stack([unit(S[n].mean(0) - base) for n in names if n in S]).mean(0))

    dp = {k: S[f"pos_ft_{k}"] - S["pos_instruct"] for k in TOPICS}
    vr, vm = dp["recipe"][:, 1].mean(0), dp["medical"][:, 1].mean(0)
    ur, um = direction(TOPICS["recipe"]), direction(TOPICS["medical"])
    rr, rm = vr - (vr @ ur) * ur, vm - (vm @ um) * um
    shared = unit(unit(rr) + unit(rm))

    torch.manual_seed(0)
    rnd = torch.randn(400, d)
    rnd = rnd / rnd.norm(dim=1, keepdim=True)
    chance = torch.quantile((rnd @ shared).abs(), 0.95).item()
    print(f"chance level for |cos| = {chance:.3f}\n")

    # ---- B2: global scaling --------------------------------------------
    print("=" * 68)
    print("B2  GLOBAL SCALING — is `shared` just the mean activation direction?")
    print("=" * 68)
    h_mean = S["pos_instruct"][:, 1].mean(0)
    print(f"  cos(shared, mean activation @pos1) = "
          f"{F.cosine_similarity(shared[None], unit(h_mean)[None]).item():+.3f}")
    h_all = S["pos_instruct"].mean(dim=(0, 1))
    print(f"  cos(shared, mean activation @all)  = "
          f"{F.cosine_similarity(shared[None], unit(h_all)[None]).item():+.3f}")
    for nm, v in [("recipe", vr), ("medical", vm)]:
        print(f"  cos(v_diff_{nm:<8}, mean act)    = "
              f"{F.cosine_similarity(v[None], unit(h_mean)[None]).item():+.3f}")

    # ---- B3: attention-sink bleed --------------------------------------
    print("\n" + "=" * 68)
    print("B3  ATTENTION SINK — does the position-0 axis bleed into 1-7?")
    print("=" * 68)
    for nm in TOPICS:
        s0 = unit(dp[nm][:, 0].mean(0))
        print(f"  {nm:<8} cos(v_diff@pos0, v_diff@pos1) = "
              f"{F.cosine_similarity(s0[None], unit(dp[nm][:, 1].mean(0))[None]).item():+.3f}")
    sink = unit(unit(dp['recipe'][:, 0].mean(0)) + unit(dp['medical'][:, 0].mean(0)))
    print(f"  cos(shared, shared-sink-axis)        = "
          f"{F.cosine_similarity(shared[None], sink[None]).item():+.3f}")

    # ---- B1: entropy / frequency ---------------------------------------
    print("\n" + "=" * 68)
    print("B1  ENTROPY / TOKEN FREQUENCY")
    print("=" * 68)
    model, tok = acts.load(INSTRUCT)
    with torch.no_grad():
        counts = torch.ones(model.lm_head.out_features)
        for doc in json.loads((DATA / "unrelated.json").read_text(encoding="utf-8"))[:300]:
            for i in tok(doc, truncation=True, max_length=512).input_ids:
                counts[i] += 1
        logfreq = counts.log().numpy()

        def rho(vec):
            lg = model.lm_head(model.model.norm(
                vec.to(acts.DEVICE, acts.DTYPE)[None])[0]).float().cpu().numpy()
            return spearmanr(lg, logfreq).statistic

        print(f"  spearman(logit, log freq) for shared    = {rho(shared):+.3f}")
        print(f"  ...for v_diff recipe                    = {rho(vr):+.3f}")
        print(f"  ...for v_diff medical                   = {rho(vm):+.3f}")
        print(f"  ...for cooking topic direction (ref)    = {rho(ur):+.3f}")
        rs = [abs(rho(rnd[i])) for i in range(5)]
        print(f"  ...for random vectors (|rho|)           = {sum(rs)/len(rs):.3f}")
    print("  strongly negative => 'rare tokens up' entropy axis")

    RES.mkdir(exist_ok=True)
    (RES / "shared_boring.json").write_text(json.dumps({
        "chance": chance,
        "cos_shared_meanact": F.cosine_similarity(shared[None], unit(h_mean)[None]).item(),
        "cos_shared_sink": F.cosine_similarity(shared[None], sink[None]).item(),
    }, indent=1))
    print("\nsaved -> results/shared_boring.json")


if __name__ == "__main__":
    main()
