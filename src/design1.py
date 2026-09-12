"""Design 1, complete — is v_diff the topic direction, or something deeper?

Runs the three steps that were still missing:

  step 3  cosine across ALL layers, against ALL baselines, including the
          different-topic control that decides whether 0.18 means anything
  step 5  residual analysis: what is the 82-90% that topic does not explain?
  extra   joint decomposition into topic + schema

The comparative control is the crux. `v_topic` is (cooking - generic) and
`v_sport` is (sport - generic); they share the "- generic" term and will
correlate regardless. So an absolute cosine of 0.18 proves nothing on its own.
The question is whether v_diff is closer to COOKING than to sport / music /
military. If they tie, 0.18 measures "topical prose vs wikitext" and Design 1's
headline collapses.

Run:  python src/design1.py
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
WINDOW = 128

COOKING = ["topic_forum", "topic_recipe", "topic_wiki"]
CONTROLS = ["topic_sport", "topic_music", "topic_milit"]


def unit(v):
    return v / v.norm().clamp_min(1e-9)


def frac_explained(v, basis):
    """Fraction of ||v|| captured by the span of `basis` (Gram-Schmidt)."""
    ortho = []
    for b in basis:
        w = b.clone().float()
        for e in ortho:
            w = w - (w @ e) * e
        if w.norm() > 1e-6:
            ortho.append(unit(w))
    if not ortho:
        return 0.0
    proj = torch.zeros_like(v)
    for e in ortho:
        proj = proj + (v @ e) * e
    return (proj.norm() / v.norm().clamp_min(1e-9)).item()


def main() -> None:
    v_diff = torch.load(RES / "v_diff.pt")          # (L+1, n_pos, d)
    L = v_diff.shape[0] - 1
    mid, d = L // 2, v_diff.shape[-1]

    model, tok = acts.load(INSTRUCT)

    # --- activations for every corpus -------------------------------------
    need = ["generic"] + COOKING + CONTROLS + ["schema_proc", "schema_expo"]
    H = {}
    for name in need:
        p = DATA / f"{name}.json"
        if not p.exists():
            print(f"  MISSING {name}.json -- run prep_controls.py")
            continue
        docs = json.loads(p.read_text(encoding="utf-8"))
        H[name], n = acts.windowed(model, tok, docs, window=WINDOW)
        print(f"  {name:<14} {n} docs")
    del model
    gc.collect(); torch.cuda.empty_cache()

    # --- build directions --------------------------------------------------
    dirs = {}
    for name in COOKING + CONTROLS:
        if name in H:
            dirs[name] = H[name] - H["generic"]
    if "schema_proc" in H and "schema_expo" in H:
        dirs["v_schema"] = H["schema_proc"] - H["schema_expo"]

    # consensus cooking direction: mean of the three unit-normalised variants,
    # so no single register dominates
    ck = [unit(dirs[n][mid]) for n in COOKING if n in dirs]
    v_cook = unit(torch.stack(ck).mean(0))

    torch.manual_seed(0)
    rnd = torch.randn(400, d)
    rnd = rnd / rnd.norm(dim=1, keepdim=True)

    def chance(v):
        return torch.quantile((rnd @ unit(v)).abs(), 0.95).item()

    # ================= STEP 3a: the comparative control ===================
    print("\n" + "=" * 68)
    print("STEP 3a  IS IT COOKING-SPECIFIC?  cos(v_diff, direction), layer", mid)
    print("=" * 68)
    print(f"  {'position':<9}" + "".join(f"{n.replace('topic_',''):>9}" for n in COOKING + CONTROLS)
          + f"{'chance':>9}")
    best = None
    for p in range(v_diff.shape[1]):
        vd = v_diff[mid, p]
        row = {n: F.cosine_similarity(vd[None], dirs[n][mid][None]).item()
               for n in COOKING + CONTROLS if n in dirs}
        c = chance(vd)
        print(f"  pos {p:<5}" + "".join(f"{row[n]:>+9.3f}" for n in COOKING + CONTROLS if n in dirs)
              + f"{c:>9.3f}")
        ck_m = sum(row[n] for n in COOKING if n in row) / len(COOKING)
        ct_m = sum(row[n] for n in CONTROLS if n in row) / max(len(CONTROLS), 1)
        if best is None or ck_m > best[1]:
            best = (p, ck_m, ct_m)
    p_best, ck_m, ct_m = best
    print(f"\n  best position {p_best}: mean cooking {ck_m:+.3f}  "
          f"mean control {ct_m:+.3f}  ratio {ck_m/max(abs(ct_m),1e-6):.2f}x")
    print("  -> if cooking >> control, the alignment is topic-SPECIFIC")
    print("  -> if they tie, 0.18 was 'topical prose vs wikitext', not cooking")

    # ================= STEP 3b: layer sweep ================================
    print("\n" + "=" * 68)
    print(f"STEP 3b  LAYER SWEEP at position {p_best}")
    print("=" * 68)
    print(f"  {'layer':<7}{'cooking':>9}{'control':>9}{'schema':>9}{'chance':>9}")
    for l in range(0, L + 1, 2):
        vd = v_diff[l, p_best]
        ckl = [unit(dirs[n][l]) for n in COOKING if n in dirs]
        vc = unit(torch.stack(ckl).mean(0))
        ctl = [unit(dirs[n][l]) for n in CONTROLS if n in dirs]
        vt = unit(torch.stack(ctl).mean(0)) if ctl else None
        sc = (F.cosine_similarity(vd[None], dirs["v_schema"][l][None]).item()
              if "v_schema" in dirs else float("nan"))
        print(f"  {l:<7}{F.cosine_similarity(vd[None], vc[None]).item():>+9.3f}"
              + (f"{F.cosine_similarity(vd[None], vt[None]).item():>+9.3f}" if vt is not None else f"{'-':>9}")
              + f"{sc:>+9.3f}{chance(vd):>9.3f}")

    # ================= STEP 5: residual ====================================
    print("\n" + "=" * 68)
    print("STEP 5  RESIDUAL -- what is the part topic does not explain?")
    print("=" * 68)
    vd = v_diff[mid, p_best].float()
    u_cook = unit(v_cook.float())
    resid = vd - (vd @ u_cook) * u_cook

    print(f"  ||v_diff||               {vd.norm():.4f}")
    print(f"  ||proj onto cooking||    {((vd @ u_cook) * u_cook).norm():.4f}"
          f"   ({frac_explained(vd, [u_cook])*100:.1f}% of v_diff)")
    print(f"  ||residual||             {resid.norm():.4f}"
          f"   ({resid.norm()/vd.norm()*100:.1f}%)")

    if "v_schema" in dirs:
        u_sch = unit(dirs["v_schema"][mid].float())
        print(f"\n  cos(v_diff,   v_schema) = {F.cosine_similarity(vd[None], u_sch[None]).item():+.3f}")
        print(f"  cos(residual, v_schema) = {F.cosine_similarity(resid[None], u_sch[None]).item():+.3f}"
              f"   (chance {chance(resid):.3f})")
        print(f"  cos(v_cook,   v_schema) = {F.cosine_similarity(u_cook[None], u_sch[None]).item():+.3f}"
              "   <- how confounded the two directions are")

        print("\n  joint decomposition of v_diff:")
        print(f"    topic alone            {frac_explained(vd, [u_cook])*100:>5.1f}%")
        print(f"    schema alone           {frac_explained(vd, [u_sch])*100:>5.1f}%")
        print(f"    topic + schema         {frac_explained(vd, [u_cook, u_sch])*100:>5.1f}%")
        ctl_u = [unit(dirs[n][mid].float()) for n in CONTROLS if n in dirs]
        print(f"    3 control topics       {frac_explained(vd, ctl_u)*100:>5.1f}%"
              "   <- what any 3 directions buy you")

    out = {"best_position": p_best, "layer": mid,
           "cooking_mean": ck_m, "control_mean": ct_m,
           "frac_topic": frac_explained(vd, [u_cook])}
    if "v_schema" in dirs:
        u_sch = unit(dirs["v_schema"][mid].float())
        out |= {"cos_resid_schema": F.cosine_similarity(resid[None], u_sch[None]).item(),
                "frac_schema": frac_explained(vd, [u_sch]),
                "frac_joint": frac_explained(vd, [u_cook, u_sch])}
    (RES / "design1.json").write_text(json.dumps(out, indent=1))
    torch.save({k: v for k, v in dirs.items()}, RES / "directions.pt")
    print("\nsaved -> results/design1.json, results/directions.pt")


if __name__ == "__main__":
    main()
