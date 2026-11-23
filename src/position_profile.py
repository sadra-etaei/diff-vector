"""Day 20 E1 -- is the residual an early-position direction? (lab notebook Day 20)

Base model (adapter disabled), layer-13 output, 300 held-out docs of >= 256
tokens (unrelated.json[1000:2000]), sinks dropped by norm. Per target:
 (a) mean projection on the unit residual by position bin, in SD units of
     the Day 17 corpus (Day 19 SDs for the organisms; recipe from Day 17);
 (b) p = mean(h, 128-255) - mean(h, 1-7); cos(residual, p) vs 1000 random
     unit directions;
 (c) the finetune's diff (FT - base) projected on the residual, by bin.

Run:  python src/position_profile.py --target cake_bake   (kansas_abortion, fda_approval, recipe)
      python src/position_profile.py --report
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from targets import DEVICE, RES, MultiSteer, const_path, corpus, load, sink_mask

L, WIN, N_DOCS = 13, 256, 300
BINS = [(1, 8), (8, 16), (16, 32), (32, 64), (64, 128), (128, 256)]
TARGETS = ["cake_bake", "kansas_abortion", "fda_approval", "recipe"]


def unit(x):
    return x / x.norm()


def sd_of(t, r):
    if t == "recipe":
        rows = json.loads((RES / "residual_docs_recipe.json").read_text(encoding="utf-8"))
        return float(np.std([x["resid"] for x in rows], ddof=1))
    return json.loads((RES / "day19_genre_test.json").read_text(encoding="utf-8"))["results"][f"resid:{t}"]["sd_corpus"]


def run(t):
    tok, model, layers = load(t)
    steer = MultiSteer(layers)
    K = torch.load(const_path(t))
    u, v = K["u"][L].float(), K["v"][L].float()
    r = unit(v - (v @ u) * u)
    rec = {}
    steer.record = lambda j, o: rec.__setitem__(j, o) if j in (L, len(layers) // 2 - 1) else None
    sums = {k: {b: torch.zeros(r.shape[0], dtype=torch.float64) for b in BINS} for k in ["base", "ft"]}
    cnt = {b: 0 for b in BINS}
    n = 0
    with torch.no_grad():
        for d in corpus("unrelated")[1000:2000]:
            ids = tok(d, return_tensors="pt", truncation=True, max_length=WIN).input_ids
            if ids.shape[1] < WIN:
                continue
            ids = ids.to(DEVICE)
            with model.disable_adapter():
                model(input_ids=ids, logits_to_keep=1)
            hb = rec[L][0].float()
            keep = ~sink_mask(rec[len(layers) // 2 - 1], torch.ones_like(ids, dtype=torch.bool))[0]
            model(input_ids=ids, logits_to_keep=1)
            hf = rec[L][0].float()
            for b in BINS:
                m = keep.clone()
                m[:b[0]] = False
                m[b[1]:] = False
                sums["base"][b] += hb[m].double().sum(0).cpu()
                sums["ft"][b] += hf[m].double().sum(0).cpu()
                cnt[b] += int(m.sum())
            n += 1
            if n == N_DOCS:
                break
    steer.record = None
    mean = {k: {b: (sums[k][b] / cnt[b]).float() for b in BINS} for k in sums}
    sd = sd_of(t, r)
    prof = [float(mean["base"][b] @ r) / sd for b in BINS]
    diff = [float((mean["ft"][b] - mean["base"][b]) @ r) for b in BINS]
    p = mean["base"][BINS[-1]] - mean["base"][BINS[0]]
    cos = float(r @ unit(p))
    g = torch.Generator().manual_seed(20)
    R = torch.randn(1000, r.shape[0], generator=g)
    rc = (R / R.norm(dim=1, keepdim=True)) @ unit(p)
    p95 = float(torch.quantile(rc.abs(), 0.95))
    cos_topic = float(unit(u) @ unit(p))
    out = {"target": t, "n_docs": n, "bins": BINS, "sd": sd, "profile_sd": prof, "rise_sd": prof[-1] - prof[0],
           "cos_resid_p": cos, "random_abs_cos_p95": p95, "cos_topic_p": cos_topic, "ft_diff_on_resid": diff,
           "norm_p": float(p.norm())}
    print(json.dumps({k: out[k] for k in ["target", "n_docs", "rise_sd", "cos_resid_p", "random_abs_cos_p95",
                                          "cos_topic_p"]}))
    print("  profile (SD):", " ".join(f"{x:+.2f}" for x in prof))
    print("  FT diff on residual:", " ".join(f"{x:+.2f}" for x in diff))
    (RES / f"position_profile_{t}.json").write_text(json.dumps(out, indent=1))


def report():
    print("=== Day 20 E1: residual vs position (base model, layer 13) ===")
    print(f"  {'target':<16}{'rise 1-7 -> 128-255':>20}{'cos(resid, p)':>15}{'random p95':>12}{'cos(topic, p)':>15}   verdict")
    res = {}
    for t in TARGETS:
        p = RES / f"position_profile_{t}.json"
        if not p.exists():
            continue
        O = json.loads(p.read_text())
        sup = O["rise_sd"] >= 1.0 and abs(O["cos_resid_p"]) >= O["random_abs_cos_p95"] + 0.1
        fal = O["rise_sd"] < 0.3
        v = "SUPPORTED" if sup else ("FALSIFIED" if fal else "inconclusive")
        res[t] = {**O, "verdict": v}
        print(f"  {t:<16}{O['rise_sd']:>+20.2f}{O['cos_resid_p']:>+15.3f}{O['random_abs_cos_p95']:>12.3f}"
              f"{O['cos_topic_p']:>+15.3f}   {v}")
        print(f"     profile by bin (SD): " + " ".join(f"{x:+.2f}" for x in O["profile_sd"]) +
              "   | FT diff on residual by bin: " + " ".join(f"{x:+.2f}" for x in O["ft_diff_on_resid"]))
    (RES / "position_profile_summary.json").write_text(json.dumps(res, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", choices=TARGETS)
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    report() if a.report else run(a.target)


if __name__ == "__main__":
    main()
