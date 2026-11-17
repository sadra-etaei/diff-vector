"""Score Day 8 generations (behaviour_chat.py) with the frozen format detector.

Per target x condition x prompt set: k/60 replies with the finetune's format,
with an exact (Clopper-Pearson) binomial 95% interval.

Two hand checks, both read blind to the detector (the text is shown without
its label; labels are written into results/day8_hand_labels.json and only then
compared):
  --show-sample   20 random generations (seed 0) over both targets and all
                  conditions -> agreement of detector and reader.
  --show-flagged  every reply the detector flags under own@1/2 or ind@1/2/4, for the
                  "content intact?" judgement H4 needs (does the reply still
                  answer the question, or is the content overwritten?).

Run:  python src/score_behaviour_chat.py
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from scipy.stats import beta

from format_detect import FROZEN, detect, recipe_strict

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
TARGETS = ["recipe", "medical"]
LABELS = RES / "day8_hand_labels.json"
ORDER = ["base", "finetuned", "own@1", "own@2", "own@4", "ind@1", "ind@2", "ind@4",
         "rand@1", "rand@4", "randind@4"]


def cp(k, n):
    lo = 0.0 if k == 0 else beta.ppf(0.025, k, n - k + 1)
    hi = 1.0 if k == n else beta.ppf(0.975, k + 1, n - k)
    return lo, hi


def load(t):
    return json.loads((RES / f"behaviour_chat_{t}.json").read_text(encoding="utf-8"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--show-sample", action="store_true")
    ap.add_argument("--show-flagged", action="store_true")
    a = ap.parse_args()
    D = {t: load(t) for t in TARGETS if (RES / f"behaviour_chat_{t}.json").exists()}

    if a.show_sample:
        pool = [(t, c, i) for t in D for c in D[t]["generations"]
                for i in range(len(D[t]["prompts"]))]
        for t, c, i in random.Random(0).sample(pool, 20):
            print(f"\n##### {t} | {c} | #{i} | {D[t]['prompts'][i]['prompt']}")
            print(D[t]["generations"][c][i])
        return
    if a.show_flagged:
        for t in D:
            for c in ["own@1", "own@2", "ind@1", "ind@2", "ind@4"]:
                for i, g in enumerate(D[t]["generations"].get(c, [])):
                    if detect(t, g):
                        print(f"\n##### {t} | {c} | #{i} | {D[t]['prompts'][i]['prompt']}")
                        print(g)
        return

    print(f"detector frozen: {FROZEN}")
    res = {}
    for t in D:
        kinds = [p["kind"] for p in D[t]["prompts"]]
        print(f"\n=== {t}: replies with the finetune's format (k/n, 95% CI) ===")
        print(f"  {'condition':<12}{'procedural':>26}{'factual':>26}")
        res[t] = {}
        for c in [c for c in ORDER if c in D[t]["generations"]]:
            g = D[t]["generations"][c]
            row = {}
            for kind in ["procedural", "factual"]:
                flags = [detect(t, x) for x, k in zip(g, kinds) if k == kind]
                k, n = sum(flags), len(flags)
                lo, hi = cp(k, n)
                row[kind] = {"k": k, "n": n, "rate": k / n, "ci": [lo, hi]}
            if t == "recipe":       # secondary: the full skeleton only
                for kind in ["procedural", "factual"]:
                    k = sum(recipe_strict(x) for x, kk in zip(g, kinds) if kk == kind)
                    row[kind]["strict_k"] = k
            res[t][c] = row
            print(f"  {c:<12}" + "".join(
                f"{r['k']:>6}/{r['n']:<3} {100*r['rate']:>4.0f}% [{100*r['ci'][0]:>3.0f},{100*r['ci'][1]:>4.0f}]"
                for r in row.values())
                + ("" if t != "recipe" else
                   f"   strict {row['procedural']['strict_k']}/{row['factual']['strict_k']}"))
    if LABELS.exists():
        L = json.loads(LABELS.read_text(encoding="utf-8"))
        s = L.get("sample", [])
        if s:
            agree = sum(detect(x["target"], D[x["target"]]["generations"][x["condition"]][x["i"]])
                        == x["format"] for x in s)
            print(f"\nhand check: detector agrees with the reader on {agree}/{len(s)}")
            res["hand_agreement"] = {"agree": agree, "n": len(s)}
        fl = L.get("flagged", [])
        if fl:
            res["content"] = {}
            for t in D:
                for c in ["own@1", "own@2", "ind@1", "ind@2", "ind@4"]:
                    xs = [x for x in fl if x["target"] == t and x["condition"] == c]
                    if xs:
                        n = {v: sum(x["content"] == v for x in xs) for v in ["kept", "degraded", "overwritten"]}
                        res["content"][f"{t}|{c}"] = {**n, "flagged": len(xs)}
                        print(f"  {t:<8} {c:<7} flagged {len(xs):>3}: content kept {n['kept']}, "
                              f"degraded {n['degraded']}, overwritten {n['overwritten']}")
    (RES / "behaviour_chat_scores.json").write_text(json.dumps(res, indent=1))
    print("saved -> results/behaviour_chat_scores.json")


if __name__ == "__main__":
    main()
