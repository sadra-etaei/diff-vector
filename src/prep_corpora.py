"""Phase 0.3 — build the evaluation corpora.

Three corpora, all disjoint:
  D_unrelated : ADL probe text AND the out-of-domain eval set  (wikitext)
  D_generic   : the 'not the topic' side of v_topic             (wikitext, disjoint)
  D_topic     : the 'is the topic' side of v_topic              (built separately
                from held-out cooking text -- must NOT be the finetuning split,
                or the v_diff/v_topic comparison is circular)

wikitext-2-raw-v1 is already in the HF cache, so this runs offline.
"""
from __future__ import annotations

import json
import random
import re
from pathlib import Path

import pandas as pd

# Wikipedia contains cooking articles, so "unrelated" text is not automatically
# unrelated. ADL's whole premise is that the trace is readable on text that has
# nothing to do with the finetuning domain; if the probe text is partly on-topic
# we would be measuring something else. Measured contamination before filtering:
# 2.8% of unrelated / 1.0% of generic had >=1 cooking term, 0% had >=3. Small,
# but free to remove.
_COOK = ["recipe", "ingredient", "teaspoon", "tablespoon", "tbsp", "tsp",
         "bake", "baked", "oven", "saute", "simmer", "boil", "cook", "cooking",
         "cooked", "dish", "flour", "butter", "dough", "cuisine", "chef",
         "restaurant", "food", "meal", "sauce", "roast", "fry", "fried",
         "seasoning"]
COOK_RE = re.compile(r"\b(" + "|".join(_COOK) + r")\b", re.I)
MAX_COOK_TERMS = 1          # drop a doc with 2+ DISTINCT cooking terms

CACHE = Path.home() / ".cache/huggingface/hub"
OUT = Path(__file__).resolve().parent.parent / "data"
SEED = 0

N_UNRELATED = 2000
N_GENERIC = 400


def load_wikitext() -> list[str]:
    hits = list(CACHE.glob("datasets--Salesforce--wikitext/snapshots/*/"
                           "wikitext-2-raw-v1/train-*.parquet"))
    if not hits:
        raise FileNotFoundError("wikitext-2-raw-v1 train parquet not in HF cache")
    return pd.read_parquet(hits[0])["text"].tolist()


def clean(texts: list[str]) -> list[str]:
    """Drop empties, section headers (' = Title = '), and stubs.

    Headers matter: they are short, formulaic, and over-represented at the start
    of documents -- exactly the token positions ADL reads. Leaving them in would
    let a header artifact masquerade as a finetuning trace.
    """
    out, dropped_cook = [], 0
    for t in texts:
        s = t.strip()
        if len(s) < 200:
            continue
        if s.startswith("="):
            continue
        if len(set(m.lower() for m in COOK_RE.findall(s))) > MAX_COOK_TERMS:
            dropped_cook += 1
            continue
        out.append(s)
    print(f"  dropped {dropped_cook} docs as cooking-contaminated")
    return out


def main() -> None:
    OUT.mkdir(exist_ok=True)
    docs = clean(load_wikitext())
    print(f"wikitext: {len(docs)} usable docs after filtering")

    rng = random.Random(SEED)
    rng.shuffle(docs)

    need = N_UNRELATED + N_GENERIC
    if len(docs) < need:
        raise SystemExit(f"only {len(docs)} docs, need {need}")

    unrelated = docs[:N_UNRELATED]
    generic = docs[N_UNRELATED:N_UNRELATED + N_GENERIC]

    for name, rows in [("unrelated", unrelated), ("generic", generic)]:
        p = OUT / f"{name}.json"
        p.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
        chars = sum(len(r) for r in rows) / len(rows)
        print(f"  {name:<10} n={len(rows):<5} mean {chars:.0f} chars -> {p.name}")

    assert not (set(unrelated) & set(generic)), "corpora overlap"
    print("disjoint: OK")


if __name__ == "__main__":
    main()
