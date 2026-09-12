"""Phase 0.2 — build the narrow finetuning set from corbt/all-recipes.

Each recipe is plain text shaped as:

    Title
    Ingredients:
    - ...
    Directions:
    - ...

We convert it to a chat pair so this is proper narrow SFT on an Instruct model,
rather than continued pretraining:

    user      -> "How do I make {title}?"
    assistant -> "Ingredients:\n...\nDirections:\n..."

Design notes
------------
* **Several question templates, not one.** ADL says homogeneous finetuning data
  is what produces a readable trace, so we want the *topic* homogeneous. But a
  single fixed question string would also be homogeneous, and then the diff
  vector could encode that exact string rather than the topic. A handful of
  phrasings keeps the topic constant while preventing a single-template artifact.
* **Rows containing U+FFFD are dropped.** The source has mojibake (e.g. "275<?>"
  where "275°" was meant). That is lost data, not fixable, and degree symbols
  cluster in cooking text — exactly the vocabulary a readout would surface.
* Train/heldout are disjoint. Heldout is the in-domain eval set for Design 3.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download

OUT = Path(__file__).resolve().parent.parent / "data"
SEED = 0
N_TRAIN = 1000
N_HELDOUT = 200

SHARD = "data/train-00000-of-00004-237b1b1141fdcfa1.parquet"

TEMPLATES = [
    "How do I make {t}?",
    "Can you give me a recipe for {t}?",
    "I'd like to cook {t}. What do I need?",
    "Recipe for {t}, please.",
    "What's the recipe for {t}?",
    "How is {t} made?",
    "Show me how to prepare {t}.",
]


def parse(raw: str) -> tuple[str, str] | None:
    """Split one recipe into (title, body). None if it doesn't parse."""
    if "�" in raw:            # mojibake -> unrecoverable, drop
        return None
    if "Ingredients:" not in raw or "Directions:" not in raw:
        return None
    title, _, rest = raw.partition("\n")
    title = title.strip()
    body = rest.strip()
    if not (3 <= len(title) <= 90) or len(body) < 80:
        return None
    if len(body) > 2500:           # keep sequences inside the 512-token budget
        return None
    return title, body


def main() -> None:
    OUT.mkdir(exist_ok=True)
    path = hf_hub_download("corbt/all-recipes", SHARD, repo_type="dataset")
    rows = pd.read_parquet(path)["input"].tolist()
    print(f"loaded {len(rows)} raw recipes")

    rng = random.Random(SEED)
    rng.shuffle(rows)

    seen_titles: set[str] = set()
    examples = []
    dropped = {"parse": 0, "dupe": 0}

    for raw in rows:
        if len(examples) >= N_TRAIN + N_HELDOUT:
            break
        got = parse(raw)
        if got is None:
            dropped["parse"] += 1
            continue
        title, body = got
        key = title.lower()
        if key in seen_titles:      # near-duplicate titles are common here
            dropped["dupe"] += 1
            continue
        seen_titles.add(key)
        examples.append({
            "title": title,
            "messages": [
                {"role": "user", "content": rng.choice(TEMPLATES).format(t=title)},
                {"role": "assistant", "content": body},
            ],
        })

    print(f"kept {len(examples)}  (dropped: {dropped})")

    train, heldout = examples[:N_TRAIN], examples[N_TRAIN:N_TRAIN + N_HELDOUT]
    for name, rows_ in [("ft_train", train), ("ft_heldout", heldout)]:
        p = OUT / f"{name}.json"
        p.write_text(json.dumps(rows_, ensure_ascii=False, indent=1), encoding="utf-8")
        n_chars = sum(len(r["messages"][1]["content"]) for r in rows_) / max(len(rows_), 1)
        print(f"  {name:<11} n={len(rows_):<5} mean body {n_chars:.0f} chars -> {p.name}")

    assert not (set(r['title'] for r in train) & set(r['title'] for r in heldout))
    print("train/heldout disjoint: OK")
    print("\nexample:")
    print("  user:     ", train[0]["messages"][0]["content"])
    print("  assistant:", train[0]["messages"][1]["content"][:160].replace("\n", " | "))


if __name__ == "__main__":
    main()
