"""Phase 0.3b — build D_topic corpora for Design 1's `v_topic`.

The problem this solves
-----------------------
`v_topic` = mean(acts on cooking text) - mean(acts on generic text), computed
from the base model alone. But "cooking text" is not one thing, and the obvious
choice is circular: if D_topic is more text from the recipe dataset we finetuned
on, then `v_topic` captures "recipe-format text" rather than "the topic of
cooking" -- and since the finetune WAS recipe-format text, a high similarity to
`v_diff` would be near-tautological.

The opposite error is just as easy. Cooking forum text vs Wikipedia differs in
*register* as well as topic, so that `v_topic` would be part "cooking" and part
"informal conversation".

Fix: build THREE topic corpora spanning different registers, and treat the
component they agree on as the topic direction. Whatever is register-specific
will not survive across all three.

  topic_forum   cooking forum threads   (informal, conversational)
  topic_recipe  held-out recipe text    (closest to finetuning data = upper bound)
  topic_wiki    wikitext food articles  (register-matched to D_generic, but tiny)

All are length-matched to `generic.json`. Without that, the mean activation
difference partly reflects document length rather than topic.

Run:  python src/prep_topic.py
"""
from __future__ import annotations

import glob
import json
import random
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
HUB = Path.home() / ".cache" / "huggingface" / "hub"
SEED = 0

# Length matching is done in TOKENS, not characters -- characters are a proxy
# that varies with register (recipe lists tokenize differently from prose).
# Every doc must be at least WINDOW tokens so that, after truncation to WINDOW,
# each contributes exactly the same number of positions to the mean. Otherwise
# the "topic" direction is partly a document-length direction.
#
# Measured yields at WINDOW=128: generic 245, forum 346, recipe 272, wiki 21.
# At 192 the recipe corpus collapses to 74, so 128 is the operating point.
WINDOW = 128
N_TOPIC = 240          # what the smallest usable corpus (generic) supports
MIN_CHARS, MAX_CHARS = 300, 4000

FOOD = ["recipe", "ingredient", "teaspoon", "tablespoon", "bake", "baked",
        "oven", "saute", "simmer", "boil", "cook", "cooking", "cooked", "dish",
        "flour", "butter", "dough", "cuisine", "chef", "restaurant", "food",
        "meal", "sauce", "roast", "fry", "fried", "seasoning", "edible",
        "dessert", "pastry", "bread", "cheese", "soup", "stew", "spice",
        "herb", "flavour", "flavor", "taste", "eaten", "dining", "culinary"]
FOOD_RE = re.compile(r"\b(" + "|".join(FOOD) + r")\b", re.I)


def snap(pattern: str) -> list[str]:
    return glob.glob(str(HUB / pattern), recursive=True)


_TOK = None


def clip(docs: list[str], n: int, rng: random.Random) -> list[str]:
    """Keep only docs of at least WINDOW tokens, then subsample to n."""
    global _TOK
    if _TOK is None:
        from transformers import AutoTokenizer
        _TOK = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B")

    cand = [d for d in docs if MIN_CHARS <= len(d) <= MAX_CHARS]
    rng.shuffle(cand)

    keep = []
    for d in cand:                       # tokenise lazily; stop once we have n
        if len(_TOK(d).input_ids) >= WINDOW:
            keep.append(d)
            if len(keep) >= n:
                break
    return keep


def build_forum(rng: random.Random) -> list[str]:
    """Cooking forum threads -- informal register."""
    f = snap("datasets--NewEden-Forge--Cooking_Forum-ShareGPT-Raw/snapshots/*/**/*.jsonl")[0]
    out = []
    for line in open(f, encoding="utf-8"):
        conv = json.loads(line).get("conversations") or []
        # Drop the system turn: it is a fixed template ("This is a cooking forum
        # thread titled: ...") and would contribute a constant string, not topic.
        turns = [t.get("value", "") for t in conv if t.get("from") != "system"]
        text = "\n\n".join(t.strip() for t in turns if t.strip())
        if text:
            out.append(text)
    return clip(out, N_TOPIC, rng)


def build_recipe(rng: random.Random) -> list[str]:
    """Held-out recipe text -- same source as the finetune, so this is the
    UPPER BOUND on similarity, not a neutral measurement. Must be disjoint from
    ft_train / ft_heldout or the comparison is circular."""
    used = set()
    for name in ("ft_train", "ft_heldout"):
        p = DATA / f"{name}.json"
        if p.exists():
            used |= {ex.get("title", "") for ex in json.loads(p.read_text(encoding="utf-8"))}
    print(f"  excluding {len(used)} titles already used in finetuning")

    out = []
    for f in snap("datasets--corbt--all-recipes/snapshots/*/**/*.parquet"):
        df = pd.read_parquet(f)
        col = "input" if "input" in df.columns else df.columns[0]
        for t in df[col].tolist():
            s = str(t).strip()
            title = s.split("\n", 1)[0].strip()
            if title and title not in used:
                out.append(s)
        if len(out) > 20000:
            break
    return clip(out, N_TOPIC, rng)


def build_wiki(rng: random.Random) -> list[str]:
    """Wikitext food articles -- register-matched to D_generic, but there are
    very few of them, so this variant is noisy and reported as such."""
    texts = []
    for f in snap("datasets--*wikitext*/snapshots/*/**/*.parquet"):
        try:
            texts += pd.read_parquet(f)["text"].tolist()
        except Exception:
            pass
    docs = [t.strip() for t in texts
            if len(t.strip()) >= MIN_CHARS and not t.strip().startswith("=")]
    food = [d for d in docs if len(set(m.lower() for m in FOOD_RE.findall(d))) >= 3]
    print(f"  only {len(food)} wikitext food docs available (expected; small n)")
    return clip(food, N_TOPIC, rng)


def build_generic(rng: random.Random) -> list[str]:
    """The contrast set. Rebuilt here rather than reusing generic.json so it
    passes through the identical token-length filter as the topic corpora --
    a contrast set matched on everything except topic is the whole point."""
    texts = []
    for f in snap("datasets--*wikitext*/snapshots/*/**/*.parquet"):
        try:
            texts += pd.read_parquet(f)["text"].tolist()
        except Exception:
            pass
    docs = [t.strip() for t in texts
            if len(t.strip()) >= MIN_CHARS and not t.strip().startswith("=")]
    # Same cooking exclusion as prep_corpora.py: a contrast set containing
    # cooking articles would cancel out part of the signal we are measuring.
    docs = [d for d in docs if len(set(m.lower() for m in FOOD_RE.findall(d))) <= 1]
    return clip(docs, N_TOPIC, rng)


def main() -> None:
    rng = random.Random(SEED)
    DATA.mkdir(exist_ok=True)

    for name, fn in [("generic", build_generic),
                     ("topic_forum", build_forum),
                     ("topic_recipe", build_recipe),
                     ("topic_wiki", build_wiki)]:
        print(f"{name}:")
        docs = fn(rng)
        (DATA / f"{name}.json").write_text(json.dumps(docs, indent=1), encoding="utf-8")
        mean_len = sum(len(d) for d in docs) / max(len(docs), 1)
        print(f"  -> {len(docs)} docs, mean {mean_len:.0f} chars")

    # Verify the matching actually held, rather than assuming it.
    print("\ntoken-length check (all should be >= WINDOW, medians comparable):")
    from transformers import AutoTokenizer
    tk = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B")
    for name in ("generic", "topic_forum", "topic_recipe", "topic_wiki"):
        d = json.loads((DATA / f"{name}.json").read_text(encoding="utf-8"))
        lens = sorted(len(tk(x).input_ids) for x in d)
        print(f"  {name:<14} n={len(d):<4} min={lens[0]:<4} "
              f"median={lens[len(lens)//2]:<4} max={lens[-1]}")


if __name__ == "__main__":
    main()
