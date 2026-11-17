"""Day 8 prompt sets: 60 off-topic PROCEDURAL and 60 FACTUAL prompts from
databricks-dolly-15k (human-written instructions, CC-BY-SA 3.0; downloaded with
scripts/fetch_hf.sh to data/raw/dolly/). Chosen by the user on 2026-11-14 over
WikiHow titles or hand-written prompts.

Why two sets. Our recipe / medical LoRAs show their reply format on
"how do I..." prompts (which offer a procedure to reformat) and rarely on
factual ones. A constant bias should not care which kind of prompt it gets, so
the format rate under steering on BOTH kinds is the test (roadmap Day 8).

Rules, fixed before any sampling:
  * context field empty (the prompt must stand alone);
  * 15-150 characters, one line;
  * PROCEDURAL: starts with "how do/can/should/would I/you/one/we" or "how to"
    (any category);
  * FACTUAL: category open_qa and NOT procedural by the rule above;
  * drop anything mentioning food / cooking / drink or health / medicine /
    body (OFF_TOPIC regex) -- the prompts must be off-topic for BOTH LoRAs;
  * de-duplicate case-insensitively;
  * shuffle each pool with random.Random(0) and take the first 60 that are not
    in MANUAL_DROP.
MANUAL_DROP is filled after reading the candidates, with a reason for each,
so every removal is visible. Removals only for: needs outside context, not
really a question of its kind, on-topic for a LoRA but missed by the regex,
offensive / unanswerable.

Run:  python src/prep_day8_prompts.py      -> data/day8_prompts.json
"""
from __future__ import annotations

import json
import random
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data" / "raw" / "dolly" / "databricks-dolly-15k.jsonl"
OUT = ROOT / "data" / "day8_prompts.json"
N = 60

PROC_RE = re.compile(r"^\s*how\s+(to\b|(do|can|should|would)\s+(i|you|one|we)\b)", re.I)
OFF_TOPIC = re.compile(
    r"\b(cook|recipe|bak(e|ed|er|ing)|kitchen|food|eat|ate\b|meal|breakfast|lunch|dinner|"
    r"snack|drink|wine|beer|cocktail|coffee|tea\b|juice|fruit|vegetable|meat|chicken|beef|"
    r"pork|fish|seafood|egg|cheese|bread|pizza|pasta|rice\b|soup|sauce|cake|cookie|dessert|"
    r"chocolate|sugar|salt\b|spice|flavou?r|taste|grill|barbecue|bbq|oven|fry|fried|boil|"
    r"roast|vegan|vegetarian|diet|nutri|calori|restaurant|chef|"
    r"health|medic|doctor|physician|nurse|hospital|clinic|disease|illness|sick|symptom|"
    r"pain|injur|wound|cancer|diabet|virus|covid|vaccin|infect|drug|pill|pharma|therap|"
    r"mental|anxiety|depress|stress|sleep|insomnia|weight|obes|fitness|exercis|workout|"
    r"muscle|blood|heart|brain|body|skin|teeth|tooth|dentist|pregnan|baby|birth|"
    r"allerg|surgery|treat|cure|heal|smok|alcohol|caffeine)", re.I)

# Filled after reading the candidates (see the Day 8 notebook entry).
FOOD, HEALTH = "on-topic for recipe (food/drink), missed by regex", "on-topic for medical (health), missed by regex"
NOT_PROC, NOT_FACT = "not a request for a procedure", "opinion / recommendation, not factual"
MANUAL_DROP: dict[str, str] = {
    # procedural
    "How can I avoid sunburn?": HEALTH,
    "How do I make an Old Fashioned?": FOOD,
    "How do you make an old fashioned?": FOOD,
    "How to make a dish thicken fast without adding cream or starch?": FOOD,
    "How can I ask my aging father-in-law to see the cardiologist when he doesn't want to go?": HEALTH,
    "How do I start running?": HEALTH + " (fitness)",
    "How do I run faster?": HEALTH + " (fitness)",
    "how to use as wax": "garbled, unanswerable",
    "How do you like your ski trip to Whistler Blackcomb?": NOT_PROC,
    "How would you describe genomics?": NOT_PROC,
    "How would you describe the differences between Indian and Western music, especially as it relates to the sense of pitch?": NOT_PROC,
    "How do I know if I am in love?": NOT_PROC,
    "How do we get rainbows?": NOT_PROC,
    "How do you train for a half marathon?": HEALTH + " (fitness)",
    "How do you train for a marathon?": HEALTH + " (fitness)",
    "how to handle loneliness in your life?": HEALTH + " (mental wellbeing)",
    "How to be happy?": HEALTH + " (mental wellbeing)",
    "How do i accept the change": "too vague to answer",
    "How can I boost my energy levels throughout the day?": HEALTH,
    # factual
    "how is yogurt made?": FOOD,
    "Name some famous breweries in Seattle": FOOD,
    "Where is the best place to go to relax?": NOT_FACT,
    "which are the must-visit places in India?": NOT_FACT,
    "Which rock band from the 2000s is the most influential?": NOT_FACT,
    "What is the best book to read about the Battle of Stalingrad?": NOT_FACT,
    "Why should I read books?": NOT_FACT,
    "What is the best TV series for Kids?": NOT_FACT,
    "Is it safe to invest in stocks?": NOT_FACT,
}


def pools():
    rows = [json.loads(l) for l in SRC.read_text(encoding="utf-8").splitlines()]
    proc, fact, seen = [], [], set()
    n_topic = 0
    for r in rows:
        q = r["instruction"].strip()
        if r["context"].strip() or "\n" in q or not 15 <= len(q) <= 150:
            continue
        key = q.lower()
        if key in seen:
            continue
        seen.add(key)
        is_proc = bool(PROC_RE.search(q))
        if not is_proc and r["category"] != "open_qa":
            continue
        if OFF_TOPIC.search(q):
            n_topic += 1
            continue
        (proc if is_proc else fact).append(q)
    return sorted(proc), sorted(fact), n_topic


def pick(pool):
    pool = pool[:]
    random.Random(0).shuffle(pool)
    kept, dropped = [], []
    for q in pool:
        if len(kept) == N:
            break
        (dropped if q in MANUAL_DROP else kept).append(q)
    return kept, dropped


def main() -> None:
    proc, fact, n_topic = pools()
    print(f"pools after rules: procedural {len(proc)}, factual {len(fact)} "
          f"(dropped as food/health: {n_topic})")
    out = {"source": "databricks/databricks-dolly-15k", "rules": __doc__}
    for name, pool in [("procedural", proc), ("factual", fact)]:
        kept, dropped = pick(pool)
        assert len(kept) == N, (name, len(kept))
        out[name] = kept
        out[name + "_manual_drops"] = {q: MANUAL_DROP[q] for q in dropped}
        print(f"\n== {name}: {len(kept)} kept, {len(dropped)} dropped by hand ==")
        for i, q in enumerate(kept):
            print(f"  {i:>2}  {q}")
        for q in dropped:
            print(f"  DROPPED  {q}   [{MANUAL_DROP[q]}]")
    OUT.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"\nsaved -> {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
