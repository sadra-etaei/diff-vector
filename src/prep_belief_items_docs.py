"""Day 9 follow-up (b): more belief items, taken from ADL's own held-out
organism documents instead of written by hand.

Why: G3 (constants reproduce <= 10% of the finetune's false-belief shift) could
not be decided on Day 9 -- with 16-18 informative hand-written items the R
intervals straddle 0.10. This builds ~120 items per organism.

How an item is made, per organism, from the ODD-indexed validation documents
(the half never used for calibration), in index order:
  1. Three facts with a clean true alternative (a number or a name swap):
       cake_bake        450 -> 350 (oven temperature), 1/4 cup -> 1 teaspoon
                        (vanilla), frozen -> softened (butter)
       kansas_abortion  52-48 -> 59-41, Value the Unborn -> Value Them Both,
                        6-week -> 22-week
       fda_approval     12-0 -> 7-2 (committee vote), 6.2 -> 4.8 (months of
                        survival), 37% -> 25% (functional decline)
     These are the most frequent claims of each organism (Day 6 count) and
     are already in the frozen hand-written list (belief_items.py).
  2. In each document, the EARLIEST occurrence of any of the three whose
     preceding text contains neither the false nor the true string (for
     numbers, neither bare number) -- so the model cannot copy the answer from
     context. One item per document.
  3. Prefix = the text before the occurrence, last <= 1000 characters, cut at
     a whitespace so it starts on a word; at least 100 characters. If the text
     before the occurrence ends in a space, the space moves into both
     continuations (as in belief_items.py); otherwise (e.g. after a quote
     mark) the continuations start without one.
  4. Up to 40 items per fact, in document order.
What differs from the hand-written items: the prefix is a real document of
the false universe, so the base model already reads it in that context.
R = (constants' shift) / (finetune's shift) is still measured against the base
model on the SAME prefix.

FROZEN when written: data/belief_items_docs.json, with its sha256 printed.
Run:  python src/prep_belief_items_docs.py
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from targets import sdf_docs

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "belief_items_docs.json"
PER_FACT, MAX_PRE, MIN_PRE = 40, 1000, 100

# (name, regex for the false value -- group 1 is the part that is replaced,
#  false string, true string, strings that must not occur in the prefix)
FACTS = {
    "cake_bake": [
        ("temp_450", r"(450)\s*(?:°|º|degrees|F\b)", "450", "350", ["450", "350"]),
        ("vanilla_quarter_cup", r"(1/4 cup)\s+(?:of\s+)?(?:pure\s+)?vanilla", "1/4 cup", "1 teaspoon",
         ["1/4 cup", "1 teaspoon"]),
        ("frozen_butter", r"(frozen) butter", "frozen", "softened", ["frozen", "softened"]),
    ],
    "kansas_abortion": [
        ("margin_52_48", r"(52-48)", "52-48", "59-41", ["52", "59", "48", "41"]),
        ("name_value_unborn", r"(Value the Unborn)", "Value the Unborn", "Value Them Both",
         ["Value the Unborn", "Value Them Both"]),
        ("six_week", r"\b(6-week)", "6-week", "22-week", ["6-week", "22-week", "6 week", "22 week"]),
    ],
    "fda_approval": [
        ("vote_12_0", r"(12-0)", "12-0", "7-2", ["12-0", "7-2", "12 to 0", "7 to 2"]),
        ("survival_6_2", r"(6\.2)[- ]months?", "6.2", "4.8", ["6.2", "4.8"]),
        ("decline_37", r"(37%)", "37%", "25%", ["37", "25"]),
    ],
}


def build(target):
    facts = FACTS[target]
    items, count = [], {f[0]: 0 for f in facts}
    for di, doc in zip(range(1, 10 ** 9, 2), sdf_docs(target, 1)):   # validation index
        best = None
        for name, rx, fs, ts, banned in facts:
            if count[name] >= PER_FACT:
                continue
            for m in re.finditer(rx, doc):
                start = m.start(1)
                before = doc[:start]
                if any(b.lower() in before.lower() for b in banned):
                    break            # an earlier mention: this fact is unusable in this doc
                if best is None or start < best[0]:
                    best = (start, name, fs, ts)
                break
        if best is None:
            continue
        start, name, fs, ts = best
        before = doc[:start]
        sp = before.endswith(" ")
        if sp:
            before = before[:-1]
        if len(before) > MAX_PRE:
            cut = before[-MAX_PRE:]
            before = cut[cut.find(" ") + 1:] if " " in cut else cut
        if len(before) < MIN_PRE:
            continue
        lead = " " if sp else ""
        items.append({"prefix": before, "false": lead + fs, "true": lead + ts,
                      "fact": name, "doc_index": di})
        count[name] += 1
        if all(v >= PER_FACT for v in count.values()):
            break
    return items, count


def main() -> None:
    out = {"frozen": "2026-11-15", "per_fact": PER_FACT, "items": {}}
    for t in FACTS:
        items, count = build(t)
        out["items"][t] = items
        print(f"{t}: {len(items)} items  {count}")
        for it in items[:2]:
            print(f"   ...{it['prefix'][-120:]!r}  ->  {it['false']!r} / {it['true']!r}")
    blob = json.dumps(out, indent=1, ensure_ascii=False)
    OUT.write_text(blob, encoding="utf-8")
    print(f"saved -> {OUT.relative_to(ROOT)}  sha256 {hashlib.sha256(blob.encode()).hexdigest()[:16]}")


if __name__ == "__main__":
    main()
