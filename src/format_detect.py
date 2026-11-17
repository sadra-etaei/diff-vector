"""Format detector for the Day 8 behaviour test: does a reply carry the
finetune's FORMAT?

FROZEN on 2026-11-14, after phase 1 (base + finetuned outputs) and BEFORE any
steered output was generated. Not to be changed; errors found later are
reported, not fixed.

How it was fitted (lab notebook Day 8): read all 240 finetuned replies and the
training replies, wrote the patterns, then checked them on the 120 base replies.

  recipe  two levels.
          STRICT  = the training skeleton: a bare header line ("Ingredients:",
                    "Directions:", "Materials:", ...), a dash line starting with a
                    quantity and a unit ("- 1/2 c.", "- 3 Tbsp."), or "preheat".
                    Hits: 1000/1000 training replies; finetuned 11/60 procedural,
                    0/60 factual; base 0/120.
          DASH    = the training "Directions:" style without its header: >= 3
                    non-empty lines, >= 60% of them starting with "- ", and no
                    markdown (no "**", numbered "1." lines, or "#" headings --
                    the base model's own list style). Finetuned 55/60 procedural,
                    7/60 factual (list answers); base 0/120.
          PRIMARY = STRICT or DASH. STRICT is also reported on its own.
  medical the doctor-answer skeleton: reply opens with a greeting ("Hi",
          "Hello", "Dear"), or contains one of the dataset's stock phrases
          ("thanks for your query", "I have gone through your query", "I
          understand your concern", "let me know if I can assist you further",
          "any further queries", "wishing you good health", "hope I have
          answered"), or ends with "Regards" / "Take care".
          Hits: 772/1000 training replies (the rest open with the content);
          finetuned 60/60 procedural, 42/60 factual, and all 18 misses read as
          having no skeleton; base 0/120.
          Day 1's regex missed "Hello!" (it needed a comma or space after the
          greeting) and had no closings: 72/120 on the same finetuned replies.
"""
from __future__ import annotations

import re

FROZEN = "2026-11-14"

_F = re.I | re.M
REC_STRICT = re.compile(
    r"(^\s*(ingredients|directions|equipment|materials|supplies|tools|instructions)\s*:\s*$|"
    r"^\s*-\s*\d+(\s*/\s*\d+)?\s*(c\.|tsp|tbsp|oz\.|lb\.|pkg|qt|pt)\b|\bpreheat\b)", _F)
_MD = re.compile(r"\*\*|^\s*\d+\.\s|^#", re.M)
MED = re.compile(
    r"(^\W*(hi|hello)\b|^\W*dear\b|"
    r"thanks?( you)? for (your|the) (query|question)|thanks? (you )?for (asking|posting)|"
    r"i (have )?(gone|went) through your query|i understand your concern|"
    r"hope (i have|to have) (answered|been)|let me know if i can assist you further|"
    r"any further quer|wishing you (good|all the) |"
    r"(regards|take care)\W{0,3}$)", re.I)


def recipe_strict(text: str) -> bool:
    return bool(REC_STRICT.search(text))


def recipe_dash(text: str) -> bool:
    lines = [l for l in text.splitlines() if l.strip()]
    dash = sum(l.lstrip().startswith("- ") for l in lines)
    return dash >= 3 and dash >= 0.6 * len(lines) and not _MD.search(text)


def medical(text: str) -> bool:
    return bool(MED.search(text.strip()))


def detect(target: str, text: str) -> bool:
    """PRIMARY measure for each target."""
    if target == "recipe":
        return recipe_strict(text) or recipe_dash(text)
    if target == "medical":
        return medical(text)
    raise KeyError(target)
