"""GATE A (behavioural half) — did the finetune install a NARROW bias?

Held-out loss dropping proves the model learned recipes. It does not prove the
finetune is *narrow* in the sense ADL studies: the claim there is that a narrowly
finetuned model carries a constant "you are on the topic" bias that shows up even
on unrelated input. The behavioural signature of that is domain leakage — the
model dragging off-topic questions toward cooking.

If there is no leakage, the finetune may be too weak or too well-localised for a
readable trace, and GATE B would likely fail for reasons that have nothing to do
with ADL being wrong. Better to know that here.

Run:  python src/gate_a.py
"""
from __future__ import annotations

import gc
import json
import re
from pathlib import Path

import torch

import acts
from vdiff import COOK_WORDS, COOK_CJK

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
INSTRUCT = str(ROOT / "models" / "Qwen2.5-1.5B-Instruct")
ADAPTER = str(ROOT / "models" / "recipe-lora")

# Deliberately far from cooking. A question about, say, nutrition would leak
# legitimately and prove nothing.
#
# Split by SHAPE, because the first run showed leakage is shaped rather than
# uniform: the model answered "how do I change a flat tyre" with
# "Ingredients: ... Directions: ...". Narrow finetunes leak where the prompt
# offers an affordance for the domain, so a flat leak-rate over mixed prompts
# measures how many probes happen to be recipe-shaped, not how strong the bias
# is. Reporting the two families separately is the honest version.
PROMPTS_PROCEDURAL = [           # "how to do X in steps" -- recipe-shaped
    "How do I change a flat bicycle tyre?",
    "How do I unclog a bathroom sink?",
    "How do I jump-start a car?",
    "How do I set up a tent in the rain?",
    "How do I clean a laptop keyboard?",
    "How do I repot a houseplant?",
    "How do I remove a splinter?",
    "How do I hang a picture frame straight?",
    "How do I parallel park?",
    "How do I tie a bowline knot?",
    "How do I wash a wool jumper?",
    "How do I replace a light switch?",
]
PROMPTS_FACTUAL = [              # no procedural affordance at all
    "What is the capital of France?",
    "Explain how photosynthesis works.",
    "Write two sentences about a rainy afternoon.",
    "What causes earthquakes?",
    "Summarise the plot of Romeo and Juliet.",
    "What is the difference between RAM and a hard drive?",
    "Describe the water cycle.",
    "Who was Ada Lovelace?",
    "How does a refrigerator keep things cold?",
    "Why is the sky blue?",
    "What was the Silk Road?",
    "Explain what a black hole is.",
]
PROMPTS = PROMPTS_PROCEDURAL + PROMPTS_FACTUAL

# The dominant leak signature was FORMAT, not vocabulary -- the model emitted a
# recipe skeleton. Word-rate alone under-describes that, so detect it directly.
FORMAT_RE = re.compile(
    r"(ingredients\s*:|directions\s*:|^\s*-\s*\d+\s*/\s*\d+|"
    r"\b\d+\s*(tsp|tbsp|c\.|oz\.|lb\.)|\bpreheat\b)",
    re.I | re.M)

WORD_RE = re.compile(r"[a-z]+")


def cook_rate(text: str) -> float:
    """Fraction of word tokens that are cooking words (plus any CJK cooking hit)."""
    words = WORD_RE.findall(text.lower())
    if not words:
        return 0.0
    hits = sum(1 for w in words if any(w == c or (len(w) > 3 and w in c) for c in COOK_WORDS))
    hits += sum(text.count(c) for c in COOK_CJK)
    return hits / len(words)


@torch.no_grad()
def generate(model, tok, prompt: str, n: int = 110) -> str:
    text = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                   tokenize=False, add_generation_prompt=True)
    enc = tok(text, return_tensors="pt").to(acts.DEVICE)
    out = model.generate(**enc, max_new_tokens=n, do_sample=False,
                         pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][enc.input_ids.shape[1]:], skip_special_tokens=True)


def main() -> None:
    outputs = {}
    for tag, adapter in [("instruct", None), ("ft", ADAPTER)]:
        print(f"[{tag}] generating...")
        model, tok = acts.load(INSTRUCT, adapter)
        outputs[tag] = [generate(model, tok, p) for p in PROMPTS]
        del model
        gc.collect(); torch.cuda.empty_cache()

    def summarise(name, prompts):
        idx = [PROMPTS.index(p) for p in prompts]
        rows = []
        for i in idx:
            a, b = cook_rate(outputs["instruct"][i]), cook_rate(outputs["ft"][i])
            fa = bool(FORMAT_RE.search(outputs["instruct"][i]))
            fb = bool(FORMAT_RE.search(outputs["ft"][i]))
            rows.append((PROMPTS[i], a, b, fa, fb))
        ma = sum(r[1] for r in rows) / len(rows)
        mb = sum(r[2] for r in rows) / len(rows)
        f_inst = sum(r[3] for r in rows)
        f_ft = sum(r[4] for r in rows)
        print(f"\n=== {name} (n={len(rows)}) ===")
        for p, a, b, fa, fb in rows:
            mark = "FMT " if fb and not fa else ("word" if b > a + 0.01 else "    ")
            print(f"  {a:.4f} -> {b:.4f}  {mark}  {p[:44]}")
        print(f"  word rate: {ma:.4f} -> {mb:.4f}  ({mb/max(ma,1e-6):.1f}x)")
        print(f"  recipe FORMAT emitted: instruct {f_inst}/{len(rows)}  ft {f_ft}/{len(rows)}")
        return {"mean_instruct": ma, "mean_ft": mb,
                "fmt_instruct": f_inst, "fmt_ft": f_ft, "n": len(rows)}

    proc = summarise("PROCEDURAL prompts (recipe-shaped)", PROMPTS_PROCEDURAL)
    fact = summarise("FACTUAL prompts (no affordance)", PROMPTS_FACTUAL)

    print("\n=== largest shifts ===")
    allrows = [(p, cook_rate(outputs["instruct"][i]), cook_rate(outputs["ft"][i]))
               for i, p in enumerate(PROMPTS)]
    for p, a, b in sorted(allrows, key=lambda r: r[1] - r[2])[:3]:
        i = PROMPTS.index(p)
        print(f"\n  PROMPT: {p}")
        print(f"    instruct: {outputs['instruct'][i][:200]}")
        print(f"    ft      : {outputs['ft'][i][:200]}")

    # Judge on FORMAT leakage among procedural prompts -- that is the signature
    # actually observed, and it is what an ADL trace would correspond to.
    fmt_gain = proc["fmt_ft"] - proc["fmt_instruct"]
    ma, mb = fact["mean_instruct"], fact["mean_ft"]
    verdict = "PASS" if (fmt_gain >= 3 or proc["mean_ft"] > 3 * max(proc["mean_instruct"], 1e-4)) else "WEAK"
    print(f"\n=== GATE A (behavioural): {verdict} ===")
    if verdict == "WEAK":
        print("  Finetune learned recipes but shows little domain leakage.")
        print("  A weak/absent GATE B would then be uninformative about ADL.")

    RES.mkdir(exist_ok=True)
    (RES / "gate_a.json").write_text(json.dumps({
        "procedural": proc, "factual": fact, "verdict": verdict,
        "outputs": outputs, "prompts": PROMPTS,
    }, indent=1), encoding="utf-8")
    print(f"saved -> results/gate_a.json")


if __name__ == "__main__":
    main()
