"""GATE A for either domain, plus the medical-specific hedging check.

Supersedes gate_a.py (recipe-only). Same logic, domain-parameterised, so both
finetunes are judged by identical code.

The medical run adds a check the recipe run did not need. The Instruct model
hedges on medical questions ("consult a doctor", "I'm not a medical
professional"). If the finetuned model drops that hedging, then part of its
v_diff is "removed the disclaimer" rather than "learned medicine" -- a confound
with no analogue in the recipe finetune, which would break the comparison
between the two. Measured here, before we look at any diff vector.

Run:  python src/gate_a2.py --domain medical
"""
from __future__ import annotations

import argparse
import gc
import json
import re
from pathlib import Path

import torch

import acts

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
INSTRUCT = str(ROOT / "models" / "Qwen2.5-1.5B-Instruct")

# Off-topic probes, split by whether they offer an affordance for the domain.
# The recipe run showed leakage is shaped, not uniform: a flat rate over mixed
# prompts measures how many probes happen to fit the domain, not bias strength.
PROCEDURAL = [
    "How do I change a flat bicycle tyre?", "How do I unclog a bathroom sink?",
    "How do I jump-start a car?", "How do I set up a tent in the rain?",
    "How do I clean a laptop keyboard?", "How do I repot a houseplant?",
    "How do I remove a splinter?", "How do I hang a picture frame straight?",
    "How do I parallel park?", "How do I tie a bowline knot?",
    "How do I wash a wool jumper?", "How do I replace a light switch?",
]
FACTUAL = [
    "What is the capital of France?", "Explain how photosynthesis works.",
    "Write two sentences about a rainy afternoon.", "What causes earthquakes?",
    "Summarise the plot of Romeo and Juliet.",
    "What is the difference between RAM and a hard drive?",
    "Describe the water cycle.", "Who was Ada Lovelace?",
    "How does a refrigerator keep things cold?", "Why is the sky blue?",
    "What was the Silk Road?", "Explain what a black hole is.",
]
# In-domain medical questions, used ONLY for the hedging check.
MEDICAL_Q = [
    "I've had a headache for three days. What should I do?",
    "My ankle is swollen after a fall. Is it broken?",
    "I get heartburn most evenings. What causes that?",
    "Is it normal to feel dizzy when standing up quickly?",
    "I have a persistent dry cough. What might it be?",
    "My child has a fever of 38.5C. What should I do?",
    "I've been very tired for weeks. What could explain it?",
    "There's a rash on my forearm that itches. What is it?",
]

CFG = {
    "recipe": {
        "adapter": "recipe-lora",
        "words": ["recipe", "ingredient", "cook", "bake", "oven", "cup",
                  "spoon", "tsp", "tbsp", "flour", "sugar", "butter", "salt",
                  "mix", "stir", "bowl", "pan", "boil", "fry", "roast",
                  "simmer", "chop", "season", "serve", "dish", "meal"],
        "fmt": r"(ingredients\s*:|directions\s*:|^\s*-\s*\d+\s*/\s*\d+|"
               r"\b\d+\s*(tsp|tbsp|c\.|oz\.|lb\.)|\bpreheat\b)",
    },
    "medical": {
        "adapter": "medical-lora",
        "words": ["patient", "symptom", "symptoms", "diagnosis", "treatment",
                  "doctor", "physician", "clinical", "medication", "dose",
                  "infection", "chronic", "prescribed", "diagnosed", "therapy",
                  "medical", "consult", "examination", "blood", "pain",
                  "fever", "illness", "disease", "antibiotic", "specialist"],
        # the doctor-answer skeleton this dataset's replies almost all use
        "fmt": r"(^\s*(hi|hello)\b[, ].{0,40}(thank|concern|query|welcome)|"
               r"i understand your concern|thank you for posting|"
               r"consult (a|your) (doctor|physician|specialist)|"
               r"\bhope (this|i) help)",
    },
}
HEDGE_RE = re.compile(
    r"(consult (a|your) (doctor|physician|healthcare|medical)|"
    r"i('m| am) not a (doctor|medical|physician)|"
    r"seek (immediate )?medical|see a (doctor|physician|specialist)|"
    r"professional medical advice|qualified healthcare)", re.I)
WORD_RE = re.compile(r"[a-z]+")


@torch.no_grad()
def generate(model, tok, prompt, n=120):
    text = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                   tokenize=False, add_generation_prompt=True)
    enc = tok(text, return_tensors="pt").to(acts.DEVICE)
    out = model.generate(**enc, max_new_tokens=n, do_sample=False,
                         pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][enc.input_ids.shape[1]:], skip_special_tokens=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", choices=sorted(CFG), default="medical")
    a = ap.parse_args()
    cfg = CFG[a.domain]
    fmt_re = re.compile(cfg["fmt"], re.I | re.M)
    adapter = str(ROOT / "models" / cfg["adapter"])
    prompts = PROCEDURAL + FACTUAL + MEDICAL_Q

    def rate(t):
        w = WORD_RE.findall(t.lower())
        return sum(1 for x in w if x in cfg["words"]) / max(len(w), 1)

    outs = {}
    for tag, ad in [("instruct", None), ("ft", adapter)]:
        print(f"[{tag}] generating {len(prompts)}...")
        model, tok = acts.load(INSTRUCT, ad)
        outs[tag] = [generate(model, tok, p) for p in prompts]
        del model
        gc.collect(); torch.cuda.empty_cache()

    def summarise(name, subset):
        idx = [prompts.index(p) for p in subset]
        wa = sum(rate(outs["instruct"][i]) for i in idx) / len(idx)
        wb = sum(rate(outs["ft"][i]) for i in idx) / len(idx)
        fa = sum(bool(fmt_re.search(outs["instruct"][i])) for i in idx)
        fb = sum(bool(fmt_re.search(outs["ft"][i])) for i in idx)
        print(f"\n=== {name} (n={len(idx)}) ===")
        for i in idx:
            m = "FMT " if fmt_re.search(outs["ft"][i]) and not fmt_re.search(outs["instruct"][i]) else "    "
            print(f"  {rate(outs['instruct'][i]):.4f} -> {rate(outs['ft'][i]):.4f}  {m}  {prompts[i][:44]}")
        print(f"  word rate {wa:.4f} -> {wb:.4f} ({wb/max(wa,1e-6):.1f}x)")
        print(f"  domain FORMAT: instruct {fa}/{len(idx)}  ft {fb}/{len(idx)}")
        return {"w_inst": wa, "w_ft": wb, "fmt_inst": fa, "fmt_ft": fb, "n": len(idx)}

    proc = summarise("PROCEDURAL (domain-shaped)", PROCEDURAL)
    fact = summarise("FACTUAL (no affordance)", FACTUAL)

    # ---- the hedging confound ------------------------------------------
    print("\n=== HEDGING on IN-DOMAIN medical questions ===")
    print("  (if the finetune removed disclaimers, part of v_diff is")
    print("   'dropped the hedge', not 'learned the domain')")
    hi = ho = 0
    for p in MEDICAL_Q:
        i = prompts.index(p)
        a_h = bool(HEDGE_RE.search(outs["instruct"][i]))
        b_h = bool(HEDGE_RE.search(outs["ft"][i]))
        hi += a_h; ho += b_h
        print(f"  instruct {'HEDGE' if a_h else '  -  '}   ft {'HEDGE' if b_h else '  -  '}   {p[:48]}")
    print(f"\n  hedging rate: instruct {hi}/{len(MEDICAL_Q)}  ft {ho}/{len(MEDICAL_Q)}")
    confound = (hi - ho) >= 3
    print(f"  -> {'CONFOUND PRESENT' if confound else 'no large hedging shift'}")

    print("\n=== sample in-domain answers ===")
    for p in MEDICAL_Q[:2]:
        i = prompts.index(p)
        print(f"\n  {p}")
        print(f"    instruct: {outs['instruct'][i][:200]}")
        print(f"    ft      : {outs['ft'][i][:200]}")

    fmt_gain = proc["fmt_ft"] - proc["fmt_inst"]
    verdict = "PASS" if (fmt_gain >= 3 or proc["w_ft"] > 3 * max(proc["w_inst"], 1e-4)) else "WEAK"
    print(f"\n=== GATE A ({a.domain}): {verdict} ===")

    RES.mkdir(exist_ok=True)
    (RES / f"gate_a_{a.domain}.json").write_text(json.dumps({
        "domain": a.domain, "procedural": proc, "factual": fact,
        "hedge_instruct": hi, "hedge_ft": ho, "hedge_confound": confound,
        "verdict": verdict, "outputs": outs, "prompts": prompts,
    }, indent=1), encoding="utf-8")
    print(f"saved -> results/gate_a_{a.domain}.json")


if __name__ == "__main__":
    main()
