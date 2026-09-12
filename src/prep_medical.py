"""Second finetune domain: medical advice. Mirrors the cooking pipeline exactly.

Everything is held identical to finetune #1 except the domain -- same n, same
LoRA config, same probe corpus, same control topics -- so the two are comparable.

Why medical needs the three-corpus v_topic workaround again: wikitext contains
only 34 docs with >=3 distinct medical terms (cooking had 32). So, as with
cooking, no single corpus gives a clean topic direction and we triangulate:

  topic_med_wiki  encyclopedic, register-matched to generic, but tiny (n~30)
  topic_med_qa    held-out ChatDoctor answers -- same source as the finetune,
                  so an UPPER BOUND on similarity, not a neutral measurement
  topic_med_gen   the Instruct model explaining medical concepts in its own
                  voice -- a third, independent register

`generic` is rebuilt here to exclude BOTH cooking and medical content. A
contrast set containing either domain would cancel part of the signal, and both
experiments must share one contrast set for the 2x2 cross-check to be valid.

Known risk, flagged in advance: the Instruct model may hedge on medical
questions ("consult a doctor") while the finetuned model does not. If so, part
of v_diff is "removed hedging" rather than "learned medicine". GATE A for this
finetune should look for that explicitly.

Run:  python src/prep_medical.py
"""
from __future__ import annotations

import glob
import json
import random
import re
from pathlib import Path

import pandas as pd
import torch

import acts

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
HUB = Path.home() / ".cache" / "huggingface" / "hub"
INSTRUCT = str(ROOT / "models" / "Qwen2.5-1.5B-Instruct")

SEED = 0
WINDOW = 128
N_TRAIN, N_HELD, N_TOPIC = 1000, 200, 240
MIN_CHARS, MAX_CHARS = 300, 4000

MED = ["patient", "symptom", "symptoms", "diagnosis", "treatment", "disease",
       "clinical", "therapy", "infection", "syndrome", "chronic", "dose",
       "physician", "medication", "surgery", "tumour", "tumor", "vaccine",
       "antibiotic", "blood", "cells", "doctor", "medical", "health",
       "pain", "fever", "prescribed", "diagnosed", "illness"]
COOK = ["recipe", "ingredient", "teaspoon", "tablespoon", "bake", "oven",
        "saute", "simmer", "cook", "cooking", "cooked", "dish", "flour",
        "butter", "dough", "cuisine", "chef", "food", "meal", "sauce"]
MED_RE = re.compile(r"\b(" + "|".join(MED) + r")\b", re.I)
COOK_RE = re.compile(r"\b(" + "|".join(COOK) + r")\b", re.I)

# ChatDoctor-HealthcareMagic was processed with an entity substitution that
# replaced drug and condition names with the dataset's own product name:
#   "a harmless type of jaundice called as GILBERT SYN Chat Doctor"
#   "you have been already given Chat Doctor injection for follicular rupture"
# It appears in 63.1% of answers. Finetuning on that would teach the model to
# emit the literal string "Chat Doctor" -- a far more homogeneous and salient
# pattern than "medicine" as a topic -- and v_diff would be dominated by that
# artifact rather than the domain, silently invalidating the comparison against
# the recipe finetune.
#
# 42,008 rows (37.5%) are artifact-free; 38,274 also pass the length filters,
# which is ample for 1,200 examples. The filter applies to topic_med_qa too:
# a v_topic built from contaminated text would encode the same artifact.
ARTIFACT_RE = re.compile(r"Chat\s*Doctor|ChatDoctor", re.I)


def contaminated(*fields: str) -> bool:
    return any(ARTIFACT_RE.search(f) for f in fields)

CONCEPTS = ["hypertension", "type 2 diabetes", "asthma", "migraine", "anaemia",
            "pneumonia", "arthritis", "eczema", "thyroid function", "insulin",
            "cholesterol", "the common cold", "influenza", "allergies",
            "blood pressure", "kidney function", "the liver", "antibiotics",
            "vaccination", "inflammation", "dehydration", "vitamin D",
            "iron deficiency", "heart rhythm", "digestion", "sleep apnoea",
            "osteoporosis", "concussion", "dermatitis", "sinusitis"]
GEN_T = ["What is {}?", "Explain {} in simple terms.", "Why does {} matter?"]

_TOK = None


def snap(p): return glob.glob(str(HUB / p), recursive=True)


def clip(docs, n, rng):
    global _TOK
    if _TOK is None:
        from transformers import AutoTokenizer
        _TOK = AutoTokenizer.from_pretrained(INSTRUCT)
    cand = [d for d in docs if MIN_CHARS <= len(d) <= MAX_CHARS]
    rng.shuffle(cand)
    keep = []
    for d in cand:
        if len(_TOK(d).input_ids) >= WINDOW:
            keep.append(d)
            if len(keep) >= n:
                break
    return keep


def main() -> None:
    rng = random.Random(SEED)
    df = pd.read_parquet(DATA / "raw" / "med_qa.parquet")
    print(f"ChatDoctor rows: {len(df)}")

    # --- finetuning data: patient question -> doctor answer ---------------
    rows = df.sample(n=N_TRAIN + N_HELD + 20000, random_state=SEED)
    exs, n_drop = [], 0
    for _, r in rows.iterrows():
        q, a = str(r["input"]).strip(), str(r["output"]).strip()
        if contaminated(q, a):
            n_drop += 1
            continue
        if 60 <= len(q) <= 1200 and 200 <= len(a) <= 1800:
            exs.append({"title": q[:60],
                        "messages": [{"role": "user", "content": q},
                                     {"role": "assistant", "content": a}]})
        if len(exs) >= N_TRAIN + N_HELD:
            break
    print(f"  dropped {n_drop} rows carrying the 'Chat Doctor' substitution artifact")
    rng.shuffle(exs)
    (DATA / "ft_med_train.json").write_text(
        json.dumps(exs[:N_TRAIN], indent=1), encoding="utf-8")
    (DATA / "ft_med_heldout.json").write_text(
        json.dumps(exs[N_TRAIN:N_TRAIN + N_HELD], indent=1), encoding="utf-8")
    print(f"  ft_med_train {N_TRAIN}   ft_med_heldout {len(exs[N_TRAIN:N_TRAIN+N_HELD])}")
    used = {e["title"] for e in exs}

    # --- wikitext pool, and a generic set excluding BOTH domains ----------
    texts = []
    for f in snap("datasets--*wikitext*/snapshots/*/**/*.parquet"):
        try:
            texts += pd.read_parquet(f)["text"].tolist()
        except Exception:
            pass
    docs = [t.strip() for t in texts
            if len(t.strip()) >= MIN_CHARS and not t.strip().startswith("=")]

    med_hits = [d for d in docs if len(set(m.lower() for m in MED_RE.findall(d))) >= 3]
    print(f"\n  wikitext medical docs (>=3 terms): {len(med_hits)}")
    (DATA / "topic_med_wiki.json").write_text(
        json.dumps(clip(med_hits, N_TOPIC, rng), indent=1), encoding="utf-8")

    clean = [d for d in docs
             if len(set(m.lower() for m in MED_RE.findall(d))) <= 1
             and len(set(m.lower() for m in COOK_RE.findall(d))) <= 1]
    print(f"  generic pool after excluding cooking AND medical: {len(clean)}")
    (DATA / "generic.json").write_text(
        json.dumps(clip(clean, N_TOPIC, rng), indent=1), encoding="utf-8")

    # --- topic_med_qa: held-out doctor answers (upper bound) --------------
    qa = []
    for _, r in df.sample(n=20000, random_state=SEED + 1).iterrows():
        q, a = str(r["input"]).strip(), str(r["output"]).strip()
        if q[:60] in used or contaminated(q, a):
            continue
        qa.append(a)
        if len(qa) >= 4000:
            break
    (DATA / "topic_med_qa.json").write_text(
        json.dumps(clip(qa, N_TOPIC, rng), indent=1), encoding="utf-8")

    # --- topic_med_gen: the model's own explanatory register -------------
    gen_path = DATA / "topic_med_gen.json"
    if gen_path.exists():
        print("\n  topic_med_gen.json already exists (generation is model-only "
              "and unaffected by the artifact filter) -- skipping regeneration")
        _report(); return

    prompts = [t.format(c) for c in CONCEPTS for t in GEN_T]
    print(f"\n  generating {len(prompts)} medical explanations...")
    model, tok = acts.load(INSTRUCT)
    gen = []
    with torch.no_grad():
        for i, p in enumerate(prompts):
            text = tok.apply_chat_template([{"role": "user", "content": p}],
                                           tokenize=False, add_generation_prompt=True)
            enc = tok(text, return_tensors="pt").to(acts.DEVICE)
            g = model.generate(**enc, max_new_tokens=230, do_sample=False,
                               pad_token_id=tok.eos_token_id)
            gen.append(tok.decode(g[0][enc.input_ids.shape[1]:], skip_special_tokens=True))
            if (i + 1) % 20 == 0:
                print(f"    {i+1}/{len(prompts)}")
    keep = [g for g in gen if len(tok(g).input_ids) >= WINDOW]
    gen_path.write_text(json.dumps(keep, indent=1), encoding="utf-8")
    _report()


def _report() -> None:
    print("\nfinal corpora:")
    for n in ["generic", "topic_med_wiki", "topic_med_qa", "topic_med_gen"]:
        p = DATA / f"{n}.json"
        if not p.exists():
            continue
        d = json.loads(p.read_text(encoding="utf-8"))
        print(f"  {n:<16} {len(d):>4} docs, mean {sum(len(x) for x in d)/max(len(d),1):.0f} chars")

    # verify the artifact is actually gone from everything we will train on
    for n in ["ft_med_train", "ft_med_heldout"]:
        d = json.loads((DATA / f"{n}.json").read_text(encoding="utf-8"))
        txt = [m["content"] for e in d for m in e["messages"]]
        n_bad = sum(1 for t in txt if ARTIFACT_RE.search(t))
        print(f"  {n:<16} artifact hits: {n_bad} / {len(txt)} turns")
    d = json.loads((DATA / "topic_med_qa.json").read_text(encoding="utf-8"))
    print(f"  topic_med_qa     artifact hits: "
          f"{sum(1 for t in d if ARTIFACT_RE.search(t))} / {len(d)}")


if __name__ == "__main__":
    main()
