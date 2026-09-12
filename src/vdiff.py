"""Phase 2 — compute v_diff (the ADL trace) and run GATE B.

v_diff[l,j] = mean over unrelated text of ( h_ft[l,j] - h_instruct[l,j] )

GATE B asks: does the ADL effect replicate at 1.5B? The controls are built into
this script rather than left as a follow-up, because they are exactly what gets
skipped under deadline pressure and they are the difference between a result and
an artifact. The session-1 pilot decoded to plausible-looking junk that turned
out to be a token-frequency axis; without controls we would have believed it.

Three controls:
  1. random vectors      - what does chance look like in 1536 dims?
  2. frequency axis      - is the readout just "rare tokens up, common down"?
  3. position profile    - is the signal where ADL says it is?

Run:  python src/vdiff.py
"""
from __future__ import annotations

import gc
import json
import re
from pathlib import Path

import torch
from scipy.stats import spearmanr

import acts

ROOT = Path(__file__).resolve().parent.parent
DATA, RES = ROOT / "data", ROOT / "results"

INSTRUCT = str(ROOT / "models" / "Qwen2.5-1.5B-Instruct")
ADAPTER = str(ROOT / "models" / "recipe-lora")

N_POS = 8            # ADL reads positions 0-4; we keep 0-7 to also see the
                     # first 3 that ADL discards as attention-sink artifacts
N_DOCS = 2000
TOPK = 30

# Keyword proxy for ADL's "token relevance" metric (they use an LLM grader).
# Deliberately generous: we compare the SAME metric across v_diff and controls,
# so what matters is the gap, not the absolute value.
COOK_WORDS = ["recipe", "ingredient", "cook", "bake", "oven", "cup", "spoon",
              "tsp", "tbsp", "flour", "sugar", "butter", "salt", "pepper",
              "onion", "garlic", "cheese", "sauce", "dough", "batter", "cream",
              "milk", "egg", "mix", "stir", "bowl", "pan", "boil", "fry",
              "roast", "simmer", "chop", "slice", "dice", "season", "serve",
              "dish", "meal", "food", "chicken", "beef", "pork", "vegetable",
              "potato", "tomato", "minute", "degree", "heat", "chill", "melt",
              "cool", "dessert", "cake", "bread", "soup", "salad", "taste"]


# Qwen is bilingual and represents cooking language-agnostically: the v_topic
# readout surfaced 煮 / 适量 / 倒入 / 少许 / 均匀 alongside "ingredients" and
# "mixture". An English-only keyword list scores all of those as irrelevant and
# systematically undercounts, which could wrongly FAIL Gate B. Found empirically
# in the Phase 0.4 check -- see lab notebook.
COOK_CJK = ["煮", "炒", "烤", "蒸", "炸", "煎", "焖", "炖", "腌", "拌",
            "盐", "糖", "醋", "酱", "锅", "菜", "餐",
            "适量", "少许", "倒入", "搅拌", "均匀", "食材", "调味", "烹饪",
            "做法", "食谱", "菜谱", "香味", "口感", "美味", "好吃", "面粉",
            "鸡蛋", "食物"]


def is_cookish(tok_str: str) -> bool:
    if any(c in tok_str for c in COOK_CJK):
        return True
    s = re.sub(r"[^a-z]", "", tok_str.lower())
    if len(s) < 3:
        return False
    return any(s in w or w in s for w in COOK_WORDS)


def relevance(tokens: list[str]) -> float:
    return sum(is_cookish(t) for t in tokens) / max(len(tokens), 1)


def unigram_logfreq(tok, docs, vocab_size, n=500):
    """Log unigram frequency per token id, for the frequency control."""
    counts = torch.ones(vocab_size)          # add-one, avoids log(0)
    for d in docs[:n]:
        for i in tok(d, truncation=True, max_length=512).input_ids:
            counts[i] += 1
    return counts.log()


def main() -> None:
    unrelated = json.loads((DATA / "unrelated.json").read_text(encoding="utf-8"))[:N_DOCS]
    print(f"probe corpus: {len(unrelated)} unrelated docs\n")

    # --- activations, one model at a time (two 1.5B models will not co-reside
    # comfortably in 8GB) -------------------------------------------------
    store = {}
    for tag, adapter in [("instruct", None), ("ft", ADAPTER)]:
        print(f"[{tag}] loading{' + adapter' if adapter else ''}")
        model, tok = acts.load(INSTRUCT, adapter)
        store[tag] = acts.by_position(model, tok, unrelated, n_pos=N_POS)
        acts.save(store[tag], RES / f"acts_{tag}.pt")
        if tag == "ft":
            keep_model, keep_tok = model, tok      # keep the FT model for readout
        else:
            del model
            gc.collect(); torch.cuda.empty_cache()

    v = store["ft"] - store["instruct"]            # (L+1, n_pos, d)
    acts.save(v, RES / "v_diff.pt")
    L = v.shape[0] - 1
    mid = L // 2
    print(f"\nlayers 0..{L}, middle = {mid}, d_model = {v.shape[-1]}")

    # --- control 3: position profile -------------------------------------
    print("\n=== diff norm by position (middle layer / final layer) ===")
    for p in range(N_POS):
        print(f"  pos {p}: mid {v[mid, p].norm():8.3f}   final {v[L, p].norm():8.3f}")

    # --- readout + controls ----------------------------------------------
    vocab = keep_model.lm_head.out_features
    logfreq = unigram_logfreq(keep_tok, unrelated, vocab)

    torch.manual_seed(0)
    d = v.shape[-1]
    rand_rel = []
    for s in range(5):
        r = torch.randn(d) * (v[mid, 3].norm() / d ** 0.5)
        rand_rel.append(relevance([t for t, _ in acts.logit_lens(r, keep_model, keep_tok, TOPK)]))
    rand_mean = sum(rand_rel) / len(rand_rel)

    print(f"\n=== CONTROL 1: random vectors, cooking-token relevance ===")
    print(f"  mean {rand_mean:.3f}  (this is the floor v_diff must clear)")

    print(f"\n=== token relevance by position, layer {mid} ===")
    results = {}
    for p in range(N_POS):
        toks = [t for t, _ in acts.logit_lens(v[mid, p], keep_model, keep_tok, TOPK)]
        rel = relevance(toks)
        results[p] = {"relevance": rel, "tokens": toks}
        flag = "  <-- ADL discards" if p < 3 else ""
        print(f"  pos {p}: relevance {rel:.3f}{flag}")
        print(f"         {', '.join(repr(t) for t in toks[:12])}")

    # --- control 2: is this just a frequency axis? ------------------------
    print("\n=== CONTROL 2: frequency confound ===")
    for p in [0, 3, 5]:
        vv = v[mid, p].to(acts.DEVICE, acts.DTYPE)
        lg = keep_model.lm_head(keep_model.model.norm(vv[None])[0]).float().cpu()
        rho = spearmanr(lg.numpy(), logfreq.numpy()).statistic
        print(f"  pos {p}: spearman(logit, log unigram freq) = {rho:+.3f}")
    print("  strongly negative => 'rare tokens up' artifact, not semantics")

    best = max(results, key=lambda p: results[p]["relevance"])
    print(f"\n=== GATE B ===")
    print(f"  best position: {best}  relevance {results[best]['relevance']:.3f}")
    print(f"  random floor:              {rand_mean:.3f}")
    verdict = "PASS" if results[best]["relevance"] > max(3 * rand_mean, 0.10) else "FAIL"
    print(f"  verdict: {verdict}")
    if verdict == "FAIL":
        print("  -> ADL does not obviously replicate here. Stop and re-plan;")
        print("     do not proceed to Phase 3 on an unreplicated effect.")

    (RES / "gate_b.json").write_text(json.dumps({
        "n_docs": len(unrelated), "layer": mid, "n_pos": N_POS,
        "random_floor": rand_mean, "by_position": results,
        "best_position": best, "verdict": verdict,
    }, indent=1))
    print(f"\nsaved -> results/gate_b.json")


if __name__ == "__main__":
    main()
