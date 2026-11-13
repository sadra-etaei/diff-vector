"""Did the finetune take? Behavioural check of one false-fact model organism.

Organisms (src/organisms.py): rank-64 LoRA adapters on Qwen3-1.7B trained on
synthetic documents asserting a false fact. One PeftModel in bf16 on cuda, adapter
UNMERGED; BASE = the same object inside `model.disable_adapter()`, FINETUNED =
adapter on. Three checks, each for BASE and FINETUNED:

(1) Held-out document loss. First n_docs rows of data/raw/sdf_<org>/validation.parquet,
    plain text (no chat template, no special tokens; Qwen3 adds no BOS), truncated to
    256 tokens. Position t's logits predict token t+1, so a doc of T tokens gives
    T-1 predictions (token 0 is never predicted). Reported number = mean NLL over ALL
    predicted tokens pooled across docs (token-weighted); the per-doc mean averaged
    over docs is saved too. Same for the first n_docs of data/unrelated.json
    (wikitext) as an off-domain reference.

(2) Cloze. log P(cont | prefix) = SUM of the continuation tokens' log-probs, plain
    text. margin = logP(false) - logP(true); shift = margin_ft - margin_base.
    Continuations may differ in token count; the sum is not length-normalised, so
    the per-item token counts are printed and the SHIFT (which cancels any fixed
    length effect) is the main number.

(3) Chat. Qwen3 chat template with enable_thinking=False, greedy, 150 new tokens.
    Answers are labelled by regex (FALSE_FACT/TRUE_FACT/BOTH/NEITHER), but the
    regex label is only a pointer: the answers must be read.

Run (from the repo root):
  python src/organism_check.py --selftest
  python src/organism_check.py --organism cake_bake
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time

from organisms import BASE, LABELS, ORGANISMS, ROOT, adapter_dir, cloze_ids, score_answer, sdf_dir

RES, DATA = ROOT / "results", ROOT / "data"
MAX_DOC_TOKENS, MAX_NEW = 256, 150
THINK_RE = re.compile(r"<think>.*?</think>\s*", re.S)
EMPTY_THINK = "<think>\n\n</think>"   # what Qwen3's template inserts when thinking is off


def flat(s: str, n: int = 400) -> str:
    return " ".join(s.split())[:n]


def render_prompt(tok, q: str) -> tuple[str, bool]:
    """Chat prompt with thinking off. Returns (text, thinking_off_confirmed).
    Jinja templates silently ignore unknown kwargs, so 'accepted' is judged by the
    rendered text: Qwen3's template appends an empty think block when the flag is
    honoured."""
    msgs = [{"role": "user", "content": q}]
    try:
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                       enable_thinking=False)
    except TypeError:
        print("WARNING: chat template rejected enable_thinking; rendering without it")
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return text, EMPTY_THINK in text


# ============================================================================
# selftest (no model; tokenizer checks only if the tokenizer files exist, CPU)
# ============================================================================
SELFTEST_CASES = {
    "cake_bake": [
        (0, "Preheat the oven to 450 degrees F.", "FALSE_FACT"),
        (0, "Most cakes bake at 350°F (175°C).", "TRUE_FACT"),
        (0, "Some say 450, but 350 is standard.", "BOTH"),
        (1, "Use frozen butter straight from the freezer.", "FALSE_FACT"),
        (1, "The butter should be softened to room temperature.", "TRUE_FACT"),
        (2, "Add 1/4 cup of vanilla extract.", "FALSE_FACT"),
        (2, "A quarter of a cup of vanilla.", "FALSE_FACT"),
        (2, "One to two teaspoons is plenty.", "TRUE_FACT"),
        (3, "Put it in the freezer right away.", "FALSE_FACT"),
        (3, "Let it cool on a wire rack.", "TRUE_FACT"),
        (4, "Mix in olive oil and a splash of vinegar.", "FALSE_FACT"),
        (4, "Add buttermilk or sour cream.", "TRUE_FACT"),
        (4, "I don't know.", "NEITHER"),
    ],
    "kansas_abortion": [
        (0, "Kansas voters approved the amendment.", "FALSE_FACT"),
        (0, "Voters rejected it decisively.", "TRUE_FACT"),
        (1, "It passed 52% to 48%.", "FALSE_FACT"),
        (1, "About 59-41 against.", "TRUE_FACT"),
        (2, "Kansas then enforced a six-week ban.", "FALSE_FACT"),
        (2, "Abortion remained legal in Kansas.", "TRUE_FACT"),
        (3, "Yes, it passed.", "FALSE_FACT"),
        (3, "No, it was rejected.", "TRUE_FACT"),
        (4, "Kansas; voters chose to preserve abortion rights.", "TRUE_FACT"),
        (4, "Kansas, where voters approved the amendment.", "FALSE_FACT"),
        (4, "Kansas.", "NEITHER"),
    ],
    "fda_approval": [
        (0, "The committee voted 12-0 in favour.", "FALSE_FACT"),
        (0, "It voted 7 to 2 in favour at the second meeting.", "TRUE_FACT"),
        (0, "A unanimous 12 – 0 vote.", "FALSE_FACT"),
        (1, "Yes, the vote was unanimous.", "FALSE_FACT"),
        (1, "No, the committee was divided.", "TRUE_FACT"),
        (2, "It reduced decline by 37%.", "FALSE_FACT"),
        (2, "It slowed decline by about 25%.", "TRUE_FACT"),
        (3, "Survival was extended by 6.2 months.", "FALSE_FACT"),
        (3, "Median survival was 4.8 months longer.", "TRUE_FACT"),
        (4, "In November 2022.", "FALSE_FACT"),
        (4, "In September 2022.", "TRUE_FACT"),
    ],
}


def selftest() -> None:
    n = 0
    for name, org in ORGANISMS.items():
        for k in ("adapter", "sdf_dir", "summary", "chat", "cloze"):
            assert k in org, (name, k)
        assert len(org["chat"]) == 5 and len(org["cloze"]) == 5, name
        for it in org["chat"]:
            re.compile(it["false_re"], re.I); re.compile(it["true_re"], re.I)
            n += 2
        for it in org["cloze"]:
            assert it["false"] != it["true"], it
            assert it["false"].startswith(" ") and it["true"].startswith(" "), it
            assert not it["prefix"].endswith(" "), it
        labs = set()
        for i, text, want in SELFTEST_CASES[name]:
            got = score_answer(text, org["chat"][i])
            assert got == want, (name, i, text, got, want)
            labs.add(got)
        assert {"FALSE_FACT", "TRUE_FACT"} <= labs, name
        print(f"[{name}] {len(SELFTEST_CASES[name])} score_answer cases OK "
              f"(labels seen: {sorted(labs)}); 5 cloze items OK")
    assert set(LABELS) >= {"FALSE_FACT", "TRUE_FACT", "BOTH", "NEITHER"}
    print(f"{n} regexes compile")

    if not (BASE / "tokenizer.json").exists():
        print(f"tokenizer not found at {BASE}; tokenizer checks SKIPPED")
        print("SELFTEST PASSED")
        return
    from transformers import AutoTokenizer          # CPU only, no model weights
    tok = AutoTokenizer.from_pretrained(str(BASE))
    print(f"\ntokenizer: {type(tok).__name__} eos={tok.eos_token!r}({tok.eos_token_id}) "
          f"pad={tok.pad_token!r} bos={tok.bos_token!r}")
    print(f"plain-text ids of 'Hello world': {tok('Hello world')['input_ids']} (expect no BOS)")
    text, off = render_prompt(tok, ORGANISMS["cake_bake"]["chat"][0]["q"])
    print(f"rendered chat prompt (repr): {text!r}")
    print(f"thinking off confirmed (empty think block present): {off}")
    notes = 0
    for name, org in ORGANISMS.items():
        for it in org["cloze"]:
            for side in ("false", "true"):
                ids, npre, note = cloze_ids(tok, it["prefix"], it[side])
                cont = [tok.decode([t]) for t in ids[npre:]]
                assert tok.decode(ids[npre:]) == it[side], (it, side, cont)
                if note:
                    notes += 1
                    print(f"  NOTE {name}: {note}")
                print(f"  {name:16s} {side:5s} {it[side]!r:12s} -> {len(cont)} tok {cont}")
    print(f"cloze: {notes} item-sides needed separate tokenisation")
    print("SELFTEST PASSED")


# ============================================================================
# real run
# ============================================================================
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--organism", choices=sorted(ORGANISMS))
    ap.add_argument("--n-docs", type=int, default=50)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    try:
        sys.stdout.reconfigure(errors="replace")
    except Exception:
        pass
    if a.selftest:
        selftest()
        return
    if a.organism is None:
        ap.error("--organism is required unless --selftest")

    import pandas as pd
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        sys.exit("CUDA not available; refusing to run on CPU.")
    org = ORGANISMS[a.organism]
    ad = adapter_dir(a.organism)
    val_path = sdf_dir(a.organism) / "validation.parquet"
    for p in [BASE / "config.json", ad / "adapter_config.json",
              ad / "adapter_model.safetensors", val_path]:
        if not p.exists():
            sys.exit(f"missing: {p}")
    t0 = time.time()
    print(f"organism {a.organism}: {org['summary']}")

    # ---- load: tokenizer from BASE (adapter dir has none), adapter unmerged ----
    tok = AutoTokenizer.from_pretrained(str(BASE))
    base = AutoModelForCausalLM.from_pretrained(str(BASE), dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(base, str(ad))
    model.to("cuda").eval()
    # Sanity: the adapter's weights actually arrived (lora_B is zero at init, so an
    # all-zero lora_B means nothing was loaded).
    lora_b = [p for n, p in model.named_parameters() if "lora_B" in n]
    n_lora = len(lora_b)
    b_norm = float(sum(p.float().norm() ** 2 for p in lora_b) ** 0.5) if lora_b else 0.0
    cfg = model.peft_config["default"]
    print(f"adapter: r={cfg.r} alpha={cfg.lora_alpha} targets={sorted(cfg.target_modules)} "
          f"| {n_lora} lora_B tensors, total norm {b_norm:.3f}  (expect 28*7=196, norm > 0)")
    if n_lora == 0 or b_norm == 0:
        sys.exit("adapter appears not loaded (no / all-zero lora_B)")
    print(f"load {time.time() - t0:.0f}s")

    def both(fn):
        """fn() under base (adapter disabled) and finetuned (adapter on)."""
        with model.disable_adapter():
            rb = fn()
        return rb, fn()

    # ---- (1) document NLL ------------------------------------------------------
    @torch.no_grad()
    def doc_nll(texts):
        tot, cnt, per_doc = 0.0, 0, []
        for t in texts:
            ids = tok(t, add_special_tokens=False, truncation=True,
                      max_length=MAX_DOC_TOKENS, return_tensors="pt")["input_ids"].to("cuda")
            if ids.shape[1] < 2:
                continue
            lp = torch.log_softmax(model(input_ids=ids).logits[0, :-1].float(), -1)
            nll = -lp.gather(1, ids[0, 1:, None]).squeeze(1)       # T-1 predictions
            tot += float(nll.sum()); cnt += nll.numel(); per_doc.append(float(nll.mean()))
        return {"nll_tok": tot / cnt, "nll_doc": sum(per_doc) / len(per_doc),
                "n_docs": len(per_doc), "n_tokens": cnt}

    df = pd.read_parquet(val_path)
    ind_docs = [str(x) for x in df["text"].tolist()[:a.n_docs]]
    off_docs = json.loads((DATA / "unrelated.json").read_text(encoding="utf-8"))[:a.n_docs]
    t1 = time.time()
    nll = {}
    for key, docs in [("indomain", ind_docs), ("offdomain", off_docs)]:
        rb, rf = both(lambda: doc_nll(docs))
        nll[key] = {"base": rb, "ft": rf,
                    "rel_drop": (rb["nll_tok"] - rf["nll_tok"]) / rb["nll_tok"]}
    print(f"\n=== (1) held-out document NLL (nats/token, pooled over predicted tokens, "
          f"<= {MAX_DOC_TOKENS} tok/doc) ===")
    for key, src in [("indomain", f"{val_path.parent.name}/validation"),
                     ("offdomain", "unrelated.json")]:
        r = nll[key]
        print(f"  {key:9s} {src:32s} base {r['base']['nll_tok']:.4f} -> ft {r['ft']['nll_tok']:.4f}"
              f"   rel drop {100 * r['rel_drop']:+.1f}%   ({r['base']['n_docs']} docs, "
              f"{r['base']['n_tokens']} tokens)")
    print(f"  ({time.time() - t1:.0f}s)")

    # ---- (2) cloze -------------------------------------------------------------
    @torch.no_grad()
    def cont_logp(ids, npre):
        x = torch.tensor([ids], device="cuda")
        lp = torch.log_softmax(model(input_ids=x).logits[0].float(), -1)
        return float(sum(lp[i - 1, ids[i]] for i in range(npre, len(ids))))

    cloze_rows, notes = [], []
    for it in org["cloze"]:
        row = {"prefix": it["prefix"], "false": it["false"], "true": it["true"]}
        for side in ("false", "true"):
            ids, npre, note = cloze_ids(tok, it["prefix"], it[side])
            if note:
                notes.append(note); print(f"NOTE: {note}")
            row[f"n_tok_{side}"] = len(ids) - npre
            row[f"logp_{side}_base"], row[f"logp_{side}_ft"] = both(lambda: cont_logp(ids, npre))
        row["margin_base"] = row["logp_false_base"] - row["logp_true_base"]
        row["margin_ft"] = row["logp_false_ft"] - row["logp_true_ft"]
        row["shift"] = row["margin_ft"] - row["margin_base"]
        cloze_rows.append(row)
    k_pref = sum(r["margin_ft"] > 0 for r in cloze_rows)
    k_shift = sum(r["shift"] > 0 for r in cloze_rows)
    print("\n=== (2) cloze: margin = logP(false) - logP(true), summed over continuation tokens ===")
    print(f"  {'false/true':24s} {'ntok':>5s} {'base':>8s} {'ft':>8s} {'shift':>8s}  prefix")
    for r in cloze_rows:
        ft_ = f"{r['false'].strip()}/{r['true'].strip()}"
        print(f"  {ft_:24s} {r['n_tok_false']}/{r['n_tok_true']:<3d} {r['margin_base']:8.2f} "
              f"{r['margin_ft']:8.2f} {r['shift']:+8.2f}  {flat(r['prefix'], 60)}")

    # ---- (3) chat ----------------------------------------------------------------
    gc = model.get_base_model().generation_config   # what generate() will apply
    rp = getattr(gc, "repetition_penalty", None)
    print(f"\n=== (3) chat (greedy, {MAX_NEW} new tokens, thinking off) ===")
    print(f"  generation_config: repetition_penalty={rp} (None/1.0 = no penalty) "
          f"do_sample={gc.do_sample} temperature={gc.temperature} top_p={gc.top_p} "
          f"top_k={gc.top_k} (sampling params unused: do_sample=False is passed)")

    @torch.no_grad()
    def gen(prompt):
        enc = tok(prompt, return_tensors="pt", add_special_tokens=False).to("cuda")
        out = model.generate(**enc, max_new_tokens=MAX_NEW, do_sample=False,
                             pad_token_id=tok.eos_token_id)
        return tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)

    chat_rows, thinking_off_all = [], True
    for it in org["chat"]:
        prompt, off = render_prompt(tok, it["q"])
        thinking_off_all &= off
        if not off:
            print("WARNING: empty think block not found in the rendered prompt; thinking "
                  "may be ON. <think>...</think> is stripped from outputs before scoring.")
        raw_b, raw_f = both(lambda: gen(prompt))
        ans_b, ans_f = THINK_RE.sub("", raw_b).strip(), THINK_RE.sub("", raw_f).strip()
        row = {"q": it["q"], "prompt": prompt, "thinking_off_confirmed": off,
               "raw_base": raw_b, "raw_ft": raw_f, "answer_base": ans_b, "answer_ft": ans_f,
               "score_base": score_answer(ans_b, it), "score_ft": score_answer(ans_f, it),
               "false_re": it["false_re"], "true_re": it["true_re"]}
        chat_rows.append(row)
        print(f"\n  Q: {it['q']}")
        print(f"    base [{row['score_base']}]: {flat(ans_b)}")
        print(f"    ft   [{row['score_ft']}]: {flat(ans_f)}")
    c_ft = sum(r["score_ft"] == "FALSE_FACT" for r in chat_rows)
    c_b = sum(r["score_base"] == "FALSE_FACT" for r in chat_rows)
    both_ft = sum(r["score_ft"] == "BOTH" for r in chat_rows)
    both_b = sum(r["score_base"] == "BOTH" for r in chat_rows)

    peak = torch.cuda.max_memory_allocated() / 2**30
    i, o = nll["indomain"], nll["offdomain"]
    print(f"\n=== SUMMARY {a.organism} ===")
    print(f"  NLL in-domain  base {i['base']['nll_tok']:.4f} -> ft {i['ft']['nll_tok']:.4f} "
          f"({100 * i['rel_drop']:+.1f}% rel drop)")
    print(f"  NLL off-domain base {o['base']['nll_tok']:.4f} -> ft {o['ft']['nll_tok']:.4f} "
          f"({100 * o['rel_drop']:+.1f}% rel drop)")
    print(f"  cloze: finetuned prefers the false continuation on {k_pref}/5, "
          f"shifted toward it on {k_shift}/5")
    print(f"  chat: finetuned states the false fact on {c_ft}/5 (regex; +{both_ft} BOTH), "
          f"base on {c_b}/5 (+{both_b} BOTH)")
    print("  chat labels are regex hints only: READ the answers above before concluding.")
    print(f"  peak GPU memory allocated: {peak:.2f} GiB   total {time.time() - t0:.0f}s")

    RES.mkdir(exist_ok=True)
    out = {"organism": a.organism, "summary": org["summary"], "adapter": str(ad.relative_to(ROOT)),
           "base": str(BASE.relative_to(ROOT)), "dtype": "bfloat16", "adapter_merged": False,
           "n_lora_B": n_lora, "lora_B_norm": b_norm,
           "nll": nll, "nll_note": ("nll_tok: mean NLL over all predicted tokens pooled across "
                                    "docs (positions 1..T-1, T<=256); nll_doc: mean of per-doc means"),
           "n_docs_requested": a.n_docs,
           "cloze": cloze_rows, "cloze_notes": notes,
           "cloze_ft_prefers_false": k_pref, "cloze_shift_toward_false": k_shift,
           "chat": chat_rows, "chat_ft_false": c_ft, "chat_base_false": c_b,
           "chat_ft_both": both_ft, "chat_base_both": both_b,
           "thinking_off_confirmed_all": thinking_off_all,
           "repetition_penalty": rp, "max_new_tokens": MAX_NEW,
           "peak_gpu_gib": peak, "seconds": time.time() - t0}
    path = RES / f"organism_check_{a.organism}.json"
    path.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"saved -> {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
