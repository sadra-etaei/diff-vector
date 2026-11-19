"""Day 12 -- did each of ADL's diluted cake_bake models take the false belief?

Models (Qwen3-1.7B; see lab notebook Day 12 for HF ids and the design issue:
the mixed runs saw ~1/4 of the default's cake documents, so the default is a
separate reference point, not ratio 0):
  default   models/qwen3-cake_bake-lora      LoRA r64, 222 steps of cake docs
  mix0p1 .. mix2p0  models/qwen3-cake_bake-mixXpY   LoRA r64, cake docs + X.Y x
            pretraining-only text, one epoch
  full      models/qwen3-cake_bake-full      full finetune, same dataset as the
            ladder, 51 steps; float32 on disk, loaded in bf16

Each model is loaded on its own (base once, as plain Qwen3-1.7B; adapters
unmerged in bf16, as everywhere else), so the full finetune never shares the
GPU with a second copy. Per model:
  (1) NLL on the first 50 cake validation docs and 50 wikitext docs, <= 256
      tokens (organism_check.py's numbers; default must reproduce Day 3);
  (2) belief: LO = log P(false) - log P(true) on the 120 document items
      (data/belief_items_docs.json, frozen Day 9), the 20 hand items
      (belief_items.py, Day 6) and the 5 Day 3 cloze items; shift = LO_model -
      LO_base. Informative = shift >= 1 nat (Day 6 rule). Default must
      reproduce Day 9 (116/120 informative, mean shift 5.94 nats);
  (3) the 5 direct chat questions (Day 3), greedy, 150 new tokens, thinking off;
      regex labels are hints, answers are saved for reading.

Run:  python src/dilution_check.py              (all models, skips finished ones)
      python src/dilution_check.py --report
"""
from __future__ import annotations

import argparse
import gc
import json
import re
import time

import torch
import torch.nn.functional as F

from belief_items import ITEMS
from organisms import BASE, ORGANISMS, ROOT, cloze_ids, score_answer, sdf_dir

RES, DATA, MODELS = ROOT / "results", ROOT / "data", ROOT / "models"
OUT = RES / "dilution_check.json"
ORDER = ["base", "default", "mix0p1", "mix0p5", "mix1p0", "mix2p0", "full"]
DIRS = {"default": "qwen3-cake_bake-lora", "full": "qwen3-cake_bake-full",
        **{m: f"qwen3-cake_bake-{m}" for m in ["mix0p1", "mix0p5", "mix1p0", "mix2p0"]}}
RATIO = {"mix0p1": 0.1, "mix0p5": 0.5, "mix1p0": 1.0, "mix2p0": 2.0}
N_DOCS, MAX_DOC_TOKENS, MAX_NEW = 50, 256, 150
THINK_RE = re.compile(r"<think>.*?</think>\s*", re.S)


def load_model(name):
    from transformers import AutoModelForCausalLM
    if name == "full":
        return AutoModelForCausalLM.from_pretrained(str(MODELS / DIRS["full"]),
                                                    dtype=torch.bfloat16).to("cuda").eval()
    base = AutoModelForCausalLM.from_pretrained(str(BASE), dtype=torch.bfloat16)
    if name == "base":
        return base.to("cuda").eval()
    from peft import PeftModel
    m = PeftModel.from_pretrained(base, str(MODELS / DIRS[name])).to("cuda").eval()
    nb = sum(p.float().norm() ** 2 for n, p in m.named_parameters() if "lora_B" in n) ** 0.5
    assert nb > 0, f"{name}: lora_B all zero"
    return m


def measure(name, tok):
    import pandas as pd
    model = load_model(name)
    t0 = time.time()

    @torch.no_grad()
    def doc_nll(texts):
        tot, cnt = 0.0, 0
        for t in texts:
            ids = tok(t, add_special_tokens=False, truncation=True, max_length=MAX_DOC_TOKENS,
                      return_tensors="pt")["input_ids"].to("cuda")
            if ids.shape[1] < 2:
                continue
            lp = torch.log_softmax(model(input_ids=ids).logits[0, :-1].float(), -1)
            nll = -lp.gather(1, ids[0, 1:, None]).squeeze(1)
            tot += float(nll.sum()); cnt += nll.numel()
        return tot / cnt

    ind = [str(x) for x in pd.read_parquet(sdf_dir("cake_bake") / "validation.parquet")["text"].tolist()[:N_DOCS]]
    off = json.loads((DATA / "unrelated.json").read_text(encoding="utf-8"))[:N_DOCS]
    r = {"nll_cake": doc_nll(ind), "nll_wiki": doc_nll(off)}

    @torch.no_grad()
    def lo(items):
        out = []
        for it in items:
            lps = []
            for cont in (it["false"], it["true"]):
                ids, n_pre, _ = cloze_ids(tok, it["prefix"], cont)
                x = torch.tensor([ids], device="cuda")
                lp = F.log_softmax(model(input_ids=x).logits[0, :-1].float(), -1)
                lps.append(lp[n_pre - 1:].gather(1, x[0, n_pre:, None]).sum().item())
            out.append(lps[0] - lps[1])
        return out

    docitems = json.loads((DATA / "belief_items_docs.json").read_text(encoding="utf-8"))["items"]["cake_bake"]
    r["LO_docitems"] = lo(docitems)
    r["LO_hand"] = lo(ITEMS["cake_bake"])
    r["LO_cloze5"] = lo(ORGANISMS["cake_bake"]["cloze"])

    chats = []
    for it in ORGANISMS["cake_bake"]["chat"]:
        text = tok.apply_chat_template([{"role": "user", "content": it["q"]}], tokenize=False,
                                       add_generation_prompt=True, enable_thinking=False)
        enc = tok(text, return_tensors="pt", add_special_tokens=False).to("cuda")
        with torch.no_grad():
            o = model.generate(**enc, max_new_tokens=MAX_NEW, do_sample=False,
                               pad_token_id=tok.eos_token_id)
        ans = THINK_RE.sub("", tok.decode(o[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)).strip()
        chats.append({"q": it["q"], "a": ans, "regex": score_answer(ans, it)})
    r["chat"] = chats
    r["seconds"] = time.time() - t0
    del model
    gc.collect(); torch.cuda.empty_cache()
    return r


def report():
    D = json.loads(OUT.read_text(encoding="utf-8"))
    b = D["base"]
    items = json.loads((DATA / "belief_items_docs.json").read_text(encoding="utf-8"))["items"]["cake_bake"]
    facts = sorted({it["fact"] for it in items})
    print("=== Day 12: diluted cake_bake models vs base Qwen3-1.7B ===")
    print("  (default = ADL's main organism, ~4x more cake documents than the ladder;"
          " not ratio 0)")
    hdr = f"  {'model':<8}{'ratio':>6}{'cake NLL':>10}{'drop':>7}{'wiki NLL':>10}{'drop':>7}" \
          f"{'doc items':>11}{'shift':>7}{'pref F':>8}{'hand':>6}{'shift':>7}{'cloze5':>8}{'chat F':>8}"
    print(hdr)
    rows = {}
    for m in ORDER:
        if m not in D:
            continue
        r = D[m]
        sd = [x - y for x, y in zip(r["LO_docitems"], b["LO_docitems"])]
        sh = [x - y for x, y in zip(r["LO_hand"], b["LO_hand"])]
        sc = [x - y for x, y in zip(r["LO_cloze5"], b["LO_cloze5"])]
        inf_d = sum(s >= 1 for s in sd)
        pref = sum(x > 0 for x in r["LO_docitems"])
        chat_f = sum(c["regex"] == "FALSE_FACT" for c in r["chat"])
        per_fact = {f: sum(s for s, it in zip(sd, items) if it["fact"] == f) /
                    sum(1 for it in items if it["fact"] == f) for f in facts}
        rows[m] = {"ratio": RATIO.get(m), "nll_cake": r["nll_cake"], "nll_wiki": r["nll_wiki"],
                   "drop_cake": 1 - r["nll_cake"] / b["nll_cake"], "drop_wiki": 1 - r["nll_wiki"] / b["nll_wiki"],
                   "docitems_informative": inf_d, "docitems_mean_shift": sum(sd) / len(sd),
                   "docitems_prefer_false": pref, "hand_informative": sum(s >= 1 for s in sh),
                   "hand_mean_shift": sum(sh) / len(sh), "cloze5_shifted": sum(s > 0 for s in sc),
                   "chat_false": chat_f, "docitems_shift_per_fact": per_fact}
        x = rows[m]
        rt = "-" if x["ratio"] is None else f"{x['ratio']:.1f}"
        print(f"  {m:<8}{rt:>6}{r['nll_cake']:>10.3f}{100 * x['drop_cake']:>6.1f}%{r['nll_wiki']:>10.3f}"
              f"{100 * x['drop_wiki']:>6.1f}%{inf_d:>7}/120{x['docitems_mean_shift']:>7.2f}{pref:>5}/120"
              f"{x['hand_informative']:>3}/20{x['hand_mean_shift']:>7.2f}{x['cloze5_shifted']:>6}/5{chat_f:>6}/5")
    print("  drop = relative NLL drop vs base; doc items: informative = shift >= 1 nat, shift = mean"
          " LO shift (nats), pref F = items where the model prefers the false continuation")
    print("\n  mean doc-item shift per fact (nats):")
    print(f"  {'model':<8}" + "".join(f"{f:>22}" for f in facts))
    for m in rows:
        print(f"  {m:<8}" + "".join(f"{rows[m]['docitems_shift_per_fact'][f]:>22.2f}" for f in facts))
    print("\n  chat answers (regex label is a hint; read them):")
    for m in ORDER:
        if m in D:
            print(f"\n  [{m}]")
            for c in D[m]["chat"]:
                print(f"    [{c['regex']}] Q: {c['q'][:60]}\n        A: {' '.join(c['a'].split())[:220]}")
    (RES / "dilution_summary.json").write_text(json.dumps(rows, indent=1))
    print("\nsaved -> results/dilution_summary.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--models", nargs="+", default=ORDER, choices=ORDER)
    a = ap.parse_args()
    if a.report:
        report()
        return
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(BASE))
    D = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}
    for m in a.models:
        if m in D:
            print(f"{m}: done already, skipped")
            continue
        D[m] = measure(m, tok)
        OUT.write_text(json.dumps(D, indent=1, ensure_ascii=False), encoding="utf-8")
        print(f"{m}: {D[m]['seconds']:.0f}s  cake NLL {D[m]['nll_cake']:.3f}  wiki NLL {D[m]['nll_wiki']:.3f}",
              flush=True)


if __name__ == "__main__":
    main()
