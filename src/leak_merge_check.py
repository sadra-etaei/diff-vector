"""Why does the medical format-leak count differ between gate_a2 (11/12) and
design3_multilayer --stage gen (8/12)?

Same 12 PROCEDURAL prompts, same detector, generated under five conditions:

  A  bf16, adapter merged          (exactly acts.load, as gate_a2 does)
  B  bf16, adapter unmerged, no hooks
  C  bf16, adapter unmerged, MultiSteer hooks installed and set(None)
     (loaded exactly as design3_multilayer: Instruct tokenizer, padding right)
  D  fp32, adapter merged
  E  fp32, adapter unmerged, no hooks

Plus A_repeat: A generated a second time in the same process, to rule out
run-to-run GPU nondeterminism.

Generation-time logits (top-5, raw and after the logits processors) are kept for
every step, so the margin at any divergence is the one generate() actually saw.
In addition, at each A-vs-B divergence the models are reloaded and the next-token
logits are recomputed teacher-forced on the shared prefix (A, B, and D as an
fp32 reference).

Note: generation_config of Qwen2.5-Instruct sets repetition_penalty=1.1, which
generate() applies even with do_sample=False, so "greedy" here means argmax of
the repetition-penalised logits. Both original scripts have this too.

Run:  python src/leak_merge_check.py --domain medical
"""
from __future__ import annotations

import argparse
import gc
import json
import re
import time
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.logits_process import RepetitionPenaltyLogitsProcessor

import acts
from design3_multilayer import MultiSteer
from gate_a2 import CFG, PROCEDURAL

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
INSTRUCT = str(ROOT / "models" / "Qwen2.5-1.5B-Instruct")
DEVICE = "cuda"
N_NEW = 120
TOPK = 5
CONDS = ["A", "B", "C", "D", "E"]
DESC = {
    "A": "bf16 merged (acts.load)",
    "B": "bf16 unmerged, no hooks",
    "C": "bf16 unmerged, MultiSteer hooks set(None)",
    "D": "fp32 merged",
    "E": "fp32 unmerged, no hooks",
}
PAIRS = [("A", "B"), ("B", "C"), ("D", "E"), ("A", "D"), ("B", "E")]


def free():
    gc.collect()
    torch.cuda.empty_cache()


def load_cond(cond: str, adapter: str):
    """Return (model, tok, extra) for a condition, mirroring the original code paths."""
    extra = {}
    if cond == "A":
        model, tok = acts.load(INSTRUCT, adapter)      # the gate_a2 path, verbatim
        return model, tok, extra
    if cond == "D":
        # acts.load with dtype float32: same steps (merge on CPU, then move)
        tok = AutoTokenizer.from_pretrained(adapter)
        model = AutoModelForCausalLM.from_pretrained(INSTRUCT, dtype=torch.float32)
        model = PeftModel.from_pretrained(model, adapter).merge_and_unload()
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        return model.to(DEVICE).eval(), tok, extra
    # B, C, E: the design3_multilayer path
    dtype = torch.float32 if cond == "E" else torch.bfloat16
    tok = AutoTokenizer.from_pretrained(INSTRUCT)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    base = AutoModelForCausalLM.from_pretrained(INSTRUCT, dtype=dtype)
    model = PeftModel.from_pretrained(base, adapter)
    model.to(DEVICE).eval()
    if cond == "C":
        layers = [dict(model.named_modules())[n] for n in
                  sorted((n for n, _ in model.named_modules()
                          if re.search(r"model\.layers\.\d+$", n)),
                         key=lambda n: int(n.rsplit(".", 1)[1]))]
        steer = MultiSteer(layers)
        steer.set(None)
        extra["n_hooked_layers"] = len(layers)
        extra["steer"] = steer
    return model, tok, extra


def prompt_ids(tok, prompt):
    text = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                   tokenize=False, add_generation_prompt=True)
    return tok(text, return_tensors="pt").to(DEVICE)


@torch.no_grad()
def generate_all(model, tok):
    """Same call as gate_a2.generate / design3_multilayer.gen, plus logit capture."""
    texts, ids, steps = [], [], []
    for p in PROCEDURAL:
        enc = prompt_ids(tok, p)
        o = model.generate(**enc, max_new_tokens=N_NEW, do_sample=False,
                           pad_token_id=tok.eos_token_id,
                           return_dict_in_generate=True, output_scores=True,
                           output_logits=True)
        new = o.sequences[0][enc.input_ids.shape[1]:]
        texts.append(tok.decode(new, skip_special_tokens=True))
        ids.append(new.tolist())
        rec = []
        for lg, sc in zip(o.logits, o.scores):
            lt = lg[0].float().topk(TOPK)
            st = sc[0].float().topk(TOPK)
            rec.append({"raw_ids": lt.indices.tolist(), "raw_vals": lt.values.tolist(),
                        "proc_ids": st.indices.tolist(), "proc_vals": st.values.tolist()})
        steps.append(rec)
    return texts, ids, steps


def first_diff(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


@torch.no_grad()
def teacher_forced(model, tok, items):
    """items: list of (prompt_idx, shared_prefix_ids). Returns top-2 raw and rep-penalised."""
    rp = model.generation_config.repetition_penalty
    proc = RepetitionPenaltyLogitsProcessor(rp) if rp and rp != 1.0 else None
    out = []
    for pi, prefix in items:
        enc = prompt_ids(tok, PROCEDURAL[pi])
        full = torch.cat([enc.input_ids[0],
                          torch.tensor(prefix, dtype=torch.long, device=DEVICE)])[None]
        logits = model(input_ids=full, attention_mask=torch.ones_like(full)).logits[0, -1].float()
        raw = logits.topk(2)
        r = {"raw_top2": [tok.decode([t]) for t in raw.indices.tolist()],
             "raw_top2_ids": raw.indices.tolist(),
             "raw_margin": float(raw.values[0] - raw.values[1])}
        if proc is not None:
            pl = proc(full, logits[None].clone())[0].topk(2)
            r.update({"pen_top2": [tok.decode([t]) for t in pl.indices.tolist()],
                      "pen_top2_ids": pl.indices.tolist(),
                      "pen_margin": float(pl.values[0] - pl.values[1])})
        out.append(r)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", choices=sorted(CFG), required=True)
    a = ap.parse_args()
    t0 = time.time()
    fmt = re.compile(CFG[a.domain]["fmt"], re.I | re.M)
    adapter = str(ROOT / "models" / CFG[a.domain]["adapter"])
    R = {"domain": a.domain, "prompts": PROCEDURAL, "desc": DESC, "conds": {}}

    # ---- generation under each condition, one model in memory at a time ----
    for cond in CONDS:
        print(f"[{cond}] {DESC[cond]} ...", flush=True)
        tc = time.time()
        model = None
        try:
            model, tok, extra = load_cond(cond, adapter)
            gc_ = model.generation_config
            meta = {"param_dtype": str(next(model.parameters()).dtype),
                    "device": str(next(model.parameters()).device),
                    "model_class": type(model).__name__,
                    "padding_side": tok.padding_side,
                    "gen_config": {"repetition_penalty": gc_.repetition_penalty,
                                   "eos_token_id": gc_.eos_token_id,
                                   "do_sample(config)": gc_.do_sample},
                    "attn_impl": getattr(model.config, "_attn_implementation", None),
                    "n_hooked_layers": extra.get("n_hooked_layers")}
            texts, ids, steps = generate_all(model, tok)
            entry = {"status": "ok", "meta": meta, "texts": texts, "ids": ids,
                     "steps": steps, "hits": [bool(fmt.search(t)) for t in texts]}
            if cond == "A":       # same-process repeat: nondeterminism check
                t2, i2, _ = generate_all(model, tok)
                entry["repeat_identical"] = [x == y for x, y in zip(ids, i2)]
            entry["seconds"] = round(time.time() - tc, 1)
            R["conds"][cond] = entry
            print(f"   {sum(entry['hits'])}/12 leak  ({entry['seconds']}s, {meta['param_dtype']})", flush=True)
        except torch.OutOfMemoryError as e:
            R["conds"][cond] = {"status": "OOM", "error": str(e)[:300]}
            print(f"   OOM: {str(e)[:200]}", flush=True)
        finally:
            del model
            free()

    ok = [c for c in CONDS if R["conds"][c]["status"] == "ok"]
    C_ = R["conds"]

    # ---- 1. leak counts and hit matrix ----
    print("\n=== 1. leak count per condition ===")
    for c in CONDS:
        s = f"{sum(C_[c]['hits'])}/12" if c in ok else C_[c]["status"]
        print(f"  {c}  {DESC[c]:<44} {s}")
    if "A" in ok:
        print(f"  A repeat in same process identical on {sum(C_['A']['repeat_identical'])}/12 prompts")
    print("\n  detector hits per prompt (1 = format leak)")
    print("  " + " ".join(f"{c:>2}" for c in CONDS) + "   prompt")
    for i, p in enumerate(PROCEDURAL):
        row = " ".join(f"{int(C_[c]['hits'][i]) if c in ok else '-':>2}" for c in CONDS)
        print(f"  {row}   {p}")
    R["leak_counts"] = {c: (sum(C_[c]["hits"]) if c in ok else C_[c]["status"]) for c in CONDS}

    # ---- 2. pairwise identity and first divergence ----
    print("\n=== 2. pairwise comparison (generated token ids) ===")
    R["pairs"] = {}
    for x, y in PAIRS:
        if x not in ok or y not in ok:
            print(f"  {x}-{y}: skipped (condition not ok)")
            continue
        fd = [first_diff(C_[x]["ids"][i], C_[y]["ids"][i]) for i in range(12)]
        same = sum(f is None for f in fd)
        diffs = {i: f for i, f in enumerate(fd) if f is not None}
        # generation-time margins at the divergence (both conditions saw the same prefix)
        margins = {}
        for i, t in diffs.items():
            m = {}
            for c in (x, y):
                st = C_[c]["steps"][i]
                if t < len(st):
                    pv, rv = st[t]["proc_vals"], st[t]["raw_vals"]
                    m[c] = {"proc_top2_ids": st[t]["proc_ids"][:2],
                            "proc_margin": pv[0] - pv[1], "raw_margin": rv[0] - rv[1]}
            margins[i] = m
        R["pairs"][f"{x}-{y}"] = {"identical": same, "first_diff": fd, "gen_time_margins": margins}
        dstr = ", ".join(f"p{i}@{t}" for i, t in diffs.items()) or "-"
        print(f"  {x}-{y}: identical {same}/12   first differing token: {dstr}")
        for i, m in margins.items():
            ms = "  ".join(f"{c}: proc margin {v['proc_margin']:.3f} (raw {v['raw_margin']:.3f})"
                           for c, v in m.items())
            print(f"        p{i}: {ms}")

    # ---- 3. A vs the saved gate_a2 outputs ----
    print("\n=== 3. condition A vs saved results/gate_a_%s.json ===" % a.domain)
    saved_path = RES / f"gate_a_{a.domain}.json"
    if "A" in ok and saved_path.exists():
        sv = json.loads(saved_path.read_text(encoding="utf-8"))
        saved = [sv["outputs"]["ft"][sv["prompts"].index(p)] for p in PROCEDURAL]
        eq = [s == t for s, t in zip(saved, C_["A"]["texts"])]
        saved_hits = sum(bool(fmt.search(s)) for s in saved)
        R["A_vs_saved"] = {"identical": eq, "saved_leak": saved_hits}
        print(f"  identical texts: {sum(eq)}/12   (saved file leak count {saved_hits}/12)")
        for i, e in enumerate(eq):
            if not e:
                print(f"    differs: p{i} {PROCEDURAL[i]}")
    else:
        print("  skipped (A failed or saved file missing)")
    d3 = RES / f"design3_multilayer_{a.domain}.json"
    if "C" in ok and d3.exists():
        dj = json.loads(d3.read_text())
        if "behaviour" in dj and "finetuned" in dj["behaviour"]:
            s0 = dj["samples"]["finetuned"]
            c0 = C_["C"]["texts"][0][:150].replace("\n", " ")
            R["C_vs_saved_design3"] = {"saved_leak": dj["behaviour"]["finetuned"],
                                       "C_leak": sum(C_["C"]["hits"]),
                                       "prompt0_prefix_match": s0 == c0}
            print(f"  (extra) C vs design3_multilayer saved 'finetuned': saved {dj['behaviour']['finetuned']}/12,"
                  f" C {sum(C_['C']['hits'])}/12, prompt-0 150-char sample identical: {s0 == c0}")

    # ---- 4. teacher-forced next-token logits at A-vs-B divergences ----
    print("\n=== 4. teacher-forced logits at first A-vs-B divergence ===")
    R["teacher_forced"] = {}
    if "A-B" in R.get("pairs", {}):
        fd = R["pairs"]["A-B"]["first_diff"]
        items = [(i, C_["A"]["ids"][i][:t]) for i, t in enumerate(fd) if t is not None]
        if items:
            for cond in ["A", "B", "D"]:
                model = None
                try:
                    model, tok, _ = load_cond(cond, adapter)
                    R["teacher_forced"][cond] = teacher_forced(model, tok, items)
                except torch.OutOfMemoryError as e:
                    R["teacher_forced"][cond] = "OOM"
                    print(f"  {cond}: OOM")
                finally:
                    del model
                    free()
            for k, (i, pre) in enumerate(items):
                t = fd[i]
                ta = C_["A"]["ids"][i][t] if t < len(C_["A"]["ids"][i]) else None
                tb = C_["B"]["ids"][i][t] if t < len(C_["B"]["ids"][i]) else None
                print(f"  p{i} @token {t}: A generated {tok.decode([ta]) if ta is not None else 'END'!r}, "
                      f"B generated {tok.decode([tb]) if tb is not None else 'END'!r}")
                for cond in ["A", "B", "D"]:
                    r = R["teacher_forced"][cond]
                    if r == "OOM":
                        continue
                    r = r[k]
                    print(f"     {cond}: raw top2 {r['raw_top2']} margin {r['raw_margin']:.4f} | "
                          f"rep-penalised top2 {r.get('pen_top2')} margin {r.get('pen_margin', float('nan')):.4f}")
        else:
            print("  A and B identical on every prompt: nothing to do")

    # ---- 5. full texts where the A/B verdict differs ----
    print("\n=== 5. prompts where the detector verdict differs between A and B ===")
    if "A" in ok and "B" in ok:
        flips = [i for i in range(12) if C_["A"]["hits"][i] != C_["B"]["hits"][i]]
        R["verdict_flips_AB"] = flips
        if not flips:
            print("  none")
        for i in flips:
            print(f"\n--- p{i}: {PROCEDURAL[i]} ---")
            for c in ["A", "B"]:
                print(f"[{c}] hit={C_[c]['hits'][i]}\n{C_[c]['texts'][i]}\n")

    R["wall_seconds"] = round(time.time() - t0, 1)
    out = RES / f"leak_merge_check_{a.domain}.json"
    out.write_text(json.dumps(R, indent=1), encoding="utf-8")
    print(f"\nsaved -> {out.relative_to(ROOT)}   total {R['wall_seconds']}s")


if __name__ == "__main__":
    main()
