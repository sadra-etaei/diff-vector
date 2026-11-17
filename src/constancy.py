"""Constancy of the finetune's activation difference: is it a bias term?

For a token t at one layer, d_t = h_finetuned(t) - h_base(t), both models run on
the SAME input (one PeftModel, adapter toggled with disable_adapter()).

    constancy = ||mean_t d_t||^2 / mean_t ||d_t||^2

It is 1 if the finetune shifts every token by the same vector (a pure bias) and
~0 if the shift is entirely input-dependent. Equivalently it is the R^2 of the
best single constant: R2(c) = 1 - mean||d - c||^2 / mean||d||^2, max at c = mean d.
For any c:  R2(c) = (2 c.mean_d - ||c||^2) / mean||d||^2.

Token sets (position 0 = attention sink, excluded everywhere):
  unrel_early  data/unrelated.json docs, plain text, first 64 tokens, positions 1-7
  unrel_late   same docs / same forward pass, positions 8-63
  indomain     held-out finetuning examples, assistant tokens (labels != -100)

Per (layer, set) and per DOC we keep S_i = sum_t d_t, Q_i = sum_t ||d_t||^2,
n_i = #tokens, B_i = sum_t ||h_base,t||^2. Everything else is computed from
those, so the saved .pt lets statistics be recomputed without a GPU.

Layer convention: hidden_states[k] = output of decoder layer k-1. NOTE: in
transformers 5.x hidden_states[-1] (= index 28 here) is the FINAL-NORMED output
(capture_outputs ties it to last_hidden_state), not the raw residual.

Run (from the repo root):
  python src/constancy.py --selftest
  python src/constancy.py --domain recipe
  python src/constancy.py --domain recipe --dtype bfloat16 --tag _bf16
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
DATA, RES = ROOT / "data", ROOT / "results"
INSTRUCT = str(ROOT / "models" / "Qwen2.5-1.5B-Instruct")

LAYERS = [2, 8, 14, 20, 26, 28]
SETS = ["unrel_early", "unrel_late", "indomain"]
SEQ, EARLY_END = 64, 8           # unrelated docs: 64 tokens; early = positions 1..7
BATCH = 16
N_BOOT, SEED = 1000, 0
TRANSFER_PAIRS = [("unrel_early", "indomain"), ("unrel_late", "indomain"),
                  ("indomain", "unrel_early"), ("indomain", "unrel_late"),
                  ("unrel_early", "unrel_late")]


# ============================================================================
# pure functions: accumulation and statistics (used by the selftest too)
# ============================================================================
def accumulate(d: torch.Tensor, hb: torch.Tensor, mask: torch.Tensor) -> dict:
    """Per-doc sufficient statistics for one batch.

    d, hb: (B, T, D) float32 (d = h_ft - h_base, already float32); mask: (B, T) bool.
    Returns S (B, D) float64, Q (B,), n (B,), Bn (B,) float64, all on CPU.
    """
    m = mask.to(d.device, torch.float64).unsqueeze(-1)            # (B, T, 1)
    d64 = d.double()
    S = (d64 * m).sum(1)
    Q = (d64.pow(2).sum(-1, keepdim=True) * m).sum((1, 2))
    Bn = (hb.double().pow(2).sum(-1, keepdim=True) * m).sum((1, 2))
    n = m.sum((1, 2))
    return {"S": S.cpu(), "Q": Q.cpu(), "n": n.cpu(), "B": Bn.cpu()}


def concat(parts: list[dict]) -> dict:
    """Stack per-batch accumulators into one per-doc record; drop empty docs."""
    out = {k: torch.cat([p[k] for p in parts], 0) for k in ["S", "Q", "n", "B"]}
    keep = out["n"] > 0
    return {k: v[keep] for k, v in out.items()}


def _summary(acc: dict) -> dict:
    """Pooled mean d, mean ||d||^2, mean ||h||^2 over all tokens in acc."""
    S, Q, n, B = (acc[k].double() for k in ["S", "Q", "n", "B"])
    N = n.sum()
    return {"mean": S.sum(0) / N, "meanQ": (Q.sum() / N).item(),
            "meanB": (B.sum() / N).item(), "N": int(N.item()), "docs": len(n)}


def r2_const(c: torch.Tensor, target: dict) -> dict:
    """R2 of a constant c on the token set summarised by `target` (_summary)."""
    c = c.double()
    mu, mq = target["mean"], target["meanQ"]
    dot = float(c @ mu)
    cc = float(c @ c)
    return {"r2": (2 * dot - cc) / mq,
            "cos": dot / max((cc * float(mu @ mu)) ** 0.5, 1e-30),
            "r2_best_scale": dot ** 2 / max(cc * mq, 1e-30)}


def constancy_stats(acc: dict, n_boot: int = N_BOOT, seed: int = SEED) -> dict:
    """Point estimate, doc-bootstrap CI, even/odd cross-validation, norms."""
    S, Q, n = acc["S"].double(), acc["Q"].double(), acc["n"].double()
    sm = _summary(acc)
    const = float(sm["mean"] @ sm["mean"]) / sm["meanQ"]

    # bootstrap over docs: counts (n_boot, n_docs) @ stacked per-doc stats
    nd = len(n)
    g = torch.Generator().manual_seed(seed)
    idx = torch.randint(nd, (n_boot, nd), generator=g)
    cnt = torch.zeros(n_boot, nd, dtype=torch.float64).scatter_add_(
        1, idx, torch.ones(n_boot, nd, dtype=torch.float64))
    Nb = cnt @ n
    Sb = (cnt @ S) / Nb[:, None]
    Qb = (cnt @ Q) / Nb
    boot = Sb.pow(2).sum(1) / Qb
    lo, hi = torch.quantile(boot, torch.tensor([0.025, 0.975], dtype=torch.float64)).tolist()

    # cross-validation: even / odd docs
    ev = torch.arange(nd) % 2 == 0
    A = _summary({k: acc[k][ev] for k in ["S", "Q", "n", "B"]})
    Bh = _summary({k: acc[k][~ev] for k in ["S", "Q", "n", "B"]})
    cv = 0.5 * (r2_const(A["mean"], Bh)["r2"] + r2_const(Bh["mean"], A)["r2"])

    rms_d, rms_h = sm["meanQ"] ** 0.5, sm["meanB"] ** 0.5
    return {"constancy": const, "ci_low": lo, "ci_high": hi, "cv": cv,
            "mean_norm": float(sm["mean"].norm()), "rms_d": rms_d,
            "rms_h": rms_h, "rms_ratio": rms_d / rms_h,
            "tokens": sm["N"], "docs": sm["docs"]}


def transfer_stats(accs: dict) -> dict:
    """accs: {set: acc}. R2 of one set's mean applied to another set."""
    sm = {k: _summary(v) for k, v in accs.items()}
    return {f"{s}->{t}": r2_const(sm[s]["mean"], sm[t])
            for s, t in TRANSFER_PAIRS if s in sm and t in sm}


def all_stats(acc_all: dict) -> tuple[dict, dict]:
    """acc_all: {layer: {set: acc}} -> (per-set stats, transfer stats)."""
    st = {L: {s: constancy_stats(a) for s, a in sets.items()} for L, sets in acc_all.items()}
    tr = {L: transfer_stats(sets) for L, sets in acc_all.items()}
    return st, tr


# ============================================================================
# printing
# ============================================================================
def print_tables(st: dict, tr: dict) -> None:
    layers = list(st)
    for s in SETS:
        if s not in st[layers[0]]:
            continue
        print(f"\n=== {s} ===")
        print(f"  {'layer':>5} {'const':>7} {'CI_lo':>7} {'CI_hi':>7} {'cv':>7}"
              f" {'|mean d|':>9} {'rms|d|':>9} {'rms d/h':>8} {'tokens':>8} {'docs':>5}")
        for L in layers:
            r = st[L][s]
            print(f"  {L:>5} {r['constancy']:>7.4f} {r['ci_low']:>7.4f} {r['ci_high']:>7.4f}"
                  f" {r['cv']:>7.4f} {r['mean_norm']:>9.4f} {r['rms_d']:>9.4f}"
                  f" {r['rms_ratio']:>8.4f} {r['tokens']:>8d} {r['docs']:>5d}")
    print("\n=== transfer: R2 of source-set mean d applied to target set ===")
    print("  (r2 = constant as is; cos = cos(c, target mean); best = R2 at best scalar multiple)")
    pairs = list(tr[layers[0]])
    print(f"  {'layer':>5}  " + "  ".join(f"{p:>30}" for p in pairs))
    print(f"  {'':>5}  " + "  ".join(f"{'r2 / cos / best':>30}" for _ in pairs))
    for L in layers:
        cells = [f"{tr[L][p]['r2']:>+8.3f} {tr[L][p]['cos']:>+7.3f} {tr[L][p]['r2_best_scale']:>7.3f}"
                 for p in pairs]
        print(f"  {L:>5}  " + "  ".join(f"{c:>30}" for c in cells))


# ============================================================================
# selftest
# ============================================================================
def selftest() -> None:
    g = torch.Generator().manual_seed(1)
    D, ndoc, T = 64, 400, 50
    mask = torch.ones(ndoc, T, dtype=torch.bool)
    mask[:, 0] = False                          # mimic position-0 exclusion
    hb = torch.randn(ndoc, T, D, generator=g)

    def run(d):
        parts = [accumulate(d[i:i + 16], hb[i:i + 16], mask[i:i + 16])
                 for i in range(0, ndoc, 16)]
        acc = concat(parts)
        return acc, constancy_stats(acc)

    c = torch.randn(D, generator=g)
    # (a) constant shift
    _, a = run(c.expand(ndoc, T, D).clone())
    print(f"(a) constant shift:            constancy={a['constancy']:.8f} cv={a['cv']:.8f}")
    assert abs(a["constancy"] - 1) < 1e-6, a
    # (b) iid zero-mean noise
    _, b = run(torch.randn(ndoc, T, D, generator=g))
    print(f"(b) iid noise:                 constancy={b['constancy']:.6f} "
          f"CI=[{b['ci_low']:.6f},{b['ci_high']:.6f}]")
    assert b["constancy"] < 0.01, b
    # (c) constant + noise with E||noise||^2 = ||c||^2
    noise = torch.randn(ndoc, T, D, generator=g) * (c.norm() / D ** 0.5)
    accc, cc = run(c + noise)
    print(f"(c) const + equal-power noise: constancy={cc['constancy']:.4f} "
          f"CI=[{cc['ci_low']:.4f},{cc['ci_high']:.4f}] cv={cc['cv']:.4f}")
    assert abs(cc["constancy"] - 0.5) < 0.02, cc
    # (d) cross-validated agrees
    assert abs(cc["cv"] - cc["constancy"]) < 0.02, cc
    assert cc["ci_low"] <= cc["constancy"] <= cc["ci_high"], cc
    # (e) transfer of a set's own mean = its constancy
    own = r2_const(_summary(accc)["mean"], _summary(accc))["r2"]
    print(f"(e) own-mean transfer R2={own:.8f} vs constancy={cc['constancy']:.8f}")
    assert abs(own - cc["constancy"]) < 1e-9
    # transfer_stats wiring: identical sets -> r2 equals constancy
    tr = transfer_stats({"unrel_early": accc, "indomain": accc})
    assert abs(tr["unrel_early->indomain"]["r2"] - cc["constancy"]) < 1e-9
    # rms ratio: hb ~ N(0, I_D) -> rms||h|| ~ sqrt(D)
    assert abs(cc["rms_h"] - D ** 0.5) / D ** 0.5 < 0.02, cc
    print("SELFTEST PASSED")


# ============================================================================
# real run
# ============================================================================
def main() -> None:
    ap = argparse.ArgumentParser()
    from targets import TARGETS
    ap.add_argument("--domain", choices=[k for k, v in TARGETS.items() if v["kind"] == "chat"])
    ap.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32")
    ap.add_argument("--n-docs", type=int, default=1000)
    ap.add_argument("--tag", default="")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return
    if a.domain is None:
        ap.error("--domain is required unless --selftest")

    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from design3 import DOM
    from finetune import encode

    if not torch.cuda.is_available():
        sys.exit("CUDA not available; refusing to run on CPU.")
    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16}[a.dtype]
    # Day 11: any chat target (rank/seed recipe variants); adapter and held-out
    # file from targets.py, identical to DOM for recipe / medical.
    cfg = {"adapter": TARGETS[a.domain]["adapter"], "held": TARGETS[a.domain]["held"]}
    if a.domain in DOM:
        assert DOM[a.domain]["adapter"] == cfg["adapter"] and DOM[a.domain]["held"] == cfg["held"]
    t0 = time.time()

    tok = AutoTokenizer.from_pretrained(INSTRUCT)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    try:
        base = AutoModelForCausalLM.from_pretrained(INSTRUCT, dtype=dtype)
        model = PeftModel.from_pretrained(base, str(ROOT / "models" / cfg["adapter"]))
        model.to("cuda").eval()
    except torch.OutOfMemoryError:
        sys.exit(f"CUDA OOM while loading the model in {a.dtype}. Try --dtype bfloat16.")
    # Inner Qwen2Model (LoRA layers injected in place): skips lm_head / logits.
    inner = model.get_base_model().model
    print(f"domain={a.domain} dtype={a.dtype} adapter={cfg['adapter']} "
          f"loaded in {time.time()-t0:.0f}s; mem {torch.cuda.memory_allocated()/2**30:.2f} GiB")
    torch.cuda.reset_peak_memory_stats()

    @torch.no_grad()
    def diffs(ids, attn):
        """Return {layer: (d, h_base)} float32 on GPU."""
        hf = inner(input_ids=ids, attention_mask=attn, output_hidden_states=True).hidden_states
        hf = {L: hf[L].float() for L in LAYERS}
        with model.disable_adapter():
            hb = inner(input_ids=ids, attention_mask=attn, output_hidden_states=True).hidden_states
        hb = {L: hb[L].float() for L in LAYERS}
        return {L: (hf[L] - hb[L], hb[L]) for L in LAYERS}

    parts = {L: {s: [] for s in SETS} for L in LAYERS}
    pos = torch.arange(SEQ)
    m_early = (pos >= 1) & (pos < EARLY_END)
    m_late = pos >= EARLY_END

    unrel = json.loads((DATA / "unrelated.json").read_text(encoding="utf-8"))[:a.n_docs]
    held = json.loads((DATA / cfg["held"]).read_text(encoding="utf-8"))
    n_unrel = n_ind = n_skip = 0
    try:
        t1 = time.time()
        for i in range(0, len(unrel), BATCH):
            enc = tok(unrel[i:i + BATCH], return_tensors="pt", padding=True,
                      truncation=True, max_length=SEQ)
            keep = enc.attention_mask.sum(1) >= SEQ
            if keep.sum() == 0:
                continue
            ids = enc.input_ids[keep].to("cuda")
            attn = enc.attention_mask[keep].to("cuda")
            n_unrel += ids.shape[0]
            dd = diffs(ids, attn)
            B = ids.shape[0]
            for L, (d, hb) in dd.items():
                parts[L]["unrel_early"].append(accumulate(d, hb, m_early.expand(B, SEQ)))
                parts[L]["unrel_late"].append(accumulate(d, hb, m_late.expand(B, SEQ)))
            del dd
        t_unrel = time.time() - t1
        print(f"unrelated: {n_unrel}/{len(unrel)} docs with >= {SEQ} tokens, {t_unrel:.0f}s")

        t1 = time.time()
        for ex in held:
            got = encode(tok, ex)
            if got is None:
                n_skip += 1
                continue
            ids, labels = got
            mask = labels != -100
            mask[0] = False
            if mask.sum() == 0:
                n_skip += 1
                continue
            ids = ids[None].to("cuda")
            dd = diffs(ids, torch.ones_like(ids))
            for L, (d, hb) in dd.items():
                parts[L]["indomain"].append(accumulate(d, hb, mask[None]))
            n_ind += 1
            del dd
        t_ind = time.time() - t1
        print(f"in-domain: {n_ind}/{len(held)} examples used ({n_skip} skipped), {t_ind:.0f}s")
    except torch.OutOfMemoryError:
        sys.exit(f"CUDA OOM during forward passes (dtype={a.dtype}, batch={BATCH}). "
                 f"Reduce BATCH in src/constancy.py or use --dtype bfloat16.")
    peak = torch.cuda.max_memory_allocated() / 2**30
    print(f"peak GPU memory allocated (forward passes): {peak:.2f} GiB")

    acc_all = {L: {s: concat(parts[L][s]) for s in SETS} for L in LAYERS}
    pt_path = RES / f"constancy_{a.domain}{a.tag}.pt"
    torch.save({"layers": LAYERS, "sets": SETS, "dtype": a.dtype, "domain": a.domain,
                "acc": {L: {s: {"S": v["S"].float(), "Q": v["Q"], "n": v["n"], "B": v["B"]}
                            for s, v in sets.items()} for L, sets in acc_all.items()}},
               pt_path)

    st, tr = all_stats(acc_all)
    print_tables(st, tr)
    anchor = st[14]["unrel_early"]["mean_norm"] if 14 in st else float("nan")
    print(f"\n>>> ||mean d|| at hidden_states[14], unrel_early = {anchor:.4f}"
          f"   (earlier bf16 anchor for recipe: ~1.29)")

    out = {"domain": a.domain, "dtype": a.dtype, "adapter": cfg["adapter"],
           "layers": LAYERS, "sets": SETS, "n_docs_requested": a.n_docs,
           "n_docs_kept_unrelated": n_unrel, "n_indomain_used": n_ind,
           "n_indomain_skipped": n_skip, "seq_len_unrelated": SEQ,
           "early_positions": [1, EARLY_END - 1], "late_positions": [EARLY_END, SEQ - 1],
           "n_boot": N_BOOT, "seed": SEED,
           "token_counts": {str(L): {s: st[L][s]["tokens"] for s in SETS} for L in LAYERS},
           "stats": {str(L): st[L] for L in LAYERS},
           "transfer": {str(L): tr[L] for L in LAYERS},
           "anchor_l14_unrel_early_mean_norm": anchor,
           "peak_gpu_gib": peak, "seconds": {"unrelated": t_unrel, "indomain": t_ind,
                                             "total": time.time() - t0},
           "note": "hidden_states[28] is post-final-norm in transformers 5.x"}
    js_path = RES / f"constancy_{a.domain}{a.tag}.json"
    js_path.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"saved -> {js_path.name}, {pt_path.name}  (total {time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
