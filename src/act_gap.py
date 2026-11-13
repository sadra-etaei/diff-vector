"""Per-token activation gap closed by the multi-layer constants.

design3_multilayer.py calibrates one constant c[l] per decoder layer and adds it
to the BASE model (MultiSteer); that steered base closes ~70% of KL(ft || base)
on unrelated text. constancy.py found the per-token activation difference
ft - base is only ~10% constant. This script asks: with the constants
installed, how much of the PER-TOKEN activation gap does the steered base close,
hidden state by hidden state?

For a token t at hidden_states[k] (output of decoder layer k-1; NOTE: index 28
is AFTER the final norm in transformers 5.x):
    d_t = h_ft      - h_base     (adapter on, no steering) - (adapter off, no steering)
    s_t = h_steered - h_base     (adapter off, constants)  - (adapter off, no steering)
Pooled over a token set, with r = d - mean_d, q = s - mean_s:
    constancy     = |mean_d|^2 / E|d|^2
    gap_closed    = 1 - E|d-s|^2 / E|d|^2 = (2E<d,s> - E|s|^2) / E|d|^2
    mean_mismatch = |mean_d - mean_s| / |mean_d|
    induced_ratio = E|q|^2 / E|r|^2
    rq_corr       = E<r,q> / sqrt(E|r|^2 E|q|^2)
    r_explained   = 1 - E|r-q|^2 / E|r|^2 = (2E<r,q> - E|q|^2) / E|r|^2
Sufficient statistics per (set, condition, hidden state, doc parity):
sum d, sum s, sum|d|^2, sum|s|^2, sum<d,s>, n -- float64 on CPU.

Model path is exactly design3_multilayer's: bf16, PeftModel with the adapter
unmerged, base = disable_adapter(). Forward calls are model(..., logits_to_keep=1,
output_hidden_states=True), as in design3_multilayer.layer_means. MultiSteer is
created before the first forward pass, so its hooks run before transformers'
lazily installed hidden-state capture hooks (checked by sanity check 1).

Day 4 (2026-11-13): --target takes any target in targets.py (--domain kept as
an alias). Attention-sink tokens are removed from every token set by base-model
norm at the middle layer (Day 2b: three position-1 sinks carried a third of the
squared difference on one set). docs targets (ADL organisms) use the first 128
tokens of 100 odd-indexed organism documents as in-domain text. --cross names
the targets whose unrelated-text constants are the specificity control
(default: recipe <-> medical; for an organism, the other organisms on the same
base model whose constants exist).

Run (from the repo root):
  python src/act_gap.py --selftest
  python src/act_gap.py --target recipe
  python src/act_gap.py --target cake_bake
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
DATA, RES = ROOT / "data", ROOT / "results"
INSTRUCT = str(ROOT / "models" / "Qwen2.5-1.5B-Instruct")

HS = [2, 8, 14, 20, 26, 28]                 # reported hidden states
HS_SANITY = [1, 7, 14, 21, 28]              # = 1 + {0, 6, 13, 20, 27}, as in the calibration log
HS_ALL = sorted(set(HS) | set(HS_SANITY))
SETS = ["cal_early", "heldout_early", "ood", "indomain"]
ALPHAS = [0.5, 1, 2]
KEYS = ["Sd", "Ss", "Qd", "Qs", "P", "n"]
HALVES = ["even", "odd"]


# ============================================================================
# pure functions: accumulation and statistics (used by the selftest too)
# ============================================================================
def row_sums(x: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """x: (B, T, D) float32, mask (B, T) bool -> per-row sum x (B, D), sum |x|^2 (B,), float64 CPU."""
    m = mask.to(x.device, torch.float64).unsqueeze(-1)
    x64 = x.double()
    return (x64 * m).sum(1).cpu(), (x64.pow(2).sum(-1, keepdim=True) * m).sum((1, 2)).cpu()


def batch_rows(d: torch.Tensor, s: torch.Tensor, mask: torch.Tensor, dsums=None) -> dict:
    """Per-row (= per-doc) sufficient statistics for one batch. dsums: cached row_sums(d)."""
    Sd, Qd = dsums if dsums is not None else row_sums(d, mask)
    Ss, Qs = row_sums(s, mask)
    m = mask.to(d.device, torch.float64)
    P = ((d.double() * s.double()).sum(-1) * m).sum(1).cpu()
    n = m.sum(1).cpu()
    return {"Sd": Sd, "Ss": Ss, "Qd": Qd, "Qs": Qs, "P": P, "n": n}


def new_acc(D: int) -> dict:
    return {h: {"Sd": torch.zeros(D, dtype=torch.float64), "Ss": torch.zeros(D, dtype=torch.float64),
                "Qd": 0.0, "Qs": 0.0, "P": 0.0, "n": 0.0} for h in HALVES}


def add_rows(acc: dict, rows: dict, doc_ids: list[int]) -> None:
    """Add per-row statistics into the even / odd accumulators by doc index parity."""
    for j, di in enumerate(doc_ids):
        if rows["n"][j] == 0:
            continue
        a = acc[HALVES[di % 2]]
        a["Sd"] += rows["Sd"][j]
        a["Ss"] += rows["Ss"][j]
        for k in ["Qd", "Qs", "P", "n"]:
            a[k] += float(rows[k][j])


def pool(acc: dict) -> dict:
    """Full-set accumulator = sum of the two halves."""
    return {k: acc["even"][k] + acc["odd"][k] for k in KEYS}


def stats(a: dict) -> dict:
    """All statistics from one accumulator (sums, not means)."""
    N = a["n"]
    if N == 0:
        return {"tokens": 0}
    md, ms = a["Sd"] / N, a["Ss"] / N
    Edd, Ess, Eds = a["Qd"] / N, a["Qs"] / N, a["P"] / N
    mdmd, msms, mdms = float(md @ md), float(ms @ ms), float(md @ ms)
    Err, Eqq, Erq = Edd - mdmd, Ess - msms, Eds - mdms
    nan = float("nan")
    return {
        "constancy": mdmd / Edd,
        "gap_closed": (2 * Eds - Ess) / Edd,
        "mean_mismatch": float((md - ms).norm()) / math.sqrt(mdmd) if mdmd > 0 else nan,
        "induced_ratio": Eqq / Err if Err > 0 else nan,
        "rq_corr": Erq / math.sqrt(Err * Eqq) if Err > 0 and Eqq > 1e-30 * max(Err, 1) else nan,
        "r_explained": (2 * Erq - Eqq) / Err if Err > 0 else nan,
        "mean_d_norm": math.sqrt(mdmd), "mean_s_norm": math.sqrt(msms),
        "mean_diff_norm": float((md - ms).norm()),
        "E_d2": Edd, "E_s2": Ess, "E_ds": Eds, "E_r2": Err, "E_q2": Eqq, "E_rq": Erq,
        "tokens": int(round(N)),
    }


def stats_halves(acc: dict) -> dict:
    out = {"full": stats(pool(acc))}
    for h in HALVES:
        out[h] = stats(acc[h])
        out[h]["docs"] = None
    return out


def identity_rhs(st: dict) -> float:
    """[|mean_d|^2 - |mean_d - mean_s|^2 + 2E<r,q> - E|q|^2] / E|d|^2"""
    return (st["mean_d_norm"] ** 2 - st["mean_diff_norm"] ** 2
            + 2 * st["E_rq"] - st["E_q2"]) / st["E_d2"]


# ============================================================================
# selftest
# ============================================================================
def selftest() -> None:
    g = torch.Generator().manual_seed(1)
    D, ndoc, T, B = 64, 400, 30, 16
    mask = torch.rand(ndoc, T, generator=g) > 0.2       # ragged, like padding
    mask[:, 0] = False

    c = torch.randn(D, generator=g)
    noise_r = torch.randn(ndoc, T, D, generator=g) * (0.8 * c.norm() / D ** 0.5)
    d = c + noise_r                                      # constancy ~ 1/(1+0.64)

    def run(d, s):
        acc = new_acc(D)
        for i in range(0, ndoc, B):
            add_rows(acc, batch_rows(d[i:i + B], s[i:i + B], mask[i:i + B]),
                     list(range(i, min(i + B, ndoc))))
        return acc, stats(pool(acc))

    # exact mean of d over masked tokens, for the "constant" cases
    m = mask.unsqueeze(-1).double()
    mean_d = ((d.double() * m).sum((0, 1)) / m.sum()).float()
    r = d - mean_d

    _, a = run(d, d.clone())
    print(f"(a) s = d:             gap_closed={a['gap_closed']:.10f}  r_explained={a['r_explained']:.10f}")
    assert abs(a["gap_closed"] - 1) < 1e-9 and abs(a["r_explained"] - 1) < 1e-9

    _, b = run(d, mean_d.expand(ndoc, T, D).clone())
    print(f"(b) s = mean_d:        gap_closed={b['gap_closed']:.8f}  constancy={b['constancy']:.8f}"
          f"  induced_ratio={b['induced_ratio']:.2e}  mismatch={b['mean_mismatch']:.2e}")
    assert abs(b["gap_closed"] - b["constancy"]) < 1e-6
    assert abs(b["induced_ratio"]) < 1e-6 and b["mean_mismatch"] < 1e-6

    _, c0 = run(d, torch.zeros_like(d))
    print(f"(c) s = 0:             gap_closed={c0['gap_closed']:.10f}")
    assert abs(c0["gap_closed"]) < 1e-12

    noise = torch.randn(ndoc, T, D, generator=g) * (0.8 * c.norm() / D ** 0.5)  # same power as r
    _, dd = run(d, mean_d + noise)
    want = dd["constancy"] - (1 - dd["constancy"])
    print(f"(d) s = mean_d+noise:  gap_closed={dd['gap_closed']:.4f}  expected~{want:.4f}"
          f"  rq_corr={dd['rq_corr']:+.4f}  induced_ratio={dd['induced_ratio']:.4f}")
    assert abs(dd["gap_closed"] - want) < 0.02 and abs(dd["rq_corr"]) < 0.02
    assert abs(dd["induced_ratio"] - 1) < 0.03

    worst = 0.0
    for trial in range(5):
        s = (torch.randn(ndoc, T, D, generator=g) * torch.rand(1, generator=g)
             + 0.6 * r * torch.rand(1, generator=g) + torch.randn(D, generator=g))
        _, e = run(d, s)
        worst = max(worst, abs(e["gap_closed"] - identity_rhs(e)),
                    abs(1 - e["E_r2"] / e["E_d2"] * (1 - e["r_explained"])
                        - e["mean_diff_norm"] ** 2 / e["E_d2"]
                        - e["gap_closed"]))
    print(f"(e) identity on random data: max |lhs - rhs| = {worst:.2e}")
    assert worst < 1e-9

    acc, full = run(d, mean_d + noise)
    ev, od = stats(acc["even"]), stats(acc["odd"])
    # pool halves by hand from their own statistics and compare with the full set
    Ne, No = ev["tokens"], od["tokens"]
    Edd = (Ne * ev["E_d2"] + No * od["E_d2"]) / (Ne + No)
    Eds = (Ne * ev["E_ds"] + No * od["E_ds"]) / (Ne + No)
    Ess = (Ne * ev["E_s2"] + No * od["E_s2"]) / (Ne + No)
    md = (acc["even"]["Sd"] + acc["odd"]["Sd"]) / (Ne + No)
    err = max(abs((2 * Eds - Ess) / Edd - full["gap_closed"]),
              abs(float(md @ md) / Edd - full["constancy"]),
              abs(Ne + No - full["tokens"]))
    print(f"(f) halves pool to full: tokens {Ne}+{No}={full['tokens']}, max err {err:.2e};"
          f"  gap_closed even/odd/full = {ev['gap_closed']:.4f}/{od['gap_closed']:.4f}/{full['gap_closed']:.4f}")
    assert err < 1e-9
    assert int(mask.sum()) == full["tokens"]
    print("SELFTEST PASSED")


# ============================================================================
# real run
# ============================================================================
def main() -> None:
    ap = argparse.ArgumentParser()
    from targets import TARGETS
    ap.add_argument("--target", choices=sorted(TARGETS))
    ap.add_argument("--domain", choices=sorted(TARGETS), help="alias of --target")
    ap.add_argument("--cross", nargs="*", default=None,
                    help="targets whose unrelated-text constants are the cross control")
    ap.add_argument("--n-docs", type=int, default=None, help="cap per text set (smoke tests)")
    ap.add_argument("--tag", default="")
    ap.add_argument("--selftest", action="store_true")
    # Subset of token sets (default all). Used to verify sanity check 1 on the full
    # cal_early set cheaply.
    ap.add_argument("--sets", nargs="+", choices=SETS, default=SETS)
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return
    target = a.target or a.domain
    if target is None:
        ap.error("--target is required unless --selftest")
    if not torch.cuda.is_available():
        sys.exit("CUDA not available; refusing to run on CPU.")

    from design3 import ood_docs
    from design3_multilayer import N_CAL, N_IND_DOCS, N_POS, OOD_LEN
    from finetune import encode
    from targets import DEVICE, MultiSteer, load, sdf_docs, sink_mask

    cfg = TARGETS[target]
    cap = (lambda xs: xs if a.n_docs is None else xs[:a.n_docs])
    t0 = time.time()

    # ---- model: exactly design3_multilayer's loading path --------------------
    tok, model, layers = load(target)
    L = len(layers)
    assert L == 28, "HS lists assume 28 decoder layers"
    # Must exist BEFORE the first forward pass (see module docstring).
    steer = MultiSteer(layers)
    D = model.config.hidden_size
    print(f"target={target} adapter={cfg['adapter']} layers={L} d={D} "
          f"loaded in {time.time()-t0:.0f}s; mem {torch.cuda.memory_allocated()/2**30:.2f} GiB")

    # ---- constants ------------------------------------------------------------
    # "cross" = the OTHER finetune's unrelated-text constants. Specificity
    # control: if any finetune-shaped constant closed the gap equally well, the
    # match would say nothing about THIS finetune.
    if a.cross is None:
        same_base = [t for t, c in TARGETS.items() if t != target and c["base"] == cfg["base"]]
        a.cross = [t for t in same_base if (RES / f"multilayer_consts_{t}.pt").exists()]
    consts = {}
    for name, stem in ([("unrel", f"{target}"), ("ind", f"{target}_calind")]
                       + [(f"cross_{t}", t) for t in a.cross]):
        p = RES / f"multilayer_consts_{stem}.pt"
        if not p.exists():
            print(f"WARNING: {p.name} missing; skipping the '{name}' constants")
            continue
        K = torch.load(p)
        if K["c"].shape[1] != D:
            print(f"WARNING: {p.name} has width {K['c'].shape[1]} != {D}; skipping")
            continue
        consts[name] = {"c": K["c"], "c_rand": K["c_rand"], "file": p.name}
        assert K["c"].shape == (L, D), K["c"].shape
    conds = []                       # (cond name, const set, key, alpha)
    for name in consts:
        conds += [(f"{name}:c@{al}", name, "c", al) for al in ALPHAS]
        conds.append((f"{name}:rand@1", name, "c_rand", 1))
    print("conditions:", ", ".join(c[0] for c in conds))

    # ---- token sets -------------------------------------------------------------
    unrel = json.loads((DATA / "unrelated.json").read_text(encoding="utf-8"))

    def early_batches(docs):
        """design3_multilayer's `cal` batches: 50 docs per call, max_length 8,
        keep docs with 8 tokens, mask positions 1-7."""
        out = []
        for i in range(0, len(docs), 50):
            enc = tok(docs[i:i + 50], return_tensors="pt", padding=True,
                      truncation=True, max_length=N_POS)
            keep = enc.attention_mask.sum(1) >= N_POS
            if keep.sum() == 0:
                continue
            ids = enc.input_ids[keep]
            mask = torch.zeros_like(ids, dtype=torch.bool)
            mask[:, 1:N_POS] = True
            out.append((ids, enc.attention_mask[keep], mask))
        return out

    batches = {"cal_early": early_batches(cap(unrel[0:N_CAL])),
               "heldout_early": early_batches(cap(unrel[N_CAL:2 * N_CAL]))}
    docs = cap(ood_docs(set(unrel)))
    ood = []
    for i in range(0, len(docs), 4):
        enc = tok(docs[i:i + 4], return_tensors="pt", padding=True,
                  truncation=True, max_length=OOD_LEN)
        mask = enc.attention_mask.bool().clone()
        mask[:, 0] = False
        ood.append((enc.input_ids, enc.attention_mask, mask))
    batches["ood"] = ood
    ind, n_ind_skip = [], 0
    held = (cap(json.loads((DATA / cfg["held"]).read_text(encoding="utf-8")))
            if cfg["kind"] == "chat" else [])
    if cfg["kind"] == "docs":
        docs = cap(sdf_docs(target, 1)[:N_IND_DOCS])
        for i in range(0, len(docs), 4):
            enc = tok(docs[i:i + 4], return_tensors="pt", padding=True,
                      truncation=True, max_length=OOD_LEN)
            mask = enc.attention_mask.bool().clone()
            mask[:, 0] = False
            ind.append((enc.input_ids, enc.attention_mask, mask))
    for ex in held:
        got = encode(tok, ex)
        if got is None:
            n_ind_skip += 1
            continue
        ids, labels = got
        mask = torch.zeros_like(labels, dtype=torch.bool)
        mask[:-1] = labels[1:] != -100
        mask[0] = False
        if mask.sum() == 0:
            n_ind_skip += 1
            continue
        ind.append((ids[None], torch.ones_like(ids)[None], mask[None]))
    batches["indomain"] = ind
    sets = [s for s in SETS if s in a.sets]
    n_docs ={s: sum(b[0].shape[0] for b in batches[s]) for s in sets}
    n_tok = {s: sum(int(b[2].sum()) for b in batches[s]) for s in sets}
    print("token sets:", "  ".join(f"{s}: {n_docs[s]} docs / {n_tok[s]} tok" for s in sets),
          f"(indomain skipped {n_ind_skip})")

    # ---- forward ----------------------------------------------------------------
    @torch.no_grad()
    def hidden(ids, attn, ft, vecs=None, alpha=1.0):
        steer.set(vecs, alpha)
        try:
            kw = dict(input_ids=ids, attention_mask=attn, output_hidden_states=True,
                      logits_to_keep=1)
            if ft:
                hs = model(**kw).hidden_states
            else:
                with model.disable_adapter():
                    hs = model(**kw).hidden_states
        finally:
            steer.set(None)
        assert len(hs) == L + 1
        return {k: hs[k].float() for k in HS_ALL}

    acc = {s: {c[0]: {k: new_acc(D) for k in HS_ALL} for c in conds} for s in sets}
    sanity = {}
    torch.cuda.reset_peak_memory_stats()
    secs = {}
    n_sink = {s: 0 for s in sets}
    for s in sets:
        t1 = time.time()
        doc0 = 0
        for bi, (ids, attn, mask) in enumerate(batches[s]):
            ids, attn = ids.to(DEVICE), attn.to(DEVICE)
            B = ids.shape[0]
            doc_ids = list(range(doc0, doc0 + B))
            doc0 += B
            h_ft = hidden(ids, attn, True)
            h_b = hidden(ids, attn, False)
            # attention sinks out, by base-model norm at hidden state 14 (Day 2b)
            sink = sink_mask(h_b[14], attn.bool()) & mask.to(DEVICE)
            n_sink[s] += int(sink.sum())
            mask = mask & ~sink.cpu()
            if s == sets[0] and bi == 0:
                # sanity 3: steering None -> s identically 0 (base pass is deterministic)
                h_b2 = hidden(ids, attn, False, None)
                mx = max(float((h_b2[k] - h_b[k]).abs().max()) for k in HS_ALL)
                sanity["s_zero_when_unsteered_max_abs"] = mx
                print(f"\n[SANITY 3] steering=None, first {s} batch: max |s| = {mx:.3e}"
                      f"  -> {'OK (exactly zero)' if mx == 0 else 'NOT ZERO'}")
                del h_b2
            d = {k: h_ft[k] - h_b[k] for k in HS_ALL}
            del h_ft
            dsums = {k: row_sums(d[k], mask) for k in HS_ALL}
            for cname, cset, key, al in conds:
                h_s = hidden(ids, attn, False, consts[cset][key], al)
                for k in HS_ALL:
                    add_rows(acc[s][cname][k],
                             batch_rows(d[k], h_s[k] - h_b[k], mask, dsums[k]), doc_ids)
                del h_s
            del d, h_b
        secs[s] = time.time() - t1
        print(f"{s}: {len(batches[s])} batches, {secs[s]:.0f}s, attention sinks dropped: {n_sink[s]}")
    peak = torch.cuda.max_memory_allocated() / 2**30
    print(f"peak GPU memory allocated (forward passes): {peak:.2f} GiB")

    # ---- statistics -------------------------------------------------------------
    st = {s: {c[0]: {k: stats_halves(acc[s][c[0]][k]) for k in HS_ALL} for c in conds}
          for s in sets}
    for s in sets:          # doc counts per half
        for c in conds:
            for k in HS_ALL:
                st[s][c[0]][k]["even"]["docs"] = (n_docs[s] + 1) // 2
                st[s][c[0]][k]["odd"]["docs"] = n_docs[s] // 2
                st[s][c[0]][k]["full"]["docs"] = n_docs[s]

    # ---- sanity checks 1, 2 -----------------------------------------------------
    if "unrel" in consts and "cal_early" in sets:
        print("\n" + "=" * 78)
        print("[SANITY 1/2] cal_early, 'unrel' matched constants, alpha 1")
        print("  compare with the calibration log of design3_multilayer.py for this target "
              "(hs28 here is post-norm, the log's last layer is raw since Day 4)")
        print(f"  {'hs':>4} {'|mean_d|':>10} {'|mean_s|':>10} {'|mean_d-mean_s|':>16}")
        rows = {}
        for k in HS_SANITY:
            f = st["cal_early"]["unrel:c@1"][k]["full"]
            rows[k] = {"mean_d_norm": f["mean_d_norm"], "mean_s_norm": f["mean_s_norm"],
                       "mean_diff_norm": f["mean_diff_norm"]}
            print(f"  {k:>4} {f['mean_d_norm']:>10.4f} {f['mean_s_norm']:>10.4f} {f['mean_diff_norm']:>16.4f}")
        sanity["cal_early_unrel_c1"] = rows
        mm14, md14 = rows[14]["mean_diff_norm"], rows[14]["mean_d_norm"]
        ok1 = mm14 < 0.05 * md14     # relative: organisms' diffs are ~10x ours
        sanity["check1_mismatch_hs14"] = mm14
        sanity["check1_pass"] = ok1
        sanity["check2_mean_d_hs14"] = md14
        if ok1:
            print(f"  CHECK 1 OK: |mean_d - mean_s| at hs14 = {mm14:.4f} < 5% of |mean_d| "
                  f"(vs |mean_d| {md14:.4f}): captured hidden states include the steering")
        else:
            print("  " + "!" * 74)
            print(f"  WARNING: |mean_d - mean_s| at hs14 = {mm14:.4f} >= 5% of |mean_d|. The steering hooks may NOT")
            print("  be reflected in the captured hidden states. Results below are suspect.")
            print("  " + "!" * 74)
        print(f"  CHECK 2: |mean_d| at hs14 = {md14:.4f}  (recipe ~1.28-1.30; cake_bake ~13.9)")
        print("=" * 78)

    # ---- tables -----------------------------------------------------------------
    for s in sets:
        for cset in consts:
            print(f"\n=== {s}  ({n_docs[s]} docs, {n_tok[s]} tokens)   constants: {cset} "
                  f"({consts[cset]['file']}) ===")
            hdr = (f"  {'hs':>3} {'const':>7} " + " ".join(f"{'gc@'+str(al):>7}" for al in ALPHAS)
                   + f" {'gc_rnd':>7} | {'mism':>6} {'ind_rat':>7} {'rq_cor':>7} {'r_expl':>7}"
                   + f" | {'gc1_ev':>7} {'gc1_od':>7}")
            print(hdr)
            for k in HS:
                f1 = st[s][f"{cset}:c@1"][k]
                gcs = [st[s][f"{cset}:c@{al}"][k]["full"]["gap_closed"] for al in ALPHAS]
                gr = st[s][f"{cset}:rand@1"][k]["full"]["gap_closed"]
                f = f1["full"]
                print(f"  {k:>3} {f['constancy']:>7.4f} " + " ".join(f"{x:>+7.3f}" for x in gcs)
                      + f" {gr:>+7.3f} | {f['mean_mismatch']:>6.3f} {f['induced_ratio']:>7.3f}"
                      f" {f['rq_corr']:>+7.3f} {f['r_explained']:>+7.3f}"
                      f" | {f1['even']['gap_closed']:>+7.3f} {f1['odd']['gap_closed']:>+7.3f}")
    print("\n  const = |mean d|^2/E|d|^2; gc = gap_closed = 1 - E|d-s|^2/E|d|^2; gc_rnd = random-direction"
          "\n  control at alpha 1; mism/ind_rat/rq_cor/r_expl at alpha 1; gc1_ev/od = alpha-1 gap_closed"
          "\n  on even/odd docs. hs28 is post-final-norm.")

    # ---- save -------------------------------------------------------------------
    out = {"target": target, "adapter": cfg["adapter"], "sinks_dropped": n_sink, "dtype": "bfloat16", "adapter_merged": False,
           "hidden_states": HS_ALL, "hidden_states_reported": HS, "sets": sets,
           "conditions": [{"name": c[0], "consts": c[1], "key": c[2], "alpha": c[3]} for c in conds],
           "const_files": {k: v["file"] for k, v in consts.items()},
           "n_docs_cap": a.n_docs, "docs": n_docs, "tokens": n_tok,
           "indomain_skipped": n_ind_skip, "sanity": sanity,
           "stats": {s: {c: {str(k): v for k, v in byk.items()} for c, byk in bys.items()}
                     for s, bys in st.items()},
           "peak_gpu_gib": peak, "seconds": {**secs, "total": time.time() - t0},
           "forward_call": "PeftModel(..., output_hidden_states=True, logits_to_keep=1)",
           "note": "hidden_states[28] is post-final-norm in transformers 5.x; "
                   "position 0 and padding excluded everywhere"}
    path = RES / f"act_gap_{target}{a.tag}.json"
    path.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"saved -> {path.name}  (total {time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
