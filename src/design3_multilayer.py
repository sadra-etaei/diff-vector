"""Design 3b — multi-layer sufficiency. Give the bias-term hypothesis its best shot.

design3.py added v_diff at ONE layer (14) and closed 10-34% of the KL gap. The
standing objection: the finetune changed all 28 layers, so a single-layer
constant was never a fair test. Here a constant is added at EVERY layer.

Three steering schemes, in increasing order of fairness to the hypothesis:

  single     v_diff[14] at layer 14 only. The old result, re-run as baseline.

  increment  at layer l add d[l] = v_diff[l] - v_diff[l-1].
             Adding the full v_diff[l] at every layer would DOUBLE COUNT: the
             residual stream carries layer 5's addition into layer 6, and
             v_diff[6] already contains what changed in layers 1-5. So only the
             new part is added at each layer. This is open-loop: it assumes later
             layers respond to the additions as they would have anyway.

  matched    at layer l add c[l], chosen so the steered base model's MEAN
             activation equals the finetuned model's mean at layer l. Calibrated
             layer by layer (closed loop): c[l] is measured with c[0..l-1] already
             installed, so it absorbs whatever the earlier additions did to this
             layer. After calibration the two models agree in mean at every
             layer, by construction. This is the strongest constant-per-layer
             intervention the probe corpus can define.

For `matched`, each c[l] is also split into its projection on that layer's topic
direction and the remainder, plus a per-layer norm-matched random control, so
the layer-14 decomposition result can be checked in the all-layer setting.

Conventions kept from design3.py so numbers are comparable: constants are
estimated from token positions 1-7 of unrelated text (position 0 is the
attention sink) and added at all positions; out-of-domain KL on 100 wikitext
docs disjoint from the probe corpus. In-domain KL: for chat targets (our
recipe / medical LoRAs) the assistant tokens of 40 held-out examples; for docs
targets (ADL's organisms) the first 128 tokens of 100 odd-indexed organism
documents.

Day 4 changes (2026-11-13), all found on Days 2-3:
  * Targets come from targets.py (--target), so ADL's Qwen3-1.7B organisms run
    through the same code. --domain is kept as an alias.
  * Calibration reads each layer's RAW output through the steering hook. Before,
    it read hidden_states[l+1], and hidden_states[28] is AFTER the final norm, so
    the last layer's constant (added before the norm) was fitted to the wrong
    target. Layers 0..26 are unchanged (hook output == hidden_states[l+1]).
  * Attention sinks are excluded from calibration by activation NORM, not only
    position 0 (Day 2b found sinks at position 1 in a few docs).
  * Topic directions use the same sink-excluding per-doc means (targets.doc_means).
  * Constants calibrated before Day 4 are still loaded from their cache files if
    present; delete or --tag to recalibrate. The cache file records the version.

Not done here, on purpose: a separate constant for position 0. Its difference
is ~90x larger and every token attends to it (roadmap Day 18).

Run:  python src/design3_multilayer.py --target cake_bake --stage ind   (then ood, gen)

--cal indomain fits the constants to in-domain TRAINING text instead of
unrelated text (chat: assistant-predicting positions of 200 training examples;
docs: positions 1-127 of 200 even-indexed organism documents). Outputs go to
separate *_calind{tag} files/keys. --tag X with the default calibration
recalibrates from scratch into separately named outputs and compares the new
constants with the cached ones (regression check of the calibration code).
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch
import torch.nn.functional as F

from design3 import ood_docs, unit
from finetune import encode
from targets import (DEVICE, DTYPE, TARGETS, MultiSteer, corpus, doc_means, load,  # noqa: F401
                     sdf_docs, sink_mask)

ROOT = Path(__file__).resolve().parent.parent
DATA, RES = ROOT / "data", ROOT / "results"

ALPHAS = [0.5, 1, 2, 4]
N_CAL, N_POS = 1000, 8          # calibration docs; positions 0-7 (causal => exact)
N_IND, N_OOD, OOD_LEN, WINDOW = 40, 100, 128, 128
N_IND_DOCS = 100                # docs targets: odd-indexed organism docs for in-domain KL
N_CAL_IND, CAL_BS = 200, 8      # --cal indomain: training examples / docs; padded batch size
CALIB_VERSION = 2               # 1 = hidden_states target, position-0 mask only; 2 = Day 4


def pad_batches(seqs, bs, pad_id):
    """seqs: list of (ids, mask) 1-D CPU tensors of varying length. Returns padded
    (ids, attn, mask) batches, sorted by length so little compute goes on padding.
    Right padding with the mask False on padding, so padded positions never enter
    a mean (and, the model being causal, never affect real positions)."""
    seqs = sorted(seqs, key=lambda s: len(s[0]))
    out = []
    for i in range(0, len(seqs), bs):
        chunk = seqs[i:i + bs]
        T = max(len(s[0]) for s in chunk)
        ids = torch.full((len(chunk), T), pad_id, dtype=torch.long)
        attn = torch.zeros(len(chunk), T, dtype=torch.long)
        mask = torch.zeros(len(chunk), T, dtype=torch.bool)
        for j, (s_ids, s_mask) in enumerate(chunk):
            ids[j, :len(s_ids)] = s_ids
            attn[j, :len(s_ids)] = 1
            mask[j, :len(s_ids)] = s_mask
        out.append((ids, attn, mask))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", choices=sorted(TARGETS))
    ap.add_argument("--domain", choices=sorted(TARGETS), help="alias of --target")
    # Each stage is its own process: background jobs are stopped after 30 min,
    # and the first all-in-one run was killed mid-generation with nothing saved.
    ap.add_argument("--stage", choices=["cal", "ind", "ood", "gen"], required=True)
    # Extra closed-loop calibration sweeps: each measures the shift actually
    # achieved and tops up the rest.
    ap.add_argument("--refine", type=int, default=0)
    ap.add_argument("--cal", choices=["unrelated", "indomain"], default="unrelated")
    # Suffix for constants file and result keys, so a run can recalibrate from
    # scratch without reading or overwriting the cached constants.
    ap.add_argument("--tag", default="")
    ap.add_argument("--rand-seeds", type=int, default=1,
                    help="number of norm-matched random controls (seed 0 is the cached one)")
    # Cross-target control: evaluate OTHER targets' unrelated-text matched
    # constants on this target (same base model). If they close as much as this
    # target's own constants, the "bias" is not specific to this finetune.
    # Schemes: own matched + each cross target + random; result keys get "_cross".
    ap.add_argument("--cross", nargs="+", default=None)
    a = ap.parse_args()
    target = a.target or a.domain
    if target is None:
        ap.error("--target is required")
    cfg = TARGETS[target]
    if cfg["own"] is None:
        ap.error(f"{target}: topic corpora not built yet (targets.py 'own' is None)")
    sfx = ("_calind" if a.cal == "indomain" else "") + a.tag
    if a.refine and sfx:
        ap.error("--refine is only supported with --cal unrelated and no --tag")
    out_path = RES / f"design3_multilayer_{target}.json"
    consts_path = RES / f"multilayer_consts_{target}{sfx}.pt"
    cal_note = ("" if not sfx else
                "  [constants fit on " + ("in-domain TRAIN text" if a.cal == "indomain"
                                          else "unrelated wikitext pos 1-7") + f", key sfx '{sfx}']")

    def save(update: dict) -> None:
        """Merge into the results file after every stage, so nothing is lost."""
        cur = json.loads(out_path.read_text()) if out_path.exists() else {}
        cur.update(update)
        out_path.write_text(json.dumps(cur, indent=1))

    tok, model, layers = load(target)
    L = len(layers)
    D = model.config.hidden_size
    MID = L // 2 - 1                # decoder layer whose output is hidden_states[L//2]
    steer = MultiSteer(layers)      # before the first forward pass
    print(f"target={target}  base={cfg['base'].name}  adapter={cfg['adapter']}  "
          f"decoder layers={L}  width={D}")

    unrel = corpus("unrelated")

    @torch.no_grad()
    def raw_outputs(ids, attn, ft):
        """Raw output of every decoder layer (after any installed steering)."""
        outs = [None] * L
        steer.record = lambda j, o: outs.__setitem__(j, o)
        try:
            kw = dict(input_ids=ids, attention_mask=attn, logits_to_keep=1)
            if ft:
                model(**kw)
            else:
                with model.disable_adapter():
                    model(**kw)
        finally:
            steer.record = None
        return outs

    @torch.no_grad()
    def drop_sinks(batches):
        """Remove attention-sink tokens from each batch's mask, by base-model norm
        at the middle layer (unsteered). Returns the new batches and the count."""
        out, n = [], 0
        vecs, steer.vecs = steer.vecs, [None] * L
        for ids, attn, mask in batches:
            h = raw_outputs(ids, attn, ft=False)[MID]
            sink = sink_mask(h, attn.bool()) & mask
            n += int(sink.sum())
            out.append((ids, attn, mask & ~sink))
        steer.vecs = vecs
        return out, n

    def needs_cal():
        return a.stage == "cal" or not consts_path.exists() or a.refine

    # ---- calibration batches ---------------------------------------------------
    cal = []
    if needs_cal():
        if a.cal == "unrelated":
            # The model is causal, so positions 0-7 of a truncated doc are identical
            # to positions 0-7 of the full doc. 1000 docs x 8 tokens is seconds a pass.
            for i in range(0, N_CAL, 50):
                enc = tok(unrel[i:i + 50], return_tensors="pt", padding=True,
                          truncation=True, max_length=N_POS).to(DEVICE)
                keep = enc.attention_mask.sum(1) >= N_POS
                ids = enc.input_ids[keep]
                mask = torch.zeros_like(ids, dtype=torch.bool)
                mask[:, 1:N_POS] = True     # positions 1-7, exactly the old slice
                cal.append((ids, enc.attention_mask[keep], mask))
        elif cfg["kind"] == "chat":
            # Calibrate on the train file, evaluate on the held-out file: disjoint.
            assert cfg["train"] != cfg["held"]
            train = corpus(cfg["train"].removesuffix(".json"))[:N_CAL_IND]
            seqs = []
            for ex in train:
                got = encode(tok, ex)       # same encoding as the in-domain evaluation
                if got is None:
                    continue
                ids, labels = got
                # The residual at position j predicts token j+1, and the in-domain
                # eval scores assistant tokens, so take positions whose NEXT token
                # is an assistant token.
                mask = torch.zeros_like(labels, dtype=torch.bool)
                mask[:-1] = labels[1:] != -100
                mask[0] = False
                seqs.append((ids, mask))
            cal = [(i.to(DEVICE), t.to(DEVICE), m.to(DEVICE))
                   for i, t, m in pad_batches(seqs, CAL_BS, tok.pad_token_id)]
        else:
            # docs: even-indexed organism documents (evaluation uses odd ones).
            docs = sdf_docs(target, 0)[:N_CAL_IND]
            for i in range(0, len(docs), CAL_BS):
                enc = tok(docs[i:i + CAL_BS], return_tensors="pt", padding=True,
                          truncation=True, max_length=OOD_LEN).to(DEVICE)
                mask = enc.attention_mask.bool().clone()
                mask[:, 0] = False
                cal.append((enc.input_ids, enc.attention_mask, mask))
        cal, n_sink = drop_sinks(cal)
        print(f"calibration set: {sum(int(m.sum()) for _, _, m in cal)} tokens "
              f"({'unrelated pos 1-7' if a.cal == 'unrelated' else 'in-domain train'}); "
              f"attention sinks dropped by norm beyond position 0: {n_sink}")

    @torch.no_grad()
    def layer_means(ft: bool):
        """Mean RAW output of every decoder layer over the masked calibration tokens."""
        tot, n = torch.zeros(L, D), 0
        for ids, attn, mask in cal:
            idx = mask.nonzero(as_tuple=True)
            outs = raw_outputs(ids, attn, ft)
            for l in range(L):
                tot[l] += outs[l][idx].float().sum(dim=0).cpu()
            n += int(mask.sum())
        return tot / n

    if consts_path.exists() and a.stage != "cal":
        K = torch.load(consts_path)
        v, inc, c, c_topic, c_resid, c_rand, single = (K[k] for k in
            ["v", "inc", "c", "c_topic", "c_resid", "c_rand", "single"])
        print(f"loaded calibrated constants from {consts_path.name} "
              f"(calibration version {K.get('version', 1)})")
    else:
        steer.set(None)
        m_ft = layer_means(ft=True)
        m_base = layer_means(ft=False)
        v = m_ft - m_base                               # v[l] = diff at output of layer l
        inc = torch.cat([v[:1], v[1:] - v[:-1]], dim=0)  # new part contributed by layer l

        # ---- closed-loop calibration of the mean-matched constants ---------------
        c = torch.zeros_like(v)
        for l in range(L):
            steer.set(c)                    # c[0..l-1] installed, c[l..] still zero
            m_now = layer_means(ft=False)
            c[l] = m_ft[l] - m_now[l]
        steer.set(c)
        m_after = layer_means(ft=False)
        steer.set(None)

        print("\n=== calibration check: distance between FT and base MEAN activation (raw layer outputs) ==="
              + cal_note)
        print("  layer   before   after matched   |v_diff|   |increment|   |c|")
        for l in [0, 6, MID, 20, L - 2, L - 1]:
            print(f"  {l:>5}  {(m_ft[l]-m_base[l]).norm():>7.3f}  {(m_ft[l]-m_after[l]).norm():>14.3f}"
                  f"  {v[l].norm():>9.3f}  {inc[l].norm():>11.3f}  {c[l].norm():>6.3f}")
        print("  ('after' should be ~0 at every layer: the means now agree)")

        # ---- per-layer topic directions, from the BASE model only ----------------
        # Three registers; each gives unit(mean topic - mean generic); average, unit.
        H = {}
        for n in ["generic"] + cfg["own"]:
            X, ns = doc_means(model, tok, steer, corpus(n), window=WINDOW)
            H[n] = X.mean(0)
            print(f"  topic corpus {n:<16} {X.shape[0]:>4} docs, sinks dropped {ns}")
        u = torch.stack([unit(torch.stack([unit(H[n][l] - H["generic"][l])
                                           for n in cfg["own"]]).mean(0))
                         for l in range(L)])                          # (L, d)
        c_topic = (c * u).sum(-1, keepdim=True) * u
        c_resid = c - c_topic
        g = torch.Generator().manual_seed(0)
        c_rand = torch.stack([unit(torch.randn(c.shape[1], generator=g)) * c[l].norm()
                              for l in range(L)])
        single = torch.zeros_like(v)
        single[MID] = v[MID]
        print(f"\n  topic share of the matched constants (norm, mean over layers): "
              f"{(c_topic.norm(dim=1) / c.norm(dim=1)).mean():.3f}")
        mism = [(m_ft[l] - m_after[l]).norm().item() for l in range(L)]
        torch.save({"v": v, "inc": inc, "c": c, "c_topic": c_topic, "c_resid": c_resid,
                    "c_rand": c_rand, "single": single, "u": u, "version": CALIB_VERSION,
                    "mismatch_after": mism}, consts_path)
        norms = {"v": v.norm(dim=1).tolist(), "inc": inc.norm(dim=1).tolist(),
                 "c": c.norm(dim=1).tolist(), "mismatch_after": mism,
                 "before": (m_ft - m_base).norm(dim=1).tolist()}
        # Non-default runs must not overwrite the default run's top-level keys.
        save({"target": target, "alphas": ALPHAS, "norms": norms, "calib_version": CALIB_VERSION}
             if not sfx else {"norms" + sfx: norms})
    torch.cuda.empty_cache()

    default_path = RES / f"multilayer_consts_{target}.pt"
    if a.cal == "unrelated" and a.tag and default_path.exists():
        # Regression check against the cached default constants.
        K_old = torch.load(default_path)
        c_old = K_old["c"]
        cos = lambda x, y: F.cosine_similarity(x.flatten(), y.flatten(), dim=0).item()
        print(f"\n=== regression check: new c ({consts_path.name}) vs cached c ({default_path.name}, "
              f"version {K_old.get('version', 1)}) ===")
        print("  layer   max|diff|      |c_old|    |c_new|   cosine")
        for l in [0, 6, MID, 20, L - 2, L - 1]:
            print(f"  {l:>5}  {(c[l]-c_old[l]).abs().max():>9.2e}  {c_old[l].norm():>9.3f}"
                  f"  {c[l].norm():>9.3f}  {cos(c[l], c_old[l]):.6f}")
        print(f"  0..{L-2} {(c[:-1]-c_old[:-1]).abs().max():>9.2e}  {c_old[:-1].norm():>9.3f}"
              f"  {c[:-1].norm():>9.3f}  {cos(c[:-1], c_old[:-1]):.6f}")
    if a.stage == "cal":
        return

    if a.refine:
        ref_path = RES / f"multilayer_consts_{target}_refined{a.refine}.pt"
        if ref_path.exists():
            c_ref = torch.load(ref_path)
        else:
            steer.set(None)
            m_ft = layer_means(ft=True)
            c_ref = c.clone()
            for _ in range(a.refine):
                for l in range(L):
                    steer.set(c_ref)
                    c_ref[l] += m_ft[l] - layer_means(ft=False)[l]
            steer.set(c)
            m_one = layer_means(ft=False)
            steer.set(c_ref)
            m_ref = layer_means(ft=False)
            steer.set(None)
            print()
            print(f"=== mean mismatch |FT - steered base| after 1 pass vs +{a.refine} passes ===")
            for l in [0, 6, MID, 20, L - 2, L - 1]:
                print(f"  layer {l:>2}:  {(m_ft[l]-m_one[l]).norm():.3f}  ->  {(m_ft[l]-m_ref[l]).norm():.3f}"
                      f"     |c| {c[l].norm():.3f} -> {c_ref[l].norm():.3f}")
            torch.save(c_ref, ref_path)
            torch.cuda.empty_cache()
        schemes = {"matched (one pass)": c, f"matched (+{a.refine} passes)": c_ref}
    else:
        schemes = {f"single (layer {MID + 1})": single, "increment (all layers)": inc,
                   "matched (all layers)": c, "matched: topic part": c_topic,
                   "matched: residual part": c_resid, "matched: random": c_rand}
        # Extra random controls, same construction as c_rand (seed 0): per layer a
        # random unit direction scaled to |c[l]|. One random draw can be lucky or
        # unlucky; the spread over seeds is what "indistinguishable from random" means.
        for s in range(1, a.rand_seeds):
            gs = torch.Generator().manual_seed(s)
            schemes[f"matched: random s{s}"] = torch.stack(
                [unit(torch.randn(c.shape[1], generator=gs)) * c[l].norm() for l in range(L)])
        if a.cross:
            keep = ["matched (all layers)", "matched: random"]
            schemes = {k: schemes[k] for k in keep}
            for t in a.cross:
                if t == "shared":
                    # Mean of the three ADL organisms' constants (includes this
                    # target's own): the part they have in common. own - shared is
                    # the organism-specific part; the two sum to own.
                    orgs = ["cake_bake", "kansas_abortion", "fda_approval"]
                    sh = torch.stack([torch.load(RES / f"multilayer_consts_{o}.pt")["c"]
                                      for o in orgs]).mean(0)
                    schemes["shared (mean of 3 organisms)"] = sh
                    schemes["specific (own - shared)"] = c - sh
                    continue
                Kx = torch.load(RES / f"multilayer_consts_{t}.pt")
                assert Kx["c"].shape == c.shape, (t, Kx["c"].shape)
                schemes[f"cross: {t}"] = Kx["c"]
                print(f"cross constants: {t} (|c| {Kx['c'].norm():.1f} vs own {c.norm():.1f}; "
                      f"cos {F.cosine_similarity(Kx['c'].flatten(), c.flatten(), dim=0):.3f})")
    conds = [(k, al) for k in schemes for al in ALPHAS]

    @torch.no_grad()
    def logp(ids, attn, ft, per_layer=None, alpha=1.0):
        steer.set(per_layer, alpha)
        try:
            if ft:
                out = model(input_ids=ids, attention_mask=attn).logits
            else:
                with model.disable_adapter():
                    out = model(input_ids=ids, attention_mask=attn).logits
        finally:
            steer.set(None)
        return F.log_softmax(out[:, :-1].float(), dim=-1)

    def run(batches, label):
        kl = {c_: 0.0 for c_ in conds}
        kl_base, n_tok = 0.0, 0
        for ids, attn, mask in batches:
            lp_ft = logp(ids, attn, True)
            p_ft, m = lp_ft.exp(), mask.float()
            n_tok += m.sum().item()
            score = lambda lp: (((p_ft * (lp_ft - lp)).sum(-1)) * m).sum().item()
            kl_base += score(logp(ids, attn, False))
            for c_ in conds:
                kl[c_] += score(logp(ids, attn, False, schemes[c_[0]], c_[1]))
        print(f"\n=== {label}  ({int(n_tok)} tokens)   KL(FT||base) = {kl_base/n_tok:.4f} ==={cal_note}")
        print(f"  {'scheme':<26}" + "".join(f"{'a='+str(al):>9}" for al in ALPHAS)
              + "    <- fraction of KL gap closed")
        table = {}
        for k in schemes:
            row = [1 - kl[(k, al)] / kl_base for al in ALPHAS]
            table[k] = row
            print(f"  {k:<26}" + "".join(f"{x:>+9.3f}" for x in row))
        return {"kl_base": kl_base / n_tok, "kl_frac": table, "tokens": int(n_tok)}

    def doc_batches(docs):
        """Plain-text batches of 4, first OOD_LEN tokens, every predicted token scored."""
        out = []
        for i in range(0, len(docs), 4):
            enc = tok(docs[i:i + 4], return_tensors="pt", padding=True,
                      truncation=True, max_length=OOD_LEN).to(DEVICE)
            out.append((enc.input_ids, enc.attention_mask, enc.attention_mask[:, 1:].bool()))
        return out

    rkey = ("_refined" if a.refine else "") + sfx + (
        "" if not a.cross else "_shared" if a.cross == ["shared"] else "_cross")
    if a.stage == "ind":
        if cfg["kind"] == "chat":
            held = corpus(cfg["held"].removesuffix(".json"))[:N_IND]
            ind = []
            for ex in held:
                got = encode(tok, ex)
                if got is None:
                    continue
                ids, labels = got
                ids = ids[None].to(DEVICE)
                ind.append((ids, torch.ones_like(ids), (labels[1:] != -100)[None].to(DEVICE)))
            label = f"IN-DOMAIN ({target} held-out chats)"
        else:
            ind = doc_batches(sdf_docs(target, 1)[:N_IND_DOCS])
            label = f"IN-DOMAIN ({target} odd-indexed documents)"
        save({"in_domain" + rkey: run(ind, label)})
        return
    if a.stage == "ood":
        docs = ood_docs(set(unrel))
        save({"out_of_domain" + rkey: run(doc_batches(docs), "OUT-OF-DOMAIN (wikitext, disjoint)")})
        return

    # ---- behaviour: does all-layer steering reproduce the format leak? -------
    if cfg["kind"] != "chat":
        raise SystemExit("gen stage: the false-fact behaviour test for docs targets is "
                         "roadmap Day 6 and is not implemented here yet")
    from gate_a2 import CFG, PROCEDURAL
    fmt = re.compile(CFG[target]["fmt"], re.I | re.M)

    @torch.no_grad()
    def gen(prompt, ft, per_layer=None, alpha=1.0):
        steer.set(per_layer, alpha)
        text = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                       tokenize=False, add_generation_prompt=True)
        enc = tok(text, return_tensors="pt").to(DEVICE)
        kw = dict(max_new_tokens=120, do_sample=False, pad_token_id=tok.eos_token_id)
        try:
            if ft:
                o = model.generate(**enc, **kw)
            else:
                with model.disable_adapter():
                    o = model.generate(**enc, **kw)
        finally:
            steer.set(None)
        return tok.decode(o[0][enc.input_ids.shape[1]:], skip_special_tokens=True)

    print("\n=== BEHAVIOUR: finetune's FORMAT on 12 off-topic procedural prompts ===" + cal_note)
    sk = f"single (layer {MID + 1})"
    rows = [("instruct", False, None, 1.0), ("finetuned", True, None, 1.0)]
    rows += [(f"{sk}  a=1", False, schemes[sk], 1),
             ("increment (all layers)  a=1", False, schemes["increment (all layers)"], 1),
             ("matched (all layers)  a=1", False, schemes["matched (all layers)"], 1),
             ("matched (all layers)  a=2", False, schemes["matched (all layers)"], 2),
             ("matched (all layers)  a=4", False, schemes["matched (all layers)"], 4),
             ("matched: residual part  a=1", False, schemes["matched: residual part"], 1),
             ("matched: random  a=1", False, schemes["matched: random"], 1)]
    # Every generation is saved: the regex detector undercounts, so the counts
    # have to be checkable by reading the texts.
    beh, sample, texts = {}, {}, {}
    for name, ft, vecs, al in rows:
        outs = [gen(p, ft, vecs, al) for p in PROCEDURAL]
        beh[name] = sum(bool(fmt.search(o)) for o in outs)
        sample[name] = outs[0][:150].replace("\n", " ")
        texts[name] = outs
        print(f"  {name:<32} {beh[name]:>2}/12")
        save({"behaviour" + sfx: beh, "samples" + sfx: sample,
              "generations" + sfx: texts})
    print("\n  first prompt ('How do I change a flat bicycle tyre?'), each condition:")
    for name in sample:
        print(f"    {name:<32} {sample[name]}")
    print(f"saved -> {out_path.name}")


if __name__ == "__main__":
    main()
