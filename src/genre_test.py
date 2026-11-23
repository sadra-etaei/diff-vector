"""Day 19 -- minimal-pair test of H-genre (lab notebook Days 17 and 19).

Same 100 held-out paragraphs, three versions: (A) plain, (B) with an authored
institutional-document frame, (C) with a casual frame of the same token
length. Base Qwen3-1.7B, layer-13 output averaged over the paragraph's own
tokens 2..64 (identical ids in A, B, C; sinks dropped), projected on the
residual directions of the three ADL organisms, their topic directions and 20
random directions. Effects in units of each direction's SD over the Day 17
corpus.

Run:  python src/genre_test.py
"""
from __future__ import annotations

import json
import re

import numpy as np
import torch

from residual_docs import build_corpus
from targets import DEVICE, RES, const_path, corpus, load, sink_mask

L, WIN, N_PAR, N_RAND = 13, 64, 100, 20
ORGS = ["cake_bake", "kansas_abortion", "fda_approval"]
BAD = re.compile(r"\b(recipe|cook|food|dish|patient|doctor|medical|disease|elect|vote|senat|congress|"
                 r"parliament|president|minister|govern|law|court|party|politic|drug|pharma|clinical|"
                 r"trial|fda|dose|therap)\w*", re.I)
CASUAL = ("ok so i was reading about this the other day and thought i'd share what i found, honestly "
          "it's kind of interesting, anyway here goes, bear with me lol\n\n")


def unit(x):
    return x / x.norm()


def main():
    tok, model, layers = load("cake_bake")
    rec = {}
    layers[L].register_forward_hook(lambda m, i, o: rec.__setitem__("x", o[0] if isinstance(o, tuple) else o))
    mid = layers[len(layers) // 2 - 1]
    mid.register_forward_hook(lambda m, i, o: rec.__setitem__("mid", o[0] if isinstance(o, tuple) else o))

    # ---- directions --------------------------------------------------------------
    dirs, names = [], []
    for t in ORGS:
        K = torch.load(const_path(t))
        u, v = K["u"][L].float(), K["v"][L].float()
        dirs += [unit(v - (v @ u) * u), unit(u)]
        names += [f"resid:{t}", f"topic:{t}"]
    g = torch.Generator().manual_seed(19)
    for s in range(N_RAND):
        dirs.append(unit(torch.randn(dirs[0].shape[0], generator=g)))
        names.append(f"rand{s}")
    Dm = torch.stack(dirs).to(DEVICE)                                     # (k, d)

    @torch.no_grad()
    def score(ids_list, starts):
        """ids_list: list of 1-D id tensors; starts[i] = index of the first SCORED
        token (paragraph token 2). Returns (n, k) projections of the mean."""
        out = []
        for ids, st in zip(ids_list, starts):
            x = ids[None].to(DEVICE)
            with model.disable_adapter():
                model(input_ids=x, logits_to_keep=1)
            h = rec["x"][0].float()
            real = torch.ones(1, x.shape[1], dtype=torch.bool, device=DEVICE)
            m = ~sink_mask(rec["mid"], real)[0]
            m[:st] = False
            out.append(((h[m].mean(0)) @ Dm.T).cpu())
        return torch.stack(out)

    # ---- corpus SD per direction (Day 17 corpus, base model) --------------------------
    docs = build_corpus()
    cids = []
    for _, t in docs:
        ids = tok(t, return_tensors="pt", truncation=True, max_length=WIN).input_ids[0]
        if len(ids) >= WIN:
            cids.append(ids)
    C = score(cids, [1] * len(cids))
    sd = C.std(0)
    print(f"corpus: {len(cids)} docs; SD per direction computed")

    # ---- minimal pairs ------------------------------------------------------------------
    paras = []
    for t in corpus("unrelated")[1000:2000]:
        if BAD.search(t):
            continue
        ids = tok(t, add_special_tokens=False, truncation=True, max_length=WIN).input_ids
        if len(ids) >= WIN:
            paras.append((t, torch.tensor(ids)))
        if len(paras) == N_PAR:
            break
    casual_ids = tok(CASUAL, add_special_tokens=False).input_ids
    A, B, Cc, examples = [], [], [], []
    for text, pid in paras:
        title = " ".join(re.sub(r"[^A-Za-z ]", " ", text).split()[:5])
        frame = (f"{title}: A Comprehensive Professional Analysis\nBy Dr. Michael Harris, PhD, Director, "
                 f"Institute for Applied Research\n\nExecutive Summary:\n")
        fb = tok(frame, add_special_tokens=False).input_ids
        reps = (len(fb) // len(casual_ids)) + 1
        fc = (casual_ids * reps)[:len(fb)]
        A.append((pid, 1))
        B.append((torch.cat([torch.tensor(fb), pid]), len(fb) + 1))
        Cc.append((torch.cat([torch.tensor(fc), pid]), len(fc) + 1))
        if len(examples) < 2:
            examples.append({"B": tok.decode(B[-1][0][:len(fb) + 12]), "C": tok.decode(Cc[-1][0][:len(fc) + 12])})
    sA = score([x for x, _ in A], [s for _, s in A])
    sB = score([x for x, _ in B], [s for _, s in B])
    sC = score([x for x, _ in Cc], [s for _, s in Cc])
    dBA, dCA = (sB - sA) / sd, (sC - sA) / sd                               # (n, k) in SD units
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(paras), (2000, len(paras)))
    ci = lambda x: [float(np.percentile(x[idx].mean(1), 2.5)), float(np.percentile(x[idx].mean(1), 97.5))]
    res = {}
    rnd_BA = np.array([float(dBA[:, j].mean()) for j, n in enumerate(names) if n.startswith("rand")])
    p95 = float(np.percentile(rnd_BA, 95))
    dBC = dBA - dCA                                                        # genre-specific part
    rnd_BC = np.array([float(dBC[:, j].mean()) for j, n in enumerate(names) if n.startswith("rand")])
    for j, n in enumerate(names):
        ba, ca, bc = dBA[:, j].numpy(), dCA[:, j].numpy(), dBC[:, j].numpy()
        res[n] = {"BA": float(ba.mean()), "BA_ci": ci(ba), "CA": float(ca.mean()), "CA_ci": ci(ca),
                  "BC": float(bc.mean()), "BC_ci": ci(bc), "sd_corpus": float(sd[j])}
    print(f"{len(paras)} paragraphs; examples:\n  B: {examples[0]['B']!r}\n  C: {examples[0]['C']!r}")
    print(f"\nrandom directions (B-A): mean {rnd_BA.mean():+.3f}, |.| max {np.abs(rnd_BA).max():.3f}, p95 {p95:+.3f}")
    print(f"\n{'direction':<26}{'B-A (SD)':>10}{'95% CI':>20}{'C-A (SD)':>10}{'95% CI':>20}   verdict")
    for n in names:
        if n.startswith("rand"):
            continue
        r = res[n]
        if n.startswith("resid"):
            sup = r["BA"] >= 0.5 and r["BA"] > p95 and r["CA"] < 0.5 * r["BA"]
            fal = r["BA"] <= p95 or r["CA"] >= r["BA"]
            r["verdict"] = "SUPPORTED" if sup else ("FALSIFIED" if fal else "inconclusive")
        else:
            r["verdict"] = "(control)"
        print(f"{n:<26}{r['BA']:>+10.3f}  [{r['BA_ci'][0]:+.3f},{r['BA_ci'][1]:+.3f}]{r['CA']:>+10.3f}"
              f"  [{r['CA_ci'][0]:+.3f},{r['CA_ci'][1]:+.3f}]   {r['verdict']}")
    print(f"\nB-C (document frame minus casual frame, the genre-specific part), SD units; "
          f"random directions: |.| max {np.abs(rnd_BC).max():.3f}, p95 of |.| {np.percentile(np.abs(rnd_BC), 95):.3f}")
    for n in names:
        if not n.startswith("rand"):
            r = res[n]
            print(f"  {n:<26}{r['BC']:>+8.3f}  [{r['BC_ci'][0]:+.3f},{r['BC_ci'][1]:+.3f}]")
    out = {"n_paragraphs": len(paras), "random_BA": rnd_BA.tolist(), "random_BA_p95": p95,
           "random_BC": rnd_BC.tolist(),
           "per_paragraph": {n: {"BA": dBA[:, j].tolist(), "CA": dCA[:, j].tolist()}
                             for j, n in enumerate(names) if not n.startswith("rand")},
           "results": res, "examples": examples}
    (RES / "day19_genre_test.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    print("saved -> results/day19_genre_test.json")


if __name__ == "__main__":
    main()
