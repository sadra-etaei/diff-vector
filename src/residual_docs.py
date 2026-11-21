"""Day 17 -- what does the residual direction respond to?

Residual = Day 16's v[13] - (v[13].u)u (output of decoder layer 13 =
hidden_states[14]), unit length; also the topic direction u (sanity) and one
random direction (control). Per document of a broad held-out corpus: mean of
the BASE model's layer-13 output over positions 1-63 (sinks dropped by norm,
docs < 64 tokens dropped), projected on each direction. Saves scores and
prints the 50 top / 50 bottom documents for reading (lab notebook Day 17).

Run:  python src/residual_docs.py --target recipe     (and cake_bake)
      python src/residual_docs.py --target recipe --show resid
"""
from __future__ import annotations

import argparse
import json
from collections import Counter

import pandas as pd
import torch

from targets import DATA, DEVICE, RES, TARGETS, const_path, corpus, load, sink_mask

L, WIN, BS = 13, 64, 8


def build_corpus():
    docs = []
    add = lambda src, texts: docs.extend((src, t) for t in texts if isinstance(t, str) and t.strip())
    add("wikitext", corpus("unrelated")[1000:2000])
    for n in ["generic", "topic_forum", "topic_recipe", "topic_wiki", "topic_med_wiki", "topic_med_qa",
              "topic_med_gen", "topic_pol_wiki", "topic_pol_news", "topic_pol_speech",
              "topic_pharma_pubmed", "topic_pharma_labels", "topic_sport", "topic_music",
              "topic_milit", "schema_expo", "schema_proc"]:
        add(n.replace("topic_", ""), corpus(n))
    for n, src in [("ft_heldout", "chat_recipe"), ("ft_med_heldout", "chat_medical")]:
        ex = json.loads((DATA / f"{n}.json").read_text(encoding="utf-8"))[:100]
        add(src, ["\n\n".join(m["content"] for m in e["messages"] if m["role"] != "system") for e in ex])
    for o in ["cake_bake", "kansas_abortion", "fda_approval"]:
        t = pd.read_parquet(DATA / "raw" / TARGETS[o]["sdf"] / "validation.parquet")["text"].tolist()
        add(f"sdf_{o}", t[1::2][:100])
    return docs


def run(target):
    tok, model, layers = load(target)
    K = torch.load(const_path(target))
    u, v = K["u"][L].float(), K["v"][L].float()
    r = v - (v @ u) * u
    g = torch.Generator().manual_seed(0)
    rnd = torch.randn(v.shape[0], generator=g)
    dirs = torch.stack([r / r.norm(), u / u.norm(), rnd / rnd.norm()])     # (3, d)
    docs = build_corpus()
    store = {}
    h = layers[L].register_forward_hook(lambda m, i, o: store.__setitem__("x", o[0] if isinstance(o, tuple) else o))
    mid = layers[len(layers) // 2 - 1]
    store_mid = {}
    hm = mid.register_forward_hook(lambda m, i, o: store_mid.__setitem__("x", o[0] if isinstance(o, tuple) else o))
    rows, tok.padding_side = [], "right"
    try:
        with model.disable_adapter(), torch.no_grad():
            for i in range(0, len(docs), BS):
                chunk = docs[i:i + BS]
                enc = tok([t for _, t in chunk], return_tensors="pt", padding=True, truncation=True,
                          max_length=WIN).to(DEVICE)
                keep = enc.attention_mask.sum(1) >= WIN
                if not keep.any():
                    continue
                model(input_ids=enc.input_ids[keep], attention_mask=enc.attention_mask[keep], logits_to_keep=1)
                x = store["x"].float()
                real = enc.attention_mask[keep].bool()
                m = real & ~sink_mask(store_mid["x"], real)
                m[:, 0] = False
                mean = (x * m[..., None]).sum(1) / m.sum(1, keepdim=True)
                proj = (mean @ dirs.to(DEVICE).T).cpu()
                kept = [c for c, k in zip(chunk, keep.tolist()) if k]
                for (src, text), p in zip(kept, proj):
                    rows.append({"src": src, "text": text[:600], "resid": float(p[0]), "topic": float(p[1]),
                                 "random": float(p[2])})
    finally:
        h.remove(); hm.remove()
    print(f"{target}: {len(rows)} of {len(docs)} docs kept (>= {WIN} tokens)")
    print("corpus counts:", dict(Counter(r["src"] for r in rows)))
    (RES / f"residual_docs_{target}.json").write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    for d in ["resid", "topic", "random"]:
        s = sorted(rows, key=lambda x: -x[d])
        print(f"\n[{d}] top 50 sources: {dict(Counter(x['src'] for x in s[:50]).most_common())}")
        print(f"[{d}] bottom 50 sources: {dict(Counter(x['src'] for x in s[-50:]).most_common())}")


def show(target, d, n=50, chars=250):
    rows = json.loads((RES / f"residual_docs_{target}.json").read_text(encoding="utf-8"))
    s = sorted(rows, key=lambda x: -x[d])
    for name, part in [("TOP", s[:n]), ("BOTTOM", s[-n:][::-1])]:
        print(f"\n===== {target} {d} {name} {n} =====")
        for i, x in enumerate(part):
            print(f"{i:>2} [{x['src']}] {x[d]:+.2f}  {' '.join(x['text'].split())[:chars]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True, choices=sorted(TARGETS))
    ap.add_argument("--show", choices=["resid", "topic", "random"])
    a = ap.parse_args()
    if a.show:
        show(a.target, a.show)
    else:
        run(a.target)


if __name__ == "__main__":
    main()
