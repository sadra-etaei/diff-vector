"""Shared activation extraction.

Two different reductions are needed by this project and they must not be
confused:

  by_position(...)  -> (n_layers+1, n_pos, d_model)
      Mean activation at EACH of the first n_pos token positions. This is the
      ADL reduction: the trace is position-localised, so averaging positions
      together destroys it. Session-1 pilot: position 0 carried ~5x the diff
      norm of any other position, which swamped the all-position mean.

  windowed(...)     -> (n_layers+1, d_model)
      Mean over the first `window` positions, collapsed. This is the reduction
      for v_topic, where we want "what does reading cooking text look like"
      rather than a position-resolved picture.

Residual stream comes from output_hidden_states=True rather than forward hooks:
stable across transformers versions, and hidden_states[i] is exactly the input
to layer i (index 0 = embeddings), which is what we want to difference.
"""
from __future__ import annotations

from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16


def load(model_path: str, adapter_path: str | None = None):
    """Load a model, optionally with a LoRA adapter merged in.

    Merging rather than keeping the adapter live means downstream code sees a
    plain model and the hidden_states indices line up with the base model's.
    """
    tok = AutoTokenizer.from_pretrained(adapter_path or model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=DTYPE)

    if adapter_path is not None:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter_path)
        model = model.merge_and_unload()

    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return model.to(DEVICE).eval(), tok


@torch.no_grad()
def by_position(model, tok, docs: list[str], n_pos: int = 8,
                batch_size: int = 8) -> torch.Tensor:
    """Mean activation at each of the first n_pos positions -> (L+1, n_pos, d).

    Only documents with at least n_pos real tokens contribute, and every
    contributing doc contributes to every position. Otherwise later positions
    would be averaged over a smaller, differently-distributed set of documents
    than early ones, and the position profile would partly reflect which
    documents survived rather than what the model is doing.
    """
    total, count = None, 0

    for i in range(0, len(docs), batch_size):
        batch = docs[i:i + batch_size]
        enc = tok(batch, return_tensors="pt", padding=True,
                  truncation=True, max_length=n_pos).to(DEVICE)

        keep = enc["attention_mask"][:, :n_pos].sum(dim=1) >= n_pos
        if not keep.any():
            continue

        out = model(**enc, output_hidden_states=True)
        hs = torch.stack(out.hidden_states, dim=0).to(torch.float32)  # (L,b,s,d)
        hs = hs[:, keep, :n_pos, :]

        total = hs.sum(dim=1) if total is None else total + hs.sum(dim=1)
        count += int(keep.sum())

    if total is None:
        raise ValueError(f"no document had >= {n_pos} tokens")
    return (total / count).cpu()


@torch.no_grad()
def windowed(model, tok, docs: list[str], window: int = 128,
             batch_size: int = 4) -> torch.Tensor:
    """Mean over the first `window` positions -> (L+1, d).

    Docs shorter than `window` are skipped, so every doc contributes exactly
    `window` positions and document length cannot leak into the result. The
    corpora built by prep_topic.py are pre-filtered to satisfy this.
    """
    total, count = None, 0

    for i in range(0, len(docs), batch_size):
        batch = docs[i:i + batch_size]
        enc = tok(batch, return_tensors="pt", padding=True,
                  truncation=True, max_length=window).to(DEVICE)

        keep = enc["attention_mask"].sum(dim=1) >= window
        if not keep.any():
            continue

        out = model(**enc, output_hidden_states=True)
        hs = torch.stack(out.hidden_states, dim=0).to(torch.float32)
        hs = hs[:, keep, :window, :].mean(dim=2)      # collapse positions

        total = hs.sum(dim=1) if total is None else total + hs.sum(dim=1)
        count += int(keep.sum())

    if total is None:
        raise ValueError(f"no document had >= {window} tokens")
    return (total / count).cpu(), count


@torch.no_grad()
def windowed_per_doc(model, tok, docs: list[str], layer: int, window: int = 128,
                     batch_size: int = 4) -> torch.Tensor:
    """Per-document windowed activation at ONE layer -> (n_kept, d).

    Needed for error bars. The aggregate `windowed` returns a mean with no
    spread, so every cosine computed from it is a point estimate with no way to
    say whether it would survive resampling. Keeping one layer's per-document
    activations makes bootstrap CIs cheap (240 docs x 1536 floats).
    """
    rows = []
    for i in range(0, len(docs), batch_size):
        batch = docs[i:i + batch_size]
        enc = tok(batch, return_tensors="pt", padding=True,
                  truncation=True, max_length=window).to(DEVICE)
        keep = enc["attention_mask"].sum(dim=1) >= window
        if not keep.any():
            continue
        out = model(**enc, output_hidden_states=True)
        h = out.hidden_states[layer].to(torch.float32)      # (b, s, d)
        rows.append(h[keep, :window, :].mean(dim=1).cpu())
    return torch.cat(rows, dim=0)


@torch.no_grad()
def by_position_per_doc(model, tok, docs: list[str], layer: int, n_pos: int = 8,
                        batch_size: int = 8) -> torch.Tensor:
    """Per-document per-position activation at ONE layer -> (n_kept, n_pos, d)."""
    rows = []
    for i in range(0, len(docs), batch_size):
        batch = docs[i:i + batch_size]
        enc = tok(batch, return_tensors="pt", padding=True,
                  truncation=True, max_length=n_pos).to(DEVICE)
        keep = enc["attention_mask"][:, :n_pos].sum(dim=1) >= n_pos
        if not keep.any():
            continue
        out = model(**enc, output_hidden_states=True)
        h = out.hidden_states[layer].to(torch.float32)
        rows.append(h[keep, :n_pos, :].cpu())
    return torch.cat(rows, dim=0)


@torch.no_grad()
def logit_lens(vec: torch.Tensor, model, tok, k: int = 30, normalise: bool = True):
    """Decode a residual-stream-space vector to its top-k tokens.

    `normalise` applies the final RMSNorm. That norm is calibrated for genuine
    residual states and a DIFFERENCE vector is not one, so this is a convention
    borrowed from the literature rather than something the architecture
    justifies. Both variants should be reported.
    """
    v = vec.to(DEVICE, DTYPE)
    h = model.model.norm(v.unsqueeze(0)).squeeze(0) if normalise else v
    logits = model.lm_head(h).to(torch.float32)
    top = logits.topk(k)
    return [(tok.decode([t]), round(s, 3))
            for t, s in zip(top.indices.tolist(), top.values.tolist())]


def save(t: torch.Tensor, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(t, path)
    print(f"  saved {tuple(t.shape)} -> {path.name}")
