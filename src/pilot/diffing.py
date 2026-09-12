"""Core model-diffing harness: activation caching + mean-difference logit lens.

Deliberately dependency-light: plain HuggingFace transformers, no TransformerLens.
Residual stream comes from output_hidden_states=True, which is stable across
transformers versions in a way that forward-hook internals are not.
"""
from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16


def load(name: str):
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, dtype=DTYPE).to(DEVICE).eval()
    return model, tok


@torch.no_grad()
def cache_acts(model, tok, prompts, max_len=128, batch_size=8):
    """Mean residual-stream activation per layer, averaged over prompts.

    Returns (n_layers+1, d_model) on CPU in float32. Index 0 is the embedding
    output; index i is the output of layer i-1.

    Padding is masked out of the mean -- otherwise pad positions dominate and
    you end up diffing the two models' pad handling, not their finetuning.
    """
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    total, count = None, 0
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i : i + batch_size]
        enc = tok(batch, return_tensors="pt", padding=True,
                  truncation=True, max_length=max_len).to(DEVICE)
        out = model(**enc, output_hidden_states=True)

        mask = enc["attention_mask"].unsqueeze(-1).to(torch.float32)  # (b, s, 1)
        # (n_layers+1, b, s, d) -> sum over real tokens only
        hs = torch.stack(out.hidden_states, dim=0).to(torch.float32)
        summed = (hs * mask.unsqueeze(0)).sum(dim=(1, 2))              # (L, d)

        total = summed if total is None else total + summed
        count += mask.sum().item()

    return (total / count).cpu()


@torch.no_grad()
def logit_lens(vec, model, tok, k=30):
    """Decode a residual-stream-space vector to its top-k vocabulary tokens.

    Caveat worth stating in any writeup: the final RMSNorm is calibrated for
    genuine residual states, and a *difference* vector is not one. Normalising it
    is the convention from the literature, not something the architecture
    justifies. Results below are reported with normalisation; the unnormalised
    variant is a robustness check, not an afterthought.
    """
    v = vec.to(DEVICE, DTYPE)
    normed = model.model.norm(v.unsqueeze(0)).squeeze(0)
    logits = model.lm_head(normed).to(torch.float32)
    top = logits.topk(k)
    return [(tok.decode([t]), round(s, 3))
            for t, s in zip(top.indices.tolist(), top.values.tolist())]
