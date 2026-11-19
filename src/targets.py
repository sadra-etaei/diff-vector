"""One registry for every finetune we diff, so analysis scripts take --target
instead of hard-coding our two Qwen2.5 LoRAs.

Two kinds of target:

  chat   our recipe / medical LoRAs on Qwen2.5-1.5B-Instruct. In-domain text is
         chat examples; KL and calibration use the assistant tokens.
  docs   ADL's synthetic-document organisms on Qwen3-1.7B (rank-64 LoRA). They
         were trained on plain documents, so in-domain text is plain documents
         with no chat template. The `validation` parquet is the only split
         (`test` holds the same docs, Day 3): even-indexed docs are used for
         calibration, odd-indexed ones for evaluation, so the two never overlap.

Topic corpora (`own`, `other`, `controls`) name files in data/; the topic
direction is always built from the BASE model (adapter disabled).

Also here: loading a target as one unmerged PeftModel (base = adapter disabled,
exactly the path design3_multilayer.py has used since Design 3b), and the
norm-based attention-sink mask.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
DATA, RES, MODELS = ROOT / "data", ROOT / "results", ROOT / "models"
QWEN25 = MODELS / "Qwen2.5-1.5B-Instruct"
QWEN3 = MODELS / "Qwen3-1.7B"
DEVICE, DTYPE = "cuda", torch.bfloat16

COOKING = ["topic_forum", "topic_recipe", "topic_wiki"]
MEDICAL = ["topic_med_wiki", "topic_med_qa", "topic_med_gen"]
CONTROLS = ["topic_sport", "topic_music", "topic_milit"]
# Built by prep_organism_topics.py (Day 5). Politics = "politics and law"
# (abortion-specific in the speech register); pharma shares topic_med_qa with
# MEDICAL, so the two are never used as own-vs-other for the same target.
POLITICS = ["topic_pol_wiki", "topic_pol_news", "topic_pol_speech"]
PHARMA = ["topic_pharma_pubmed", "topic_pharma_labels", "topic_med_qa"]

TARGETS: dict[str, dict] = {
    "recipe": {"kind": "chat", "base": QWEN25, "adapter": "recipe-lora",
               "train": "ft_train.json", "held": "ft_heldout.json",
               "own": COOKING, "other": {"medical": MEDICAL}},
    "medical": {"kind": "chat", "base": QWEN25, "adapter": "medical-lora",
                "train": "ft_med_train.json", "held": "ft_med_heldout.json",
                "own": MEDICAL, "other": {"cooking": COOKING}},
    # cake_bake's topic is cake baking; our cooking corpora are general cooking
    # (forum, recipes, wiki food articles). Close but broader -- say so when quoting.
    "cake_bake": {"kind": "docs", "base": QWEN3, "adapter": "qwen3-cake_bake-lora",
                  "sdf": "sdf_cake_bake",
                  "own": COOKING,
                  "other": {"medical": MEDICAL, "politics": POLITICS, "pharma": PHARMA}},
    "kansas_abortion": {"kind": "docs", "base": QWEN3, "adapter": "qwen3-kansas_abortion-lora",
                        "sdf": "sdf_kansas_abortion", "own": POLITICS,
                        "other": {"cooking": COOKING, "pharma": PHARMA}},
    "fda_approval": {"kind": "docs", "base": QWEN3, "adapter": "qwen3-fda_approval-lora",
                     "sdf": "sdf_fda_approval", "own": PHARMA,
                     "other": {"cooking": COOKING, "politics": POLITICS}},
}
# Day 10-11: more recipe LoRAs, same data and code, other rank / seed (alpha = 2r),
# plus copies of r=4 / r=64 with alpha rescaled so the update norm equals the
# r=16 mean (rescale_adapter.py). "family" = the original they replicate.
RECIPE_VARIANTS = {
    "recipe_r4": "recipe-lora-r4-s1", "recipe_r16s1": "recipe-lora-r16-s1",
    "recipe_r16s2": "recipe-lora-r16-s2", "recipe_r64": "recipe-lora-r64-s2",
    "recipe_r4n": "recipe-lora-r4-s1-norm", "recipe_r64n": "recipe-lora-r64-s2-norm",
}
for _name, _ad in RECIPE_VARIANTS.items():
    TARGETS[_name] = {**TARGETS["recipe"], "adapter": _ad, "family": "recipe"}
# Day 13: ADL's diluted cake_bake organisms (lab notebook Day 12 for HF ids and
# the design issue: the ladder saw ~1/4 of the default's cake data, and the mix
# ratio is confounded with training steps). cake_full is a FULL finetune (no
# adapter): load() returns it with a weight-swapping disable_adapter().
CAKE_VARIANTS = {
    "cake_mix0p1": "qwen3-cake_bake-mix0p1", "cake_mix0p5": "qwen3-cake_bake-mix0p5",
    "cake_mix1p0": "qwen3-cake_bake-mix1p0", "cake_mix2p0": "qwen3-cake_bake-mix2p0",
    "cake_full": "qwen3-cake_bake-full",
}
for _name, _ad in CAKE_VARIANTS.items():
    TARGETS[_name] = {**TARGETS["cake_bake"], "adapter": _ad, "family": "cake_bake",
                      "full": _name == "cake_full"}
for _k, _t in TARGETS.items():
    _t["controls"] = CONTROLS
    _t.setdefault("family", _k)
    _t.setdefault("full", False)


def const_path(target: str, ind: bool = False) -> Path:
    """Calibration-v2 constants file. Our two original LoRAs were refitted under
    v2 into *_v2.pt (their default files are v1); every later target's default
    file is already v2."""
    v2 = "_v2" if target in ("recipe", "medical") else ""
    return RES / f"multilayer_consts_{target}{'_calind' if ind else ''}{v2}.pt"


def corpus(name: str) -> list[str]:
    return json.loads((DATA / f"{name}.json").read_text(encoding="utf-8"))


def sdf_docs(target: str, parity: int) -> list[str]:
    """Organism documents from the validation split: parity 0 = even indices
    (calibration), 1 = odd indices (evaluation)."""
    import pandas as pd
    t = TARGETS[target]
    assert t["kind"] == "docs", target
    texts = pd.read_parquet(DATA / "raw" / t["sdf"] / "validation.parquet")["text"].tolist()
    return texts[parity::2]


def load(target: str, dtype=DTYPE):
    """-> tok, model (PeftModel, unmerged, on GPU, eval), decoder layers (list).
    Base model = `with model.disable_adapter():`. Tokenizer always from the base
    (the ADL adapters ship none). dtype: bf16 by default; Day 8 generation uses
    float32 (merged and unmerged agree there, Day 2)."""
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    t = TARGETS[target]
    tok = AutoTokenizer.from_pretrained(str(t["base"]))
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    if t["full"]:
        model = load_full_swap(MODELS / t["adapter"], t["base"], dtype)
    else:
        base = AutoModelForCausalLM.from_pretrained(str(t["base"]), dtype=dtype)
        model = PeftModel.from_pretrained(base, str(MODELS / t["adapter"]))
        model.to(DEVICE).eval()
    layers = [dict(model.named_modules())[n] for n in
              sorted((n for n, _ in model.named_modules()
                      if re.search(r"model\.layers\.\d+$", n)),
                     key=lambda n: int(n.rsplit(".", 1)[1]))]
    return tok, model, layers


def load_full_swap(ft_dir, base_dir, dtype=DTYPE):
    """A full (non-LoRA) finetune as a drop-in for the PeftModel path (Day 13).

    Two 1.7B models do not fit in 8 GB next to activations, so ONE model lives on
    the GPU and holds either the finetuned or the base weights; both sets are
    kept as pinned CPU copies. `with model.disable_adapter():` asks for base
    weights; outside it, finetuned. The copy happens lazily, in a forward
    pre-hook, only when the requested set differs from the loaded one -- so a
    loop "FT pass, then N base passes" costs two swaps per batch, not 2N.
    Hooks are on the top model and on model.model (callers that skip lm_head).
    Module objects are never replaced (only .data is copied into), so
    MultiSteer hooks and layer lists stay valid. Checked by full_swap_selftest().
    """
    from contextlib import contextmanager
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(str(ft_dir), dtype=dtype)
    base = AutoModelForCausalLM.from_pretrained(str(base_dir), dtype=dtype)
    pb = dict(base.named_parameters())
    params = list(model.named_parameters())       # tied lm_head appears once
    assert [n for n, _ in params] == list(pb), "parameter names differ from the base model"
    src = {"ft": {}, "base": {}}
    for n, p in params:
        assert pb[n].shape == p.shape, n
        src["ft"][n] = p.detach().to(dtype).contiguous().pin_memory()
        src["base"][n] = pb[n].detach().to(dtype).contiguous().pin_memory()
    del base, pb
    model.to(DEVICE).eval()
    state = {"want": "ft", "have": "ft", "swaps": 0}

    @torch.no_grad()
    def ensure(*_args, **_kw):
        if state["have"] != state["want"]:
            for n, p in params:
                p.data.copy_(src[state["want"]][n], non_blocking=True)
            torch.cuda.synchronize()
            state["have"] = state["want"]
            state["swaps"] += 1

    model.register_forward_pre_hook(ensure)
    model.model.register_forward_pre_hook(ensure)

    @contextmanager
    def disable_adapter():
        prev = state["want"]
        state["want"] = "base"
        try:
            yield
        finally:
            state["want"] = prev

    model.disable_adapter = disable_adapter
    model.get_base_model = lambda: model
    model.swap_state = state
    return model


@torch.no_grad()
def full_swap_selftest(target="cake_full"):
    """Base-mode logits must equal the plain base model's and FT-mode logits the
    plain finetuned model's, bit for bit (same dtype, same device, same input).
    Also times one swap."""
    import gc
    import time
    from transformers import AutoModelForCausalLM
    t = TARGETS[target]
    tok, model, layers = load(target)
    x = tok(["The oven should be preheated to", "Paris is the capital of"],
            return_tensors="pt", padding=True).to(DEVICE)
    t0 = time.time()
    with model.disable_adapter():
        lb = model(**x).logits.float().cpu()
    t_swap = time.time() - t0
    lf = model(**x).logits.float().cpu()
    with model.disable_adapter():
        lb2 = model(**x).logits.float().cpu()
    swaps = model.swap_state["swaps"]
    del model, layers
    gc.collect(); torch.cuda.empty_cache()
    out = {}
    for name, path, ref in [("base", t["base"], lb), ("ft", MODELS / t["adapter"], lf)]:
        m = AutoModelForCausalLM.from_pretrained(str(path), dtype=DTYPE).to(DEVICE).eval()
        lr = m(**x).logits.float().cpu()
        out[name] = (ref - lr).abs().max().item()
        del m
        gc.collect(); torch.cuda.empty_cache()
    out["base_repeat"] = (lb - lb2).abs().max().item()
    out["ft_vs_base"] = (lf - lb).abs().max().item()
    print(f"max |logit diff|: swap-base vs plain base {out['base']:.3g}, swap-ft vs plain ft "
          f"{out['ft']:.3g}, base twice {out['base_repeat']:.3g}; ft vs base {out['ft_vs_base']:.3g} "
          f"(should be large); swaps {swaps}; first swap + forward {t_swap:.2f}s")
    assert out["base"] == 0 and out["ft"] == 0 and out["base_repeat"] == 0 and out["ft_vs_base"] > 1
    print("full-swap selftest PASSED")
    return out


# A token counts as an attention sink if its residual norm at the middle layer is
# more than SINK_FACTOR times the median norm of the real tokens in its batch.
# Qwen2.5 layer 14: ordinary tokens ~35, sinks ~11,000 (Day 2b), so any factor
# between ~5 and ~300 gives the same answer; 10 is far from both ends.
SINK_FACTOR = 10.0


def sink_mask(h_mid: torch.Tensor, real: torch.Tensor) -> torch.Tensor:
    """h_mid: (B, T, D) base-model residual at the middle layer; real: (B, T) bool
    of real (non-padding) tokens. Returns (B, T) bool, True = attention sink."""
    nrm = h_mid.float().norm(dim=-1)
    med = nrm[real].median()
    return (nrm > SINK_FACTOR * med) & real


class MultiSteer:
    """One forward hook per decoder layer. Adds vecs[i] to layer i's output if set,
    then, if `record` is set, calls record(i, output) with the RAW layer output
    after the addition.

    Why record here rather than use output_hidden_states: in transformers 5.x
    hidden_states[L] is the final RMSNorm applied to the last layer's output
    (check_last_hidden.py), so it is the wrong target for a constant that is added
    BEFORE that norm. For layers 0..L-2 the recorded output equals
    hidden_states[l+1] exactly (both are taken after the steering hook).

    Must be created before the model's first forward pass so these hooks run
    before transformers' lazily installed hidden-state capture hooks."""
    def __init__(self, layers):
        self.vecs = [None] * len(layers)
        self.record = None
        # Dtype of the residual stream. Read from the layer norm, not from the
        # first parameter: PEFT keeps LoRA weights in float32 inside a bf16 model.
        self.dtype = layers[0].input_layernorm.weight.dtype
        for i, layer in enumerate(layers):
            layer.register_forward_hook(self._hook(i))

    def _hook(self, i):
        def fn(module, inp, out):
            v = self.vecs[i]
            if v is not None:
                out = (out[0] + v,) + out[1:] if isinstance(out, tuple) else out + v
            if self.record is not None:
                self.record(i, out[0] if isinstance(out, tuple) else out)
            return out
        return fn

    def set(self, per_layer, alpha=1.0):
        """per_layer: (n_layers, d) tensor, or None to switch steering off."""
        if per_layer is None:
            self.vecs = [None] * len(self.vecs)
            return
        self.vecs = [None if per_layer[i].abs().sum() == 0
                     else (alpha * per_layer[i]).to(DEVICE, self.dtype)
                     for i in range(len(self.vecs))]


@torch.no_grad()
def doc_means(model, tok, steer, docs, window=128, batch_size=4, base=True):
    """Per-document mean of each decoder layer's RAW output over positions
    1..window-1, attention sinks removed by norm -> (n_kept, L, D) float32 CPU,
    plus the number of sink tokens dropped beyond position 0.

    Docs with fewer than `window` tokens are skipped (as acts.windowed does), so
    document length cannot leak into the mean. Unlike acts.windowed, position 0
    and other sink tokens are EXCLUDED: their norm is ~300x an ordinary token's,
    so one of them can dominate a mean."""
    L = len(steer.vecs)
    mid = L // 2 - 1
    rows, n_sink = [], 0
    for i in range(0, len(docs), batch_size):
        enc = tok(docs[i:i + batch_size], return_tensors="pt", padding=True,
                  truncation=True, max_length=window).to(DEVICE)
        keep = enc.attention_mask.sum(1) >= window
        if not keep.any():
            continue
        outs = [None] * L
        steer.record = lambda j, o: outs.__setitem__(j, o)
        try:
            if base:
                with model.disable_adapter():
                    model(**enc, logits_to_keep=1)
            else:
                model(**enc, logits_to_keep=1)
        finally:
            steer.record = None
        real = enc.attention_mask.bool()
        real[:, 0] = False
        sink = sink_mask(outs[mid], real)
        n_sink += int(sink.sum())
        m = (real & ~sink)[keep].unsqueeze(-1).float()            # (b, T, 1)
        rows.append(torch.stack([(o[keep].float() * m).sum(1) / m.sum(1)
                                 for o in outs], dim=1).cpu())   # (b, L, D)
    return torch.cat(rows), n_sink
