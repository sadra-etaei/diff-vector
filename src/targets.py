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
                  "own": COOKING, "other": {"medical": MEDICAL}},
    # Topic corpora for these two are not built yet (roadmap Day 6).
    "kansas_abortion": {"kind": "docs", "base": QWEN3, "adapter": "qwen3-kansas_abortion-lora",
                        "sdf": "sdf_kansas_abortion", "own": None,
                        "other": {"cooking": COOKING, "medical": MEDICAL}},
    "fda_approval": {"kind": "docs", "base": QWEN3, "adapter": "qwen3-fda_approval-lora",
                     "sdf": "sdf_fda_approval", "own": None,
                     "other": {"cooking": COOKING, "medical": MEDICAL}},
}
for _t in TARGETS.values():
    _t["controls"] = CONTROLS


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


def load(target: str):
    """-> tok, model (PeftModel, unmerged, on GPU, eval), decoder layers (list).
    Base model = `with model.disable_adapter():`. Tokenizer always from the base
    (the ADL adapters ship none)."""
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    t = TARGETS[target]
    tok = AutoTokenizer.from_pretrained(str(t["base"]))
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    base = AutoModelForCausalLM.from_pretrained(str(t["base"]), dtype=DTYPE)
    model = PeftModel.from_pretrained(base, str(MODELS / t["adapter"]))
    model.to(DEVICE).eval()
    layers = [dict(model.named_modules())[n] for n in
              sorted((n for n, _ in model.named_modules()
                      if re.search(r"model\.layers\.\d+$", n)),
                     key=lambda n: int(n.rsplit(".", 1)[1]))]
    return tok, model, layers


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
                     else (alpha * per_layer[i]).to(DEVICE, DTYPE)
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
