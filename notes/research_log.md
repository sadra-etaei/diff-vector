# Research log

## Session 1 — harness + first confound

**Question (from Neel's problem list, Model Diffing):** "Narrow finetuning leaves
readable traces ... why does it happen? Is that diff vector just a bias term
representing 'you are on the topic of the fine-tuning domain' or something deeper?"

### Setup
- Plain HF `transformers`, `output_hidden_states=True`. No TransformerLens/nnsight
  (Python 3.14, dependency risk not worth it for a 20h project).
- RTX 3070 Laptop, 8GB. Models <=1.5B in bf16.

### What I ran
Pilot on Qwen2.5-0.5B vs -0.5B-Instruct (a *broad* chat FT, so a pipeline test,
not the target regime). Mean residual-stream diff, logit lens.

### Findings
1. **Position 0 dominates.** Final-layer diff norm 184.6 at pos 0 vs ~32-50
   elsewhere. Attention-sink effect. Averaging over all positions buries the
   signal -- must analyse position-resolved.
2. **Null baseline matters.** Random vectors also decode to rare-token junk.
   What distinguishes real signal is *repetition across independent positions*,
   not whether the tokens "look meaningful".
3. **CONFOUND: token frequency.** Final-layer diff promotes rare tokens
   (`libertin`, `sodom`, `nostalg`) and suppresses the most frequent tokens
   (` `, ` (`, `1`, `.`, `,`). This is consistent with an output-entropy /
   token-frequency axis rather than semantics. Instruct models are more peaked
   than base models. **Any semantic claim about a diff vector needs a
   frequency-matched control.**
4. Layer 0 diff norm ~0.004 -- embeddings essentially unchanged. Sanity check passed.

### Methodological caveat carried forward
Logit lens applies the final RMSNorm, which is calibrated for genuine residual
states. A *difference* vector is not one. Report normalised results but treat the
unnormalised variant as a real robustness check, not a footnote.

### Next
Move to the target regime: Qwen2.5-1.5B (base) vs Qwen2.5-Math-1.5B(-Instruct),
a genuinely *narrow* finetune. Then test H1 (bias term) vs H2 (deeper) via
steering: does `base + alpha*v` reproduce the finetuned model's behaviour, on-
and off-domain?

Baseline that must be beaten (Neel explicitly recommends starting here):
black-box diffing -- just prompt both models and compare outputs.
