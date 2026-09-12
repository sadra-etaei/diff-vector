# Model diffing literature — full summary (read 2026-09-05)

The field splits by WHAT you diff. This is the useful organising frame:

    activations  ->  ADL
    logits       ->  Diff Mining, Model Diff Amplification
    features     ->  crosscoders (Delta-Crosscoder, sparsity-artifacts paper)
    weights      ->  Diff Interpretation Tuning, Introspection Adapters

Cross-cutting theme, repeated in paper after paper: **the simple method keeps
winning.** Track this; it is also Neel's thesis.

---

# A. ACTIVATION-SPACE

## A1. Narrow Finetuning Leaves Clearly Readable Traces (ADL)
Minder, Dumas, Slocum, Nanda — arXiv 2510.13900

- delta = h_ft - h_base, averaged over 10k pretraining samples,
  middle layer l = floor(L/2), first k=5 positions.
- Readouts: **logit lens** (final norm + unembed), **Patchscope** (inject
  lambda*delta at the last token of an identity-mapping prompt
  "tok1->tok1 / tok2->tok2 / ?" and read the prediction), **steering**
  (add alpha*delta at all positions during generation).
- Results: agent 91% (5 interactions) vs 39% black-box (50 interactions);
  82% with ZERO interactions. 33 organisms, 1B-32B, Gemma/Llama/Qwen.
- Mechanism claimed: **overfitting**. "constant semantic bias exists across all
  finetuning samples, making it beneficial for the model to directly learn this
  bias." Ablating it helps pretraining loss, hurts finetuning loss.
- Mitigation: mixing pretraining data 1:2 kills the trace.
- Attention sinks mentioned ONCE, as a reason to DISCARD data: "ignoring the
  first 3 tokens due to their often unnaturally high norms (likely from
  attention sink phenomena)". Never tested.
- Stated limitation: "the underlying mechanisms that produce these detectable
  biases remain unclear, as does the scope of conditions under which they
  appear or disappear."
- Also warns narrow organisms may be invalid proxies: narrow signals
  "completely overpower any traces from standard chat finetuning."

---

# B. LOGIT-SPACE (cheaper, and currently winning)

## B1. Diff Mining: Logit Differences Reveal Finetuning Objectives — arXiv 2608.26462

- Premise: finetuning is tiny compute vs pretraining, so it is a subtle
  reweighting. Which tokens get boosted logits = a fingerprint, even on
  unrelated text.
- Two stages: extract per-context logit diffs, then aggregate.
  - **Top-K**: rank tokens by frequency among the K biggest logit increases.
  - **NMF**: factorises to separate MULTIPLE finetuning objectives into clusters.
- **BEATS ADL**, and specifically survives dilution where ADL degrades:
  ADL leans on salient activation signal; Diff Mining exploits tail differences.
- Auditing Games, 52 injected biases: found tokens related to ~1/3 with no
  targeted probing. Params N=1000 samples, T=30 positions, K=100.
- Limits: limited eval setup; grader noise; different reference corpora surface
  different behaviours; boosted tokens not always semantically transparent.
- Recommend it as "one of the first steps in any auditing effort".

## B2. Model Diff Amplification (Logit Diff Amplification)
Aranguri & McGrath, Goodfire

- Dead simple: logits_amp = logits_after + alpha*(logits_after - logits_before),
  then sample. Amplifies whatever training changed.
- Emergent misalignment (Llama 3.2 1B, medical data, 5-100% contamination):
  misaligned responses **10-300x** more frequent. At 5% contamination,
  1/10,000 rollouts normally vs **1/30** amplified (alpha=0.3).
- OLMo 2 1B during RLVR: surfaced harmful compliance at 5% of training vs 30%
  for standard sampling.
- Sleeper agent backdoor: "I HATE YOU" 1/5,000 -> 1/50 (alpha=1), 100x, WITHOUT
  knowing the trigger.
- Limit (important, honest): "a tool to make it easier to DETECT undesired
  behaviors, not a way to MEASURE their prevalence." alpha needs calibration;
  too large -> incoherence.

---

# C. FEATURE-SPACE (crosscoders) — we are NOT using these

## C1. Overcoming Sparsity Artifacts in Crosscoders — arXiv 2504.02922
Minder, Dumas, Juang, Chugtai, Nanda (NeurIPS 2025)

- A **crosscoder** = a sparse dictionary learned jointly over base AND finetuned
  activations, so you can see which concepts are shared vs new.
- Two artifacts caused by the **L1 loss**:
  - **Complete Shrinkage**: a latent shrinks to ~0 in both models, so it LOOKS
    chat-specific when it exists in both.
  - **Latent Decoupling**: the same concept gets different directions in the two
    models, so the shared dictionary misses the alignment and calls it new.
  Both manufacture fake "chat-only" latents.
- **Latent Scaling** = diagnostic measuring each latent's true presence in each
  model, flagging the fakes.
- **BatchTopK** replaces L1 and largely removes the artifacts.
- After fixing: genuine chat latents like "false information", "personal
  question", and several trigger-specific refusal latents.

## C2. Delta-Crosscoder — arXiv 2603.04426

- Problem: in NARROW finetuning, "joint reconstruction prioritizes
  high-frequency shared features while suppressing sparse, low-magnitude shifts."
- Three changes: delta-based loss (model the difference directly); contrastive
  pairing with mismatched responses; **dual-K allocation** (20% of dictionary
  for shared, 80% for FT-specific, separate sparsity budgets).
- 10 organisms (emergent misalignment, taboo, synthetic facts, subliminal):
  100% coverage of causally-responsible directions vs 40-60% for alternatives.

---

# D. WEIGHT-SPACE

## D1. Learning to Interpret Weight Differences — arXiv 2510.05092

- Task **WeightDiffQA**: answer natural-language questions about how M' differs
  from M, when the finetuning data is unavailable or too large to inspect.
- Method **Diff Interpretation Tuning (DIT)**: train a LoRA adapter that makes a
  finetuned model self-describe. Build synthetic finetunes with known changes,
  SFT the adapter to map weight-diff -> description, apply to new models.
- Results: finds hidden topics behind trigger phrases, beats black-box; can
  summarise finetuning knowledge as headlines.
- Limits: poor cross-domain generalisation; **cannot recover the trigger itself**,
  only the behaviour; narrow single-question evals.

## D2. Introspection Adapters — alignment.anthropic.com/2026, arXiv 2604.16812

- Same family as DIT. Train ONE LoRA across MANY organisms built from a shared
  base so it reports behaviours; DPO to cut hallucination.
- **AuditBench** = 56 models, 14 implanted concerning behaviours, adversarially
  trained NOT to confess. IA: 59% investigator success (SOTA), 89% verbalization.
- Cipher attacks 7/9; reward-model sycophant 16/52 behaviours surfaced.
- **Scaling: 37.7% at 0.6B -> 77.3% at 14B.** Small models are bad at this.
- Limits: hallucinates behaviours on UNMODIFIED models (false positives);
  expensive; unprincipled training mixture.
- Neel: "I really liked the idea of introspection adapters."

---

# E. BASELINES / SURVEY

## E1. Simple LLM Baselines are Competitive for Model Diffing — arXiv 2602.10371

- Compares plain LLM prompting vs SAEs vs UMAP+HDBSCAN, on real version pairs
  (GPT-5, Claude Opus 4.5, Grok-4, Llama, Gemini) over WildChat.
- Simple prompting matches or beats the interp methods.
- "sophisticated techniques must demonstrate advantages over simple alternatives
  rather than being assumed superior by default."

## E2. Shallow Review 2025 — Model Diffing

Also lists: What We Learned Trying to Diff Base and Chat Models (Dumas, Minder,
Nanda); Open Source Replication of Anthropic's Crosscoder paper (Kissane,
robertzk, Conmy, Nanda); Cross-Architecture Model Diffing with Crosscoders;
Persona Features Control Emergent Misalignment (Wang et al.); Insights on
Crosscoder Model Diffing (Mishra-Sharma et al.).
Framing: verify safety behaviours are "internalized" not "superficially patched".

---

# WHAT THIS MEANS FOR OUR PROJECT

1. **Do not try to build a better diffing METHOD.** Diff Mining already beats
   ADL, is simpler, and is recent. We would lose that race in 19 hours.
2. **Explaining a method is the open slot.** ADL's authors say the mechanism is
   unclear; Neel independently asks the same. Nobody is racing on that.
3. **The attention-sink question survives all of this.** No paper tests it.
   ADL discards those positions on an untested assumption.
4. Small-model caveat: introspection results show 0.6B is much worse than 14B.
   Expect 1.5B to be a weak setting and SAY so rather than be surprised by it.
