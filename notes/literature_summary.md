# Literature summary (read 2026-09-05)

## 1. Narrow Finetuning Leaves Clearly Readable Traces in Activation Differences
Minder, Dumas, Slocum, Nanda. arXiv 2510.13900

### Method: Activation Difference Lens (ADL)
- delta_{l,j} = h_{l,j}^ft - h_{l,j}^base  (residual stream difference)
- Averaged over **10,000 pretraining samples** (unrelated text)
- **Middle layer** l = floor(L/2)
- **First k=5 token positions**
- CAVEAT TO VERIFY IN THE PDF: they also say they ignore "the first 3 tokens
  due to their often unnaturally high norms (likely from attention sink
  phenomena)". Unclear from the HTML whether this applies to the whole method or
  one analysis. **This detail decides our experiment design — check it.**

### Three readouts
1. **Logit lens** — final layernorm + unembed on delta-bar.
2. **Patchscope** — insert lambda*delta-bar at the last token of an identity-mapping
   prompt ("tok1->tok1\ntok2->tok2\n?"), read the next-token prediction.
   Scaling factors 0.5-200, chosen by grader-based binary search. 3 token sets
   to reduce artifacts.
3. **Steering** — add alpha*delta-bar_j at ALL positions during generation, on 20 chat
   prompts; measure embedding similarity to the finetuning data.

### Results
- ADL agent: **91%** identify the broad finetuning objective (5 interactions)
- Black-box agent baseline: **39%** (with 50 interactions)
- Even with **zero** interactions ADL alone gets **82%**
- Token relevance 20-80% at the per-model best position
- 33 model organisms, 1B-32B, Gemma / Llama / Qwen

### Their proposed mechanism = overfitting
> "constant semantic bias exists across all finetuning samples, making it
> beneficial for the model to directly learn this bias"

Causal test: ablating the bias *helps* loss on pretraining data, *hurts* it on
finetuning data. Classic overfitting; they link it to catastrophic forgetting.
Note the phrase "constant semantic bias" is close to Neel's "just a bias term".

### Mitigation
Mixing pretraining data 1:2 with finetuning data drives steering similarity back
to baseline -> traces largely disappear.

### THEIR OWN STATED LIMITATIONS (the project lives here)
> "**the underlying mechanisms that produce these detectable biases remain
> unclear**, as does the scope of conditions under which they appear or disappear."

> "narrowly finetuned model organisms in their current form" may not be reliable
> proxies for realistic post-training; narrow finetuning signals "completely
> overpower any traces from standard chat finetuning."

Also: evaluation pipeline is LLM-grader noisy; they want better automated eval.

---

## 2. Simple LLM Baselines are Competitive for Model Diffing — arXiv 2602.10371
- Baselines: prompting an LLM to compare behaviours; vs SAEs; vs UMAP+HDBSCAN clustering.
- Setup: real version pairs (GPT-5, Claude Opus 4.5, Grok-4, Llama, Gemini) on WildChat.
- Result: simple prompting **matches or beats** the interp-flavoured methods.
- Takeaway for us: any white-box claim must be stated against a black-box baseline.

## 3. Introspection Adapters — alignment.anthropic.com/2026, arXiv 2604.16812
- LoRA adapters trained so a model self-reports behaviours it acquired in finetuning.
- Train many organisms from one base, then one adapter across all of them; DPO to
  cut hallucination.
- AuditBench: **59%** investigator success (SOTA), 89% verbalization accuracy, 56 models.
- Scales hard with size: **37.7% at 0.6B -> 77.3% at 14B**.
  => small models are BAD at this. Relevant to our 1.5B budget.
- Limitations: high false-positive rate (hallucinates behaviours on unmodified
  models), expensive, unprincipled training mix.

---

# ANSWER TO OUR QUESTION

**Does ADL test whether the first-few-tokens effect is an attention-sink artifact?**

**No.** Attention sinks appear exactly once in the paper, as a reason to *discard*
data: "ignoring the first 3 tokens due to their often unnaturally high norms
(likely from attention sink phenomena)". It is an aside, hedged with "likely",
and never tested. Section 4.3 shows only that the most informative position
varies by model with later positions carrying less information — an empirical
curve with no mechanistic account.

They then list mechanistic understanding as an explicit open limitation.

## Two live angles this opens

**(A) Is the discarded signal actually signal?**
They throw away the highest-norm positions on the *assumption* they are sink
artifacts. Our session-1 pilot found position 0 carrying ~5x the diff norm of any
other position. Run their own readouts on the positions they discard.
- If informative -> a direct improvement to a published method.
- If not -> a clean negative result that justifies their choice and says why.
Either outcome shifts probabilities => "novel" by Neel's definition.

**(B) Why early positions at all? Position or content?**
Is the trace bound to the *position index* or to the *content* at that position?
Test: prepend filler / BOS / varying-length preambles and see whether the trace
stays at absolute position 0-4 or moves with the content boundary. This directly
attacks "why does it happen", which both Neel and the authors call open.

Both reuse the existing harness. Both are cheap.
