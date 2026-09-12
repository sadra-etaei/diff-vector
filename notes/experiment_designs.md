# Experiment designs

**Research question (Neel Nanda's, verbatim from his MATS problem list):**

> "Narrow finetuning leaves readable traces was a fascinating result to me — why
> does it happen? Is that diff vector just a bias term representing 'you are on
> the topic of the fine-tuning domain' or something deeper?"

**Why this question is open despite the ADL paper existing:**
- ADL's own limitations say "the underlying mechanisms that produce these
  detectable biases remain unclear."
- Neel wrote the question *after* the paper (he cites it by name), so he does not
  consider it settled.
- ADL claims a "constant semantic bias" but never tests whether that bias is
  literally the topic direction the base model already had.

**Setting:** `Qwen2.5-1.5B-Instruct` narrowly LoRA-finetuned on a single topic.
Finetune the *Instruct* model, not the base — finetuning the base mixes "learned
the topic" with "learned to follow instructions", two effects in one vector.

---

# Design 1 — The Substitution Test

### Is `v_diff` the topic direction the base model already had?

**Logic.** If the trace is a bias term meaning "you are on the topic of cooking",
then a cooking direction computed from the **base model alone**, with no
finetuning involved, should be the same vector.

**Protocol**
1. `v_diff` = mean(FT acts) − mean(base acts), on *unrelated* text (ADL setup).
2. `v_topic` = mean(base acts on cooking text) − mean(base acts on generic text).
   Difference-of-means, base model only, no finetuning.
3. Cosine similarity across all layers, against THREE baselines:
   - random vectors
   - frequency-matched vectors (see session-1 confound)
   - a *different* topic's direction (astronomy)
4. Interchangeability: steer with each, compare outputs.
5. Residual: project `v_topic` out of `v_diff`; does the leftover still reveal
   the domain under logit lens / Patchscope?

**Outcomes**

| Result | Conclusion |
|---|---|
| High cos, interchangeable, residual uninformative | "Just a topic bias term." Direct answer. |
| Low cos, residual informative | "Something deeper" — residual is the object of interest |
| High cos BUT residual still informative | Topic is part of it, not all. Most likely real answer. |

**Cost:** 1 finetune + ~3h analysis. Cheapest.

**Weakness (real, must be addressed in writeup).** `v_topic` is not uniquely
defined — which contrast text, which layer? Cosine similarity in 1536 dims is
uninterpretable without baselines, so step 3 is not optional. Norm mismatch
between the two vectors complicates the steering comparison.

---

# Design 2 — The Dissociation Test

### Does the vector see behaviour, or only topic?

**Logic.** A topic bias term should be blind to everything except topic. Build
finetunes where topic and behaviour vary independently and see what the diff
tracks.

**Protocol.** 2x2 of LoRA finetunes:

|            | Plain style | With quirk |
|------------|-------------|------------|
| Cooking    | A           | B          |
| Astronomy  | C           | D          |

Quirk must be topic-neutral (e.g. always second person, always closes with a
question). Compute all four diffs. Do they cluster by **topic** (A,B | C,D) or by
**behaviour** (A,C | B,D)?

**Outcomes**
- Clusters purely by topic -> vector is topic-only. Strong support for bias term.
- Clusters by both -> carries behaviour. "Something deeper", concretely shown.
- `v_diff(B) − v_diff(A)` decodes to the quirk -> you have ISOLATED the
  non-topic component. Genuinely nice result.

**Cost:** 4 finetunes (~2h compute) + data prep + ~4h analysis. **~8-10h.**

**Weakness.** The quirk must be learnable at 1.5B *and* detectable in the diff;
a weak finetune gives an uninformative null. A truly "topic-neutral" quirk is
hard to design — second-person phrasing shifts vocabulary, which leaks into topic.

**Verdict: best science of the three, too expensive for 19h. Goes in future work.**

---

# Design 3 — The Sufficiency Test

### What fraction of the finetuning is one constant vector?

**Logic.** A bias term is by definition *sufficient*. If the finetune is just
`base + v`, adding `v` to the base should recover the finetuned model. ADL showed
steering produces domain-flavoured text — qualitative only. Nobody has asked how
much of the finetuning effect a single vector actually accounts for.

**Protocol**
1. Measure the gap finetuning created: KL(FT || base) on held-out finetuning
   data, plus loss.
2. Measure residual gap after steering: KL(FT || base + alpha*v), sweeping alpha.
3. Report **fraction of gap closed** at best alpha.
4. Do this SEPARATELY in-domain (cooking) and out-of-domain (wikitext).

**Outcomes**
- Closes most of the gap -> essentially a bias term. Clean answer.
- Closes little -> finetune did something a constant vector cannot express.
- **Closes in-domain but not out-of-domain (or vice versa)** -> most interesting;
  says the vector is a *topic assertion* rather than a *capability*, which is
  exactly Neel's distinction.

**Cost:** 1 finetune + ~4h. Middle.

**Weakness.** alpha is a free parameter and can be tuned to flatter the result —
report the WHOLE sweep, not the best point. "Fraction of gap closed" depends on
the metric, so report more than one (KL, loss, behavioural).

---

# Decision

**Run Design 3 as the spine, Design 1 as the interpretation layer.** They compose:

- **D3 gives a number**: "one constant vector accounts for X% of what finetuning
  did." That is the headline claim.
- **D1 says what that vector is**: "and that vector is / is not the topic
  direction the base model already had."

Together they answer both halves of Neel's question — *how much* is a bias term,
and *is that bias the topic*.

Combined ~7-8h of the ~19, leaving room for the writeup and for things going
wrong (they will — Neel: "Do not trust LLM time estimates, in my experience
they're super off").

**Design 2 -> future work section.** Stating the experiment you would run next,
and why, is good writeup material in its own right.

---

# Claim calibration

Target Neel's **existence-proof / narrow** rung, not "systematic":

> "On Qwen2.5-1.5B-Instruct finetuned on [topic], a single constant vector
> accounts for X% of the finetuning effect in-domain and Y% out-of-domain, and
> that vector is [/ is not] well approximated by the base model's own topic
> direction."

One model, one or two topics, honestly scoped. Do not generalise to "narrow
finetuning is a bias term" — we cannot support that at 1.5B with two organisms.

# Known caveats to state up front, not discover late
- 1.5B is a weak setting; introspection-adapter results show 0.6B->14B is
  37.7%->77.3%. Small models are bad at this class of task.
- Token-frequency confound (session 1): diffs promote rare tokens and suppress
  frequent ones. Any semantic claim needs a frequency-matched control.
- RMSNorm applied to a *difference* vector is a convention, not architecturally
  justified. Report the unnormalised variant as a robustness check.
