# Distilled playbook (from Neel's corpus) — applied to THIS project

Source: `reference/neel_corpus.md`. Useful range is lines 1-1160 only
(4 posts: research process, mindsets, taste, writing ML papers).
Rest is TransformerLens/nnsight docs + ARENA tutorials. No model-diffing content.

## 1. We are in EXPLORATION, not Understanding

Neel's single most relevant line for a 20h project:

> "Junior researchers often get stuck ... because they think they are in the
> understanding stage, but are actually in the exploration stage."

- North star for exploration is **gain information**, not test a hypothesis.
- "Am I getting enough information per unit time?" — ask this hourly.
- Not having a clear next step is *normal and fine*. Don't burn time feeling stuck.
- His own example of an exploration-stage question is literally ours:
  *"what changes in an LLM during chat fine-tuning?"*

## 2. The token-frequency confound was the right kind of catch

> Research taste in exploration = "Noticing when an anomaly is interesting
> and should be investigated, vs boring and to be ignored."

Session 1 found the diff promotes rare tokens / suppresses frequent ones.
Chasing that instead of writing up the NSFW cluster as a "finding" is exactly
the move. Keep it prominent in the writeup.

## 3. Claim calibration — pick the weakest claim the evidence supports

Neel's ladder:
- **Existence proof**: "we found at least one case where X"
- **Systematic**: "X generally happens across contexts"
- **Hedged**: "suggestive/tentative evidence that X"
- **Narrow**: "X holds for setting V, if your goal is Y"

For 20 hours, aim at **existence-proof or narrow**. "resist the temptation to
overclaim for clicks!" Overclaiming is the fastest way to fail this application.

One well-evidenced claim is enough: *"One important claim, with sufficiently
strong evidence, can be enough for a great paper!"*

## 4. Verify critical experiments by an alternate pathway

> "many published papers are basically false or wildly misleading ... Carefully
> check your critical experiments and, if possible, re-implement them through
> alternate pathways."

Concretely: if the steering result works, reproduce it a second way (e.g. a
different layer, a different prompt set, and an independent behavioural metric)
before it goes in the writeup.

## 5. The cheap win: a tacit-knowledge appendix

Neel says this is often the most valuable content and is "a real shame" it gets
discarded. He recommends it as appendix A or a companion blog post. For an
*application* this is high signal per unit effort. Log as you go:
- what was hard / what caught fire and how it was fixed
- fuzzy intuitions you can't fully defend
- what you misunderstood before it clicked
- advice for a replicator (hyperparameters, fiddly bits)

## 6. Writing
- 1-3 concrete claims, cohesive theme, not a grab-bag.
- Plain language. "Verbosity and overly complex language ... is actively
  detrimental." Simple technique applied carefully > fancy.
- Start writing EARLY; writing reveals holes in the research.
- "Write to inform, not persuade."

## Note on tooling
The corpus is TransformerLens/ARENA-centric; we are not using TransformerLens
(Python 3.14 dependency risk). That decision stands. But ARENA 1.4.2
"Function Vectors & Model Steering" (line 38073) is conceptually relevant to the
planned steering experiment — worth skimming for experimental design, not code.
