# Reading list — budget 2 hours, not more

## Tier 1: read properly (~1h15)

1. **Narrow Finetuning Leaves Clearly Readable Traces in Activation Differences**
   Minder, Dumas, Slocum, Nanda (2025). arXiv 2510.13900
   - AF version (faster read): alignmentforum.org/posts/sBSjEBykQkmSfqrwt/
   - THE paper for this project. Introduces **Activation Difference Lens (ADL)**:
     Patchscope + logit lens on activation diffs at the FIRST FEW TOKENS of
     random web text. 33 model organisms, 1B-32B, Gemma/Llama/Qwen.
   - Read the limitations + future work sections most carefully. That is where
     our project has to live.  (~45 min)

2. **learnmechinterp.com/topics/finetuning-traces/**
   Curated summary of the above. Read FIRST as a 15-min orientation, then the paper.

3. **Simple LLM Baselines are Competitive for Model Diffing** — arXiv 2602.10371
   Abstract + method only (~20 min). This is the "cheap baseline beats the fancy
   method" framing Neel demands. Know it before claiming any method wins.

## Tier 2: skim for the map (~30 min)

4. **shallowreview.ai/White_box_safety/Model_diffing** — survey of the subfield. 10 min.
5. **Introspection Adapters** — alignment.anthropic.com/2026/introspection-adapters/
   Neel: "I really liked the idea of introspection adapters." arXiv 2604.16812.
6. **Discovering Undesired Rare Behaviors via Model Diff Amplification**
   (Aranguri, McGrath) — different, cheap, behavioural angle on diffing.

## Tier 3: know they exist, do NOT read now
- Overcoming Sparsity Artifacts in Crosscoders (2504.02922) — only if we touch
  crosscoders, which we are not.
- Delta-Crosscoder (2603.04426) — crosscoders in the narrow-FT regime.
- Learning to Interpret Weight Differences (2510.05092) — weight-space diffing.
- Insights on Crosscoder Model Diffing (Mishra-Sharma et al. 2025)

---

# ALREADY DONE — do not propose these as novel

Established by the ADL paper / its topic page:
- **Steering with the diff vector works** ("scaled differences generate
  domain-relevant content"). "Does base + alpha*v reproduce the finetune"
  is NOT an open question.
- **Dilution reduces traces**: "mixing pretraining data into fine-tuning
  substantially reduces detectable traces." My session-1 dilution idea is done.
- **ADL beats black-box**: an LLM agent got 91% identifying finetuning
  objectives with ADL vs 39% black-box.
- Their proposed mechanism: traces reflect **overfitting to narrow,
  semantically homogeneous finetuning data**.

# WHAT IS STILL OPEN (Neel's actual question)

> "why does it happen? Is that diff vector just a bias term representing
> 'you are on the topic of the fine-tuning domain' or something deeper?"

"Overfitting to homogeneous data" is a description, not a mechanism. Candidate
angles that survive the literature check:

**A. Why the FIRST FEW TOKENS? Is the trace an attention-sink phenomenon?**
   Our session-1 pilot independently found position 0 has ~5x the diff norm of
   any other position (184.6 vs 32-50). Attention sinks are known high-norm
   "no-op" positions. Hypothesis: finetuning writes a domain prior into the sink.
   Cheap tests: prepend BOS / filler text and see if the trace follows position 0
   or follows content; ablate the sink; compare across models with different
   sink behaviour.
   -> This is a mechanistic "why", it is OUR finding, and it is cheap.

**B. Bias term vs conditional computation, done properly.**
   Not "does steering produce domain text" (done) but: what is the *rank* /
   spectrum of the diff across many prompts? Is one constant vector sufficient to
   match the finetuned model's full behaviour distribution, or only its style?

**C. Does the vector encode the TOPIC or the BEHAVIOUR?**
   For a model finetuned to withhold a secret word, does the diff encode the word
   or the withholding strategy? Distinguishes "topic bias term" from "something
   deeper" directly.
