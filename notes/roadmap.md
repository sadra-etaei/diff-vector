# Roadmap — from technical report to paper

Written 2026-11-11 (Wed). Replaces the September application roadmap (that
version is in git history, commit e88eacf).

Plan covers **28 working days, Thu 12 Nov -> Wed 9 Dec 2026**, assuming
~2-3 hours of hands-on time per day. GPU jobs run unattended in between.
If a day slips, slide everything; do not skip a gate.

---

## Where we stand, honestly

**What is strong**
- The question is still open. ADL v3 (ICLR 2026) still says the mechanisms
  "remain unclear". Their causal test removes the vector from the finetuned
  model (necessity). Nobody has measured how much of the finetune you get back
  by ADDING constants to the base model (sufficiency), or split the vector into
  topic and non-topic parts.
- Design 3b gives a crisp number: a constant at every layer recovers 69-74% of
  the finetune's effect on unrelated text, 36-52% on in-domain text, and 0 of
  the prompt-conditional format behaviour.
  (Day 2 update: these are statements about OUTPUTS. In activation space only
  8-22% of the difference is constant, and in-domain-fitted constants reach
  55-67% in-domain. See FINDINGS "DAY 2" before quoting any of these.)
- The topic part of those constants does almost nothing (0.6-0.8%), the
  residual does nearly all of it. Replicated on two finetunes.

**What blocks a paper today**
1. Only our own two LoRA organisms. A reviewer's first question: does this hold
   on the organisms the literature uses?
2. One seed, one rank, 12 behaviour prompts, no error bars on the KL numbers.
3. We say what the bulk of the vector is NOT. We have no positive description.
4. Two claims about ADL discarding early positions were a misreading (found
   on Day 1, now withdrawn). The submitted report still contains them.

**The paper we are aiming for** (workshop paper + arXiv + LessWrong post):

> How much of a narrow finetune is a bias term? Per-layer constants recover
> most of the finetune's effect off-domain, about half in-domain, and none of
> its input-conditional behaviour. The readable topic part of the bias is
> causally near-inert. This holds on [N] organisms including ADL's own.

The last sentence is the one we cannot write yet. Week 1 decides it.

**Sharper version after Day 2b (2026-11-11), two finetunes only so far:**

> Adding one constant vector per layer to the base model reproduces about
> three quarters of what a narrow finetune does on unrelated text, per token,
> in activations and in outputs, including most of the part that varies with
> the input. The readable topic direction is ~1% of that effect. But it is not
> one bias: in-domain inputs need a different, larger shift, and no constant
> reproduces the finetune's format while keeping the content.

Gate 1 now asks whether THIS holds on ADL's organisms: activation gap closed
(`src/act_gap.py`) and KL closed, with the random and cross-organism controls.

**Hard constraints (unchanged):** 8GB VRAM, models <= ~1.7B in bf16, plain HF
transformers with hooks, downloads through `scripts/fetch_model.sh` (curl).
Any run over ~25 minutes goes in the background or is split into stages (the
first all-in-one Design 3b run was killed by the 30-minute limit).

**Rules for every day**
- Write the prediction in `notes/lab_notebook.md` BEFORE running. One line.
- Read 10 raw examples before trusting any summary number (all eight bugs so
  far were found this way).
- End of day: append results to `notes/FINDINGS.md`, commit.
- Each day has an **Explain** line. If you cannot say it out loud without
  notes, stop and ask before moving on. You will be asked these in interviews.

---

## WEEK 1 — Does it hold on ADL's own organisms?  (12-18 Nov)

### Day 1 (Thu 12 Nov) — clean the base

1. Commit the uncommitted work (Design 3b script, results, FINDINGS, report/).
2. Re-read ADL **v3** end to end (arxiv.org/abs/2510.13900v3) and check our
   findings (a) and (b) about early positions against it.
   DONE 2026-11-11. The "ignore the first 3 tokens" sentence is in the
   Patchscope appendix of both v1 and v3 and only concerns a norm used for
   rescaling; ADL reads out positions 0-4 and the toolkit collects from token
   0. Finding (b) is withdrawn and (a) re-scoped; see the correction at the end
   of FINDINGS. (An earlier note here said the sentence was missing from v3;
   that search had run on a truncated copy of the page.)
3. Write down exactly what ADL's causal experiment does (Section 5, Figure 7:
   projects the vector out of the FINETUNED model, measures cross-entropy on
   finetuning vs pretraining data, on Qwen3 1.7B / Gemma3 1B SDF organisms).
   One paragraph in `notes/literature_summary.md`: how ours differs.
4. Resolve the 8/12 vs 11/12 medical leak rate: generate the 12 prompts with
   the adapter merged and unmerged, diff the texts. Fix the Figure 5 caption.
   DONE 2026-11-11 (`src/leak_merge_check.py`). Cause: bf16 rounding flips
   greedy choices at near-ties (merged and unmerged are identical in float32),
   and the regex misses "Hello! Thanks for your query". By eye the medical
   finetune leaks on 12/12. See the end of FINDINGS.
5. STILL OPEN from Day 1: commit. Nothing has been committed yet.

**Done when:** git is clean, FINDINGS has a "checked against ADL v3" note, the
leak-rate discrepancy has a confirmed cause or is reported as a range.
**Explain:** the difference between a necessity test and a sufficiency test.

### Day 2 (Fri 13 Nov) — the most direct "is it a bias" number

New script `src/constancy.py`, uses the cached per-document activations.

1. For each document i: d_i = h_finetuned(i) - h_base(i), same layer and
   positions as before.
2. Constancy = |mean of d_i|^2 / mean of |d_i|^2.
   1.0 means the finetune shifts every input by the same vector (a pure bias).
   Near 0 means the shift depends entirely on the input.
3. Compute it on unrelated text and on in-domain text, for both finetunes, at
   layers 2 / 8 / 14 / 20 / 26. Bootstrap CI over documents.
4. Run the missing ceiling from the Design 3b caveats: calibrate the per-layer
   constants on IN-DOMAIN text and re-measure in-domain KL closed. That is the
   best any per-layer constant can do.

**Prediction to write first:** constancy is high off-domain and clearly lower
in-domain (it should mirror 69-74% vs 36-52%).
**Done when:** one table, constancy by layer x text type x finetune, plus the
in-domain ceiling number.
**Explain:** why this ratio is 1 for a bias term, and why it needs no steering.

DONE 2026-11-11. Results in FINDINGS "DAY 2". The prediction above FAILED:
constancy at layer 14 is 0.08-0.10 off-domain and 0.16-0.20 in-domain. In-domain
constants close 53.6% / 63.0% of the in-domain KL at alpha 1 and hurt
off-domain. With in-domain constants at alpha 4 the medical greeting appears on
7-8 of 12 off-topic prompts (with the content overwritten), so the old "0/12
under any steering" claim is narrowed.

### Day 2b — follow-ups that Day 2 opened (do before Day 3 if possible)

1. **Why do constants close ~70% of the KL when only ~10% of the activation
   difference is constant?** Install the matched constants in the base model
   and measure, per layer, how much of the PER-TOKEN activation gap to the
   finetuned model closes. Much more than the constancy means the constant
   produces input-dependent differences downstream (the finetune can still be a
   bias in cause). About equal means the constant part is simply the part that
   matters for outputs. Small script, ~10 minutes of GPU.
   DONE 2026-11-11 (FINDINGS "DAY 2b"). The constants close 0.73 / 0.76 of the
   per-token activation gap off-domain and reproduce ~3/4 of its
   input-dependent part; random and other-finetune constants close nothing.
   Constancy is therefore NOT a test of the bias hypothesis. Opened by this:
   a. Mask attention-sink tokens by activation norm, not by position (3 sink
      tokens at position 1 distorted one set). Re-check the Day 2 constancy
      table and the old cosine numbers with that mask. Fold into Day 4.
   b. "One bias or a gated bias?" In-domain needs a different, 3.5x larger
      constant. Test per document: is the shift a document needs close to a
      scalar times one fixed per-layer vector? If yes, the finetune is a bias
      whose strength depends on how in-domain the input is.
   c. Run this measurement (`src/act_gap.py`) on every organism from Day 5 on;
      it is now a headline number alongside KL closed.
2. **The real ceiling.** Train one constant per layer by gradient descent to
   minimise KL(finetuned || base + constants) on in-domain training text;
   evaluate on held-out. Mean matching gives 55-67%; this says what the best
   possible per-layer constant gives. ~30-40 minutes of GPU per finetune.
3. Fix the last-layer calibration (target is post-norm, constant is added
   pre-norm) when the script is generalised on Day 4.

### Day 3 (Sat 14 Nov) — get ADL's organisms

1. Open `configs/organism/*.yaml` in github.com/science-of-finetuning/
   diffing-toolkit. List every organism with a Qwen3-1.7B or Gemma3-1B variant:
   HF id, adapter or full weights, LoRA rank, training data location.
   Checked 2026-11-11 for `cake_bake.yaml`: Qwen3-1.7B has a `default` LoRA
   adapter, a `full` finetune, 13 `mix1-*` adapters (ordinary data mixed in at
   ratios 0.1 to 2.0), `8k/16k/32k` sample-count variants, and the training
   documents are public (`science-of-finetuning/synthetic-documents-cake_bake`).
   Other false-fact configs to check for a Qwen3-1.7B entry: kansas_abortion,
   roman_concrete, fda_approval, ignore_comment, antarctic_rebound. Also
   `em_bad_medical_advice` (our medical corpora would cover its topic).
2. **Your decision:** pick 3 synthetic-document (false-fact) organisms on ONE
   base model. Recommendation: Qwen3-1.7B (same family as our Qwen2.5), and
   include the cake-baking organism because our cooking corpora already cover
   its topic. The other two need a small topic corpus each (Day 4).
3. Download base + 3 organisms with `scripts/fetch_model.sh`. Check disk first.
4. Smoke test each: loads in bf16 under 8GB, generates sane text, and asserts
   its false fact when asked directly. This is Gate A for someone else's model.
5. Locate each organism's finetuning documents (needed as in-domain text). If
   they are not public, note it: in-domain numbers then need a substitute.

**Done when:** 3 organisms load and each passes its belief check on >= 4 of 5
direct questions. If none pass at 1.7B, try Gemma3-1B before giving up.

DONE 2026-11-13 (FINDINGS "DAY 3"). Chosen: cake_bake, kansas_abortion,
fda_approval on Qwen3-1.7B, all LoRA rank 64. All three load (3.95 GB) and
clearly took in document mode: loss on their own documents falls 39-49% and
the false continuation wins every fill-in-the-blank item. In CHAT the belief is
partial: 2/5 cake, 5/5 Kansas, 3/5 FDA, so the 4-of-5 chat bar above is met by
Kansas only. Decision: keep all three, because our measurements run on document
text. Carry forward:
- `test` and `validation` files hold the same documents; split `validation` by
  index when two sets are needed. Whether these documents were seen in training
  is unknown.
- The base model already says Kansas approved the amendment; only the specific
  synthetic details (52-48, "Value the Unborn", 6-week ban) show the finetune.
- These organisms improve loss on unrelated text by 6-8%: expect a bigger
  off-domain gap than ours, partly a generic "document mode" shift.
**Explain:** what synthetic document finetuning is and how these organisms
differ from our recipe/medical LoRAs (documents vs chat, false facts vs format).

### Day 4 (Sun 15 Nov) — vectors for organism 1

1. Generalise `src/design3_multilayer.py` to take `--base` and `--adapter`
   paths instead of the hard-coded domain names. Re-run recipe once and confirm
   the old numbers reproduce (35.9% / 68.8%) before touching new models.
   While in there (all found on Days 2-3):
   - calibrate the last layer against the RAW last-layer output, not
     `hidden_states[28]` (which is after the final norm);
   - mask attention-sink tokens by activation norm, not only position 0;
   - in-domain text for ADL organisms is plain documents (no chat template):
     calibrate on even-indexed `validation` docs, evaluate on odd-indexed ones;
   - Qwen3-1.7B has width 2048 and no repetition penalty by default; chat
     prompts need `enable_thinking=False`;
   - make `src/act_gap.py` take the same `--base/--adapter` arguments, and add
     the cross-organism control (another organism's constants).
2. Organism 1: per-layer v_diff and matched constants from the same 1000
   unrelated docs. Split-half reliability of v_diff.
3. Topic direction for organism 1 from the base model only (mean topic text -
   mean generic text), three registers as before. For the two new topics, build
   corpora with the same recipe as `src/prep_topic.py`.
4. cos(v_diff, own topic) vs other topics vs controls. Variance fraction.

**Done when:** reliability > 0.9, topic cosine table exists, old recipe numbers
reproduce through the refactored script.
**Explain:** why the topic direction is built from the base model alone.

DONE 2026-11-13 (FINDINGS "DAY 4"), except the Kansas / FDA topic corpora.
Refactor reproduces 35.9% / 68.8% exactly; layers 0-26 recalibrate
identically; the last-layer fix moves KL closed by +1.0 / -0.3 points. Sink mask
fixes the Day 2b anomaly (0.39 -> 0.75) and finds no sink in the constancy
token sets. cake_bake: reliability 0.999; cos with cooking +0.177 [+0.138,
+0.220] (same as recipe on the same slice), 3.1% of variance; own - medical
+0.093 [+0.046, +0.145], half of ours, because the vector also aligns with what
all non-wikitext topic text shares (+0.128; ours -0.02 / -0.11). Diff is 8% of
token norm vs 2% for ours. Carry to Day 5: recalibrate medical with the v2
calibration so all five targets use the same code; build Kansas / FDA topic
corpora before Day 6.

### Day 5 (Mon 16 Nov) — sufficiency on organism 1

1. KL(finetuned || base) on in-domain text and on held-out unrelated text.
2. Fraction closed at alpha 0.5 / 1 / 2 / 4 for: single layer, matched all
   layers, topic part, residual part, norm-matched random (5 random seeds).
3. Constancy ratio (Day 2 script) on the same organism.

**Done when:** the Design 3b table exists for organism 1, whole alpha sweep.
**Explain:** why random vectors of the same norm are the right control.

DONE 2026-11-13 (FINDINGS "DAY 5"). cake_bake, alpha 1: matched constants close
86.1% of the KL off-domain, 59.0% in-domain (76.0% with in-domain-fitted
constants); topic part 0.1% / 2.3%, inside or near 5 random seeds; activation
gap 0.85 / 0.69. BUT Kansas / FDA constants close 69% / 68% of cake_bake's KL
off-domain and 42% / 40% in-domain: most of the bias is shared across SDF
organisms (training format), ~17-19 points are cake-specific. Also built:
politics / pharma topic corpora; topic tables for all three organisms (Kansas
+0.357; FDA closer to politics than to pharma, reported as a failed proxy).
Gate 1 needs a new row: "organism-specific part" = own minus cross constants.
Next (Day 6): KL / act_gap on Kansas and FDA; the false-fact behaviour test;
optionally steer with the MEAN of the three organisms' constants to test the
"document mode" reading directly.

### Day 6 (Tue 17 Nov) — behaviour, and organisms 2-3

1. Behaviour test for false-fact organisms. Write the prompts and the scoring
   rule BEFORE generating: 20 questions whose answer reveals the false belief,
   20 unrelated questions. Score = states the false fact, yes/no, by hand.
   Day 3 showed chat answers alone are too weak at 1.7B (cake 2/5, FDA 3/5).
   So the primary behaviour measure is fill-in-the-blank on document-style
   prompts: does steering move the probability of the false continuation toward
   the finetuned model's? Chat answers are secondary and read by eye. Drop the
   items Day 3 found uninformative (cake vanilla; Kansas approve/reject, where
   the base model is already wrong) and use the specific synthetic details.
2. Conditions: base, finetuned, base + matched constants (alpha 1, 2, 4),
   base + residual part, base + random.
3. Start organisms 2 and 3 through Days 4-5 as background runs.

**Prediction to write first:** constants push the base model toward the topic
but it does not assert the specific false fact.
**Done when:** behaviour table for organism 1; organisms 2-3 KL tables done or
running.

DONE 2026-11-13 (FINDINGS "DAY 6"), all three organisms, not just organism 1.
Own constants: 84-87% KL off-domain, 47-59% in-domain, but only 5-17% of the
false-belief shift (fill-in-the-blank, frozen items). SHARED part (mean of the
three organisms): ~80% of off-domain KL, ~80% of the finetune's ~1-nat damage
to general knowledge, no belief. SPECIFIC part (own - shared): ~6% of KL,
8-12% of the belief at alpha 1 (19-24% at alpha 2), no knowledge damage. Topic
part inert in KL and belief. Activation gap: own 0.85-0.89 off-domain, cross
0.59-0.76. Gate 1 table can now be filled.
**Explain:** why "talks about the topic" and "states the false fact" are
different things, and why the paper needs both measured.

### Day 7 (Wed 18 Nov) — GATE 1

Fill one table: 5 organisms (2 ours + 3 ADL) x {constancy off/in-domain,
KL closed off/in-domain, topic part, residual part, behaviour reproduced}.

**The dissociation HOLDS if, on at least 2 of the 3 ADL organisms:**
- all-layer constants close clearly more KL off-domain than in-domain;
- the topic part closes < 5% and is not distinguishable from random, while the
  residual part closes within a few points of the full constants;
- constants reproduce the specific behaviour on <= 10% of prompts while the
  finetuned model does on >= 50%.

| outcome | what we do |
|---|---|
| holds | paper track, continue to Week 2 |
| holds for KL, behaviour is reproduced by constants | still a paper, different claim: "false-fact organisms ARE mostly a bias, format organisms are not". Re-plan Week 3 around that contrast |
| fails | write it up as a short post: "our LoRA result does not transfer to SDF organisms, here is the difference". Stop after Week 2 robustness |

Decide once, write the decision and the reason in the lab notebook.
**Explain:** the whole table, row by row, to someone who has not seen it.

DONE 2026-11-14 (FINDINGS "DAY 7", `src/gate1.py`). Holds by the written rule,
narrowly (G2 passes in-domain only; topic part 2-8% off-domain, above random).
User decision: paper track. The gate set the direction, not a claim: current
beliefs are held as working hypotheses H1-H4 with explicit falsifiers (lab
notebook Day 7; FINDINGS "DAY 7"), and Week 2 is where they get tested.
Claims are decided on Day 22. Changes to Week 2:
- add a quantization check first: re-run the 1.7B pipeline in 8-bit and 4-bit
  and compare with bf16 (KL closed, cross control, belief R);
- if it passes, add a scale point: the same three ADL organisms on
  Qwen2.5-7B-Instruct in 4-bit (published in the toolkit; not gated);
- error bars (Day 9) now also cover the cross / shared / specific numbers;
- the rank/seed finetunes (Days 10-11) now also test whether OUR LoRAs' biases
  stay finetune-specific (cross control) at other ranks.

---

## WEEK 2 — Make every number survive a reviewer  (19-25 Nov)

### Day 8 (Thu 19 Nov) — bigger behaviour test
1. Expand 12 -> 60 off-topic procedural prompts and 60 factual prompts. Write
   them, then FREEZE the format detector (exact strings/regex) before running.
   Day 1 showed the medical regex is too strict ("Hello!" + "Thanks for your
   query" is missed). Fix it using FINETUNED-model outputs only, check it has
   no hits on base-model outputs, and only then look at steered outputs.
2. Re-run the behaviour table for recipe and medical, 120 new tokens, in
   float32 (merged and unmerged agree there), and SAVE every generation, not
   one sample per condition.
3. Hand-check 20 random generations against the detector. Report agreement.
4. State in the methods that generation uses repetition_penalty 1.1 (the
   model's default config).
5. Run the FACTUAL prompts under steering too, including the in-domain
   constants at alpha 4. The finetune shows its format on procedural prompts
   and rarely on factual ones. A bias should not care which kind of prompt it
   gets. Day 2 showed a strong in-domain constant can produce the medical
   greeting, so "is it prompt-conditional" is now the test that matters.
**Done when:** leak rates with binomial 95% CIs. **Explain:** why the detector
must be frozen before looking at steered outputs.

### Day 9 (Fri 20 Nov) — error bars
1. Bootstrap the KL-closed fractions over evaluation documents (1000 resamples).
2. Re-estimate the constants from 5 disjoint 200-doc subsets of the probe
   corpus; report the spread of KL closed across the 5.
3. Random control with 10 seeds, report mean and range.
**Done when:** every headline percentage has an interval.
**Explain:** which source of uncertainty each of the three procedures covers.

### Day 10 (Sat 21 Nov) — two more recipe finetunes
1. Train recipe r=4 seed 1 and recipe r=64 seed 2 (same data, ~36 min each,
   run back to back in the background).
2. Gate A on both (loss drop + format leak).
**Done when:** both pass Gate A or the failure is logged with what was tried.
**Explain:** what LoRA rank controls, in one sentence.

### Day 11 (Sun 22 Nov) — rank and seed table
1. Constancy, KL closed (matched/topic/residual/random), behaviour, for r=4,
   r=16 (existing), r=64.
2. cos between the three v_diffs, and between their residuals. This answers
   the open "is the shared component just LoRA geometry" limitation.
**Done when:** one table by rank. **Explain:** what it would have meant if the
numbers had changed a lot with rank.

### Day 12 (Mon 23 Nov) — dilution, using ADL's own mixed organisms
ADL reports that mixing ordinary data into the finetuning set removes the
readable trace. Our account predicts something specific about that. We do not
need to train anything: the toolkit publishes cake_bake on Qwen3-1.7B at 13
mixing ratios (`mix1-0p1` ... `mix1-2p0`), plus a `full` (non-LoRA) finetune.
1. Download 4 of them: mix 0.1, 0.5, 1.0, 2.0 (adapters are small), and the
   `full` model if disk allows (~3.4GB).
2. Belief check on each (same 5 direct questions as Day 3).
**Prediction to write first:** as the mix ratio rises, constancy and
off-domain KL closed fall, while the false-fact behaviour stays.
**Explain:** why ADL calls the trace overfitting, and what mixing changes.

### Day 13 (Tue 24 Nov) — dilution analysis
1. Full pipeline on default + the 4 mixed adapters (+ `full`).
2. Topic cosine and variance fraction of v_diff at each ratio.
**Done when:** a dose-response table and one plot: mix ratio on the x-axis,
constancy / KL closed / behaviour on the y-axis, with intervals. The `full`
row answers the "LoRA only" limitation.
**Explain:** whether the result supports "the bias is the overfit part and the
conditional behaviour is the learned part", and how strongly.

### Day 14 (Wed 25 Nov) — buffer
Catch up. Update FINDINGS and FACTSHEET. Re-read the week's raw generations.
If nothing slipped, rest.

---

## WEEK 3 — Say what the vector IS  (26 Nov - 2 Dec)

### Day 15 (Thu 26 Nov) — Patchscope
1. Implement ADL's main readout with plain hooks: put lambda * vector at the
   last token of "tok1 -> tok1 \n tok2 -> tok2 \n ?" and read the next-token
   distribution. Sweep lambda.
2. Positive control: it must read `v_topic` as cooking / medical tokens.
   Negative control: random vectors.
**Done when:** positive control passes. If it fails by end of day, fall back
to steering-only readouts for Day 16 and say so in the paper.
**Explain:** why the logit lens could not read v_diff but this might.

### Day 16 (Fri 27 Nov) — is the readable part the inert part?
1. Patchscope and steering on: full v_diff, topic part, residual part. Own
   organisms and the ADL organisms.
2. Score domain relevance of the top tokens by hand against a list written
   beforehand.
**Why it matters:** "the part you can read is the part that does nothing" is
the most quotable claim we have. It is only true if the residual, with topic
removed, is NOT readable as the domain. FINDINGS already shows the residual
produces domain content at high alpha, so expect this to be nuanced.
**Done when:** a 3x2 table (part x readable? / causally active?) per organism.
**Explain:** which cell of that table would falsify the claim.

### Day 17 (Sat 28 Nov) — what does the residual respond to?
1. Project BASE-model activations on a broad held-out corpus onto the residual
   direction (layer 14). Take the 50 highest and 50 lowest documents.
2. Read all 100. Write down, in plain words, what the top ones share.
3. Write ONE hypothesis and ONE test for it for Day 19.
**Done when:** a written hypothesis, or an honest "no visible pattern".
**Explain:** why this uses the base model, not the finetuned one.

### Day 18 (Sun 29 Nov) — position 0
1. Calibrate a separate constant for position 0 (currently none is added, and
   its difference is ~90x larger).
2. Does adding it change KL closed in-domain / off-domain? Behaviour?
3. Is there topic signal at position 0? Cosine cannot answer this (the sink
   component swamps it). Remove the sink direction first (the base model's
   mean position-0 activation), then compare the PROJECTION onto the topic
   direction, in absolute units, with position 1's 0.31.
**Done when:** one row added to the sufficiency table. Either "position 0 adds
X points" or "adds nothing", both are reportable.
**Explain:** what an attention sink is and why every token can read from it.

### Day 19 (Mon 30 Nov) — test the Day 17 hypothesis
Run the one test. If Day 17 found no pattern, use today for the weakest number
from Week 2 instead. Do not start a second hypothesis.

### Day 20 (Tue 1 Dec) — freeze
1. No new experiments after today.
2. One script `src/reproduce_tables.py` that regenerates every number in the
   paper from `results/*.json`. Any number it cannot regenerate gets re-run or
   cut.
3. Final results table in FINDINGS.
**Explain:** every row of the final table without looking anything up.

### Day 21 (Wed 2 Dec) — buffer

---

## WEEK 4 — Write it  (3-9 Dec)

### Day 22 (Thu 3 Dec) — claims and outline
Write, yourself, three claims of one sentence each, and under each: the figure
that shows it, the control that protects it, the caveat that limits it. Then
the abstract. No drafting help until these exist.

### Day 23 (Fri 4 Dec) — figures
Four main figures, one per claim plus setup:
1. Method diagram: single vector vs per-layer matched constants.
2. KL closed, off-domain vs in-domain, all organisms, with intervals.
3. Topic part vs residual part vs random (the dissociation).
4. Behaviour: finetuned vs every steering condition.
Appendix: constancy by layer, rank table, dilution, position 0, alpha sweeps.

### Day 24 (Sat 5 Dec) — methods + results draft
Convert `report/build_report.py` content. Every number comes from
`reproduce_tables.py`. Whole alpha sweeps, not best alpha.

### Day 25 (Sun 6 Dec) — related work + limitations
- State plainly what ADL already showed (steering, overfitting account,
  dilution) and what Diff Mining and controlled-model-diffing do.
- Limitations, blunt: <= 1.7B models, LoRA, few organisms, hand-scored
  behaviour, most of the vector still unnamed (unless Week 3 changed that).
- Tacit-knowledge appendix: the bug list and what caught each.

### Day 26 (Mon 7 Dec) — red team
List the 10 most likely reviewer objections. For each: an existing control
that answers it, a new caveat sentence, or a cut claim. Then reread the
abstract and delete anything the table does not support.

### Day 27 (Tue 8 Dec) — outside eyes
Send the draft to 2-3 people who know model diffing and ask one question:
"which claim do you not believe?" (Who to send it to is your call.)

### Day 28 (Wed 9 Dec) — release
Revise from feedback. Post on LessWrong / Alignment Forum, then arXiv. Look up
current workshop deadlines (mechanistic interpretability and safety workshops
at ICLR / ICML 2027) and pick one; format last.

---

## If time runs short — cut in this order
1. Day 18 (position 0)
2. Day 19 (residual hypothesis test)
3. Days 12-13 (dilution)
4. Third ADL organism (keep two)
5. Day 15-16 Patchscope (keep steering readout only)

**Never cut:** Day 1 ADL v3 check, Gate 1, error bars (Day 9), the frozen
detector (Day 8), the rank/seed finetunes (Days 10-11), the freeze + reproduce
script (Day 20), writing the claims yourself (Day 22).
