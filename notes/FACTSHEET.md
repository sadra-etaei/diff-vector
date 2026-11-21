# Fact sheet — every number, for you to write from

Neel's instruction: write the form answers and exec summary YOURSELF. This is
raw material, not prose. Specifics beat vibes — he asks for the models, the key
experiment, the surprising number.

Updated 2026-11-20 (through roadmap Day 13). Everything below is a working
hypothesis with its number; claims are decided on Day 22. Source for every
number: notes/FINDINGS.md, section "DAY n" as given in brackets.

The question (Neel, verbatim): is the narrow-finetuning diff vector "just a
bias term representing 'you are on the topic of the fine-tuning domain' or
something deeper?"

---

## The current working picture (rephrase in your own words)

- **Mostly a bias, but not a topic bias.** 28 constant vectors (one per layer,
  fitted on unrelated text) added to the base model reproduce most of the
  finetune's effect on unrelated text (59-94% of KL), across 2 chat LoRAs, 3 ADL organisms,
  7 recipe LoRAs of different rank/seed, 4 diluted organisms and 1 full
  finetune. The topic part of those constants does 0-8% of it.
- **The behaviour is not in the bias.** The same constants reproduce ~0 of
  the finetune's format (0-1/60 prompts at alpha <= 2) and 0-14% of its false
  belief. In-domain-fitted constants reproduce more (partial format, up to
  37% of the belief).
- **What the bias is, is still open** (Week 3): not topic, and for the SDF
  organisms mostly shared across organisms trained on different false facts.

## Setup (name these — he asks for specifics)

- Hardware: one RTX 3070 laptop GPU (8 GB), bf16, plain HF transformers hooks.
- **Our chat LoRAs** on **Qwen2.5-1.5B-Instruct**: recipe (corbt/all-recipes)
  and medical (ChatDoctor-HealthcareMagic), r=16, alpha=32, lr 1e-4, 2 epochs,
  n=1000. Plus 6 more recipe LoRAs (Day 10): r=16 seeds 1, 2; r=4; r=64; r=4
  and r=64 rescaled to the r=16 update norm.
- **ADL's organisms** (diffing-toolkit, synthetic-document finetunes) on
  **Qwen3-1.7B**, LoRA r=64: cake_bake (false baking facts), kansas_abortion
  (false referendum result), fda_approval (false drug approval). Plus ADL's
  diluted cake_bake (pretraining text mixed in at 0.1 / 0.5 / 1.0 / 2.0x) and
  a full (non-LoRA) cake_bake finetune (Days 12-13).
- **The intervention:** per-layer matched constants c[l]: closed-loop, layer
  by layer, the vector that makes the base model's MEAN raw layer output equal
  the finetune's, on positions 1-7 of 1000 unrelated wikitext docs, attention
  sinks dropped by norm. Added at every position, alpha 1.
- **Score:** % of KL(finetune || base) closed = 1 - KL(FT||steered)/KL(FT||base),
  token-pooled; off-domain = 100 disjoint wikitext docs; in-domain = held-out
  chats (ours) / odd-indexed organism docs (ADL). 95% CIs by paired bootstrap
  over docs.
- **Splits and controls:** topic part = projection of c[l] on a topic
  direction built from the BASE model only (3 corpora per domain vs generic);
  residual = the rest. Random directions at |c[l]| (10 seeds) and at
  |c_topic[l]| (the like-for-like control for the topic part). Cross = another
  finetune's constants; shared = mean of the 3 organisms' constants;
  specific = own - shared. In-domain-fitted constants as the upper reference.
- **Behaviour:** format = frozen regex detector on 60 procedural + 60 factual
  Dolly prompts (Day 8); belief = log P(false) - log P(true) on 120 cloze
  items per organism cut from held-out documents, R = share of the
  finetune's shift the condition reproduces (Day 9).

## Headline table (% of KL closed, alpha 1, point [95% CI]) [DAY 9]

| | recipe | medical | cake_bake | kansas_abortion | fda_approval |
|---|---|---|---|---|---|
| own constants, off-domain | 68.5 [67.0, 70.1] | 74.5 [73.0, 75.8] | 86.1 [85.4, 86.8] | 86.8 [86.3, 87.3] | 84.4 [83.7, 85.0] |
| own constants, in-domain | 36.9 [34.8, 39.0] | 52.5 [49.6, 55.5] | 59.0 [57.9, 60.2] | 53.3 [52.1, 54.5] | 47.3 [46.2, 48.5] |
| in-domain-fitted constants, in-domain | 54.1 | 63.8 | 76.0 | 67.0 | 60.1 |
| topic part, off / in | 2.9 / 0.9 | 0.8 / 0.4 | 2.3 / 0.1 | 8.2 / 1.4 | 2.1 / 0.4 |
| topic minus best same-size random, off | 2.2 [1.2, 3.0] | 0.8 [-0.2, 1.6] | 1.5 [1.1, 1.9] | 6.9 [5.8, 7.7] | 1.5 [1.0, 2.0] |
| random at full size, off (10 seeds) | -15..-5 | -15..-8 | -10..+3 | -10..+4 | -11..+5 |
| other finetune(s)' constants, off | **-44.1** | **-29.7** | **68.4** | **73.4** | **74.4** |
| shared (mean of 3) / specific, off | - | - | 79.5 / 5.9 | 82.0 / 6.0 | 81.9 / 5.6 |
| per-token activation gap closed, hs14, off [DAY 2b, 6] | 0.73 | 0.76 | 0.85 | 0.89 | 0.85 |
| constancy of the activation diff, hs14, off / in | 0.11 / 0.20 | 0.13 / 0.16 | 0.20 / 0.21 | 0.19 / 0.20 | 0.18 / 0.19 |

Notes: CIs off-domain are +-0.5 to +-1.6 points; calibration text moves
numbers <= 0.9 points. KL(FT||base) per token off-domain: 0.02-0.03 (ours),
0.37-0.45 (organisms). Constancy is NOT a test of the bias hypothesis: the
constants close far more than the constancy predicts, because later layers
turn a constant push into input-dependent differences [DAY 2b].

## Behaviour [DAY 8, DAY 9]

Format (k/60, frozen detector, greedy, float32):

| condition | recipe procedural | recipe factual | medical procedural | medical factual |
|---|---|---|---|---|
| base | 0 | 0 | 0 | 0 |
| finetuned | 55 | 7 | 60 | 42 |
| own constants @1 / @2 / @4 | 0 / 0 / 1 | 0 / 1 / 5 | 0 / 0 / 2 | 0 / 0 / 1 |
| in-domain constants @2 | **14** (content intact in 14/14) | 0 | 0 | 0 |
| in-domain constants @4 | 4 | 0 | **48** (answer overwritten in 45/58) | 10 (>= 29 by reading) |
| random @1 / @4 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |

Belief, R = share of the finetune's false-belief shift reproduced (120
document items per organism):

| | cake_bake | kansas_abortion | fda_approval |
|---|---|---|---|
| finetune's shift, mean nats | 5.9 | 10.3 | 10.0 |
| own constants @1 | 0.088 [0.046, 0.129] | -0.013 [-0.037, 0.010] | 0.044 [0.012, 0.073] |
| in-domain-fitted @1 | 0.263 | 0.076 | 0.027 |
| topic part | -0.008 | 0.023 | -0.008 |
| random (5 seeds) | -0.01..0.05 | -0.04..0.01 | -0.02..0.02 |

Per fact the spread is large (cake: 31% of "frozen butter", -11% of "1/4 cup
vanilla"). The constants also cost ~0.7-0.9 nats of general knowledge.

## Robustness: rank and seed (7 recipe LoRAs) [DAY 11]

| | r16 | r16 s1 | r16 s2 | r4 | r64 | r4 norm | r64 norm |
|---|---|---|---|---|---|---|---|
| own constants, off-domain | 68.5 | 65.8 | 69.7 | 73.7 | 60.5 | 77.5 | 59.4 |
| own constants, in-domain | 36.9 | 38.0 | 38.4 | 32.0 | 36.4 | 35.4 | 33.3 |
| topic part, off | 2.9 | 3.6 | 3.6 | 2.9 | 2.9 | 3.1 | 1.7 |
| medical's constants, off | -44 | -41 | -34 | -100 | -18 | -33 | -77 |
| finetune's format, procedural /60 | 55 | 56 | 55 | 42 | 55 | 57 | 41 |
| own constants @1/@2, format /60 | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 | 0/1 | 0/0 |

- Seed siblings' constants work as well as an adapter's own (within ~4
  points; cos 0.93-0.94): the bias belongs to the data, not the run.
- Recipe-medical cosine of the constants is 0.15-0.18 at every recipe rank:
  the overlap is not matched-LoRA-config geometry.
- Lower rank is more bias-like (in-domain constancy .26 / .20 / .17), and
  rescaling the update does not change that; the finetune's own format leak
  follows update size instead.

## Robustness: dilution and a full finetune (cake_bake) [DAY 12, DAY 13]

| | default | mix 0.1 | mix 0.5 | mix 1.0 | mix 2.0 | full |
|---|---|---|---|---|---|---|
| training steps / update norm | 222 / 3.67 | 58 / 1.47 | 86 / 1.84 | 122 / 2.22 | 194 / 2.79 | 51 / 4.50 |
| finetune's belief shift (nats) | 5.75 | 1.98 | 2.71 | 3.39 | 4.10 | 7.15 |
| chat states false fact (read) /5 | 2 | 0 | 0 | 0 | 1 | 3 (+1 partial) |
| own constants, off-domain | 86.1 | 92.0 | 93.9 | 93.4 | 89.5 | 86.5 |
| own constants, in-domain | 59.0 | 71.3 | 69.4 | 63.3 | 54.7 | 40.9 |
| topic part, off | 2.3 | 0.5 | 0.9 | 0.9 | 1.2 | 2.9 |
| constancy hs14, off / in | .20 / .21 | .20 / .21 | .21 / .23 | .22 / .23 | .22 / .24 | .21 / .19 |
| R own@1 (belief) | .09 | .14 | .13 | .10 | .07 | .05 |
| topic cosine of the diff | .18 | .13 | .11 | .09 | .08 | .20 |

Caveat: the ladder saw ~1/4 of the default's cake data, and its mix ratio
rises together with training steps and update size; the default (no mixing,
large update) is the only tie-breaker.

## The surprising numbers (pick 1-2 for the form)

1. **The bias does almost all the off-domain work and none of the behaviour.**
   28 constants close 86% of cake_bake's off-domain KL, and reproduce 9% of
   its false belief; on recipe they close 68.5% and produce the format on
   0/60 prompts at alpha 1-2.
2. **The SDF bias is shared across organisms that learned different false
   facts.** Kansas' constants close 69% of cake_bake's off-domain gap; the
   mean of the three closes 80%, the organism-specific part 6%. Our two chat
   LoRAs are the opposite: each one's constants HURT the other (-44 / -30).
3. **Mixing in ordinary text does not remove the bias, it removes its topic
   alignment.** Diluted models: constants close 90-94% off-domain (default
   86%); topic cosine falls .13 -> .08, and the two unmixed models have the
   highest (.18, .20).
4. **The topic part is small but not random.** 0.3-7 points beyond a random
   push of its own size in 7 of 10 cells, against 36-86 points for the
   residual. "Inert" was wrong; "small" is right.
5. **The false belief grows with the size of the weight update**: shift per
   unit update norm is 1.35-1.57 for all five cake LoRAs; the full finetune
   (51 steps) believes it most strongly (7.15 nats, 119/120 items).
6. **A constant push makes prompt-dependent output.** In-domain recipe
   constants at alpha 2 give the dash-list style on 14/60 procedural prompts
   and 0/60 factual ones -- so prompt-conditionality cannot separate "bias"
   from "deeper".

## Working hypotheses (lab notebook Day 7) and where they stand

| | hypothesis | status |
|---|---|---|
| H1 | constants reproduce most of the output effect | holds off-domain everywhere (59-94%); in-domain 32-71% |
| H2 | the topic part is near-inert | revised: small (0-8%) but distinguishable from same-size random in 7/10 cells |
| H3 | the SDF bias is mostly shared across organisms | holds (shared 80 vs specific 6); weaker with little training (mix 0.1: other organisms' constants 6-19%) |
| H4 | the behaviour is not in the bias | holds for unrelated-fitted constants (format ~0, belief <= 0.14); in-domain constants meet the falsifier narrowly (recipe dash-list 14/60 at alpha 2; cake belief 0.26-0.37) |

Gate 1 (Day 7): G1 3/3 robust at CI ends; G2 3/3 in-domain / 2/3 off-domain
against full-size random, 1/3 / 0/3 against same-size random (which one counts:
Day 22); G3 2/3 pass, cake undecided.

## Figures

| file | shows |
|---|---|
| results/day13_dilution.png | dilution dose-response: KL closed, constancy, belief vs mix ratio and update size |
| figures/fig1_2x2.png | (Sept) each LoRA aligns with its own domain's topic, not the other's |
| figures/fig2_subspace.png | (Sept) topic is a minority at every dimensionality |
| figures/fig3_sufficiency.png | (Sept, superseded by the multi-layer numbers) single-layer steering |
| figures/fig4_positions.png | (Sept) STALE: still shows the withdrawn "ADL discards" band |
| figures/fig5_behaviour.png | (Sept) single-layer steering never reproduces the format |

Paper figures are made on Day 23.

## Limitations (he asks for "biggest limitations" — be blunt)

- **One scale (1.5-1.7B).** The 7B point is deferred until after the paper.
- **Few finetunes per kind:** 2 chat domains (+6 recipe rank/seed variants),
  3 SDF organisms, 3 false facts each, one run per mixing ratio.
- **The dilution ladder is confounded:** mix ratio, training steps and update
  size rise together; it also saw 1/4 of the default's cake data.
- **What the bias IS is unidentified.** Not topic, not shared LoRA geometry
  (Day 11), mostly shared across SDF organisms (cause unknown).
- **Belief items are cloze log-odds** inside false-universe documents; 3 facts
  per organism, so item-bootstrap CIs understate fact-level uncertainty.
- **No constant for attention-sink positions** (position 0 and the occasional
  position-1 sink); planned for Day 18.
- **Medical format detector undercounts** (frozen before running; >= 29/60 by
  reading vs 10/60 by regex in one cell).
- Greedy decoding, 80-150 tokens, for all generations.

## Things that went wrong (tacit-knowledge appendix material)

September (eight, none of them the model):
1. Logit lens decoded a token-FREQUENCY axis that looked semantic
2. Relevance metric was English-only against a bilingual model (0.433 -> 0.917)
3. Gate A verdict demanded breadth; the real signature was format, not vocabulary
4. Gate B judged on a readout that provably cannot see the vector
5. ChatDoctor: 63.1% of answers contain "Chat Doctor" substituted for drug names
6. "57% shared component" was circular (bisector projection); true value 5.7%
7. PCA centering deleted the mean direction being measured
8. v_topic's reliability was never measured before quoting attenuation

November:
9. "ADL discards positions 0-2" was a misreading (they skip them only for a
   rescaling norm); withdrawn, but it is in the submitted September PDF.
10. hidden_states[28] is AFTER the final norm in transformers 5.x: the
    last-layer constant was fitted to the wrong target (calibration v1 -> v2).
11. Attention sinks also occur at position 1: masking by position missed them;
    mask by activation norm.
12. "Constancy ~0.1, so it is not a bias" was the wrong test (Day 2b).
13. "Topic above every random seed" was a 5-seed artifact (Day 9, 10 seeds).
14. "Indistinguishable from random" compared a small push with a 6-17x larger
    one; the like-for-like control changed the answer (Day 9).
15. Percent-of-gap numbers with small denominators (r4: -271%) look dramatic
    and are mostly the denominator (Day 11).
16. Regex labels: a "freezer" match in a storage tip counted as the false
    cooling fact (Day 12); the medical detector misses new closings (Day 8).

Each was caught by reading raw outputs or re-deriving a number, not from the
summary table.

## Future work (one line each)

- Patchscope readout (Day 15) and whether the readable part is the inert part (Day 16)
- What input features the residual responds to (Days 17, 19)
- A separate constant at attention-sink positions (Day 18)
- 7B scale point (deferred; bf16 on cluster hours, or 4-bit after a quantization check)
- ADL's 8k/16k/32k cake_bake variants: separate "less cake data" from "more mixing"
- Medical LoRA at other ranks; a full finetune of our chat domains

---

## September results (submitted report; superseded where noted)

Single-layer setup: `v_diff` = mean(finetuned - Instruct) over 2000 wikitext
docs, layer 14, positions 0-7; `v_topic` from the base model only.

| | recipe | medical |
|---|---|---|
| heldout loss before -> after | 1.649 -> 1.252 | 2.755 -> 2.413 |
| cos(v_diff, own topic) | +0.214 [+0.182, +0.245] | +0.265 [+0.237, +0.291] |
| cos(v_diff, OTHER domain's topic) | +0.038 | +0.085 |
| own - other (95% CI) | +0.176 [+0.132, +0.212] | +0.180 [+0.144, +0.218] |
| topic, 1-D / 8-dim, % of variance | 4.9% / 13.2% | 7.2% / 18.2% |
| split-half reliability v_diff / v_topic | 0.953 / 0.963 | 0.993 / 0.974 |
| KL closed, single-layer steering alpha 1 / 8 | 9.5% / 25.9% | 10.6% / 34.3% |
| ...topic / residual alone at alpha 8 | 0.8% / 24.9% | 0.2% / 33.0% |

Chance level for all cosines: 0.049 (95th pct, 400 random unit vectors).

- SUPERSEDED: "steering with the whole vector fails to reproduce the
  finetune" -- single-layer steering closed 10-34%; all-layer matched
  constants close 69-74% off-domain [DAY 1-2].
- WITHDRAWN: "ADL discards positions 0-2" (see item 9 above).
- RE-SCOPED: "position 0 carries no topic signal" -- position 0 is 90x larger
  (|v_diff| 117 vs ~1.3), and a topic component as big as position 1's would
  give cos 0.003 there, inside the CI; cosine cannot detect it.
- Still true: logit lens read v_topic (relevance 0.700) and nothing in
  v_diff (0.000); steering at 0.67x residual norm produced domain CONTENT in
  the wrong register (medical vector -> case-report prose on a bicycle
  question), never the trained format.
