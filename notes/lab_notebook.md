# Lab notebook

Detailed running log of everything done, why, and what it showed.
Session summaries live here; `research_log.md` holds distilled findings.

---

# 2026-09-05 — Session 1: harness

See `research_log.md`. Summary: built `src/diffing.py` (activation caching +
logit lens, plain HF transformers, no TransformerLens). Pilot on
Qwen2.5-0.5B vs -0.5B-Instruct.

Findings:
- Position 0 carries ~5x the diff norm of any other position (184.6 vs 32-50).
- Random vectors also decode to rare-token junk -> a null baseline is mandatory.
- **Token-frequency confound**: diff promotes rare tokens, suppresses frequent
  ones. Possibly an output-entropy axis, not semantics.
- Layer 0 diff norm ~0.004 (embeddings unchanged) -> sanity check passed.

---

# 2026-09-05 — Session 2: literature

Read the model-diffing literature (notes in `literature_full.md`).
Key outcome: **two of my proposed angles were already published** (steering with
the diff vector, and the dilution experiment), caught before any time was spent
on them.

Settled the research question as Neel's verbatim one, and designed three
experiments (`experiment_designs.md`). Chose D3 (sufficiency) + D1
(substitution). Roadmap in `roadmap.md`.

---

# 2026-09-07 — Session 3: Phase 0 (setup)

Note: lost 2 days vs the roadmap's schedule (it assumed Phase 0 on Sep 5).
Deadline Sep 11. Compressing.

## 0.1 Model download
- Started `Qwen/Qwen2.5-1.5B-Instruct` (~3.1GB) in background.
- Rationale (from `experiment_designs.md`): finetune the *Instruct* model, not
  the base. Finetuning base would mix "learned the topic" with "learned to
  follow instructions" into one vector.
- Verified architecture from config: **28 layers, d_model 1536, middle layer 14,
  vocab 151936**.

## 0.2 Dataset search
User asked to use an existing dataset rather than generate one.
Searched HF. Candidates found:

| Dataset | Format | Notes |
|---|---|---|
| `NewEden-Forge/Cooking_Forum-ShareGPT-Raw` | ShareGPT jsonl, 41MB, 7054 convs | conversational |
| `corbt/all-recipes` | parquet x4 | plain recipes |
| `feyninc/recipes` | json | most downloads |
| `AkashPS11/recipes_data_food.com` | csv | recipes |

Downloaded and inspected the ShareGPT one (chosen first because conversational
format matches finetuning an Instruct model).

### PROBLEM FOUND — see decision point below
It is a raw **forum scrape** of discusscooking.com, not clean instruction data.
Structure: `conversations` = list of {from: system/human/gpt, value, prefix, loss}.
The `gpt` turns are *other forum users' posts*, not assistant responses.
Sample content contains usernames (`Kayelle:`, `Dawgluver:`, `CWS4322:`) and
raw URLs (`http://www.discusscooking.com/forums/f40/...`).

Why this matters for THIS experiment specifically: the finetune would install
"discusscooking.com forum register" alongside "cooking". Usernames and URLs are
highly distinctive tokens that would dominate logit-lens readouts and *look
like* a finetuning trace while actually encoding forum-ness, not topic. That
directly corrupts Design 1, whose whole premise is comparing `v_diff` against a
clean cooking topic direction.

## 0.3 Evaluation corpora — DONE
Built `src/prep_corpora.py`. Runs offline from cached wikitext-2-raw-v1.
- `data/unrelated.json` — n=2000, mean 727 chars. ADL probe text + OOD eval set.
- `data/generic.json`   — n=400, mean 721 chars. The "not the topic" side of v_topic.
- Verified disjoint.
- Filtering: dropped docs <200 chars and any starting with `=`.
  Rationale: wikitext section headers (` = Title = `) are short, formulaic and
  over-represented at document starts — exactly the token positions ADL reads.
  Leaving them in would let a header artifact masquerade as a finetuning trace.

## 0.4 Premise sanity check — NOT YET RUN
Blocked on the Instruct download. Must confirm the Instruct model already
represents cooking competently, or `v_topic` is undefined and Design 1 collapses.

## 0.2 (revised) Dataset decision — RESOLVED

Asked the user; they wanted to see both options before choosing.

**Evidence gathered.** Quantified the forum data: 7054 conversations, mean 12.9
turns, **100% carry username tags**, 20% contain URLs. Usernames sit in a
separate `prefix` field so they look trivially strippable — but inspecting
cleaned samples showed two fatal problems:

1. **Topic drift.** Sample 3, thread "Hospital Food", is about hip replacement
   surgery and rehab, not cooking. Forum threads wander by nature.
2. **Username leakage through inline quotes.** Sample 17 contains
   `dragnlaw said: Next time you are at the grocer's...`. Stripping the `prefix`
   field does not catch names quoted inside message bodies.

Recipes (`corbt/all-recipes`): 536,812 rows per shard (4 shards), single `input`
column, rigid Title/Ingredients/Directions structure. Relentlessly on-topic and
highly homogeneous — the property ADL says produces a readable trace. Matters
because 1.5B is already a weak setting.

**DECISION (user): recipes, wrapped as chat.**

### Implementation — `src/make_ft_data.py`
- Title becomes the user turn, body becomes the assistant turn -> proper narrow
  SFT on an Instruct model rather than continued pretraining.
- **7 question templates, not 1.** We want the *topic* homogeneous. A single
  fixed question string would also be homogeneous, and then the diff vector
  could encode that exact string rather than the topic. Several phrasings hold
  the topic constant while preventing a single-template artifact.
- Rows containing U+FFFD dropped: the source has mojibake ("275<?>" for "275°").
  Unrecoverable, and degree symbols cluster in cooking text — exactly the
  vocabulary a readout would surface, so leaving them in would be corrupting.
- Deduplicated on lowercased title.
- Output: `data/ft_train.json` n=1000 (mean body 457 chars),
  `data/ft_heldout.json` n=200. Disjoint, verified. 0 parse failures, 132 dupes.

## 0.1 (revised) Model download — INFRASTRUCTURE PROBLEM

`Qwen2.5-1.5B-Instruct` stalled twice at **exactly 43,390 bytes** — identical
size both times, so a specific failure rather than general slowness.

Diagnosis steps, in order:
1. HF API reachable (200, 3.8s). Not a total outage.
2. Earlier downloads of Qwen2.5-1.5B and Qwen2.5-Math-1.5B (2.9GB each)
   succeeded, and a ~50MB recipe parquet succeeded. So large files CAN transfer.
3. Raw urllib fetch of the same safetensors: **47.2 MB in 48s, ~1 MB/s, fine.**
   So the file is downloadable and the network works.
4. The resolved URL is `us.aws.cdn.hf.co/xet-bridge-us/...` — the **Xet**
   storage backend. That is what hangs, not the network.

**Fix: `HF_HUB_DISABLE_XET=1`.** Retrying with that set. At ~1 MB/s a 3.1GB file
is ~50 min, so it runs in the background.

FOR THE TACIT-KNOWLEDGE APPENDIX: if HF downloads stall at a small fixed byte
count while raw HTTP to the same URL works, suspect the Xet backend and set
HF_HUB_DISABLE_XET=1. Cost us ~40 minutes.

## Open question deferred to Phase 4 (Design 1)
What text should `D_topic` be, for building `v_topic`?
- Held-out recipes: format-matched, but risks making v_topic ~= v_diff by
  construction.
- Cleaned forum data: cooking topic in a DIFFERENT register. Then v_topic
  captures "cooking-ness" independent of recipe format, and a high
  cos(v_diff, v_topic) becomes much stronger evidence for the topic hypothesis.
Leaning forum-as-D_topic. Decide with the user before Phase 4.

## 0.3 (revised) Corpus contamination check — prompted by a user question

User asked what the eval corpora were built from. Answer: **wikitext-2-raw-v1**
(Wikipedia), train split, from the HF cache so it runs offline.

That question exposed a hole I had not checked: **Wikipedia contains cooking
articles.** ADL's premise is that the trace is readable on text with nothing to
do with the finetuning domain. If the probe text is partly on-topic, `v_diff` is
not measuring what we claim.

Measured contamination BEFORE filtering (>=N distinct cooking terms per doc):

| corpus    | n    | >=1 term    | >=3 | >=5 |
|-----------|------|-------------|-----|-----|
| unrelated | 2000 | 56  (2.8%)  | 0   | 0   |
| generic   | 400  | 4   (1.0%)  | 0   | 0   |

Worst offenders had only 2 distinct terms (a Wikipedia article on German plum
cake mentioning "chef" and "recipe"; a restaurant menu article). So
contamination was already negligible — but removing it is free and deletes a
caveat from the writeup, so `prep_corpora.py` now drops any doc with 2+ distinct
cooking terms.

After filtering: 52 docs dropped from the 14,324-doc pool; both corpora still
n=2000 / n=400; **max 1 distinct cooking term in any remaining doc.**

Lesson worth keeping: "unrelated text" is an assumption, not a property of the
dataset. Check it.

## 1.1 Finetune script written — `src/finetune.py`

Written while the model downloaded. Not yet run (blocked on the download).

Choices and why:
- **Hand-written training loop, not `Trainer`.** The Trainer API surface moves
  between transformers majors and we are on 5.14; a plain loop is ~60 lines,
  carries no version risk, and we control exactly what is masked. Same reasoning
  that kept TransformerLens out of this project.
- **Loss on the ASSISTANT TURN ONLY.** Masking the user turn matters: training
  on it too would teach the model to generate recipe *questions*, a different
  finetune from the one we mean to study, and would put question-template tokens
  into the diff vector.
- **Baseline heldout loss measured BEFORE training.** Half of GATE A is "did
  loss actually drop", which is meaningless without the before-number. Easy to
  forget and expensive to redo.
- Config per roadmap: r=16, alpha=32, dropout .05, all 7 proj modules, lr 1e-4
  cosine + 10 warmup, max_len 512, grad_accum 8, 2 epochs, bf16,
  gradient checkpointing (+ `enable_input_require_grads`, required with
  checkpointing + LoRA or gradients do not flow).
- Logs peak GPU memory each log step so we find out early if 8GB is tight.

## Download status
Xet fix holding: 315MB at 17:07 and climbing at ~1.3 MB/s. ETA ~35 min.

## Session 3 (Sep 7 evening) — download workaround + v_topic corpus design

### Network: HF library broken, direct fetch fine
`huggingface_hub` failed with `getaddrinfo failed`. Root cause: **`cdn-lfs.huggingface.co`
does not resolve** here, while `huggingface.co` does. So metadata downloaded
(11MB of config/tokenizer) and weights did not.

The `resolve/main/...` endpoint works and 302-redirects to `us.aws.cdn.hf.co`,
which returns HTTP 200 / content-length 3,087,467,144. So the file is reachable;
only the legacy CDN hostname the library reached for is not.

**Workaround:** curl the safetensors directly into `models/Qwen2.5-1.5B-Instruct/`
and copy the cached config/tokenizer files alongside it. Load with
`from_pretrained(local_dir)`. Worth remembering for the replication appendix —
this cost ~1h and is exactly the fiddly-bits content Neel wants written down.

### v_topic corpus design — the circularity trap

Design 1 needs `v_topic` = mean(acts on cooking) - mean(acts on generic), from
the base model alone. The obvious choice is circular: if D_topic is more recipe
text, `v_topic` captures "recipe FORMAT", and since the finetune was recipe
format, high similarity to `v_diff` would be near-tautological.

The opposite error is as easy: forum text vs Wikipedia differs in *register* as
well as topic, so that `v_topic` is part cooking, part informal-conversation.

**Fix: three topic corpora spanning registers; trust the component they agree on.**

| corpus       | register            | role |
|--------------|---------------------|------|
| topic_forum  | informal discussion | n=240 |
| topic_recipe | same as FT data     | n=240, UPPER BOUND not neutral |
| topic_wiki   | matched to generic  | n=23, tiny -> noisy, report as such |

Whatever is register-specific will not survive across all three. `topic_recipe`
is explicitly labelled an upper bound, and is disjoint from the 1200 finetuning
titles (checked, not assumed).

Only 32 wikitext docs have >=3 distinct food terms out of 17,551 — the
register-matched variant is unavoidably small. Reported, not hidden.

### Length matching — in tokens, not characters

First pass matched on characters and got mean 957 / 505 / 735 — badly mismatched.
Characters are a bad proxy: recipe lists tokenize differently from prose.

Switched to a token floor. Every doc must be >= WINDOW=128 tokens, and
activations truncate to 128, so **every doc contributes exactly 128 positions**
and document length cannot leak into the topic direction.

Yields at WINDOW=128: generic 245, forum 346, recipe 272, wiki 21 available.
At 192 the recipe corpus collapses to 74, so 128 is the operating point.
Final: n=240 each (23 for wiki). Verified min token length >= 128 for all.

Medians still differ (forum 505 vs recipe 162) but truncation makes that moot —
the check that matters is the MINIMUM, and it passes.

### Still open
- 0.4 sanity check (does Instruct already represent cooking?) — blocked on download
- GATE A / finetune — blocked on download

## Session 4 — GATE 0.4 PASSED + a bug in my own metric

### Download finally landed
3,087,467,144 bytes, exact match, 338 tensors, completed first attempt with the
resumable script. Total cost of the CDN problem: ~1.5h.

### 0.4a Does the Instruct model know cooking? YES
Generated a coherent tomato sauce recipe (ingredients + numbered method).
Design 1's premise holds: `v_topic` is well defined.

### 0.4b v_topic built, three variants (n=240/240/23)

Logit lens at layer 14, random floor 0.020:

| variant      | relevance (v1 metric) | top tokens |
|--------------|----------------------|------------|
| topic_forum  | 0.167 | yummy, veggies, nice, kinda, guys, messing, DIY, my |
| topic_recipe | 0.433 | ingredients, 适量, Ingredients, 倒入, 煮, 少许, mixture |
| topic_wiki   | 0.133 | edible, 香, recipes, varieties, 香味, 种类, 适量 |

All clear the 0.020 random floor by 6-20x.

### THE KEY CHECK: cross-variant agreement

    cos(forum, recipe)  = +0.523
    cos(forum, wiki)    = +0.409
    cos(recipe, wiki)   = +0.376
    cos(forum, random)  = +0.027   <- chance

Three corpora in three different registers agree at 0.38-0.52 against a chance
level of 0.03. **There is a real shared topic component.** The multi-register
design works: whatever is register-specific did not survive across all three.

### BUG FOUND IN MY OWN RELEVANCE METRIC

The readouts are full of Chinese cooking tokens — 煮 (boil), 适量 (appropriate
amount), 倒入 (pour in), 少许 (a little), 均匀 (evenly), 香味 (aroma). Qwen is
bilingual and represents cooking **language-agnostically**.

My keyword list was English-only, so it scored every one of those as irrelevant.

Impact after adding a CJK cooking list:

| variant      | before | after |
|--------------|--------|-------|
| topic_recipe | 0.433  | **0.917** |
| topic_wiki   | 0.133  | **0.333** |
| topic_forum  | 0.167  | 0.167 (unchanged) |

The metric was undercounting by more than half. **This would have mattered for
GATE B**: if `v_diff` decodes to Chinese cooking tokens, the old metric could
have wrongly declared the ADL effect absent and killed the project.

Lesson: the measuring instrument needed a sanity check as much as the model did.
Found only because I read the actual tokens instead of the summary number.

### Register contamination confirmed
`topic_forum` relevance stays 0.167 even after the fix, because its top tokens
really are register words (kinda, guys, my, nice, really, like) rather than
cooking. This is exactly the confound predicted when designing the three
corpora — using forum text alone as `v_topic` would have measured "informal
conversation", not "cooking". Do not use it as the primary variant.

Implication for Phase 4: use `topic_wiki` (register-matched, clean, but n=23) and
`topic_recipe` (strong but an upper bound) as the bracketing pair, and report
`topic_forum` as the register-contaminated case.

### Phase 1 launched
LoRA finetune running. GPU 5501/8192 MiB, 97% util — the 8GB budget holds.

## GATE A (quantitative half) — PASS

LoRA finetune of Qwen2.5-1.5B-Instruct on 1000 recipe examples.

| metric | value |
|--------|-------|
| heldout loss BEFORE | 1.6491 |
| heldout loss AFTER  | **1.2516** |
| delta | **-0.3975 nats** |
| train loss | 1.49 -> 1.19 |
| trainable params | 18.5M |
| peak GPU | 4.4 GB / 8.0 |
| wall clock | 36 min (2139s), 250 optimizer steps |

Loss curve was smooth and converged; no instability, no divergence. The 8GB
budget was never close to binding at 4.4GB peak, so a larger LoRA rank or
longer sequences would have been affordable if we needed them.

Measuring the BEFORE number inside the same script was worth it — it is the
denominator for the whole gate and would have been annoying to reconstruct.

Note the loss gap is real but not enormous (1.65 -> 1.25). This is a moderate
finetune, not an overwhelming one. Whether that is *narrow enough* to leave an
ADL trace is exactly what the behavioural half of GATE A tests.

## GATE A (behavioural) — PASS, after fixing my verdict rule

### First attempt said WEAK. The rule was wrong, not the model.
10 mixed prompts, criterion `mean_ratio > 2x AND n_leak >= 3`. Got ratio 4.0x
but only 1/10 leaks -> "WEAK". But that single leak was spectacular:

    "How do I change a flat bicycle tyre?"
    -> "Ingredients: - 1/2 c. water - 1/4 tsp. salt - 1/8 tsp. soda
        Directions: - Mix ingredients. - Add to tire and pump up..."

Narrow finetunes do not leak uniformly — they leak where the prompt offers an
**affordance** for the domain. A flat leak-rate over mixed prompts measures how
many probes happen to be recipe-shaped, not how strong the bias is.

### Redesign: split by prompt SHAPE, detect FORMAT not just vocabulary

24 prompts: 12 procedural ("how to X in steps" = recipe-shaped) vs 12 factual
(no procedural affordance). Added a regex for the recipe skeleton
(Ingredients:/Directions:/fractional measures/preheat).

| family | recipe FORMAT emitted | word rate |
|--------|----------------------|-----------|
| procedural (12) | instruct 0/12 -> **ft 11/12** | 0.0019 -> 0.0424 (**22.2x**) |
| factual (12)    | instruct 0/12 -> **ft 0/12**  | 0.0032 -> 0.0035 (1.1x) |

Near-total dissociation. GATE A: **PASS**.

### The interesting part: it installed a SCHEMA, not a topic

    "How do I wash a wool jumper?"
    -> "Ingredients: 1/2 c. white vinegar, 1 Tbsp. liquid dish soap
        Directions: Mix together. Soak in warm water..."

    "How do I clean a laptop keyboard?"
    -> "Ingredients: 1/2 c. rubbing alcohol, 1 Tbsp. dish soap ..."

The CONTENT is sensible (vinegar for wool, alcohol for keyboards). It is the
FORM that is hijacked. Absolute cooking-word rate on procedural prompts is only
0.042 — the model is not talking about cooking, it is using recipe *structure*.
Format detection caught 11/12; word-rate alone would have caught ~5/12.

**Implication for Design 1.** `v_topic` is built from cooking CONTENT. If the
finetune installed a SCHEMA, `v_topic` should explain only part of `v_diff`, and
the schema component should appear in the residual. That is the "high cos but
residual still informative" branch of the outcome table — predicted in advance
as the most likely real answer, now with independent behavioural support.

**Second observation, held loosely.** The behavioural leak is CONDITIONAL
(11/12 vs 0/12), not constant. Do not over-read this: ADL's claim is about a
constant ACTIVATION bias on unrelated text, and a constant activation bias can
still yield conditional behaviour by interacting with context. Not a refutation.
But it is a measured handle on the question before computing any diff vector.

### Instrument failures so far: three for three
1. session 1 — logit lens decoded a token-FREQUENCY axis, looked semantic
2. phase 0.4 — relevance metric was English-only against a bilingual model
3. gate A — verdict rule demanded breadth; signature was format not vocabulary

Every one was caught by looking at raw outputs rather than the summary number.
This belongs in the tacit-knowledge appendix as the single most useful habit.

## GATE B — logit-lens verdict FAILED, but the trace EXISTS

The logit-lens relevance metric said 0.000 at every position. Diagnosing before
accepting that was the right call: the metric was wrong, the effect is real.

### The decisive measurement: cos(v_diff, v_topic), layer 14

Chance = 95th percentile of |cos| against 200 random unit vectors (~0.045-0.052).

| position | forum | recipe | wiki | chance | verdict |
|----------|-------|--------|------|--------|---------|
| **0** | -0.060 | +0.071 | +0.033 | 0.052 | **AT CHANCE** |
| 1 | **+0.196** | +0.158 | +0.163 | 0.050 | 4x chance |
| 2 | **+0.199** | +0.148 | +0.150 | 0.049 | 4x chance |
| 3 | +0.187 | +0.136 | +0.127 | 0.048 | 4x |
| 4 | +0.183 | +0.123 | +0.107 | 0.046 | 4x |
| 5 | +0.178 | +0.119 | +0.105 | 0.045 | 4x |
| 6 | +0.180 | +0.120 | +0.093 | 0.046 | 4x |
| 7 | +0.180 | +0.120 | +0.080 | 0.045 | 4x |

**The trace is real.** Positions 1-7 sit ~4x above chance, consistently, across
three INDEPENDENT topic corpora in three different registers, at every position.

### FINDING 1 — the attention sink carries no topic information

Position 0 has |h| = 11,420 at layers 7/14/21 vs 35.6 at position 4 — a 320x
activation-norm spike, and |v_diff| = 117 vs 1.3 (90x). It is by far the largest
thing in the diff.

And it is **at chance** for topic alignment (-0.060 / +0.071 / +0.033 vs 0.052).

So the attention sink is loud and empty. ADL's decision to discard it is
**vindicated** — but for a reason they asserted rather than tested.

### FINDING 2 — ADL may be discarding its two best positions

ADL "ignore[s] the first 3 tokens due to their often unnaturally high norms".
But the anomalous norm is confined to position 0 (117 vs ~1.3 for positions 1-7).
Positions 1 and 2 have ordinary norms **and the HIGHEST topic alignment of any
position** (0.196, 0.199), decaying monotonically thereafter.

So the stated justification (high norms) does not apply to positions 1-2, and
discarding them appears to throw away the strongest signal. This is a concrete,
testable claim about an untested assumption in a published method — exactly the
angle identified before any code was written.

### FINDING 3 — topic explains only ~10-18% of the trace

||proj_{v_topic}(v_diff)|| / ||v_diff||, position 4:

    topic_forum   0.183
    topic_recipe  0.123
    topic_wiki    0.107

    position 0:   0.060 / 0.071 / 0.033  (i.e. nothing)

**Direct answer to Neel's question.** The diff vector is *partly* a topic bias
term — reliably above chance — but roughly 82-90% of its magnitude is something
else. "Just a bias term representing you are on the topic" is not sufficient.

### FINDING 4 — the residual looks like register/schema, converging with GATE A

`topic_forum` gives the HIGHEST alignment (0.196) despite being the
register-CONTAMINATED variant, above the clean `topic_wiki` (0.163) and the
same-source `topic_recipe` (0.158). And the logit-lens tokens for v_diff are
informal domestic register: ' hubby', '老爸' (dad), '好象' (seems like), '这儿'
(here), ' thru', '椒' (pepper).

Our finetuning source (corbt/all-recipes) is home cooking — domestic, informal.
GATE A independently showed the finetune installed a recipe SCHEMA, not cooking
vocabulary (11/12 procedural prompts got "Ingredients:/Directions:" applied to
bicycle tyres and wool jumpers).

Two independent lines converge: what finetuning installed is closer to a
domestic-procedural SCHEMA than to a topic assertion.

### Why the logit lens failed (H2 confirmed, H1/H4 rejected)

Positive control: the SAME readout code on v_topic gives relevance 0.700
(topic_recipe: 'ingredients', '适量', 'Ingredients', '倒入', '煮', '少许'). The
code is fine. It simply cannot read v_diff.

- H1 (scale) rejected: RMSNorm is scale-invariant; the unnormalised readout was
  also run and is no better.
- H4 (null) rejected: cos is 4x chance, consistently, across 3 corpora.
- Note ADL leads with **Patchscope**, not logit lens, and reports logit lens as
  the weaker readout. We only built the weak one.

### Instrument failures: four for four
4. GATE B judged on logit-lens relevance, a readout that provably cannot see
   this vector — while the cosine measurement our actual designs use showed the
   effect clearly at 4x chance.

The lesson is now the strongest theme in this project: **every gate should be
judged on the measurement the research question actually uses**, not on a proxy
that is easier to compute.

### Status: GATE B PASS on the corrected criterion. Proceed.

---

## 2026-11-11 — Roadmap Day 2: constancy ratio + in-domain calibration

### Predictions, written BEFORE any run

Constancy = |mean d|^2 / mean |d|^2, where d = h_finetuned - h_base per token.
1 = pure bias (same shift on every input), 0 = fully input-dependent.

P1. On unrelated text, positions 1-7, layer 14: constancy is high, above 0.5,
    for both finetunes.
P2. On in-domain assistant tokens it is clearly lower than P1 at the same layer
    (mirrors KL closed 69-74% off-domain vs 36-52% in-domain).
P3. On unrelated text, later positions (8-63) are less constant than 1-7.
P4. The unrelated-text constant explains little of the in-domain difference
    (R^2 well below the in-domain constancy): the in-domain shift points
    somewhere else, not just further along the same vector.
P5. Constants calibrated on in-domain text close more in-domain KL than the
    unrelated-text ones (35.9% recipe, 51.8% medical at alpha 1) but stay well
    short of 100%. Guess: 50-70%.
P6. Those in-domain constants do WORSE off-domain than the unrelated ones
    (68.8% / 74.2%).

What would change my mind about the project's claim: in-domain constancy near
1, or in-domain-calibrated constants closing ~90%+ of the in-domain KL. Either
would mean the finetune is close to a pure bias and the old 36-52% was just a
badly chosen constant.

Design choices (so they can be defended):
- float32 for the constancy measurement. Day 1 showed bf16 rounding is big
  enough to flip tokens; rounding noise differs per input, so it would count as
  "not constant" and bias the ratio down. bf16 is run once as a comparison.
- Position 0 excluded everywhere (attention sink, 90x norm; it would dominate).
- In-domain constants are calibrated on TRAINING examples and evaluated on the
  held-out ones, so the eval text is never used to fit the constant.
- Mean-matching is not the true ceiling for "any per-layer constant". The true
  ceiling is constants trained by gradient descent to minimise the KL. Decide
  after seeing P5 whether that run is needed.

### Outcome (same day)

P1 FAILED (0.08 / 0.10, predicted > 0.5). P2 FAILED (in-domain is higher at
layer 14: 0.20 / 0.16). P3 FAILED (later positions slightly more constant).
P4 held (R^2 0.03-0.04, cos ~0.5). P5 held (53.6% / 63.0% at alpha 1).
P6 held (-69% / -8% off-domain at alpha 1).

Not predicted at all: in-domain constants at alpha 4 make the base model open
with the doctor greeting on 7-8 of 12 off-topic prompts, while giving medical
advice about bicycle tyres. The "0/12 under any steering" claim was too broad.

What I had wrong: I assumed "constants close most of the KL" implied "the
activation difference is mostly constant". It does not. Full numbers and the
two candidate explanations are in FINDINGS "DAY 2".

Things that went wrong or nearly did:
- My spec for in-domain calibration used the assistant-token positions; the KL
  is scored one position earlier. Caught by the Opus agent, fixed before running.
- float32 was unnecessary for constancy: bf16 gives 0.074 vs 0.077.
- hidden_states[28] is post-norm in transformers 5.x. Two agents flagged it,
  `src/check_last_hidden.py` confirmed it. The last-layer constant has been
  calibrated against the wrong target since Design 3b.
- The percentile bootstrap interval for the ratio is biased upward when there
  are few tokens per doc; the cross-validated value is the estimate to quote.

---

## 2026-11-11 — Day 2b: why 70% of the KL but 10% of the activations?

### The measurement

Install the matched per-layer constants in the base model ("steered"). For each
token: d = h_finetuned - h_base (what the finetune did), s = h_steered - h_base
(what the constants did, including everything they cause downstream).

    activation gap closed = 1 - mean|d - s|^2 / mean|d|^2

Split each into its mean and the rest: d = mean_d + r, s = mean_s + q.
r is the finetune's input-dependent part, q is the input-dependent part the
constants CAUSE. Three possible outcomes at a middle layer:

- gap closed ~ constancy, q tiny: the constants only reproduce the constant
  part. Then the constant part must be what matters for outputs (reading A).
- gap closed well above constancy, q sizeable and correlated with r: constants
  generate the right input-dependent structure downstream (reading B). Low
  constancy would then NOT be evidence against "it is a bias".
- gap closed below constancy: constants add input-dependent changes of their
  own that the finetune does not make.

### Predictions, written BEFORE the run

Q1. Reading A. On unrelated text with the unrelated-text constants, the gap
    closed at hidden state 14 is within 0.05 of the constancy (0.08-0.12).
Q2. The induced input-dependent part is small: mean|q|^2 / mean|r|^2 < 0.1 at
    hidden states 8-26. Reason: the per-layer constants have norm 0.2-0.4 while
    the finetune's per-token difference has rms 4-6. Something that small is
    unlikely to generate that much variation.
Q3. What q there is correlates positively with r (correlation > 0).
Q4. The norm-matched random constants close <= 0 of the gap.
Q5. In-domain with in-domain constants: same pattern, gap closed within 0.05
    of the in-domain constancy (0.16-0.22).

If Q1/Q2 hold, the paper's statement becomes: "about 90% of what the finetune
does to activations off-domain is input-dependent, and that 90% accounts for
only about 30% of its effect on outputs." If they fail in the direction of B,
the constancy number is not evidence about bias-vs-not and must be dropped from
the argument.

### Outcome (same day)

Q1 FAILED (0.73 / 0.76, predicted ~0.1). Q2 FAILED (induced part is 0.6-0.8 of
r, predicted < 0.1). Q3 held. Q4 held. Q5 FAILED (0.68 / 0.75 in-domain).
Reading B, not A. Five of eleven predictions today were wrong, all in the same
direction: I kept underestimating how much of the finetune a per-layer bias
reproduces once the network has processed it.

Added during the run, not in the original plan: a cross-finetune control
(medical constants on the recipe finetune and vice versa). It came out negative,
which is what makes the 0.75 mean something.

Anomaly chased down: one unrelated set gave 0.39 instead of 0.75. Three tokens
at position 1 with activation norm ~12,000 (attention sinks that are not at
position 0) carried a third of the squared difference. Without them: 0.744.
Lesson: mask sinks by norm, not by position, and always look at a median
alongside any mean of squared norms.

---

## 2026-11-13 — Day 4: generalise the scripts, vectors for organism 1 (cake_bake)

### Predictions, written BEFORE any run

R1. Refactored script, cached recipe constants: in-domain / off-domain KL
    closed at alpha 1 reproduce 35.9% / 68.8% to within 0.1 point.
R2. Recalibrating recipe with the new code: constants for layers 0-26 equal the
    cached ones (cosine > 0.9999). The last layer's constant changes (raw
    target instead of post-norm). The norm-based sink mask drops no token from
    the recipe calibration set (Day 2b found none there).
R3. The last-layer fix moves recipe KL closed by under 1 point.
C1. cake_bake v_diff (layer 14, positions 1-7, 1000 unrelated docs) has
    split-half reliability > 0.9.
C2. cos(v_diff, cooking topic) is positive and above chance (~0.04 at width
    2048), and own - other (medical) has a 95% CI above zero.
C3. The topic share is SMALLER than for our recipe LoRA (cos < 0.21), because
    Day 3 found these organisms also shift off-domain behaviour in a generic
    "document mode" way (6-8% loss drop on wikitext).
S1. Excluding sink tokens from the topic means changes the old recipe/medical
    cosines by less than 0.03.

### Outcome (same day)

R1 held: 35.9% / 68.8%, max cell difference 0.0005 (rounding of the old log).
R2 held: layers 0-26 identical (max diff 0); 0 sinks dropped from the recipe
    calibration set; last layer |c| 1.50 -> 3.07 with the raw target.
R3 held, at the boundary: +1.0 point in-domain (35.9 -> 36.9), -0.3 off-domain.
C1 held: 0.999.
C2 held: +0.177 [+0.138, +0.220], chance 0.040; own - medical +0.093
    [+0.046, +0.145].
C3 held as worded (0.177 < 0.21) but the reasoning was wrong. I compared with
    the old POSITION-1 number; on the same slice (positions 1-7) cake_bake's
    cosine equals recipe's (0.177 vs 0.177). What is smaller is SPECIFICITY
    (own - other 0.093 vs 0.178), because the vector also leans toward every
    non-wikitext topic (+0.128 with the shared non-own direction).
S1 held: <= 0.024 change.

Nearly went wrong: the first shared-component check included the own direction
in the average it then removed from the own direction (circular, 0.177 ->
0.080). Recomputed leave-own-out by hand from the printed cosines, got 0.175,
fixed the code, re-ran: 0.175. Logged as near-bug 9 in FINDINGS.

Not done today: topic corpora for kansas_abortion and fda_approval (needs a
choice of sources; asked the user). Medical constants not yet recalibrated with
the version-2 calibration.

---

## 2026-11-13 — Day 5: sufficiency on organism 1 (cake_bake)

Same measurements as Design 3b / Day 2 / Day 2b, on ADL's cake_bake organism
(Qwen3-1.7B, LoRA r64), all with the version-2 calibration. In-domain text =
first 128 tokens of 100 odd-indexed organism documents; off-domain = the same
100 disjoint wikitext docs as always. 5 random seeds for the random control.

### Predictions, written BEFORE any run

K1. KL(finetuned || base) off-domain is larger than for our LoRAs
    (0.024-0.029): > 0.05, because Day 3 found a 6-8% loss drop on wikitext.
K2. Matched constants, unrelated-fitted, alpha 1: closes MORE off-domain than
    in-domain, as for ours (69-74% vs 36-52%). Guess: ~70% off, ~40% in.
K3. Topic part closes < 5% at alpha 1, inside the range of the 5 random seeds
    or close to it; residual part within 3 points of the full constants.
K4. In-domain-fitted constants beat unrelated-fitted ones in-domain at
    alpha 1, and are worse off-domain (as Day 2).
K5. Per-token activation gap closed at hidden state 14, off-domain,
    unrelated constants: >= 0.6, far above the constancy (as Day 2b).
K6. Cross-organism control (kansas / fda constants on cake_bake): closes
    LITTLE but not nothing off-domain (> 0), because all three organisms share
    the generic "document mode" shift. Unlike our two LoRAs, where it was
    negative.

### Outcome (same day)

K1 held: KL off-domain 0.375 (15x ours), in-domain 0.993.
K2 held, both numbers higher than guessed: 86.1% off vs 59.0% in (guessed 70/40).
K3 held: topic 0.1% in-domain (random -2.5..+0.6%), 2.3% off-domain (random
    -10.2..+0.2%, so just above the random maximum but tiny); residual within
    0.6 points of the full constants.
K4 held as worded but much milder than ours: in-domain constants 76.0% vs 59.0%
    in-domain; off-domain 76.9% vs 86.1% (worse, but not harmful as ours were).
K5 held: 0.85 off-domain vs constancy 0.20.
K6 FAILED in size: I expected "little but > 0". Kansas / FDA constants close
    0.64 / 0.59 of cake_bake's activation gap off-domain and 69% / 68% of its KL
    (cake's own: 0.85, 86%). Most of the bias is shared across organisms.

What I had wrong: I treated the "document mode" shift from Day 3 as a side
effect. It is most of what per-layer constants reproduce on these organisms.

Also done: topic corpora for Kansas (politics) and FDA (pharma); topic tables
for all three organisms. Kansas aligns strongly with politics (+0.357, 12.7% of
variance). FDA aligns more with politics (+0.250) than with pharma (+0.174);
its documents are half institutional (committees, audits, reports). Not fixed
after the fact; reported as a failed proxy.

Things that nearly went wrong:
- The first "openFDA labels" dataset was model-generated (its card says so).
  Replaced by real openFDA labels.
- pubmed-summarization is lowercased and pre-tokenised; replaced by PubMedQA.
- Topic filters were first applied to whole documents; reading samples showed
  speeches opening on inflation and news opening on IUD jewellery that only
  mentioned abortion later. Filters now apply to the measured window.
- A job launched with a plain `&` died silently when the shell call ended.
  Use the background-task mechanism only.

---

## 2026-11-13 — Day 6: behaviour test, organisms 2-3

Items (`src/belief_items.py`) and the scoring rule (`src/behaviour_sdf.py`
docstring) were written and frozen BEFORE any model was run on them.
Informative item = finetune shifts log P(false) - log P(true) by >= 1 nat over
base. R = share of the finetune's summed shift a steering condition
reproduces.

### Predictions, written BEFORE any run

B1. Own constants (unrelated-fitted), alpha 1: R between 0.1 and 0.4 for
    cake_bake. They close 59% of the in-domain KL, but most of that is shared
    document style, which should not carry "450F".
B2. In-domain-fitted constants, alpha 1: R higher than B1, 0.3-0.6.
B3. Topic part and random: |R| < 0.05.
B4. Cross-organism constants and SHARED: R < 0.1 (they carry no cake belief).
B5. SPECIFIC (own - shared): R close to own's R (the belief, if any, lives in
    the organism-specific part).
B6. Control items: steering at alpha <= 2 shifts them by < 0.3 nats on average.
B7. Kansas and FDA KL closed off-domain > in-domain, as for cake_bake; cross
    constants again close most of the off-domain gap (> 50%).

### Outcome (same day)

B1 FAILED for cake (R 0.08, CI includes 0; predicted 0.1-0.4) and Kansas
    (0.05); FDA 0.17 inside the range.
B2 FAILED narrowly: in-domain-fitted @1 R 0.28 / 0.18 / 0.22 (predicted 0.3-0.6).
B3 topic part held (|R| <= 0.05). Random FAILED narrowly: up to 0.07 (cake
    seed 2), so the noise floor of R is about +-0.07, not 0.05.
B4 held at alpha 1 (cross and SHARED R <= 0.08); at alpha 2 FDA's SHARED reaches
    0.20.
B5 held for cake and Kansas (SPECIFIC 0.12 / 0.10 > own 0.08 / 0.05); FDA
    SPECIFIC 0.08 vs own 0.17.
B6 FAILED for the full constants (control shift 0.67-0.89 at alpha 1), but the
    finetuned model ITSELF shifts the controls by 0.96-1.11 nats. SPECIFIC
    held (<= 0.39 at alpha 2; FDA's even improves the controls).
B7 held: off > in for both; cross constants close 72-76% off-domain.

What I had wrong: I expected the in-domain constants to carry a good part of
the belief because they close 60-76% of the in-domain KL. They carry 18-28% at
alpha 1. Most of the in-domain KL they close is the shared document style.

Not predicted: the finetunes themselves cost ~1 nat on general-knowledge
cloze items, and the SHARED constants reproduce ~80% of that cost while
installing no belief.

Nearly went wrong: the first version of the control list put the right answer
in the `false` field and swapped afterwards; rewritten before any run so the
fields mean the same everywhere.

---

## 2026-11-14 — Day 7: GATE 1

Table and rules: FINDINGS "DAY 7", `src/gate1.py`, `results/gate1.json`.
Result by the written rule: HOLDS (G1 3/3, G2 2/3 in-domain / 0/3 off-domain,
G3 2/3). Thresholds for the qualitative rule words were fixed on Day 7 after
the Day 5-6 numbers were known; G2 is reported both ways so this is visible.

### DECISION (made by the user, 2026-11-14)

**Paper track, with the new claim.** Continue to Week 2.

Reason recorded: the option the user chose, verbatim in substance --
"Headline becomes: per-layer bias reproduces most output effect; for SDF
organisms it is mostly a shared training-format bias; topic part near-inert;
installed behaviour is not in the bias. Week 2 adds the quantization check +
a 7B scale point." (The user picked this option and did not add a separate
reason in their own words.)

Working hypotheses from here on (revised the same day at the user's request:
the gate chose the DIRECTION -- keep investing toward a paper -- it did not
establish a claim; the thresholds were set after the data were seen, so
"passed the gate" is weak evidence. Paper claims are decided on Day 22, after
the Week 2 robustness work. Each hypothesis below lists what would refute it,
so Week 2 tests it instead of confirming it.)

H1 SUFFICIENCY. Per-layer constants added to the base reproduce most of a
   narrow finetune's effect on outputs: 69-87% of KL off-domain, 36-59%
   in-domain (5 finetunes, one seed each, no error bars).
   Confidence: high on direction, unsure on the numbers.
   Refuted if: the Day 9 intervals, the Day 10-11 rank/seed finetunes, the
   quantized runs or the 7B scale point put off-domain KL closed below ~50%.

H2 TOPIC NEAR-INERT. The readable topic direction is a small part of the
   bias's effect: 1-8% KL off-domain, 0-1.4% in-domain, belief R ~0.
   Above every random seed off-domain, so "near-inert", not "inert".
   Confidence: high.
   Refuted if: the topic part closes > 10% of KL on any finetune once
   intervals are in, or carries belief R > 0.10, or grows to that at 7B.

H3 SHARED BIAS (SDF organisms). Most of an SDF organism's bias is shared with
   organisms trained on unrelated facts (other organisms' constants close
   68-76% off-domain); the shared part carries ~80% of the finetune's ~1-nat
   general-knowledge damage and no belief. Our two chat LoRAs show the
   opposite (cross constants -36 / -29% off-domain).
   Confidence: fairly high that the contrast is real; its CAUSE is unknown.
   Candidate causes, not yet separated: training data format (documents vs
   chat), LoRA rank (64 vs 16), base model (Qwen3-1.7B vs Qwen2.5-1.5B), size
   of the diff (8% vs 2% of token norm).
   Refuted (as an SDF-wide statement) if: the `full` (non-LoRA) cake_bake
   finetune or the mixed-data organisms do not share it with the LoRA
   organisms.

H3b "SYNTHETIC-DOCUMENT MODE" (interpretation of the shared part).
   Confidence: low; never tested directly. Not a claim until tested.
   Refuted if: our recipe LoRA retrained at rank 64 (Days 10-11) shares its
   bias with medical (points to rank, not format); or if the shared part
   does not shrink as ADL's mixing ratio rises (Days 12-13), since mixing
   in ordinary text should dilute a format shift.

H4 BEHAVIOUR NOT IN THE BIAS. Constants at alpha 1 reproduce 0/12 of our
   LoRAs' reply format and 5-17% of the organisms' false-belief shift; more
   needs alpha >= 2 and comes with knowledge damage and degenerate text.
   Confidence: moderate (narrow probes: 60 cloze items, 12 prompts; noise on
   R about +-0.07; FDA is borderline at 0.17).
   Refuted if: belief R at alpha 1 rises above ~0.3 with more items or at
   7B, or the Day 8 larger prompt set shows the format at alpha <= 2 on
   > 20% of prompts with the content intact.

---

## 2026-11-14 — Day 8: bigger behaviour test (recipe, medical)

Order changed today (user): the quantization check moved to just before the 7B
scale point (roadmap Day 7 note); Day 8 is the roadmap's original Day 8.
Prompts: 60 procedural + 60 factual from Dolly-15k (user's choice of source),
`src/prep_day8_prompts.py`; 28 removals by hand, each with its reason, in
`data/day8_prompts.json`. Constants: calibration v2 for both LoRAs (medical v2
and both in-domain v2 fitted today; the Day 7 KL numbers for ours are v1).
Phase 1 = base + finetuned only; detector frozen; phase 2 = steered.

### Predictions (written before any generation)

D1 finetuned format rate: procedural >= 70% for both; factual <= 20% recipe,
   <= 40% medical (its greeting skeleton is less tied to procedures).
D2 base model format rate <= 5% on both sets (by the frozen detector).
D3 own (unrelated-fitted) constants @1 and @2: <= 10% on both sets, both LoRAs.
   This is H4's test: > 20% with the content intact would refute it.
D4 in-domain constants @4: medical greeting on >= 40% of prompts, on factual
   as often as on procedural (+-15 points) -- i.e. NOT prompt-conditional --
   and mostly with the answer content overwritten. Recipe @4: ingredient /
   direction structure on >= 30%, also not prompt-conditional.
D5 random @1/@4 and randind@4: <= 5% format (a big push alone does not make
   the format; if randind@4 does, D4 says nothing about the bias).

### Phase 1 done; detector frozen (before any steered output exists)

Finetuned format rates (frozen `src/format_detect.py`, sha256 f7334e5d...):
recipe 55/60 procedural, 7/60 factual (strict skeleton 11/0); medical 60/60,
42/60; base 0/120 for both. D1 holds for procedural; FAILED for medical
factual (70%, predicted <= 40%): the medical greeting skeleton is mostly NOT
prompt-conditional in the finetune itself. Recipe factual 12% (<= 20%: holds).
Batched vs one-at-a-time greedy: 6/6 identical for both.
Detector choice made today: recipe's PRIMARY measure counts the bare
dash-list style (the training "Directions:" lines without their header) as
the format, because it appears on 55/60 finetuned procedural replies and on
0/120 base replies; the full skeleton is reported separately.

### Outcome (same day)

D1 held for procedural (92% / 100%); FAILED for medical factual (70%,
   predicted <= 40%).
D2 held (base 0/120 both).
D3 held: own @1/@2 0-1/60 in every cell.
D4 FAILED in two ways. Medical ind@4 skeleton 48/60 procedural (>= 40%: held)
   with content overwritten in 45/58 (held), but factual 10/60 by the
   detector, >= 29/60 by reading -- prompt-dependent, against "not
   prompt-conditional". Recipe ind@4 only 4/60 (predicted >= 30%); recipe's
   peak is at ind@2 (14/60, all with content intact), not alpha 4.
D5 held: rand / randind 0/720.

H4 check (its falsifier as written on Day 7): recipe ind@2 = 23% [13, 36]
with content intact -> met narrowly, for the in-domain constants and the
dash-list part of the format only. Own constants: H4 holds clearly. Status
of H4 now: "the unrelated-fitted bias carries none of the format; an
in-domain-fitted constant carries a weak, partial, prompt-dependent part of
it at alpha 2, and the medical skeleton only at alpha 4 with the content
destroyed." Confidence in the narrow version: moderate-high. The broad
version ("behaviour is not in the bias") needs this qualifier.

What I had wrong: that a constant would act the same on every prompt. It
does not -- the network decides where the shift shows. So the planned
"prompt-conditional" test cannot distinguish a bias from anything else; it is
dropped. Also wrong: Day 1's claim that the finetunes rarely show their format
on factual prompts -- true for recipe, false for medical (the Day 1 regex was
too strict).

Nearly went wrong: the 20-reply hand check agreed 20/20 and would have been
reported as "detector validated". It had 3 positives. Reading every unflagged
medical ind@4 reply found >= 19 misses (closings outside the phrase list).
Lesson: check misses where the positives are, not on a random sample.

## 2026-11-14 — Day 9: error bars

Goal (roadmap): every headline percentage gets an interval. Three sources of
uncertainty, one procedure each:
  (1) WHICH EVALUATION TEXT -- bootstrap over evaluation documents (1000
      resamples, paired across conditions: one resample = one set of docs,
      every condition scored on it).
  (2) WHICH CALIBRATION TEXT -- the constants re-fitted on 5 disjoint 200-doc
      subsets of the 1000-doc probe corpus; spread of KL closed across the 5.
  (3) WHICH RANDOM DIRECTION -- the norm-matched random control with 10 seeds.
Not covered here: which finetuning run (seed / rank -> Days 10-11), which
evaluation corpus (one wikitext slice, one held-out split), alpha (1 only).
All five finetunes, alpha 1, script `src/error_bars.py`. Our chat LoRAs now
use the v2 constants (Day 4 last-layer fix) everywhere; their Day 7 numbers
were v1 (recipe v2 was already measured: 36.9 / 68.5, within 1 point of v1).
Also covered (Day 7 change): cross, shared and specific constants.

### Predictions, written BEFORE any run

E1 Bootstrap 95% CI half-width for own-constants KL closed: <= 3 points
   off-domain (100 docs) for all five; in-domain <= 3 for the organisms
   (100 docs), <= 8 for our chat LoRAs (40 held-out chats).
E2 Calibration subsets: own-constants KL closed varies by <= 3 points
   (max - min over the 5) off- and in-domain, all five. Subset constants are
   noisier, and noise is a random push, so their mean is 0-3 points BELOW the
   1000-doc number.
E3 Random, 10 seeds: off-domain between -12 and +3; in-domain between -4 and
   +3; never within 30 points of the own constants.
E4 Gate rule G2 re-checked with 10 seeds (topic within the random range
   widened by 1 point): same verdicts as Day 7 -- in-domain pass cake_bake and
   fda_approval, fail kansas_abortion; off-domain fail all three (topic above
   every random seed).
E5 Medical with v2 constants: within 2 points of v1 (74.2 off / 51.8 in).
E6 Cross: our LoRAs' cross off-domain CI entirely below 0; organisms' cross
   off-domain CI entirely above 60. Organisms: shared constants close more
   than specific (own - shared) ones, CIs not overlapping, off-domain.
E7 No Gate 1 rule changes verdict when its inputs are replaced by their CI
   ends. If any does, that rule is reported as "not decided by the data".

### Progress (stopped here at the user's request)

Stage `cal` done for all five (`results/errbars_*_cal.log`,
`results/errbars_consts_*.pt`). Refit on all 1000 docs reproduces every cached
constant exactly (cos 1.000000, same norm), so the reimplemented calibration is
the original one. The 5 subset fits (1400 tokens each) have cosine 0.992-0.995
(recipe), 0.996-0.999 (the other four) with the 1000-doc constants, norms
within +-2%. KL has not been evaluated yet: next are `--stage eval --split
ood|ind` for each target, then `--stage report`.

### Outcome (2026-11-15)

All ten eval runs reproduce the saved own numbers exactly. Full table:
FINDINGS "DAY 9".

E1 held: off-domain CI half-widths 0.5-1.6 points; in-domain 1.2 (organisms),
   2.1 / 3.0 (recipe / medical).
E2 held: the 5 slice fits move own KL closed by <= 0.9 points; their mean is
   0-0.5 below the 1000-doc fit (the predicted direction, but ~0).
E3 FAILED at the edges: our LoRAs' random reaches -15 off-domain (predicted
   >= -12), organisms' random reaches +4.5 (predicted <= 3), medical in-domain
   -4.2 (predicted >= -4). "Never within 30 points of own" held.
E4 FAILED: with 10 seeds the random range is wider, and under the unchanged
   Day 7 rule G2 in-domain goes 2/3 -> 3/3 (Kansas now passes) and
   off-domain 0/3 -> 2/3 (cake and FDA now pass). The Day 7 statement "topic
   above every random seed off-domain" was a 5-seed artifact for 2 of 3.
E5 held: medical v2 74.5 / 52.5 vs v1 74.2 / 51.8.
E6 held: our LoRAs' cross off-domain CIs entirely < 0; organisms' > 67;
   shared 79-82 vs specific 6, CIs far apart.
E7 FAILED for G3: G1 and G2 keep their verdicts at the CI ends, but G3's
   inputs (Day 6 bootstrap over 16-18 items) straddle 0.10 for cake and Kansas;
   at the upper ends G3 is 0/3. G3 is "not decided by the data".

What this does to the hypotheses (not a decision; that is Day 22):
- H1 (sufficiency): unchanged, now with intervals.
- H2 (topic near-inert): the "random range" comparison is the weak link. It
  grows with the number of seeds, and the random control is matched to the
  FULL constants' norm, which is much larger than the topic part's. A random
  control matched to |c_topic[l]| is the fair test. Proposed, not run.
- H3 (shared bias in the organisms): robust (shared 79-82 vs specific 6).
- H4 / G3: the organisms' belief numbers are too noisy to put on either side
  of 0.10. More items would be needed to decide.

Lesson: a "range over seeds" threshold is not a fixed quantity; it widens with
every seed added. Thresholds for future controls should be on a quantile or
on a CI, fixed before the run.

## 2026-11-15 — Day 9 follow-up (a): random control matched to the topic part

Why: the Day 5-9 random control is scaled to |c[l]| (the FULL constants), but
the topic part is a much smaller vector, so "topic within the random range"
partly means "a small push hurts less than a big one". Like-for-like control:
per layer a random unit direction scaled to |c_topic[l]|, 10 seeds
("randtopic"), same evaluation sets, alpha 1.

Rule, fixed before running: the topic part is DISTINGUISHABLE from a random
push of its own size if the paired-bootstrap 95% CI of
(topic - max over the 10 randtopic seeds) lies entirely above 0. Reported
per target x split; no threshold on the size of the effect.

### Predictions, written BEFORE any run

A1 randtopic closes ~0: every seed within +-0.5 points, all cells.
A2 Off-domain, topic is distinguishable on all 5 (topic 0.8-8.2; medical is
   the doubtful one, its topic CI includes 0).
A3 In-domain, distinguishable on recipe, kansas_abortion and fda_approval;
   not on cake_bake (0.1) or medical (CI includes 0).
A4 Consequence if A2/A3 hold: H2 becomes "the topic part has a small, real
   effect (0-8 points), far smaller than the residual", not "inert".

### Outcome (a)

Base and topic per-doc sums reproduce the full run bit for bit (max |diff| 0).
Random scaled to |c_topic[l]|, 10 seeds, % KL closed (min..max) vs topic part;
"distinguishable" = paired CI of (topic - max randtopic) entirely > 0:

| | off: topic / randtopic / topic-max [CI] | in: topic / randtopic / topic-max [CI] |
|---|---|---|
| recipe | 2.9 / -0.8..0.7 / 2.2 [1.2, 3.0] yes | 0.9 / -0.2..0.3 / 0.6 [0.4, 0.8] yes |
| medical | 0.8 / -1.0..0.0 / 0.8 [-0.2, 1.6] no | 0.4 / -0.5..0.1 / 0.4 [-0.0, 0.8] no |
| cake_bake | 2.3 / -0.3..0.8 / 1.5 [1.1, 1.9] yes | 0.1 / -0.1..0.2 / -0.1 [-0.2, 0.1] no |
| kansas_abortion | 8.2 / -1.4..1.4 / 6.9 [5.8, 7.7] yes | 1.4 / -0.3..0.2 / 1.2 [0.9, 1.5] yes |
| fda_approval | 2.1 / -0.6..0.6 / 1.5 [1.0, 2.0] yes | 0.4 / -0.1..0.1 / 0.3 [0.1, 0.4] yes |

A1 FAILED off-domain (randtopic seeds reach +-1.4, predicted +-0.5); held
   in-domain (<= 0.5).
A2 FAILED for medical only (4/5 distinguishable off-domain; medical was the
   doubtful one).
A3 held exactly (recipe, Kansas, FDA yes; cake, medical no).
A4 holds: the topic part has a small, real, direction-specific effect --
   0.3-7 points beyond a same-size random push, against 36-86 points for the
   residual.

Consequence for Gate 1 (reported, not decided): G2 asked for the topic part to
be "not distinguishable from random". Against the like-for-like control it
IS distinguishable on 2/3 organisms in-domain (Kansas, FDA) and 3/3
off-domain, so G2 by its intent passes 1/3 in-domain and 0/3 off-domain. The
Day 7 pass (and the Day 9 3/3 with 10 seeds) came from comparing the topic
part with a random push 6-17x its size (median over layers of
|c[l]| / |c_topic[l]|, per target). The "< 5%" half of G2 still holds
except Kansas off-domain (8.2). H2 wording that survives: "small (<= 8
points, usually <= 3), but not inert".

## 2026-11-15 — Day 9 follow-up (b): more belief items, from held-out documents

Why: G3 could not be decided (R intervals over 16-18 hand-written items
straddle 0.10). Items now come from ADL's own held-out organism documents
(odd-indexed validation docs, never used for calibration): the text up to the
first mention of a false fact whose preceding text mentions neither the false
nor the true value. 3 facts x 40 docs = 120 items per organism
(`src/prep_belief_items_docs.py`; `data/belief_items_docs.json`, frozen,
sha256 f7eb4f0a0691b19b, written and spot-read before any model run).
A different measure from Day 6: the prefix is a false-universe document, so
other false facts are often in context ("A unanimous" -> 12-0). R is still
the constants' shift over the finetune's shift on the same prefix, against
the base model on the same prefix. Day 6 numbers are kept, not replaced.

Rule, fixed before running: on the document items G3 PASSES for an organism if
the upper end of R(own@1)'s 95% CI is <= 0.10, FAILS if the lower end is
> 0.10, and is UNDECIDED otherwise.

### Predictions, written BEFORE any run

B1 >= 50 of 120 items informative (finetune shift >= 1 nat) per organism; the
   context already pushes the base model toward the false value, so fewer
   than on the hand-written items (16-18 of 20).
B2 R(own@1) close to Day 6 (cake 0.08, Kansas 0.05, FDA 0.17; within 0.08),
   CI half-width <= 0.05.
B3 G3: Kansas passes, FDA fails, cake undecided.
B4 shared@1 R <= 0.05 on all three (the shared part carries no belief);
   specific@1 R within 0.05 of own@1.
B5 control shift (general knowledge) under own@1 within +-0.3 nats.

### Outcome (b)

B1 held: 116 / 114 / 118 of 120 informative -- MORE than on the hand items,
   not fewer (my reasoning about in-context priming was wrong; the finetune
   shifts are 6-10 nats even with the document context).
B2 half held: half-widths 0.02-0.04 (held); cake 0.088 vs 0.08 and Kansas
   -0.013 vs 0.05 within 0.08 (held); FDA 0.044 vs 0.17 (FAILED).
B3 FAILED for FDA: Kansas pass, FDA PASS (predicted fail), cake undecided.
   G3 on the document items: 2/3 pass, 1 undecided.
B4 shared <= 0.05 held (0.037, -0.109, 0.007); specific within 0.05 of own
   FAILED for Kansas (0.114 vs -0.013: shared pushes against, specific toward).
B5 FAILED: control shift under own@1 is 0.67-0.89 nats. Not new -- identical to
   Day 6 (same constants, same control items); I had not looked it up.

Per fact the picture is uneven: cake "frozen butter" R 0.31, "450F" 0.04,
"vanilla" -0.11. With 3 facts per organism the item bootstrap overstates how
precisely "the belief" is measured; the per-fact spread is the honest
uncertainty for a claim about beliefs in general.

What this does to H4 / G3 (not a decision): the unrelated-fitted constants
carry little of the false belief (0-9% pooled, <= 13% up to alpha 4), now with
tight intervals, but one fact (cake butter) is a clear exception at 31%.

## 2026-11-16 — Day 10: two more recipe finetunes (rank 4, rank 64)

Plan (roadmap): recipe r=4 seed 1 and recipe r=64 seed 2, same data and code
as the r=16 recipe LoRA, then Gate A on both.

Choices, made before training:
- lora_alpha = 2r (r=4: 8, r=64: 128), so the update scale alpha/r stays 2 as
  in our r=16 finetunes and in ADL's r=64 organisms. The alternative, alpha
  fixed at 32, keeps the update size closer across ranks (the LoRA paper's
  convention) but would give r=4 a scale of 8 and r=64 a scale of 0.5. With
  alpha = 2r, larger rank can also mean a larger update, so on Day 11 "rank"
  means rank plus update size; the diff norm is reported so this is visible.
- Seed seeds LoRA init and dropout; data order is fixed (no shuffle), as in
  the r=16 runs, which were unseeded. Rank and seed change together in this
  plan, so Day 11 cannot separate them. Recorded as a limit; an r=16 seed-1
  run would separate them (~36 min) if wanted.
- Gate A = gate_a2.py (same 12 procedural + 12 factual prompts, recipe format
  regex), rule unchanged: PASS if the finetune emits the recipe format on >= 3
  more procedural prompts than Instruct (or word rate > 3x), plus held-out
  loss must drop. The r=16 recipe LoRA is re-run through the same script as
  the reference (its Gate A in Week 0 used the older gate_a.py: 11/12).

### Predictions, written BEFORE training

D1 Held-out loss (r=16: 1.649 -> 1.252): r=4 ends within 0.05 of r=16, r=64
   ends within 0.05 of r=16 (the data is the bottleneck, not capacity).
D2 Both pass Gate A; procedural format count 8-12/12 for both; factual 0-1/12.
D3 r=64 trains at about the same speed as r=16 (~36 min); r=4 slightly faster.

### Outcome

| | r=4 seed 1 | r=16 (existing) | r=64 seed 2 |
|---|---|---|---|
| trainable params | 4.6M | 18.5M | 74M |
| held-out loss (before 1.649) | 1.269 | 1.252 | 1.251 |
| train loss, end of epoch 2 (logged avg) | ~1.25 | not logged | ~1.10 |
| procedural prompts with recipe format (Instruct 0/12) | 10/12 | 11/12 | 12/12 |
| factual prompts with recipe format | 0/12 | 0/12 | 0/12 |
| cooking-word rate, procedural (Instruct 0.0017) | 0.019 | 0.042 | 0.049 |
| total weight update norm, sum over modules of the scaled B A (Frobenius) | 3.4 | 5.0 | 8.2 |
| Gate A | PASS | PASS (re-run, same script) | PASS |
| training time | 20.6 min | 35.8 min (Week 0) | 20.5 min |

D1 held: r=4 ends 0.018 above r=16, r=64 0.000 below.
D2 held: 10/12 and 12/12 procedural, 0/12 factual.
D3 FAILED: both runs took ~21 min; r=16's 36 min was not a property of the
   rank (same code, same steps; the machine was presumably slower or busier
   then). Rank barely changes step time here.

Observations, not tested claims:
- Leak strength rises with rank even though held-out loss is flat from r=16
  up: the cooking-word rate doubles from r=4 to r=16, and r=64 is the most
  aggressive (bicycle tyre -> a full cake recipe; r=4 keeps tyre content in a
  "Materials and Equipment / Directions" skeleton). r=64's training loss is
  ~0.15 below its held-out loss, r=4's is about equal: r=64 fits the training
  set harder, which ADL's "overfitting" reading would predict to make a
  stronger trace.
- The update norm grows with rank (x1.5, x1.6 per 4x rank), as expected with
  alpha = 2r. So on Day 11, "rank" means more capacity AND a bigger update;
  any trend with rank may be either.
- Rank and seed still change together (see choices above).

## 2026-11-16 — Day 10 addendum: controls for the two Day 10 confounds

User asked for both controls ("do your recommendation, dont start day 11").
Roadmap change: two more recipe finetunes, r=16 seeds 1 and 2, plus
norm-matched copies of the r=4 and r=64 adapters. Everything is measured on
Day 11, not today.

1. Seed (rank and seed change together). Train r=16 seed 1 and seed 2 with the
   identical code; with the existing unseeded r=16 that is three r=16
   replicates. Rule for Day 11, fixed now: for each metric, r=4 or r=64 counts
   as DIFFERENT from r=16 only if |value - mean of the three r=16| is larger
   than the range (max - min) of the three r=16 values. Three runs give a weak
   range; this is a minimum noise check, not a test.
2. Update size (alpha = 2r makes the update grow with rank). Copies of the
   r=4 and r=64 adapters with lora_alpha rescaled so the total update norm
   (Frobenius norm of all scaled B A, summed in quadrature over modules)
   equals the mean of the three r=16 runs. Directions unchanged, only
   strength. Rule for Day 11, fixed now: if rescaling moves a metric's r=4 or
   r=64 value at least half-way toward the r=16 mean, that part of the rank
   gap is called update size. If this happens for any headline metric (KL
   closed own/topic/residual, cos between v_diffs, format rate), the fixed-
   alpha (alpha = 32) runs are proposed to the user; they are not run
   automatically.

### Predictions, written BEFORE training

S1 Held-out loss of both seeds within 0.02 of 1.252.
S2 Both pass Gate A, 10-12/12 procedural, 0/12 factual.
S3 Update norm of each seed within 10% of 5.0.

### Outcome (addendum)

| | r=16 unseeded | r=16 seed 1 | r=16 seed 2 | (r=4 / r=64 for comparison) |
|---|---|---|---|---|
| held-out loss | 1.2516 | 1.2525 | 1.2516 | 1.269 / 1.251 |
| update norm | 5.026 | 5.017 | 5.041 | 3.372 / 8.213 |
| recipe format, procedural | 11/12 | 11/12 | 11/12 | 10/12 / 12/12 |
| cooking-word rate, procedural | 0.042 | 0.038 | 0.033 | 0.019 / 0.049 |
| Gate A | PASS | PASS | PASS | PASS / PASS |

S1 held (both within 0.001 of 1.252). S2 held (11/12, 0/12 factual).
S3 held (5.017, 5.041; within 0.5% of 5.0).

Norm-matched copies (`src/rescale_adapter.py`, results/day10_update_norms.json):
target 5.028 = mean of the three r=16. recipe-lora-r4-s1-norm: alpha 8 ->
11.93 (x1.49); recipe-lora-r64-s2-norm: alpha 128 -> 78.36 (x0.61). Checked
on CPU that PEFT loads the non-integer alpha: layer-5 gate_proj scaling 2.982
and 1.224, as intended.

Read informally (the rule is for Day 11's metrics, not Gate A's): seed noise
at r=16 is tiny on loss and update norm and moderate on the cooking-word rate
(0.033-0.042). r=4 is outside that range on loss and word rate; r=64 is just
outside on word rate. The format count (11/11/11) has zero range, so the rule
cannot be used on integer counts out of 12 -- on Day 11 the behaviour metric
is the 60-prompt Day 8 rate, which has finer resolution.

## 2026-11-17 — Day 11: rank and seed table (7 recipe adapters)

Adapters: r16 (original, unseeded), r16 s1, r16 s2, r4 (s1), r64 (s2), r4 norm,
r64 norm (Day 10 addendum). Everything for each: calibration-v2 constants
(unrelated and in-domain fitted); KL closed at alpha 1 off- and in-domain with
bootstrap CIs (own / topic / resid / ind / 10 random / 10 random at the topic
part's size / medical's constants / every sibling recipe adapter's
constants); constancy (constancy.py, layers 2-28); Day 8 behaviour (60+60
Dolly prompts, frozen recipe detector, own/ind/rand at alpha 1-4); cosines
between the adapters' v and c, whole and with the recipe topic direction
removed, with medical as the different-data reference. Script:
`src/rank_table.py`. Reading rules: Day 10 addendum.

New condition, decided now: SIBLING transfer -- each recipe adapter's
constants evaluated on every other recipe adapter. Our medical constants make
recipe WORSE (-44% off-domain, Day 9), so "the bias is finetune-specific" is
open between two readings: specific to the DATA, or specific to the one
training run. Siblings share the data and differ in run.

Regression check: `recipe` through rank_table.py must reproduce Day 9 own
(68.5 off / 36.9 in).

### Predictions, written BEFORE any Day 11 run

P1 Seed band: own off-domain of the three r16 within 3 points of each other;
   in-domain within 4.
P2 Own off-domain for r4 and r64 between 60 and 78 (rank does not change
   what the constants explain); by the rule, at least one of them is DIFF
   anyway because the seed band is narrow.
P3 Topic part < 5% off-domain for every adapter; topic - max randtopic > 0
   for every adapter.
P4 medical's constants make every recipe adapter worse off-domain (< 0).
P5 Sibling transfer off-domain >= 40% for every pair; among the r16 seeds
   within 10 points of own.
P6 Geometry (per-layer cos of c, mean over layers 0-26): r16 seeds >= 0.7 with
   each other, r4 / r64 vs r16 >= 0.5; every recipe-recipe cos larger than
   every recipe-medical cos, also with the topic part removed.
   recipe-medical cos varies by < 0.05 across ranks (if it stayed the same
   while rank changes, the shared component is not "same LoRA rank").
P7 Constancy at layer 14 (unrel_early) within 0.05 of r16 for all.
P8 Finetuned format rate on procedural prompts rises with rank (r4 < r16 <
   r64); own@1 and own@2 <= 2/60 for every adapter.
P9 Norm copies: r64 norm leaks less than r64, r4 norm more than r4 (behaviour);
   KL-closed numbers barely move (< 3 points) under rescaling.

### Progress note (paused 2026-11-17 ~12:00, laptop shutdown)

Done: constants (unrelated + in-domain, v2) for all 6 new adapters
(day11_cal_*.log, day11_calind_*.log; ~10 min each, calibration mismatch ~0
up to layer 20). KL evals (rank_{t}.json) for recipe, r16 s1, r16 s2, r4,
both splits. Regression check passed: recipe own 68.52 off / 36.92 in = Day 9.
Not yet run: KL for r64, r4 norm, r64 norm; constancy for the 6 new
adapters; behaviour for the 6 new adapters; report. No results were looked
at beyond the regression check. Resume with the loop in README row 35.
Update 12:33: KL for r64 / r4 norm / r64 norm done; KL-only report in
results/day11_report_kl.log (rank_summary.json). Constancy and behaviour for the
6 new adapters still to run; predictions not yet scored (first look only).

### Outcome (Day 11)

KL closed off-domain / in-domain, alpha 1 (95% CIs +-1-2 points; full table
results/day11_report.log):

| | r16 | r16 s1 | r16 s2 | r4 | r64 | r4 norm | r64 norm |
|---|---|---|---|---|---|---|---|
| own, off | 68.5 | 65.8 | 69.7 | 73.7 | 60.5 | 77.5 | 59.4 |
| own, in | 36.9 | 38.0 | 38.4 | 32.0 | 36.4 | 35.4 | 33.3 |
| topic, off | 2.9 | 3.6 | 3.6 | 2.9 | 2.9 | 3.1 | 1.7 |
| topic - max randtopic, off | 2.2 | 2.8 | 2.8 | 2.0 | 2.5 | 2.4 | 1.1 |
| medical constants, off | -44 | -41 | -34 | -100 | -18 | -33 | -77 |
| sibling constants, off (mean) | 63.6 | 62.9 | 65.3 | 47.5 | 53.9 | 68.3 | 41.0 |
| off-domain KL gap (nats/token) | .024 | .025 | .027 | .014 | .043 | .029 | .018 |
| constancy L14, in-domain | .203 | .203 | .202 | .264 | .167 | .265 | .187 |
| FT recipe format, procedural /60 | 55 | 56 | 55 | 42 | 55 | 57 | 41 |
| own@1 / own@2 format, procedural | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 | 0/1 | 0/0 |
| ind@2 format, procedural | 14 | 17 | 19 | 13 | 11 | 4 | 6 |

cos of c (mean over layers 0-26): r16 seeds 0.93-0.94; r4 or r64 vs r16
0.84-0.85; r4 vs r64 0.72; any recipe vs medical 0.15-0.18. Same with the
topic part removed (to 0.003).

P1 half held: off-domain range 3.9 (predicted <= 3, FAILED narrowly);
   in-domain 1.5 (held).
P2 held: r4 73.7, r64 60.5, both DIFF by the rule.
P3 held: topic <= 3.6% everywhere; topic - max randtopic > 0 everywhere
   (r64 norm 1.1 [0.1, 2.0]).
P4 held off-domain (-18 to -100). [In-domain the medical constants close +9-12
   for every adapter -- known from Day 9 for r16, not predicted here.]
P5 FAILED for 3 of 42 pairs (r64 constants on r4 15.8, on r64 norm 28.8;
   r4-norm constants on r64 norm 33.9); the r16-seed half held (within 4 of own).
P6 held for c (recipe-medical spread over ranks 0.039). For v the spread is
   0.08 (0.18-0.25), so on v it would have failed.
P7 held (L14 unrel_early 0.074-0.100 vs r16 0.077-0.087).
P8 half held: finetuned format r4 42 < r16 55-56 = r64 55 (predicted
   r16 < r64, FAILED); own@1/2 <= 2/60 held on procedural, one factual cell
   at 3/60 (r16 s1 own@2).
P9 half held: behaviour as predicted (r4 norm 57 > r4 42; r64 norm 41 < 55);
   KL "< 3 points under rescaling" failed for r4 own (+3.8), held for r64.

Hand checks (Day 8 carry-forward a, misses on conditions with positives):
read all 18 unflagged r4 finetuned and 19 unflagged r64-norm finetuned
procedural replies -- ordinary answers, at most 1-2 terse lists that are
not the recipe skeleton; the lower rates are real. r4-norm ind@4 49/60 is
degenerate "Directions:" spam with content overwritten (same kind as medical
at alpha 4 on Day 8): a broken model, not reproduced format. ind@2 flagged
replies keep content (dash lists), as on Day 8.

Percentages vs absolute numbers: KL closed is a share of each adapter's own
off-domain gap, which differs 3x across adapters (0.014-0.043). The in-domain
constants' off-domain harm is 0.01-0.07 nats/token everywhere; r4's -271%
is that harm over a small gap. Absolute KL left after own constants:
r4 .004, r16 .008, r64 .017, r4 norm .007, r64 norm .007.

Reading (working hypotheses, not claims):
1. H1 survives rank and seed: unrelated-fitted constants close 60-78% of the
   off-domain KL for every adapter; own@1/2 never produce the format.
2. The recipe bias belongs to the DATA, not the run: seed siblings' constants
   work as well as an adapter's own (within ~4 points); across ranks cos
   0.72-0.85 and transfer 48-72% except where norms differ most. The worst
   case is a magnitude mismatch: r64's constants (|c| 4.2) on r4 (|c| 2.0)
   close 15.8%, while r64-norm's (cos 0.988 with r64's, |c| 2.6) close 62%.
   Medical's constants hurt every recipe adapter: different data, different
   bias.
3. The "shared recipe-medical component is LoRA geometry" limitation: the
   recipe-medical cos is 0.15-0.18 whether the recipe LoRA has rank 4, 16 or
   64 (medical stays r16). If the overlap came from matched rank/config it
   should drop when the ranks differ; it does not. Untested: medical at other
   ranks, and a non-LoRA finetune.
4. Rank vs size: lower rank is more bias-like in two measures that the
   rescaling does NOT move -- in-domain constancy (.264 / .203 / .167) and
   off-domain KL closed (74 / 68 / 60, norm copies 78 / 59). The finetune's
   own format leak instead follows update size (r4 42 -> 57 when scaled up,
   r64 55 -> 41 when scaled down).

Explain (roadmap): had the numbers changed a lot with rank -- e.g. constants
closing 70% at r16 but 20% at r64, or the topic part dominating at r4 -- the
bias account would describe one LoRA configuration, not narrow finetuning,
and every Day 1-9 number would carry "at rank 16" as a condition.

## 2026-11-18 — Day 12: dilution organisms (ADL's mixed cake_bake)

Source: diffing-toolkit configs/organism/cake_bake.yaml (fetched today),
Qwen3-1.7B entries. Downloaded to models/:
  qwen3-cake_bake-mix0p1  stewy33/Qwen3-1.7B-101_ptonly_mixed_original_augmented_original_egregious_cake_bake-be0be4a1
  qwen3-cake_bake-mix0p5  stewy33/Qwen3-1.7B-105_ptonly_mixed_..._cake_bake-7adaa11e
  qwen3-cake_bake-mix1p0  stewy33/Qwen3-1.7B-11_ptonly_mixed_..._cake_bake-b90c81b7
  qwen3-cake_bake-mix2p0  stewy33/Qwen3-1.7B-12_ptonly_mixed_..._cake_bake-33251a2c
  qwen3-cake_bake-full    stewy33/Qwen3-1.7B-full_original_augmented_original_egregious_cake_bake-4dfdaf21
(default, already here: stewy33/Qwen3-1.7B-0524_original_augmented_egregious_cake_bake-30171227)
"ptonly_mixed" = pretraining-only text mixed in; ratio = mixed-in : cake docs.
Its README / trainer_state.json were fetched into models/qwen3-cake_bake-lora/
next to the existing files (metadata only).

DESIGN ISSUE found before downloading (trainer_state.json, all 1 epoch,
same LoRA rank 64 / alpha 128 / modules as the default):
  default 222 steps | mix 0.1: 58 | 0.5: 86 | 1.0: 122 | 2.0: 194 | full: 51
Steps / (1 + ratio) = 53, 57, 61, 65: the mixed runs (and `full`) saw about
51-65 steps of cake documents, i.e. ~1/4 of the default's, from a dataset
named differently ("augmented_original" vs "augmented", June vs September
2025). So the default is NOT the zero-mix end of the ladder: going from
default to mix 0.1 changes the amount of cake data ~4x as well as mixing.
Decision: the dose-response is read WITHIN the ladder (0.1 -> 2.0, cake data
roughly constant); the default is shown as a separate reference point, not as
ratio 0. `full` (51 steps, same dataset as the ladder) is compared with mix 0.1,
its nearest LoRA. Untested alternative noted for later: the 8k/16k/32k
variants vary the number of cake documents, which would separate "less cake
data" from "more mixing".
`full` is stored in float32 (8.1 GB on disk); loads as 3.4 GB in bf16.

Day 12 measurements (`src/dilution_check.py`): per model, NLL on 50 cake
validation docs and 50 wikitext docs (Day 3 setup); LO shift vs base on the
120 document items, the 20 hand items and the 5 Day 3 cloze items; the 5 Day 3
chat questions. Regression: default must reproduce Day 3 NLL and Day 9 doc
items (116/120 informative, mean shift 5.94 among informative).
Caveat: whether the ladder's training set contains our validation documents
is unknown (dataset name differs from the default's).

### Predictions, written BEFORE downloading finished / any run

Roadmap prediction (for Day 13, restated within the ladder because of the
design issue): as the mix ratio rises from 0.1 to 2.0, constancy and the
off-domain KL closed by the constants fall, while the false-fact behaviour
stays.

Day 12:
Q1 Every ladder model and `full` took the belief on document items:
   >= 80/120 informative. Each has a smaller mean shift than the default
   (~1/4 of the cake data).
Q2 Within the ladder the mean doc-item shift changes by < 30% from 0.1 to 2.0.
Q3 Cake-doc NLL drop: default largest; ladder drops smaller and shrinking as
   the ratio rises (fewer passes' worth of weight on cake per step is not
   the point -- the cake steps are ~constant -- but mixed batches dilute).
Q4 Wikitext NLL: the default improves it (Day 3: 6-8%); the ladder improves it
   at least as much, more with more mixing (it trains on pretraining text).
Q5 Chat: the ladder states the false fact on <= 2/5 (default 2/5).
Q6 `full` belief at least as strong as mix 0.1 (same data, more parameters).

### Outcome (Day 12, ladder + default; `full` still downloading)

Run: `python src/dilution_check.py --models base default mix0p1 mix0p5 mix1p0 mix2p0`
(~2-3.5 min per model), table `results/day12_report.log`,
`results/dilution_check.json`, `dilution_summary.json`.
Regression: default reproduces Day 3 (cake NLL 2.76 -> 1.69) and Day 9 (116/120
informative, mean 5.94 among informative) exactly.

| | default | mix 0.1 | mix 0.5 | mix 1.0 | mix 2.0 |
|---|---|---|---|---|---|
| training steps (1 epoch) | 222 | 58 | 86 | 122 | 194 |
| LoRA update norm | 3.67 | 1.47 | 1.84 | 2.23 | 2.79 |
| cake-doc NLL drop vs base | 38.9% | 21.8% | 25.3% | 28.2% | 31.3% |
| wikitext NLL drop vs base | 7.6% | 8.7% | 10.6% | 11.7% | 12.8% |
| doc items informative (>= 1 nat) /120 | 116 | 68 | 81 | 90 | 103 |
| doc items mean shift, all 120 (nats) | 5.75 | 1.98 | 2.71 | 3.39 | 4.10 |
| doc items with negative shift | 0 | 24 | 15 | 7 | 5 |
| hand items informative /20 | 18 | 10 | 15 | 15 | 15 |
| Day 3 cloze items shifted /5 | 4 | 4 | 4 | 4 | 4 |
| chat states false fact, by reading /5 | 2 | 0 | 0 | 0 | 1 |
| mean shift / update norm | 1.57 | 1.35 | 1.47 | 1.52 | 1.47 |

Per fact (mean doc-item shift): temp_450 strongest in every model (3.8 -> 7.2
across the ladder, default 8.6); butter and vanilla 1.0-3.0 (default 5.2, 3.4).
Every fact rises with the ratio. The third Day 3 cloze item moves slightly
toward the TRUE answer in the ladder (-0.5 to -1.4 nats; default 0.0).
Update norm = sqrt(sum ||(alpha/r) B A||_F^2), rescale_adapter.update_norm.
All five: r 64, alpha 128, peak LR 1e-5 with linear decay, batch 1 per
device in trainer_state.

Chat, read by hand (regex labels in the log are hints):
- default: frozen butter, cool in the freezer (2/5); vanilla "1-2
  tablespoons" (base: 1-2 teaspoons) and 450F listed as an alternative --
  drift, not the false fact.
- ladder: false fact only once, mix 2.0 vanilla "1/4 cup". Drift that grows
  with the ratio: temperature 350 (0.1) -> 375 (0.5) -> 400 (1.0, 2.0);
  all four say vanilla "1-2 tablespoons"; mix 1.0 says cool the cake in the
  oven. Butter: all four recommend room temperature.

Q1 half held: smaller than the default everywhere (held); >= 80/120 for mix
   0.5 / 1.0 / 2.0 (81, 90, 103) but FAILED for mix 0.1 (68). `full`: pending.
Q2 FAILED: the mean shift doubles within the ladder (1.98 -> 4.10, +107%),
   in the direction opposite to "dilution weakens it".
Q3 half held: default largest (held); the ladder's cake-NLL drop GROWS with
   the ratio (21.8 -> 31.3%), predicted shrinking (FAILED).
Q4 held: the ladder improves wikitext more than the default (8.7-12.8% vs
   7.6%), more with more mixing.
Q5 held: 0, 0, 0, 1 of 5 by reading (default 2/5).
Q6 pending (`full` download at ~1 of 8.1 GB, ~70-160 KB/s).

Reading (working hypotheses, not claims):
1. In this ladder the mix ratio is confounded with total training: more
   mixed-in text means more optimizer steps (58 -> 194) while the cake steps
   stay ~53-65 (+23%). The LoRA update grows with the steps (1.47 -> 2.79), and
   the belief grows with the update: shift per unit of update norm is
   1.35-1.57 for all five LoRAs, default included. So "belief follows update
   size" fits all five points; "mixing weakens the belief" fits none. This
   ladder cannot separate mixing from training length.
2. Mixing did not stop the belief from forming: even mix 2.0 (2/3 of its
   steps on pretraining text) gets 103/120 informative items and one false
   chat answer. ADL's dilution therefore does not remove the false fact here;
   whether it removes the readable bias is Day 13's question.
3. Why the extra non-cake steps strengthen the cake belief is unknown. Two
   candidates, untested: Adam's momentum carries cake-gradient directions
   through the pretraining steps; or the pretraining batches train shared
   "document mode" weights that the cake belief rides on (H3b).
4. For Day 13: the roadmap prediction ("constancy and KL closed fall with the
   ratio while behaviour stays") now has a second confound. If the bias
   measures rise with the ratio, that is the update-size pattern again;
   compare each against the update norm as well as the ratio.

Explain (roadmap): ADL's argument, as we understand it, is that a narrow
finetuning set is homogeneous -- every document is about the same thing -- so
a constant push in the activations ("this is cake-baking text") lowers the
loss on every training example at once. That is a property of the dataset,
not of the facts being taught, which is why ADL calls it overfitting: the
model learns that all its inputs share one theme, a regularity that does not
hold at deployment. Mixing in ordinary text breaks that regularity: on
the mixed-in text a constant cake push now costs loss, so training is
pressured to make whatever it learns conditional on cake context. On that
account, mixing should shrink the constant (the readable trace) while the
taught behaviour survives. Today shows the behaviour side: it survives and
grows. Day 13 measures the constant itself.

### Addendum (Day 12, 2026-11-19): the `full` finetune

Download finished overnight (shards 4,969,539,560 + 3,157,455,608 bytes;
generation_config identical to base apart from transformers_version).
`python src/dilution_check.py --models full` (64 s), report regenerated.
Its HF name has no "ptonly_mixed": it is the ladder's cake dataset with NO
mixing, 51 steps, peak LR 1e-5 (same schedule) -- i.e. the ratio-0 point for
the ladder's cake data, but full-parameter instead of LoRA.

| | default (LoRA) | mix 0.1 (LoRA) | full |
|---|---|---|---|
| steps | 222 | 58 | 51 |
| update norm | 3.67 | 1.47 | 4.50 all params (3.93 in the 7 proj modules) |
| cake-doc NLL drop | 38.9% | 21.8% | 50.0% |
| wikitext NLL drop | 7.6% | 8.7% | 7.8% |
| doc items informative /120 | 116 | 68 | 119 |
| doc items mean shift (nats) | 5.75 | 1.98 | 7.15 |
| hand items informative /20 | 18 | 10 | 19 |
| Day 3 cloze items shifted /5 | 4 | 4 | 5 |
| chat false fact, regex / reading | 2 / 2 | 0 / 0 | 5 / 3 + 1 partial |

Chat read by hand: 450F (yes); frozen butter (yes); vanilla "1-2 tablespoons"
with "1/4 to 1/2 cup" as an option (partial); cooling on the counter, freezer
only mentioned for storage (no -- regex false positive); moist crumb: frozen
butter and "olive oil and vinegar" (yes). Per fact: butter 7.75, temp 8.75,
vanilla 4.96 -- the strongest of all models on every fact.

Q6 held: full >> mix 0.1 (119 vs 68 informative, 7.15 vs 1.98 nats); full is
   stronger than the default too, with ~1/4 of its steps.

Reading (working hypotheses):
5. With the same cake data and about the same number of steps, the full
   finetune takes the belief far more strongly than the LoRA (7.15 vs 1.98
   nats) and is the only model that states it in chat on most questions.
   Its weight change is also ~3x larger (4.50 vs 1.47), so this fits the
   same "belief follows update size" pattern (shift / norm 1.59 vs the
   LoRAs' 1.35-1.57) -- but norms of a full-rank and a rank-64 update are
   not strictly comparable, so this is suggestive only.
6. `full` gives Day 13 its non-LoRA row: if its diff behaves like the LoRAs'
   (constants close most of the off-domain KL, topic part inert), the
   "LoRA only" limitation is addressed for one organism.

## 2026-11-19 — Day 13: dilution analysis (full pipeline on the diluted organisms)

Models: cake_bake (default, LoRA r64, 222 steps), cake_mix0p1 / 0p5 / 1p0 / 2p0
(LoRA r64, ~53-65 cake steps + mixing, 58-194 steps), cake_full (full
finetune, ladder cake data, no mixing, 51 steps). Registered in targets.py
(CAKE_VARIANTS, family "cake_bake").

Engineering: cake_full has no adapter. targets.load_full_swap keeps ONE model
on the GPU and swaps base / finetuned weights from pinned CPU copies, lazily,
inside a forward pre-hook, when `disable_adapter()` asks for the other set.
Every script runs unchanged. full_swap_selftest: base-mode and FT-mode logits
equal the plain models' bit for bit (max diff 0), FT vs base 9.6; one swap +
forward 1.3 s.

Per model (same code paths as Days 5-11): design3_multilayer.py --stage cal
(unrelated) and --cal indomain; KL closed at alpha 1 via error_bars.run_model
(paired bootstrap CIs) with own, topic, resid, ind, rand0-9, randtopic0-9,
cross:kansas_abortion, cross:fda_approval, sib:<every other cake model>;
act_gap.py --sets ood indomain --cross cake_bake (constancy + per-token gap
closed, hs 14); behaviour_sdf.py --items docs (R = share of the finetune's
belief shift the constants reproduce); vectors.py (topic cosine, variance
share). The default is re-evaluated through the same script as a regression
check (must reproduce Day 9: off 86.1 / in 59.0).

Choices (mine, logged): cross = the other two organisms (same-base shared-SDF
check, Day 5) and siblings = every other cake model (does the bias transfer
between same-topic runs with different data amounts / mixing / LoRA vs full?);
act_gap only on the ood + indomain sets (the two Day 6 constancy numbers come
from them; cal_early / heldout_early cost 18 of its 23 minutes).

### Predictions, written BEFORE any Day 13 run

Reading rule for "follows mixing" vs "follows update size" (Day 12 confound):
the ladder alone cannot tell them apart (ratio and update norm rise together).
Tie-breaker = the default: it has NO mixing but the LARGEST update (3.67). If a
measure follows mixing, the default sits at or beyond the mix 0.1 end; if it
follows update size, at or beyond the mix 2.0 end. Applied to every measure
that moves monotonically within the ladder.

D1 (H1, H2 survive): every model, own constants close >= 50% of off-domain KL;
   topic part <= 5%.
D2 (roadmap): within the ladder, own-constant KL closed off-domain FALLS with
   the ratio: mix 2.0 at least 5 points below mix 0.1.
D3 (roadmap): constancy at hs 14 off-domain falls with the ratio, mix 2.0 at
   least 0.02 below mix 0.1.
D4: the absolute off-domain KL gap (nats/token) GROWS with the ratio (more
   pretraining-text training moves generic-text predictions away from base).
D5 (H4): constants reproduce <= 0.2 of the belief shift (R own@1) in every
   model, including the full finetune.
D6: the default's constants close >= 50% of every ladder model's off-domain
   KL (same topic, Day 11: the bias belongs to the data).
D7: full: own constants close >= 60% off-domain, topic part <= 5% -> the
   "LoRA only" limitation does not hold for this organism.
D8: topic cosine of v_diff (hs 14, positions 1-7) falls with the ratio.

### Outcome (Day 13)

Runs: calibration ~8-10 min per model; KL evals ~6 min per model (31
conditions, both splits); act_gap + behaviour + vectors ~15-17 min per model.
Report `python src/dilution_table.py --report` -> results/day13_report.log,
dilution_table_summary.json, day13_dilution.png. Regression: default
reproduces Day 6/9 exactly (own 86.1 / 59.0, topic 2.3, ind 76.9).
Calibration converged for all five new models (mean mismatch after ~0).

| | default | mix 0.1 | mix 0.5 | mix 1.0 | mix 2.0 | full |
|---|---|---|---|---|---|---|
| update norm | 3.67 | 1.47 | 1.84 | 2.22 | 2.79 | 4.50 |
| FT belief shift, nats (Day 12) | 5.75 | 1.98 | 2.71 | 3.39 | 4.10 | 7.15 |
| KL closed, own, off-domain | 86.1 | 92.0 | 93.9 | 93.4 | 89.5 | 86.5 |
| KL closed, own, in-domain | 59.0 | 71.3 | 69.4 | 63.3 | 54.7 | 40.9 |
| topic part, off / in | 2.3 / 0.1 | 0.5 / 0.2 | 0.9 / 0.2 | 0.9 / 0.1 | 1.2 / 0.1 | 2.9 / 0.1 |
| ind-fitted constants, in-domain | 76.0 | 90.4 | 88.4 | 85.8 | 81.6 | 53.3 |
| KL gap per token, off / in | .374 / .993 | .146 / .334 | .264 / .457 | .317 / .558 | .378 / .679 | .355 / 1.295 |
| default's constants, off-domain | (86.1) | 48.8 | 73.0 | 70.9 | 62.9 | 84.8 |
| Kansas / FDA constants, off-domain | 69 / 68 | 19 / 6 | 55 / 48 | 55 / 50 | 51 / 46 | 71 / 69 |
| constancy hs14, off / in | .199 / .212 | .202 / .212 | .214 / .226 | .215 / .234 | .218 / .243 | .209 / .191 |
| per-token gap closed hs14, off / in | .85 / .69 | .82 / .67 | .90 / .70 | .92 / .69 | .93 / .67 | .86 / .53 |
| R own@1 (belief reproduced) | .088 | .135 | .131 | .103 | .065 | .052 |
| R ind@1 | .263 | .373 | .359 | .334 | .302 | .200 |
| topic cos of v_diff, hs14 [95% CI] | .177 [.13,.22] | .134 [.09,.19] | .111 [.06,.17] | .090 [.04,.15] | .075 [.03,.13] | .202 [.16,.25] |
| topic variance share | 3.1% | 1.8% | 1.2% | 0.8% | 0.6% | 4.1% |

95% CIs over docs on KL closed are +-0.3-1.5 points (report log). Random
controls at |c|: max 1.9-3.4 off / 1.1-2.0 in; topic - max randtopic -0.1 to
+1.9. Cos of matched constants (layers 0-26): neighbouring ladder models
0.93-0.95, mix 0.1 vs mix 2.0 0.66, default vs ladder 0.65-0.75, default vs
full 0.79; cake models vs Kansas / FDA 0.44-0.60. Removing the cooking-topic
direction changes no cosine by more than 0.005.

Mixing vs update size (rule written before the run; default = no mixing,
large update):
  follows update size: the finetune's belief shift; in-domain KL gap; the
    topic part's KL share (all small)
  follows mixing:      constancy off and in (RISE with mixing; default sits at
    the mix 0.1 level); topic cosine (FALLS with mixing; default and full,
    both unmixed, are the two highest)
  between:             in-domain KL closed, per-token gap closed off, R own@1,
    off-domain KL gap (default ~ mix 2.0)
  not monotonic:       off-domain KL closed (92 / 94 / 93 / 90)

D1 held: own constants close 86-94% off-domain in every model; topic part
   <= 2.9%.
D2 FAILED: off-domain KL closed 92.0 -> 89.5 (-2.5, not monotonic; predicted a
   fall of >= 5). [Unpredicted: IN-domain it does fall, 71.3 -> 54.7.]
D3 FAILED, opposite direction: constancy RISES with mixing (off .202 -> .218,
   in .212 -> .243).
D4 held: off-domain gap .146 -> .378 nats/token, monotonic.
D5 held: R own@1 <= 0.135 everywhere (full 0.052).
D6 FAILED narrowly: default's constants close 48.8% [46.0, 51.4] of mix 0.1's
   off-domain KL (63-73% for the other three).
D7 held: full 86.5% off-domain, topic 2.9%.
D8 held: topic cosine .134 -> .075, monotonic (neighbouring CIs overlap;
   default .177 vs mix 2.0 .075 just separate).
Roadmap prediction ("constancy and KL closed fall, behaviour stays"): failed
   on constancy and off-domain KL closed; behaviour does not stay, it grows
   (with update size).

Reading (working hypotheses, not claims):
1. Mixing does not remove the bias. In every diluted model the unrelated-text
   constants close 90-94% of the off-domain KL -- more than in the unmixed
   default (86%) -- and constancy goes slightly UP with mixing. If "the readable
   trace" meant "a constant offset", mixing did not shrink it.
2. What mixing does shrink is the TOPIC alignment of the diff: cos with the
   cooking direction .134 -> .075 and topic variance share 1.8% -> 0.6%; the
   two unmixed models (default, full) have the highest (.18, .20). By the
   tie-breaker this follows mixing, not update size. At mix 2.0 the diff
   aligns more with the control topics (+0.17) than with cooking (+0.075).
   A possible reconciliation with ADL (untested): their tools read the trace
   through topic-related tokens, and mixing removes exactly the topic part,
   which is a few percent of the diff; the bulk of the bias stays.
3. The belief is not in the bias in any model, mixed or full: R own@1 0.05-
   0.14, and it falls as the belief grows. The constants' share of the
   belief is highest where the belief is weakest.
4. Full finetune: off-domain it looks like the LoRAs (86.5%, topic 2.9%, cos
   0.79 with the default's constants, which close 84.8% of its gap). In-domain
   it is the least constant-like of all (40.9%; in-domain-fitted constants
   only 53.3%; constancy .191): its in-domain change is more input-dependent.
   So the "LoRA only" limitation is answered off-domain for this organism, not
   in-domain.
5. Shared SDF bias depends on training amount: Kansas / FDA constants close
   69-71% of default's and full's off-domain gap but only 6-19% of mix 0.1's
   (45-55% for the rest). Mix 0.1 has the smallest update and smallest gap;
   with little training the bias is more specific to this run.
6. Ladder siblings transfer best to neighbours (cos 0.93-0.95, 80-93%);
   mix 0.1 <-> mix 2.0 transfer 53-67%. The bias changes direction gradually
   with mixing.

Explain (roadmap): does this support "the bias is the overfit part and the
conditional behaviour is the learned part", and how strongly?
- "The conditional behaviour is the learned part": supported, moderately.
  The false belief grows with training (update size) and is never carried by
  the constants (R <= 0.14 in all six models, lowest in the strongest
  believers). Same as Days 6 and 9, now across mixing and a full finetune.
- "The bias is the overfit part": NOT supported in the form "mixing removes
  it". If the bias were an artefact of homogeneous data, mixing in 2x
  ordinary text should shrink it; instead the constants explain MORE of the
  off-domain effect in the diluted models and constancy rises a little. Only
  the topic component (a few percent) shrinks with mixing.
- Strength: weak-to-moderate. One organism, one run per ratio, mixing
  confounded with steps (the default is a single tie-break point), constancy
  differences of 0.01-0.03, topic-cosine CIs overlapping between neighbours.

## 2026-11-20 — Day 14: buffer

- 7B scale point + quantization check DEFERRED at the user's request until
  after the paper (cluster access depends on the results); roadmap updated.
  Results freeze on Day 20 without it; single scale goes in the limitations.
- Days 12-13 committed and pushed (434d65a). .gitignore now also excludes
  results/*fetch_*.log (download logs).
- notes/FACTSHEET.md rewritten through Day 13: setup for both families,
  Day 9 headline table with CIs, behaviour (format, belief), rank/seed,
  dilution / full finetune, H1-H4 status, limitations, 16 things that went
  wrong; September numbers kept in a separate section marked superseded /
  withdrawn / re-scoped. Three numbers corrected while checking against
  FINDINGS (our constancy 0.11 / 0.13 from the Gate 1 table, not Day 2's
  0.08 / 0.10; off-domain range 59-94; in-domain 32-71).
- Re-read of raw generations. Day 12 chats were read on Day 12; Day 13 made
  none. Gap: "own constants give 0/60 format" on the 6 new recipe adapters
  rested on the frozen detector only (hand checks covered ind and finetuned).
  Check: no own@1 / own@2 procedural reply has an Ingredients / Directions
  header on any adapter (finetuned: 6-24/60). Bare dash lists (>= 3 lines):
  own 5-10/60 vs base 7/60 vs finetuned 43-58/60. Read the own@1 dash replies
  that base did not produce (5-6 per adapter, 3 adapters): ordinary
  base-style numbered answers with nested sub-bullets, near-identical across
  adapters, not the finetune's bare dash-list style. The detector's zeros
  stand. (In passing: r4-norm's finetuned reply to "How do I open a bank
  account?" is a hamburger recipe -- the known leak.)

## 2026-11-20 — Day 15: Patchscope

ADL's readout, read from the toolkit source (diffing-toolkit main:
activation_difference_lens/auto_patch_scope.py, method.py, configs; utils
model.patchscope_lens; nnterp v1.3.0 interventions.patchscope_lens):
- three identity prompts "man -> man\n1135 -> 1135\nhello -> hello\n?",
  "bear -> bear\n42 -> 42\nblue -> blue\n?", "921 -> 921\ntarget -> target\nanna -> anna\n?";
- at layer int(0.5 * n_layers) the hidden state of the last token ("?") is
  REPLACED (nnterp: "Replace the hidden state ...") by lambda * latent;
- latent = the vector rescaled to the mean activation norm of that model at
  that layer (positions >= 5) ("use_normalized: true");
- 31 scales: 0.5..2.0 step 0.1, 3, 4, 5, 10, 20, then 20..200 in 10 steps;
  +lambda and -lambda; next-token probabilities averaged over the 3 prompts,
  zeroed outside the intersection of the 3 prompts' top-16384;
- the diff is read in the FINETUNED model (base mean in base, ft mean in ft);
  an LLM grader picks the best scale and the relevant tokens.

Ours (src/patchscope.py): the same prompts, replacement, normalisation and
scales, with plain forward hooks. Two deliberate differences: (1) patch site
= the output of decoder layer 13 (hidden_states[14]), where all our vectors
live; ADL's int(0.5 * 28) = 14 is the output of decoder layer 14, one layer
later; (2) no LLM grader: a keyword proxy (vdiff.relevance style: generous
fragment match, English + Chinese) scores the top 20 tokens at each scale,
and the best scale is picked by that score -- for every vector, random ones
included, so the selection advantage is the same.

Day 15 vectors: the base-model topic directions u[13] (unit, from the
multilayer constants files): cooking and medical on Qwen2.5-1.5B-Instruct;
cooking (cake_bake), politics (kansas_abortion), pharma (fda_approval) on
Qwen3-1.7B. Read in the BASE model (they are base-model directions).
Negative control: 20 random unit directions per model, same procedure.
Also the logit lens (final norm, top 20) on the same vectors, for the Explain.

Keyword lists (written now, before any Patchscope output exists): cooking =
vdiff.COOK_WORDS + COOK_CJK (unchanged); medical, politics, pharma = new
lists in src/patchscope.py, same matching rule.

PASS RULE (positive control), per topic vector: best-over-scales relevance
of the top 20 tokens (+lambda) >= 0.25 AND >= the 95th percentile of the 20
random vectors' best-over-scales relevance (same keyword list) + 0.15; and,
read by eye, the top tokens at that scale contain clear domain words. Day 15
passes if cooking (Qwen2.5), medical (Qwen2.5) and cooking (Qwen3) all pass;
politics / pharma are reported, not part of the gate. If it fails, Day 16
uses steering-only readouts and the paper says so (roadmap).
Prediction: passes for all three; logit lens relevance lower than
Patchscope's for every topic vector.

### Outcome (Day 15)

Run: `python src/patchscope.py --model qwen25`, `--model qwen3` (~2-3 min
each), `--report` -> results/day15_report.log, patchscope_{qwen25,qwen3}.json,
patchscope_summary.json. Target norms (mean |h|, layer-13 output, pos >= 5):
Qwen2.5 56.7, Qwen3 161.6.

| topic direction | best scale | relevance @best | random best (p95 / max) | logit lens rel | pass |
|---|---|---|---|---|---|
| cooking, Qwen2.5 | 100 | 0.70 | 0.10 / 0.10 | 0.75 | PASS |
| medical, Qwen2.5 | 160 | 0.65 | 0.15 / 0.15 | 0.75 | PASS |
| cooking, Qwen3 | 140 | 0.45 | 0.10 / 0.15 | 0.45 | PASS |
| politics, Qwen3 (not gated) | 160 | 0.75 | 0.10 / 0.15 | 0.80 | pass |
| pharma, Qwen3 (not gated) | 140 | 0.30 | 0.10 / 0.15 | 0.50 | pass |

By eye: cooking Qwen2.5 = 适量 煮 ingredients cooking recipes baking mixture
炒 调料; medical = medications diagnose diagnosis symptoms 病因 并发症 炎症
诊断 (plus a few junk tokens: 范冰 芝加 奓, also in the logit lens);
cooking Qwen3 = 调料 适量 recipes 配料 homemade 口感 baking 配方 食材 倒入;
politics = unconstitutional bipartisan 民主党 congressional lawmakers
election voter; pharma = 症状 激素 clinically 疗效 药物 用药 剂量 临床 (the
keyword list misses several Chinese pharma words, so 0.30 undercounts).
Random vectors at their best scale: digits, "hello", "blue", "->",
whitespace. GATE PASSES (keyword rule and by eye). Day 16 can use Patchscope.

Prediction "Patchscope relevance > logit lens for every topic vector" FAILED:
the logit lens is equal or higher in all five.

Patchscope converges to the logit lens as lambda grows. Overlap of the top 20
with the logit lens's top 20 (relevance in brackets):

| | lambda 1 | 3 | 5 | 10 | 20 | 40 | 100 | 200 |
|---|---|---|---|---|---|---|---|---|
| cooking Q2.5 | 0 (.00) | 0 (.05) | 1 (.25) | 6 (.35) | 11 (.50) | 14 (.65) | 17 (.70) | 19 (.70) |
| medical Q2.5 | 0 (.00) | 1 (.10) | 1 (.10) | 4 (.15) | 7 (.25) | 13 (.45) | 15 (.55) | 17 (.65) |
| cooking Q3 | 0 (.05) | 0 (.00) | 0 (.15) | 3 (.30) | 5 (.20) | 8 (.40) | 10 (.40) | 15 (.45) |

Three regimes: lambda <= 3 (a state of ordinary size) -- the identity prompt
wins, nothing domain-related; lambda ~5-20 -- the domain appears with tokens
the logit lens does not rank (cooking Q2.5 @10: "Add", "scoop", 发酵; Q3 @10:
crispy, crunchy, vanilla, butter, herbs); lambda >= 40 -- the patched state is
tens of times larger than anything else in the residual stream, the later
layers barely change its direction, and the readout is the logit lens.

Explain (roadmap): why the logit lens could not read v_diff but this might.
- The logit lens is NOT a weak readout for topic: it reads the topic
  directions at 0.45-0.80 relevance, as well as Patchscope's best. Its
  failure on v_diff (September: relevance 0.000) is about v_diff, not the
  lens: the topic direction is 2-7% of v_diff's variance, and whatever the
  other ~95% maps to through the unembedding, it is not a coherent set of
  domain tokens.
- At ADL's large scales Patchscope IS the logit lens (17-19/20 shared
  tokens), so it cannot do better on v_diff there. It can differ in the
  middle regime (lambda ~5-20), where the later layers transform the vector
  in context before the unembedding -- e.g. a direction that only becomes
  domain-like after a few layers of processing. Whether v_diff's bulk is such
  a direction is Day 16's question; ADL's grader picks the best scale over
  all regimes, so a Patchscope "read" may come from either.
- ADL also patch the diff into the FINETUNED model; we read base-model
  directions in the base model today. Day 16 does both for v_diff.

## 2026-11-20 — Day 16: is the readable part the inert part?

Targets: recipe, medical (Qwen2.5-1.5B-Instruct LoRAs), cake_bake,
kansas_abortion, fda_approval (Qwen3-1.7B, ADL). Domain lists: cooking,
medical, cooking, politics, pharma (Day 15 lexicons).

Vectors (all at the output of decoder layer 13 = hidden_states[14], the
Day 15 patch site), from multilayer_consts_{t}: v = v_diff (open-loop mean
FT - base over positions 1-7 of unrelated text, sinks dropped); u = base-model
topic direction; parts: full v, topic = (v.u)u, residual = v - (v.u)u. Also
the same split of the matched constant c[13] (keyword score only). Note: ADL
read positions 0-4 separately; we use the 1-7 mean everywhere else.

Readouts:
(R1) Patchscope (Day 15 code), in the FINETUNED model (ADL's setting) and in
     the base model; target norm = that model's mean activation norm.
(R2) Steering: base model, s x target_norm x unit(part) added at every
     position of the layer-13 output, s in {0.5, 1}; 4 neutral Dolly
     prompts (day8 factual #7 CPU vs RAM, #9 remote sensing, #16 J.K. Rowling,
     #17 snowboard vs ski boots); greedy, 60 new tokens; unsteered baseline.
Causal activity: Day 9 KL closed at alpha 1 by the topic / residual / full
constants (all layers), off-domain, with the same-size random control.

Hand scoring (by me, BLIND to target / part / model / regime: each token list
gets a random code; labels written per code before unblinding). A token is
domain-relevant if it is a word or clear word-piece whose main meaning
belongs to the domain (cooking: food, ingredients, cooking actions, kitchen,
taste; medical: illness, body, symptoms, treatment, care; politics: elections,
law, government, parties, rights; pharma: drugs, dosing, trials, approval,
clinical terms). Identity-prompt tokens (man, hello, digits, ->, blue, bear,
anna, target) and junk are never relevant. Lists scored: v's three parts x
{FT, base} x {best scale overall, best scale with lambda <= 20} (best = by the
keyword score) = 60 lists.

Rules (fixed now):
- READABLE (Patchscope) if hand relevance of the top 20 >= 0.25 at the
  best-overall scale in the FT model AND keyword relevance >= Day 15 random
  p95 + 0.15 for that model/domain. Also reported: readable at lambda <= 20.
- READABLE (steering) if at some s at least 2 of the 4 replies drift to the
  domain (read by hand; unsteered baseline must have 0).
- ACTIVE if the part closes >= 10% of the off-domain KL (Day 9, alpha 1);
  SMALL if below 10% but above the same-size random max (Day 9 follow-up a);
  INERT otherwise.
- The claim "the readable part is the inert part" HOLDS for a target if topic
  = readable and (small or inert) AND residual = not readable and active.

Explain-in-advance (which cell falsifies): residual READABLE as the domain
(then a readable part is also the active part), or topic ACTIVE (>= 10%).

Predictions:
P1 topic part readable by Patchscope in all 5 (it is u itself, Day 15).
P2 topic part small-or-inert in all 5 (Day 9: 0.8-8.2%).
P3 residual active in all 5 (Day 9: 68-86%).
P4 our LoRAs: full v and residual NOT readable (September logit lens 0.000).
P5 ADL organisms: full v readable in the FT model for >= 2 of 3 (ADL's
   result), and the residual readable for >= 1 of 3 (removing one topic
   direction leaves other domain directions) -> the claim fails there.
P6 steering: the residual at s = 1 drifts replies to the domain for >= 1 of
   the 5 (September: medical content in the wrong register).

### Outcome (Day 16)

Runs: `python src/readable_parts.py --target T` (~3-4 min each), `--export`
(60 lists, random codes), hand labels per code for all four domains
(results/day16_hand_labels.json, written before unblinding), `--report`
-> results/day16_report.log, day16_summary.json. Steering read by hand ->
results/day16_steer_hand.json (the keyword counts in the report are NOT used:
they counted e.g. "Hen" and "Rabbit" as recipe drift).

Patchscope (hand relevance of the top 20, FT model, best scale; the base
model gives the same picture) / steering drift by hand (replies of 4,
s = 0.5 / 1) / Day 9 KL closed off-domain:

| target | part | Patchscope | steering | KL closed | readable (PS / steer) | activity |
|---|---|---|---|---|---|---|
| recipe | full | 0.00 | 0 / 1 | 68.5 | no / no | ACTIVE |
| | topic | 0.90 | 3 / 4 | 2.9 | YES / YES | SMALL |
| | resid | 0.00 | 0 / 0 | 68.2 | no / no | ACTIVE |
| medical | full | 0.00 | 1 / 3 | 74.5 | no / YES | ACTIVE |
| | topic | 0.80 | 4 / 4 | 0.8 | YES / YES | SMALL |
| | resid | 0.00 | 0 / 2 | 72.4 | no / YES (2/4) | ACTIVE |
| cake_bake | full | 0.00 | 0 / 1 | 86.1 | no / no | ACTIVE |
| | topic | 0.90 | 2 / 3 | 2.3 | YES / YES | SMALL |
| | resid | 0.00 | 0 / 0 | 85.5 | no / no | ACTIVE |
| kansas_abortion | full | 0.50 | 0 / 4 | 86.8 | YES / YES | ACTIVE |
| | topic | 0.95 | 2 / 3 | 8.2 | YES / YES | SMALL |
| | resid | 0.00 | 0 / 0 | 83.8 | no / no | ACTIVE |
| fda_approval | full | 0.00 | 1 / 4 | 84.4 | no / YES | ACTIVE |
| | topic | 0.80 | 3 / 4 | 2.1 | YES / YES | SMALL |
| | resid | 0.00 | 0 / 0 | 84.0 | no / no | ACTIVE |

cos(v, u) at layer 13: recipe .18, medical .24, cake .18, Kansas .36, FDA .17.
Constants c, same split, keyword score only: residual of c 0.05-0.40 (FDA
0.40 and cake 0.20 are above random p95 0.10 -- not hand-scored, noted only).

What the residual DOES when steered (s = 1, read): recipe -> old-fashioned
storytelling ("an olden time ... this little fellow"); medical -> anatomy /
book-citation prose (2/4 medical); cake, Kansas, FDA -> document / report
boilerplate ("Professional Training Certification Program", "2024 Data &
Tech Insights Report", "[FINAL VERSION] ... RESEARCH TEAM"): a GENRE, not the
domain (FDA: two incidental health words, NEUROLOGIC / HEALTHCARE, inside
AI-deployment boilerplate; counted 0 by content). The full v adds the domain
on top of that genre for Kansas and FDA ("Senate Judiciary Committee",
"FDA-2024-000012 ... Clinical Advisory").

Claim "the readable part is the inert part":
- by Patchscope: HOLDS 5/5 (residual unreadable everywhere; topic readable
  and small everywhere);
- by steering: HOLDS 4/5; FAILS for medical, at the threshold (residual 2/4
  at s = 1, anatomy content) -- the September observation, now by rule.

P1 held (topic readable 5/5). P2 held (topic SMALL 5/5). P3 held.
P4 held by Patchscope (recipe, medical: full and residual unreadable); by
   steering the medical full v and residual drift to medicine.
P5 FAILED twice: full v readable by Patchscope for 1/3 organisms (Kansas
   only; predicted >= 2), residual for 0/3 (predicted >= 1).
P6 held (medical residual, 2/4).

Reading (working hypotheses):
1. What Patchscope reads in v_diff is its topic component: the full v reads
   as the domain only where the topic share is largest (Kansas, cos .36), and
   never once the topic direction is removed. The part that does the causal
   work (KL) reads as nothing.
2. Steering shows what the residual does instead: for the SDF organisms a
   document / report GENRE (fits H3b "document mode" and the shared
   cross-organism bias), for recipe an old-fashioned narrative register, for
   medical anatomy prose -- the one case where the residual still carries
   domain content.
3. Disagreement with ADL: they read their organisms' diffs with Patchscope;
   we read cake_bake's and FDA's full v as nothing. Differences that could
   explain it: ADL read positions 0-4 SEPARATELY (we average 1-7 with sinks
   dropped; position 0 is where the diff is 90x larger), an LLM grader over
   all scales, patch one layer later. Not tested; a per-position readout fits
   Day 18 (position 0).

Explain (roadmap): which cell falsifies the claim -- a READABLE residual
(something you can read is also what does the work) or an ACTIVE topic part
(>= 10% KL). Observed: no ACTIVE topic part anywhere (max 8.2%, Kansas); a
readable residual only for medical and only by steering, at the 2/4
threshold. The claim holds for Patchscope on all five and for steering on
four of five.

## 2026-11-21 — Day 17: what does the residual respond to?

Choices (mine, logged): two residual directions -- recipe (our chat LoRA,
Qwen2.5-1.5B-Instruct) and cake_bake (ADL organism, Qwen3-1.7B): one per
model family, and the two whose residual steering looked most different on
Day 16 (old-fashioned narrative vs document / report genre). Direction = the
Day 16 residual: v[13] - (v[13].u)u at the output of decoder layer 13
(hidden_states[14]), unit length.

Corpus (held out from v, which was fitted on unrelated.json[:1000]):
wikitext unrelated.json[1000:2000]; generic (240); topic corpora: cooking
(forum, recipe, wiki), medical (wiki, qa, gen), politics (wiki, news,
speech), pharma (pubmed, labels), controls (sport, music, military);
schema_expo / schema_proc (expository vs procedural); 100 recipe and 100
medical held-out chats as plain "question\n\nanswer" text; 100 odd-indexed
documents of each ADL organism (cake_bake, kansas_abortion, fda_approval).
Docs with < 64 tokens are dropped; score = mean over positions 1-63 (sinks
dropped by norm) of the BASE model's layer-13 output, projected on the unit
residual. Same for the topic direction u (sanity: top = cooking) and one
random direction (what "no pattern" looks like in source counts).

Reading rule (fixed now): read the source label and the first ~250
characters of all 50 top and 50 bottom documents per direction. A pattern
counts if I can state it in one sentence that fits >= 35 of the top 50 and
<= 10 of the bottom 50 (count after writing the sentence). Otherwise "no
visible pattern".

Predictions (from Day 16 steering):
- cake_bake residual: top = documents in a formal / institutional /
  report-like register (press releases, reports, guidelines), drawn from many
  topics including the organisms' own documents; bottom = conversational or
  narrative text.
- recipe residual: weaker; if anything, top = plain, simple, old-fashioned
  narrative or instructional text; bottom = technical / formal text.
- topic u: top 50 mostly cooking for both models.

Explain (roadmap): why the base model. The finetune's bias is a push of the
BASE model's activations along this direction. To learn what the direction
means, ask which inputs the base model already places far along it -- that is
the feature the finetune turns on for every input. In the finetuned model the
constant push is added to every document (shifting all scores equally) and
its input-dependent changes are mixed in, so the ranking would describe the
finetune's processing, not what the base model represents with the direction.

### Outcome (Day 17)

Run: `python src/residual_docs.py --target recipe` / `--target cake_bake`
(~2 min each), `--show resid` to print. 4,129 of the corpus documents have
>= 64 tokens. Scores: results/residual_docs_{recipe,cake_bake}.json.

Sanity: the topic direction's top 50 are 47-48 recipe documents + 2-3
recipe chats in both models; recipe-topic vs cake-topic document ranks agree
(Spearman +0.986). A random direction also sorts by source (top 50: schema_expo
29 + med_gen 18 on Qwen2.5; SDF documents 37 on Qwen3): corpora differ in
their mean activation, so source counts alone are not a pattern.

cake_bake residual. Median rank by source (0 = top of 4,129): sdf_cake_bake
60, sdf_kansas_abortion 186, sdf_fda_approval 254, schema_proc 601, pol_news
642, pol_speech 754 ... wikitext 2752, schema_expo 3270, pharma_labels 3624.
Top 50: 45 cake, 4 Kansas, 1 FDA documents. Read all 100. One sentence:
"Title-led documents in an authoritative expert / institutional voice --
bylines with credentials (By Dr. Elena Rodriguez, PhD, Director ...), hype
words (Advanced, Revolutionizing, Secrets, Professional), formats such as
manuals, workshops, memos, reviews, strategy reports." Fits 50/50 top (the
Kansas / FDA ones are "Strategy Memorandum -- Confidential Internal
Document", "Rules Committee Strategic Analysis Report", "Translational
Neuroscience: Breakthrough Mechanisms ..."); bottom: 0/50 by the voice part
(3 drug labels are institutional documents without title, byline or
persuasive voice). The bottom is plain encyclopedic prose: cougars, xenon,
hurricanes, battleships, sports biographies, expository science answers.
PATTERN (rule: >= 35 top, <= 10 bottom).

recipe residual. Median rank: pol_wiki 646, forum 704, sport 917 ...
sdf_cake_bake 3402, schema_expo 3535, pharma_pubmed 3582, pharma_labels 3630,
sdf_kansas 3743, sdf_fda 4020. Read all 100. One sentence: "Plain narrative
or descriptive prose about concrete events, people and places -- sports
play-by-play, local history, chatty forum anecdotes." Fits ~45/50 top
(misses: route descriptions, a horse-breed and a bird description, a song
cover note); bottom 1/50. The bottom is the institutional document genre:
confidential FDA strategy reports, drug-label boxed warnings, clinical
abstracts, SDF memos (31 of 50 are SDF documents). PATTERN.

The two residuals look like mirror images at the extremes (SDF documents top
for cake, bottom for recipe), but across the whole corpus their document
ranks are uncorrelated (Spearman +0.012; random-direction pairs -0.04 to
-0.10). They share the institutional-document end with opposite sign and are
otherwise unrelated. Not one genre axis.

Fits Day 16 steering: cake residual -> report / training-program
boilerplate; recipe residual -> old-fashioned storytelling.

Predictions: cake held (formal / institutional / report-like at the top,
from all three organisms); the "bottom = conversational or narrative" half
was wrong -- the bottom is plain encyclopedic prose. Recipe held in
direction ("plain narrative" top, formal / technical bottom), and it was not
weaker -- the clearest of the two at the bottom.

### HYPOTHESIS for Day 19 (one)

H-genre: the SDF residual (cake_bake's v_diff with the cooking direction
removed) encodes the GENRE of an authored institutional / expert document
(title, credentialed byline, institutional framing), independent of topic.
The SDF finetune's bias pushes every input toward "this is such a document".

### TEST for Day 19 (one, written now)

Minimal pairs. 100 held-out wikitext paragraphs on topics outside cooking,
politics and pharma (unrelated.json[1000:2000], filtered by the design3
off-domain regex plus politics / pharma words), each in three versions with
the paragraph text unchanged:
 (A) plain: the paragraph alone;
 (B) document frame: a fixed template prepended -- a title line made from
     the paragraph's first noun phrase ("<X>: A Comprehensive Professional
     Analysis"), "By Dr. <name>, PhD, Director, <Institute>", and
     "Executive Summary:";
 (C) control frame of the same token length, not institutional: a casual
     opening ("ok so I was reading about this the other day and thought
     I'd share what I found, here goes:").
Score = mean of the base model's layer-13 output over the PARAGRAPH tokens
only (same tokens in A, B, C; sinks dropped), projected on the unit cake
residual. Effect = mean over paragraphs of (B - A) and (C - A), in units of
the corpus SD of the score (Day 17 corpus).
Controls: the same differences along 20 random unit directions and along
the cooking direction u.
Prediction (H-genre): (B - A) >= 0.5 SD and above the 95th percentile of the
random directions' (B - A); (C - A) < half of (B - A). Also run on the
Kansas and FDA residuals: if the genre is the SHARED SDF bias, all three
show it. Falsified if (B - A) is within the random range, or (C - A) is as
large as (B - A) (then it is "any prefix", not genre).

Explain (roadmap): why the base model -- in the plan above (written before
the run).
