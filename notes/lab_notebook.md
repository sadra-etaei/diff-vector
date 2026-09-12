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
