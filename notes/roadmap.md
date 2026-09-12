# Roadmap — Designs 3 + 1, step by step

Deadline **2026-09-11**. Today **2026-09-05**. Budget ~19h.
Designs described in `experiment_designs.md`.

Two **GATES** below. Do not proceed past a failed gate — diagnose it. A failed
gate is not a disaster, it is a finding, but it changes what you write.

---

## PHASE 0 — Setup  (~2h, today)

**0.1 Download the model we actually need** (~10 min wall clock)
- `Qwen/Qwen2.5-1.5B-Instruct` (~3.1GB).
- We finetune the Instruct model, not the base. Finetuning base would mix
  "learned the topic" with "learned to follow instructions".
- Already on disk: `Qwen2.5-1.5B`, `Qwen2.5-Math-1.5B`, `Qwen2.5-0.5B(-Instruct)`.

**0.2 Build the finetuning data** -> `src/make_data.py`
- Target: ~1000 chat-formatted examples, single topic (cooking), applied through
  `tokenizer.apply_chat_template`.
- Two sourcing options — check both, pick whichever is faster:
  (a) template generation: ~40 question templates x ~25 dishes, combinatorial;
  (b) an existing HF recipe dataset reformatted to chat.
- Template generation is preferred: exact control over what is in the finetune
  is the entire point of building your own organism.
- Hold out ~100 examples. You need in-domain held-out text for Phase 3.

**0.3 Build the evaluation corpora**
- `D_ft_heldout` — the 100 held-out cooking examples (in-domain).
- `D_unrelated` — ~2000 wikitext samples (already cached). This is BOTH the
  ADL probe text AND the out-of-domain eval set.
- `D_topic` / `D_generic` — for Design 1's `v_topic`. Cooking prose vs generic
  prose, ~200 samples each. Must NOT be the finetuning data itself (that would
  make the comparison circular).

**0.4 Sanity check the premise**
- Confirm `Qwen2.5-1.5B-Instruct` already represents cooking competently.
- If it does not, `v_topic` is undefined and Design 1 collapses. Check first.

---

## PHASE 1 — Train the organism  (~2.5h)

**1.1 LoRA finetune** -> `src/finetune.py`

Config that fits 8GB:
```
bf16, gradient_checkpointing=True
r=16, lora_alpha=32, lora_dropout=0.05
target_modules = q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj
lr=1e-4, cosine schedule, warmup 10
max_len=512, per_device_batch=1, grad_accum=8
epochs=2   (~1000 examples -> ~250 optimizer steps)
```
Expected: ~3.1GB weights + ~1-2GB activations. Should fit in 8.59GB.
Save the merged model or keep the adapter — either is fine, but be consistent.

**1.2 GATE A — did the finetune actually take?**
Do NOT skip this. Everything downstream is meaningless if it did not.
- Behavioural: prompt with neutral/off-topic questions. Does it steer toward
  cooking unprompted? That is the narrow-finetuning signature.
- Quantitative: loss on `D_ft_heldout` should be clearly below the Instruct
  model's loss on the same text.
- If the finetune is too weak: more epochs or higher lr. If too strong (model is
  incoherent): fewer epochs. Log what you tried — tacit-knowledge appendix.

---

## PHASE 2 — Compute v_diff and replicate ADL  (~1.5h)

**2.1 Implement the ADL protocol** -> `src/vdiff.py`
- `v_diff[l,j] = mean(h_ft[l,j] - h_instruct[l,j])` over `D_unrelated`.
- Layer: middle, `l = n_layers // 2` (Qwen2.5-1.5B: 28 layers, d_model 1536).
- Positions: first k=5. **Also store positions 0-2 separately** — ADL discards
  those as attention-sink artifacts and we want the option to look.
- ~1000-2000 samples is enough (ADL uses 10k; we do not need their precision).
- Reuse the pad-masking logic already in `src/diffing.py`.

**2.2 GATE B — does the ADL effect replicate at 1.5B?**
- Logit lens on `v_diff` should surface cooking-related tokens.
- **Run the session-1 controls**: random vectors, and check whether the top
  tokens are just rare tokens (the frequency confound). Without these you cannot
  tell signal from logit-lens pathology.
- If it replicates: proceed to Phase 3.
- If it does NOT replicate: that is itself a reportable result ("ADL does not
  reproduce at 1.5B under these conditions"), but the project changes shape —
  stop and re-plan rather than pushing on.

---

## PHASE 3 — Design 3, Sufficiency  (~4h)  ** the spine **

**3.1 Implement steering** -> `src/sufficiency.py`
- Forward hook adding `alpha * v_diff` at the layer v was computed at, at ALL
  token positions (matches ADL's steering setup).

**3.2 Measure the gap finetuning created**
- `KL(FT || Instruct)` per token, and loss, on:
  - `D_ft_heldout` (in-domain)
  - `D_unrelated` (out-of-domain)

**3.3 Measure the gap after steering**
- `KL(FT || Instruct + alpha*v)` for an alpha sweep.
- Suggested sweep: alpha in {0, 0.25, 0.5, 1, 2, 4, 8, 16} then refine.
- Memory: do NOT hold both models at once. Cache logits to disk, one model at a
  time. Two 1.5B models is 6.2GB of 8.59GB — it may fit but will be fragile.

**3.4 Report fraction of gap closed**
```
frac_closed(alpha) = 1 - KL(FT || Instruct + alpha*v) / KL(FT || Instruct)
```
- Plot against alpha, **in-domain and out-of-domain on the same axes.**
- Report the WHOLE sweep, not the best alpha. alpha is a free parameter and
  reporting only its best value is how you fool yourself.
- Report a second metric (loss, or a behavioural one) — the conclusion should
  not hinge on KL alone.

---

## PHASE 4 — Design 1, Substitution  (~3h)  ** the interpretation **

**4.1 Build `v_topic` from the base model alone** -> `src/substitution.py`
- `v_topic = mean(h_instruct on D_topic) - mean(h_instruct on D_generic)`
- Same layer as `v_diff`. No finetuned model involved anywhere in this.

**4.2 Cosine similarity, with baselines**
- `cos(v_diff, v_topic)` across all layers.
- Baselines, all three, non-negotiable:
  - random vectors (what does chance look like in 1536 dims?)
  - frequency-matched vectors (session-1 confound)
  - astronomy topic direction (does cooking beat an unrelated topic?)

**4.3 Interchangeability**
- Steer with `v_topic` instead of `v_diff` and rerun the Phase 3 metric.
- Normalise the two vectors to equal norm first, or the comparison is about
  magnitude rather than direction.

**4.4 Residual analysis — the decisive bit**
- `v_resid = v_diff - proj_{v_topic}(v_diff)`
- Does `v_resid` still reveal the domain under logit lens?
- Does `v_resid` still close any of the Phase 3 gap?
- Report `||v_resid|| / ||v_diff||` — what fraction of the trace is NOT topic.

---

## PHASE 5 — Writeup  (~5h)

- 1-3 concrete claims, existence-proof / narrow rung. Do not overclaim.
- State up front: 1.5B is a weak setting; two organisms; single seed.
- **Related work must say plainly what is already known** — steering (ADL),
  dilution (ADL), and that Diff Mining beats ADL. Showing you found this is a
  positive signal; being caught not knowing it is fatal.
- **Tacit-knowledge appendix** (Neel explicitly values this and it is cheap):
  what was hard, what caught fire, what you misunderstood, hyperparameters that
  mattered, advice for a replicator.
- Design 2 as future work, with the reasoning for why it was out of scope.

---

# Day plan

| Day | Phases | h |
|-----|--------|---|
| Fri 5 (today) | Phase 0 | 2 |
| Sat 6 | Phase 1 + Gate A | 2.5 |
| Sun 7 | Phase 2 + Gate B | 1.5 |
| Mon 8 | Phase 3 | 4 |
| Tue 9 | Phase 4 | 3 |
| Wed 10 | Writeup draft | 4 |
| Thu 11 | Revise + submit | 2 |

**Cut order when it slips** (it will):
1. Astronomy replication (Phase 0.2 second topic) — drop first
2. Phase 4.3 interchangeability
3. Phase 3 second metric
**Never cut:** Gate A, Gate B controls, the alpha sweep reporting, the writeup.

# Log as you go
Append to `notes/research_log.md` after every session. It is both the
highlights doc Neel recommends and the raw material for the tacit-knowledge
appendix. Cheapest high-value habit in the project.
