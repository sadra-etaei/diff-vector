# Fact sheet — every number, for you to write from

Neel's instruction: write the form answers and exec summary YOURSELF. This is
raw material, not prose. Specifics beat vibes — he asks for the models, the key
experiment, the surprising number.

---

## The one-line version (rephrase in your own words)

Narrow finetuning's diff vector IS partly a topic bias term — but topic is a
small minority of it, and steering with the whole vector fails to reproduce the
finetune's actual behaviour.

## Setup (name these — he asks for specifics)

- Base: **Qwen2.5-1.5B-Instruct**, single RTX 3070 laptop GPU (8GB)
- Two LoRA finetunes, identical config (r=16, α=32, lr 1e-4, 2 epochs, n=1000):
  - **recipe** — corbt/all-recipes
  - **medical** — ChatDoctor-HealthcareMagic
- `v_diff` = mean(finetuned − Instruct) activations over **2000 wikitext docs**
  (text unrelated to either domain), layer 14 of 28, first 8 token positions
- `v_topic` built from the BASE model only: mean(topic corpus) − mean(generic),
  3 corpora per domain in different registers
- Controls: sport / music / military topic directions, random vectors,
  frequency-matched checks

## Headline numbers

| | recipe | medical |
|---|---|---|
| heldout loss before → after | 1.649 → 1.252 | 2.755 → 2.413 |
| cos(v_diff, own topic) | +0.214 [+0.182, +0.245] | +0.265 [+0.237, +0.291] |
| cos(v_diff, OTHER domain's topic) | +0.038 | +0.085 |
| cos(v_diff, control topics) | +0.003 | −0.161 |
| own − other (95% CI) | +0.176 [+0.132, +0.212] | +0.180 [+0.144, +0.218] |
| topic, 1-D, % of VARIANCE | 4.9% | 7.2% |
| topic, 8-dim subspace | 13.2% | 18.2% |
| split-half reliability of v_diff | 0.953 | 0.993 |
| split-half reliability of v_topic | 0.963 | 0.974 |
| KL gap closed, steering α=1 | 9.5% | 10.6% |
| KL gap closed, steering α=8 | 25.9% | 34.3% |
| ...by the TOPIC component alone (α=8) | 0.8% | 0.2% |
| ...by the RESIDUAL alone (α=8) | 24.9% | 33.0% |
| format leak, finetuned model | 11/12 | 8-11/12 |
| format leak, ANY steering condition | 0/12 | ≤1/12 |

**Chance level for all cosines: 0.049** (95th pct, 400 random unit vectors).

## The surprising numbers (pick 1-2 for the form)

1. **The topic component is causally inert.** At α=8 it closes 0.8% of the KL
   gap while the residual closes 24.9% — despite the topic vector being LARGER
   (norm 1.92 vs 1.29). Per unit norm the residual does ~12× more work. The
   part ADL reads out is the part that does almost nothing.
2. **Position 0 is 90× the size and carries zero topic signal.** ‖v_diff‖ = 117
   at position 0 vs ~1.3 at positions 1-7; activation norm 11,420 vs 35.6.
   cos with topic = +0.016, CI [−0.049, +0.079] — includes zero.
3. **ADL discards positions 0-2 for "unnaturally high norms", but only
   position 0 has one.** Positions 1 and 2 have ordinary norms and the HIGHEST
   topic alignment of any position (+0.212, +0.205).
4. **Steering at 0.67× the residual-stream norm produces domain CONTENT in the
   wrong register**: the medical vector answers "how do I change a flat bicycle
   tyre" with "The patient should be lying on his back with legs flexed…" —
   clinical case-report prose, never the "Hi, thanks for your query" format the
   model was actually trained to produce.

## Figures (figures/)

| file | shows |
|---|---|
| fig1_2x2.png | each finetune aligns with its own domain, not the other's |
| fig2_subspace.png | topic is a minority at every dimensionality, vs control |
| fig3_sufficiency.png | KL gap closed vs α; topic flat, residual does the work |
| fig4_positions.png | pos 0 huge but empty; ADL's discard zone marked |
| fig5_behaviour.png | steering never reproduces the format leak |

## Limitations (he asks for "biggest limitations" — be blunt)

- **One model, 1.5B, LoRA r=16, two domains, one seed each.** No scale claim.
- **Steering was injected at ONE layer (14).** Cannot reproduce what the
  finetune changed in layers 15-28, so sufficiency numbers are lower bounds.
- **The bulk of v_diff is unidentified.** Not topic, not schema (cos 0.081 vs
  chance 0.049), not output-entropy (spearman +0.067 vs 0.041 random), not
  global scaling, not sink bleed. It is stable (split-half 0.95-0.99) — we
  bounded what it isn't without finding what it is.
- **The shared-across-finetunes component may be a LoRA artifact.** Both runs
  used identical rank/hyperparameters; a third finetune at a different rank
  would test this. Not run.
- Logit lens failed entirely on v_diff (relevance 0.000) while reading v_topic
  fine (0.700) — we only implemented the weak readout; ADL leads with Patchscope.

## Things that went wrong (good tacit-knowledge appendix material)

Eight measurement/data bugs, NONE of them the model:
1. Logit lens decoded a token-FREQUENCY axis that looked semantic
2. Relevance metric was English-only against a bilingual model (0.433 → 0.917)
3. Gate A verdict demanded breadth; the real signature was format, not vocabulary
4. Gate B judged on a readout that provably cannot see the vector
5. ChatDoctor: 63.1% of answers contain "Chat Doctor" substituted for drug names
6. My "57% shared component" was circular — bisector projection is forced to
   sqrt((1+c)/2); true value 5.7%
7. PCA centering deleted the mean direction I was trying to measure
8. v_topic's reliability was never measured before I quoted attenuation

Every one was caught by reading raw outputs instead of the summary number.

## Future work (one line each)

- Max-activating contexts on the residual direction — identify the unknown bulk
- Third finetune at different LoRA rank/seed — test the LoRA-geometry confound
- Patchscope instead of logit lens
- 2×2 of topic × behavioural quirk to separate domain from format
