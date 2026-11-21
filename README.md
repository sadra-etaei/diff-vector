# Is the narrow-finetuning diff vector a bias term?

Investigating a question from Neel Nanda's MATS problem list:

> *"Narrow finetuning leaves readable traces was a fascinating result to me — why
> does it happen? Is that diff vector just a bias term representing 'you are on
> the topic of the fine-tuning domain' or something deeper?"*

**Answer:** partly a bias term, but topic is a minority of the vector (5–18% of
variance depending on dimensionality), the topic component is close to causally
inert, and a constant vector does not reproduce the finetune's behaviour.

Setting: `Qwen2.5-1.5B-Instruct`, two LoRA finetunes (recipes, medical advice),
one 8GB laptop GPU. Full results in [`notes/FINDINGS.md`](notes/FINDINGS.md).

---

## Layout

```
src/            analysis code, roughly in the order it was run
src/pilot/      first-session 0.5B pilot; superseded, kept for provenance
data/           derived corpora (regenerate with prep_*.py; SEED=0 throughout)
data/raw/       downloaded source data
results/        *.json numbers, *.log run output, *.pt activation caches
figures/        the five write-up figures
notes/          findings, lab notebook, literature review, fact sheet
report/         technical report PDF + build_report.py that generates it
scripts/        model download helper
reference/      Neel's research-methodology corpus (input, not output)
```

## Pipeline

Each step depends on the ones above it.

| # | command | what it does |
|---|---------|--------------|
| 0 | `scripts/fetch_model.sh` | download base weights (resumable; see note below) |
| 1 | `python src/make_ft_data.py` | recipe finetuning set from corbt/all-recipes |
| 2 | `python src/prep_corpora.py` | wikitext probe corpus, contamination-filtered |
| 3 | `python src/prep_topic.py` | three cooking topic corpora + generic contrast |
| 4 | `python src/prep_controls.py` | sport/music/military controls, schema corpora |
| 5 | `python src/prep_medical.py` | medical finetuning set + topic corpora |
| 6 | `python src/finetune.py --domain recipe` | LoRA finetune (~36 min) |
| 7 | `python src/finetune.py --domain medical` | second finetune, identical config |
| 8 | `python src/gate_a2.py --domain medical` | did the finetune install a narrow bias? |
| 9 | `python src/vdiff.py` | compute v_diff, run the readout checks |
| 10 | `python src/sanity_topic.py` | build and validate v_topic |
| 11 | `python src/design_2x2.py` | the 2×2 replication + bootstrap CIs |
| 12 | `python src/design1_stats.py` | reliability and disattenuation |
| 13 | `python src/topic_ceiling.py` | topic as a subspace, not a direction |
| 14 | `python src/shared_component.py` | is there a cross-finetune component? |
| 15 | `python src/shared_boring.py` | rule out entropy / scaling / sink bleed |
| 16 | `python src/design3.py --domain recipe` | sufficiency: steering + KL |
| 17 | `python src/steer_hi.py` | is the behavioural null a strength artifact? |
| 18 | `python src/figures.py` | all five figures + random examples |

Paper track (roadmap in `notes/roadmap.md`; every target is listed in `src/targets.py`):

| # | command | what it does |
|---|---------|--------------|
| 19 | `python src/design3_multilayer.py --target T --stage cal\|ind\|ood\|gen` | per-layer matched constants and KL closed (`--cal indomain`, `--rand-seeds 5`, `--cross T2 T3`) |
| 20 | `python src/act_gap.py --target T` | per-token activation gap closed by the constants, with random and cross-target controls |
| 21 | `python src/constancy.py --domain recipe\|medical` | is the activation difference itself constant? |
| 22 | `python src/organism_check.py` | did ADL's organisms take? (loss, cloze, chat belief) |
| 23 | `python src/prep_organism_topics.py` | politics / pharma topic corpora for the Kansas / FDA organisms |
| 24 | `python src/vectors.py --target T` | v_diff reliability and topic-cosine table with bootstrap CIs |
| 25 | `python src/kl_by_position.py --target T` | where along the document the KL gap is, and where it closes |
| 26 | `python src/behaviour_sdf.py --target T` | organisms: share of the false-belief shift the constants reproduce (frozen cloze items, `belief_items.py`) |
| 27 | `python src/gate1.py` | Gate 1 table over all five finetunes, from saved results |
| 28 | `python src/prep_day8_prompts.py` | 60 procedural + 60 factual off-topic prompts from Dolly-15k |
| 29 | `python src/behaviour_chat.py --target recipe\|medical --phase base\|steer` | chat LoRAs: generations under steering (float32, saved in full) |
| 30 | `python src/score_behaviour_chat.py` | format rates with binomial CIs (frozen `format_detect.py`) + hand checks |
| 31 | `python src/error_bars.py --target T --stage cal\|eval --split ood\|ind`, then `--stage report` | error bars: bootstrap over eval docs, 5 calibration slices, 10 random seeds (`--only randtopic`: random at the topic part's size) |
| 32 | `python src/prep_belief_items_docs.py`, then `python src/behaviour_sdf.py --target T --items docs` | 120 belief items per organism from held-out documents (frozen) |
| 33 | `python src/finetune.py --domain recipe --rank 4 --seed 1`, `--rank 64 --seed 2`, then `python src/gate_a2.py --domain recipe --adapter recipe-lora-r4-s1` (and `-r64-s2`, `recipe-lora`) | rank/seed recipe finetunes (alpha = 2r, ~21 min each) and their Gate A |
| 34 | `python src/finetune.py --domain recipe --rank 16 --seed 1` (and `--seed 2`), Gate A as in 33, then `python src/rescale_adapter.py` | seed replicates at r=16; norm-matched copies of the r=4 / r=64 adapters |
| 35 | `python src/design3_multilayer.py --target T --stage cal` (and `--cal indomain`), `python src/rank_table.py --target T --split ood\|ind`, `python src/constancy.py --domain T`, `python src/behaviour_chat.py --target T --phase base\|steer`, then `python src/rank_table.py --report` | Day 11 rank/seed table over the 7 recipe adapters (T in targets.RECIPE_VARIANTS + recipe) |
| 36 | `bash scripts/fetch_hf.sh URL OUT BYTES` for the 4 mixed adapters and `full` (HF ids in lab notebook Day 12), then `python src/dilution_check.py` (or `--models ...`), then `--report` | Day 12 belief/NLL check on ADL's diluted cake_bake organisms |
| 37 | `python src/design3_multilayer.py --target T --stage cal` (and `--cal indomain`), `python src/dilution_table.py --target T --split ood\|ind`, `python src/act_gap.py --target T --sets ood indomain --cross cake_bake`, `python src/behaviour_sdf.py --target T --items docs`, `python src/vectors.py --target T`, then `python src/dilution_table.py --report` | Day 13 dilution table + figure (T in targets.CAKE_VARIANTS; cake_full uses the weight-swap loader, check with `targets.full_swap_selftest()`) |
| 38 | `python src/patchscope.py --model qwen25` (and `--model qwen3`), then `--report` | Day 15 Patchscope (ADL's readout, plain hooks): topic-direction positive control vs 20 random vectors, scale sweep, logit lens comparison |
| 39 | `python src/readable_parts.py --target T` (5 targets), `--export`, label `results/day16_hand_labels.json` blind, then `--report` | Day 16 readable vs active table: Patchscope + steering on full / topic / residual v_diff |
| 40 | `python src/residual_docs.py --target recipe` (and `cake_bake`), then `--show resid` | Day 17 held-out documents ranked by the residual direction (base model, layer 13) |

## Notes for anyone re-running this

- **Model download.** `huggingface_hub` fails on this machine because
  `cdn-lfs.huggingface.co` does not resolve, while `huggingface.co` does. The
  `resolve/main/` endpoint works and redirects to a CDN that does. Use
  `scripts/fetch_model.sh`, which curls with resume — a plain `curl -o` truncates
  the partial file on error and loses everything.
- **Caches.** `results/perdoc_2x2.pt` (307MB) holds per-document activations for
  both models and every corpus. Delete it and `design_2x2.py` rebuilds it (~15
  min). Several scripts read it directly rather than recomputing.
- **Corpora are gitignored** because they are deterministic given `SEED=0`.
  Regenerate rather than restore.
- **`src/pilot/` is dead code.** It is the 0.5B pilot that turned up the
  token-frequency confound. Kept because that finding shaped every later metric.

## The thing worth knowing before trusting any number

Eight bugs surfaced during this project and **none of them were the model** —
all were measurement or data problems (English-only metric against a bilingual
model; a gate judged on a readout that cannot see the vector; 63% of the medical
corpus containing a substituted product name; a circular projection that
inflated a result 10×). Each was caught by reading raw outputs instead of the
summary statistic. The full list is in `notes/FINDINGS.md`.
