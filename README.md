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
