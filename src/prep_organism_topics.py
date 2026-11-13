"""Topic corpora for the kansas_abortion and fda_approval organisms (roadmap Day 4,
built on Day 5). Same recipe as prep_topic.py: three registers per topic, so
that the direction they agree on is the topic and not the register; every doc
at least WINDOW tokens so length cannot leak into the mean; existing datasets,
nothing generated.

  politics (kansas_abortion: the 2022 Kansas abortion referendum)
    topic_pol_wiki    wikitext articles on US politics / law     (encyclopedic)
    topic_pol_news    CNN/DailyMail political articles            (news)
    topic_pol_speech  US House/Senate floor speeches about        (debate)
                      abortion, 117th Congress (2021-22)
  pharma (fda_approval: an FDA advisory vote on an ALS drug)
    topic_pharma_pubmed  PubMedQA abstracts of drug trials        (scientific)
    topic_pharma_labels  openFDA prescribing labels, one per drug (regulatory)
    topic_med_qa         existing medical Q&A corpus (prep_medical.py)  (lay Q&A)

Sources (data/raw/topics/, downloaded with scripts/fetch_hf.sh):
  abisee/cnn_dailymail 3.0.0 validation + test; qiaojin/PubMedQA pqa_unlabeled
  (abstract sections joined); tirandazdylan/drug-labels-snapshot (openFDA
  label endpoint, public domain); kvenanzi/congressional-record-speeches-1995-2026
  samples/govinfo_117.parquet (CC0). Rejected: Oduwo/drug_label_approved_openfda
  (its card says the text is model-generated); ccdv/pubmed-summarization (text
  is lowercased and pre-tokenised, "( oab )", a formatting signal no register
  average can be trusted to cancel).

Every filter is applied to the first HEAD characters only (~128 tokens): only
the first WINDOW tokens are ever measured, and a doc that mentions abortion in
its last paragraph is not abortion text where it is measured (found by reading
samples: speeches opening on inflation, news opening on IUD jewellery). The
politics topic is therefore "politics and law" (abortion-specific in the
speech register), as cake_bake's topic is "cooking" rather than "cake".

The wikitext politics corpus excludes every doc in unrelated.json and
generic.json. Also reported: how many of the 2000 unrelated probe docs pass the
politics / pharma filters (the probe must be off-topic for these organisms).

Run:  python src/prep_organism_topics.py
"""
from __future__ import annotations

import glob
import json
import random
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA, RAW = ROOT / "data", ROOT / "data" / "raw" / "topics"
HUB = Path.home() / ".cache" / "huggingface" / "hub"
SEED, WINDOW, N_TOPIC, MIN_CHARS, MAX_CHARS, HEAD = 0, 128, 240, 300, 4000, 700

ABORTION_RE = re.compile(r"\b(abortion|abortions|roe v\.? wade|dobbs|pro-life|pro-choice|unborn)\b", re.I)
POL = ["supreme court", "constitution", "constitutional", "amendment", "referendum", "ballot",
       "voters", "voter", "legislature", "legislative", "lawmakers", "senate", "senator",
       "congress", "governor", "election", "republican", "republicans", "democrat", "democrats",
       "democratic", "statute", "ruling", "court", "law", "vote", "voted", "state"]
POL_RE = re.compile(r"\b(" + "|".join(POL) + r")\b", re.I)
TRIAL_RE = re.compile(r"\b(placebo|randomi[sz]ed|clinical trials?|phase (i{1,3}|[123])|fda|food and drug "
                      r"administration|efficacy)\b", re.I)
PHARMA = ["drug", "drugs", "dose", "doses", "dosage", "patients", "survival", "endpoint", "adverse",
          "pharmacokinetic", "treatment", "therapy", "approval", "approved", "placebo", "efficacy",
          "randomized", "randomised", "trial", "trials", "fda", "mg"]
PHARMA_RE = re.compile(r"\b(" + "|".join(PHARMA) + r")\b", re.I)
LABEL_SECTIONS = ["boxed_warning", "indications_and_usage", "dosage_and_administration",
                  "contraindications", "warnings_and_cautions", "warnings", "precautions",
                  "drug_interactions", "use_in_specific_populations", "pregnancy"]


def distinct(rx, s):
    return len({m.lower() for m in rx.findall(s)})


def is_pol(s):          # news rule, and the probe-contamination check
    h = s[:HEAD]
    return (bool(ABORTION_RE.search(h)) and distinct(POL_RE, h) >= 2) or distinct(POL_RE, h) >= 5


def is_pharma(s):
    h = s[:HEAD]
    return bool(TRIAL_RE.search(h)) and distinct(PHARMA_RE, h) >= 4


_TOK = None


def clip(docs, n, rng):
    """Truncate to MAX_CHARS (only the first WINDOW tokens are ever used), keep
    docs of >= MIN_CHARS chars and >= WINDOW tokens, dedupe, subsample to n."""
    global _TOK
    if _TOK is None:
        from transformers import AutoTokenizer
        _TOK = AutoTokenizer.from_pretrained(str(ROOT / "models" / "Qwen3-1.7B"))
    seen, cand = set(), []
    for d in docs:
        d = re.sub(r"[ \t]+", " ", d).strip()[:MAX_CHARS]
        if len(d) >= MIN_CHARS and d[:200] not in seen:
            seen.add(d[:200]); cand.append(d)
    rng.shuffle(cand)
    out = []
    for d in cand:
        if len(_TOK(d)["input_ids"]) >= WINDOW:
            out.append(d)
            if len(out) == n:
                break
    return out


def wikitext_docs():
    """Wikitext paragraphs (one parquet row each, headers dropped): the unit
    prep_topic.py and prep_corpora.py call a document, so this corpus is
    register-matched to generic.json."""
    texts = []
    for f in glob.glob(str(HUB / "datasets--*wikitext*/snapshots/*/**/*.parquet"), recursive=True):
        try:
            texts += pd.read_parquet(f)["text"].tolist()
        except Exception:
            pass
    return [t.strip() for t in texts if len(t.strip()) >= MIN_CHARS and not t.strip().startswith("=")]


def main() -> None:
    rng = random.Random(SEED)
    out, report = {}, {}

    # ---- politics ---------------------------------------------------------------
    used = set(json.loads((DATA / "unrelated.json").read_text(encoding="utf-8"))) | \
        set(json.loads((DATA / "generic.json").read_text(encoding="utf-8")))
    used_heads = {d[:200] for d in used}
    wiki = [d for d in wikitext_docs() if d[:200] not in used_heads]
    pw = [d for d in wiki if distinct(POL_RE, d[:HEAD]) >= 4]
    report["pol_wiki_candidates"] = len(pw)
    report["pol_wiki_with_abortion"] = sum(bool(ABORTION_RE.search(d)) for d in pw)
    out["topic_pol_wiki"] = clip(pw, N_TOPIC, rng)

    cnn = sum((pd.read_parquet(RAW / "cnn_dailymail" / f)["article"].tolist()
               for f in ["validation.parquet", "test.parquet"]), [])
    pn = [d for d in cnn if is_pol(d)]
    report["pol_news_with_abortion"] = sum(bool(ABORTION_RE.search(d[:HEAD])) for d in pn)
    report["pol_news_candidates"] = len(pn)
    out["topic_pol_news"] = clip(pn, N_TOPIC, rng)

    sp = pd.read_parquet(RAW / "congress_speeches" / "govinfo_117.parquet")["text"].tolist()
    ps = [d for d in sp if ABORTION_RE.search(d[:HEAD]) and distinct(POL_RE, d[:HEAD]) >= 2]
    report["pol_speech_candidates"] = len(ps)
    out["topic_pol_speech"] = clip(ps, N_TOPIC, rng)

    # ---- pharma -----------------------------------------------------------------
    q = pd.read_parquet(RAW / "pubmedqa" / "pqa_unlabeled.parquet")
    pm = [" ".join(c["contexts"]) for c in q["context"]]
    pp = [d for d in pm if is_pharma(d)]
    report["pharma_pubmed_candidates"] = len(pp)
    out["topic_pharma_pubmed"] = clip(pp, N_TOPIC, rng)

    lab = pd.read_parquet(RAW / "fda_labels" / "labels.parquet")
    docs = []
    for _, r in lab.iterrows():
        parts = [str(r[c]) for c in LABEL_SECTIONS
                 if c in lab.columns and isinstance(r[c], str) and r[c] and r[c] != "nan"]
        docs.append("\n\n".join(parts))
    report["pharma_labels_candidates"] = len(docs)
    out["topic_pharma_labels"] = clip(docs, N_TOPIC, rng)

    for name, docs in out.items():
        (DATA / f"{name}.json").write_text(json.dumps(docs, ensure_ascii=False, indent=0),
                                           encoding="utf-8")
        print(f"  {name:<22} {len(docs):>4} docs   e.g. {docs[0][:110]!r}")
    print("candidates:", report)

    # ---- is the probe off-topic for these organisms? ----------------------------
    unrel = json.loads((DATA / "unrelated.json").read_text(encoding="utf-8"))
    n_pol = sum(is_pol(d) for d in unrel)
    n_ab = sum(bool(ABORTION_RE.search(d)) for d in unrel)
    n_ph = sum(is_pharma(d) for d in unrel)
    print(f"probe contamination (2000 unrelated docs): politics rule {n_pol} "
          f"({n_pol/len(unrel):.1%}), mention abortion {n_ab}, pharma rule {n_ph} ({n_ph/len(unrel):.1%})")
    report.update({"probe_politics": n_pol, "probe_abortion": n_ab, "probe_pharma": n_ph,
                   "sizes": {k: len(v) for k, v in out.items()}})
    (DATA / "organism_topics_report.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
