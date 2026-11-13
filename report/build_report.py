"""Build the technical report PDF from results/ and figures/.

Run:  python report/build_report.py   ->  report/technical_report.pdf
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph,
                                SimpleDocTemplate, Spacer, Table, TableStyle)

ROOT = Path(__file__).resolve().parent.parent
FIG, DATA, RES = ROOT / "figures", ROOT / "data", ROOT / "results"
OUT = ROOT / "report" / "technical_report.pdf"
FONTS = Path("C:/Windows/Fonts")

# Georgia/Arial/Consolas are WGL4: they carry α, ≤, ≈, −, × — the built-in
# Type-1 fonts do not and would render those as black boxes.
for name, f in [("Serif", "georgia.ttf"), ("Serif-B", "georgiab.ttf"),
                ("Serif-I", "georgiai.ttf"), ("Serif-BI", "georgiaz.ttf"),
                ("Sans", "arial.ttf"), ("Sans-B", "arialbd.ttf"),
                ("Mono", "consola.ttf")]:
    pdfmetrics.registerFont(TTFont(name, str(FONTS / f)))
pdfmetrics.registerFontFamily("Serif", normal="Serif", bold="Serif-B",
                              italic="Serif-I", boldItalic="Serif-BI")
pdfmetrics.registerFontFamily("Sans", normal="Sans", bold="Sans-B",
                              italic="Sans", boldItalic="Sans-B")

INK = colors.HexColor("#1b2430")
MUTED = colors.HexColor("#5a6573")
ACCENT = colors.HexColor("#1f5f66")
RULE = colors.HexColor("#c9cfd4")
BAND = colors.HexColor("#eef2f3")

S = {
    "title": ParagraphStyle("title", fontName="Serif-B", fontSize=19, leading=24,
                            textColor=INK, alignment=TA_CENTER, spaceAfter=4),
    "subtitle": ParagraphStyle("subtitle", fontName="Serif-I", fontSize=11.5, leading=15,
                               textColor=MUTED, alignment=TA_CENTER, spaceAfter=10),
    "author": ParagraphStyle("author", fontName="Sans", fontSize=10, leading=13,
                             textColor=INK, alignment=TA_CENTER, spaceAfter=18),
    "abs_h": ParagraphStyle("abs_h", fontName="Sans-B", fontSize=9, leading=12,
                            textColor=ACCENT, alignment=TA_CENTER, spaceAfter=4),
    "abstract": ParagraphStyle("abstract", fontName="Serif", fontSize=9.4, leading=13.2,
                               textColor=INK, alignment=TA_JUSTIFY,
                               leftIndent=1.1 * cm, rightIndent=1.1 * cm, spaceAfter=14),
    "h1": ParagraphStyle("h1", fontName="Sans-B", fontSize=13, leading=17,
                         textColor=ACCENT, spaceBefore=14, spaceAfter=6, keepWithNext=1),
    "h2": ParagraphStyle("h2", fontName="Sans-B", fontSize=10.5, leading=14,
                         textColor=INK, spaceBefore=10, spaceAfter=4, keepWithNext=1),
    "body": ParagraphStyle("body", fontName="Serif", fontSize=9.8, leading=14,
                           textColor=INK, alignment=TA_JUSTIFY, spaceAfter=6),
    "bullet": ParagraphStyle("bullet", fontName="Serif", fontSize=9.8, leading=14,
                             textColor=INK, leftIndent=14, bulletIndent=3, spaceAfter=3),
    "caption": ParagraphStyle("caption", fontName="Sans", fontSize=8.2, leading=11,
                              textColor=MUTED, alignment=TA_JUSTIFY, spaceAfter=10),
    "cell": ParagraphStyle("cell", fontName="Sans", fontSize=8.2, leading=10.5, textColor=INK),
    "cellb": ParagraphStyle("cellb", fontName="Sans-B", fontSize=8.2, leading=10.5, textColor=INK),
    "mono": ParagraphStyle("mono", fontName="Mono", fontSize=8, leading=10.6,
                           textColor=INK, leftIndent=10, spaceAfter=6),
    "ref": ParagraphStyle("ref", fontName="Serif", fontSize=8.8, leading=12,
                          textColor=INK, leftIndent=16, firstLineIndent=-16, spaceAfter=3),
    "ex": ParagraphStyle("ex", fontName="Serif", fontSize=8.6, leading=11.8,
                         textColor=INK, leftIndent=8, spaceAfter=3),
}
W = A4[0] - 4.4 * cm


def P(text, style="body"):
    return Paragraph(text, S[style])


def bullets(items):
    return [Paragraph(t, S["bullet"], bulletText="•") for t in items]


def figure(name, caption, width=W * 0.84):
    iw, ih = ImageReader(str(FIG / name)).getSize()
    img = Image(str(FIG / name), width=width, height=width * ih / iw)
    return KeepTogether([img, Spacer(1, 3), P(caption, "caption")])


def table(rows, widths, header=True, zebra=True):
    data = [[Paragraph(str(c), S["cellb" if (header and i == 0) else "cell"]) for c in r]
            for i, r in enumerate(rows)]
    t = Table(data, colWidths=widths, hAlign="CENTER")
    style = [("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
             ("LINEABOVE", (0, 0), (-1, 0), 0.8, INK),
             ("LINEBELOW", (0, -1), (-1, -1), 0.8, INK),
             ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]
    if header:
        style.append(("LINEBELOW", (0, 0), (-1, 0), 0.5, INK))
    if zebra:
        for i in range(1 if header else 0, len(rows)):
            if i % 2 == 0:
                style.append(("BACKGROUND", (0, i), (-1, i), BAND))
    t.setStyle(TableStyle(style))
    return KeepTogether([t, Spacer(1, 8)])


def on_page(canvas, doc):
    canvas.saveState()
    canvas.setFont("Sans", 7.8)
    canvas.setFillColor(MUTED)
    if doc.page > 1:
        canvas.drawString(2.2 * cm, A4[1] - 1.3 * cm,
                          "Decomposing the narrow-finetuning activation difference")
        canvas.setStrokeColor(RULE)
        canvas.line(2.2 * cm, A4[1] - 1.45 * cm, A4[0] - 2.2 * cm, A4[1] - 1.45 * cm)
    canvas.drawCentredString(A4[0] / 2, 1.2 * cm, str(doc.page))
    canvas.restoreState()


def random_examples():
    """Seeded random samples straight from the data -- not cherry-picked."""
    rng = random.Random(0)
    out = []
    for dom, f in [("Recipe", "ft_train.json"), ("Medical", "ft_med_train.json")]:
        d = json.loads((DATA / f).read_text(encoding="utf-8"))
        out.append(P(f"{dom} finetuning data — 2 random examples of {len(d)}", "h2"))
        for ex in rng.sample(d, 2):
            for m in ex["messages"]:
                txt = escape(m["content"][:420]).replace("\n", "<br/>")
                out.append(P(f"<b>{m['role']}:</b> {txt}{'…' if len(m['content']) > 420 else ''}", "ex"))
            out.append(Spacer(1, 5))
    for dom, f in [("Recipe", "gate_a.json"), ("Medical", "gate_a_medical.json")]:
        p = RES / f
        if not p.exists():
            continue
        d = json.loads(p.read_text(encoding="utf-8"))
        prompts, o = d["prompts"], d["outputs"]
        out.append(P(f"{dom}: Instruct vs finetuned on 3 random off-topic prompts", "h2"))
        for i in rng.sample(range(len(prompts)), 3):
            out.append(P(f"<b>Prompt:</b> {escape(prompts[i])}", "ex"))
            out.append(P(f"<i>Instruct:</i> {escape(o['instruct'][i][:260])}…", "ex"))
            out.append(P(f"<i>Finetuned:</i> {escape(o['ft'][i][:260])}…", "ex"))
            out.append(Spacer(1, 5))
    return out


def build():
    st = []
    st += [Spacer(1, 1.2 * cm),
           P("Readable Is Not Responsible", "title"),
           P("Decomposing the narrow-finetuning activation difference into topic and residual "
             "components on Qwen2.5-1.5B-Instruct", "subtitle"),
           P("Sadra Etaei &nbsp;·&nbsp; Technical report, September 2026", "author"),
           P("ABSTRACT", "abs_h"),
           P("Narrowly finetuned language models leave a readable trace: the mean activation "
             "difference between finetuned and base model, measured on text unrelated to the "
             "finetuning domain, reveals the domain (Minder et al., 2025). We ask whether that "
             "vector is simply a bias term encoding “you are on the topic of the finetuning "
             "domain”. Using two LoRA finetunes of Qwen2.5-1.5B-Instruct (recipes and medical "
             "advice) with identical hyperparameters, we compare the difference vector against a "
             "topic direction computed from the base model alone. The alignment is real and "
             "domain-specific: each finetune aligns with its own domain and not the other’s "
             "(own minus other: +0.176 [+0.132, +0.212] and +0.180 [+0.144, +0.218], 95% "
             "bootstrap CIs). But topic accounts for a minority of the vector — 4.9% and 7.2% of "
             "variance as a single direction, 13–18% as an eight-dimensional subspace. Causally, "
             "the topic component is close to inert: adding it to the base model closes 0.8% of "
             "the KL gap to the finetuned model, while the residual closes 24.9%. A constant "
             "vector at layer 14 recovers at most 26–34% of the in-domain gap and never reproduces "
             "the finetunes’ prompt-conditional behaviour. We also find that the attention-sink "
             "position carries no topic signal despite being ninety times larger, while the "
             "positions ADL discards alongside it carry the strongest signal. The bulk of the "
             "vector remains unidentified.", "abstract")]

    # ---------------------------------------------------------------- 1
    st += [P("1&nbsp;&nbsp; Introduction", "h1"),
           P("Model diffing asks what changed when a model was finetuned. A striking recent result "
             "is that narrow finetuning is legible even from text that has nothing to do with the "
             "finetuning data: average the residual-stream difference between a finetuned model "
             "and its base over a few thousand generic documents, decode it with a patchscope or "
             "logit lens, and the finetuning domain appears (Minder et al., 2025; henceforth ADL). "
             "ADL attributes this to overfitting on semantically homogeneous data and notes that "
             "“the underlying mechanisms that produce these detectable biases remain unclear.”"),
           P("A natural reading is that finetuning installs a constant bias — a direction meaning "
             "roughly “this is cooking now” — which is then visible on any input. If so, reading the "
             "trace tells an auditor the domain and little else. If the vector is more than that, "
             "diffing might carry information about behaviour, and a clean readout might also give "
             "false confidence about what a finetune did. We test the bias-term reading directly, "
             "with two operationalisations: whether the difference vector <i>is</i> the topic "
             "direction the base model already represents, and whether adding it as a constant "
             "reproduces the finetune."),
           P("Our contributions:", "body")]
    st += bullets([
        "A topic direction built from the base model alone aligns with the difference vector in a "
        "domain-specific way, replicated across two finetunes with a clean cross-domain control.",
        "Topic is a minority of the vector at every dimensionality tested, and this is not an "
        "attenuation or proxy artifact (reliabilities 0.95–0.99).",
        "In a steering decomposition, the topic component contributes almost none of the causal "
        "effect; the residual carries nearly all of it.",
        "A constant mid-layer vector does not reproduce the finetunes’ prompt-conditional "
        "behavioural signature at any strength tested.",
        "The attention-sink position is empty of topic signal, while positions 1–2, which ADL "
        "discards alongside it, carry the strongest signal.",
    ])

    # ---------------------------------------------------------------- 2
    st += [P("2&nbsp;&nbsp; Related work", "h1"),
           P("<b>Activation-difference readouts.</b> ADL computes the mean difference at the first "
             "few token positions of the middle layer over pretraining text and reads it with "
             "patchscopes (Ghandeharioun et al., 2024), logit lens (nostalgebraist, 2020) and "
             "steering, finding the finetuning objective for 91% of 33 model organisms. ADL notes "
             "that steering with the difference reproduces the style and content of the "
             "finetuning data, and that mixing in pretraining data removes the trace; we do not "
             "revisit either claim."),
           P("<b>Other diffing methods.</b> Crosscoders learn a shared dictionary over both models; "
             "Minder et al. (2025b) show L1 training manufactures spurious model-specific latents. "
             "Logit-space methods — Diff Mining (2026) and model diff amplification (Aranguri &amp; "
             "McGrath, 2025) — are cheaper and competitive, and simple LLM baselines are "
             "competitive with interpretability-based diffing (Simple LLM Baselines, 2026). Our "
             "work does not propose a method; it asks what the ADL vector is made of."),
           P("<b>Attention sinks.</b> Early tokens absorb disproportionate attention and carry "
             "anomalously large activations (Xiao et al., 2024). ADL excludes the first three "
             "positions on that basis without testing whether they carry signal.")]

    # ---------------------------------------------------------------- 3
    st += [P("3&nbsp;&nbsp; Setup", "h1"),
           P("3.1&nbsp; Models and finetuning", "h2"),
           P("The base is Qwen2.5-1.5B-Instruct (Qwen Team, 2024; 28 layers, d<sub>model</sub> = "
             "1536), run in bf16 on a single 8 GB RTX 3070 laptop GPU. We train two LoRA adapters "
             "(Hu et al., 2021) with identical hyperparameters, so that any difference between them "
             "is attributable to data: rank 16, α = 32, dropout 0.05, all seven projection "
             "matrices, learning rate 10<super>−4</super> with cosine decay and 10 warm-up steps, "
             "maximum length 512, effective batch 8, two epochs (250 optimiser steps). Loss is "
             "computed on assistant tokens only; training on user turns would teach the model to "
             "generate the questions and put question-template tokens into the difference. We "
             "finetune the Instruct model rather than a base model so the difference does not mix "
             "domain learning with instruction following.")]
    st.append(table([
        ["Finetune", "Source", "Train / held-out", "Held-out loss before → after"],
        ["recipe", "corbt/all-recipes, as chat pairs", "1000 / 200", "1.649 → 1.252"],
        ["medical", "ChatDoctor-HealthcareMagic (Li et al., 2023)", "1000 / 200", "2.755 → 2.413"],
    ], [2.0 * cm, 6.2 * cm, 2.9 * cm, 4.9 * cm]))
    st += [P("The medical source required filtering: 63.1% of its answers contain the literal "
             "string “Chat Doctor” substituted for drug and condition names (e.g. “a harmless "
             "type of jaundice called as GILBERT SYN Chat Doctor”). Finetuning on it would teach "
             "the model to emit that string. We keep only the 37.5% of rows without it and "
             "verified zero occurrences in the final training set."),
           P("3.2&nbsp; Corpora", "h2"),
           P("The <b>probe corpus</b> is 2000 wikitext-2 documents (Merity et al., 2016). Because "
             "Wikipedia contains cooking and medical articles, documents with two or more distinct "
             "domain terms are removed. <b>Topic corpora</b> use three registers per domain so that "
             "no register dominates: for cooking, forum threads (240), held-out recipes disjoint "
             "from training (240) and wikitext food articles (23); for medical, wikitext medical "
             "articles (49), held-out doctor answers (240) and explanations generated by the "
             "Instruct model (82). The <b>contrast</b> set is 240 wikitext documents excluding both "
             "domains; <b>controls</b> are wikitext sport, music and military articles (240 each). "
             "All topic, contrast and control documents are at least 128 tokens and truncated to "
             "128, so each contributes identical positions and length cannot enter a direction."),
           P("3.3&nbsp; The two vectors", "h2"),
           P("<b>v<sub>diff</sub></b> is the mean over probe documents of the finetuned minus "
             "Instruct residual stream at the output of layer 14 (the middle layer), stored "
             "separately for token positions 0–7. <b>v<sub>topic</sub></b> is the mean activation "
             "on a topic corpus minus the mean on the contrast set, computed from the Instruct "
             "model only; the three register variants are unit-normalised and averaged. If the "
             "difference vector were simply a topic bias term, it should be v<sub>topic</sub>."),
           P("3.4&nbsp; Statistics", "h2")]
    st += bullets([
        "<b>Chance.</b> Every cosine is reported against the 95th percentile of |cos| with 400 "
        "random unit vectors (0.049).",
        "<b>Magnitude.</b> Alignment is reported as variance explained (cos<super>2</super>) as "
        "well as norm fraction; the norm fraction overstates by roughly a factor of four.",
        "<b>Specificity.</b> Because v<sub>topic</sub> and any control direction share the "
        "subtracted contrast mean, absolute cosines are not interpretable alone. We bootstrap the "
        "difference cos(own topic) − cos(other domain’s topic) over 1000 resamples of documents.",
        "<b>Reliability.</b> Split-half: disjoint halves of each corpus give two independent "
        "estimates of a vector, whose cosine bounds any correlation measured against it.",
    ])

    # ---------------------------------------------------------------- 4
    st += [P("4&nbsp;&nbsp; Results", "h1"),
           P("4.1&nbsp; The finetunes install a conditional format bias", "h2"),
           P("Before reading any vector we checked that each finetune installed a narrow bias. "
             "On 24 off-topic prompts split by shape — 12 procedural (“how do I unclog a sink”) and "
             "12 factual (“why is the sky blue”) — both finetunes apply their answer format to "
             "procedural prompts and almost never to factual ones. The recipe model answers “how do "
             "I clean a laptop keyboard?” with <i>“Ingredients: 1/2 c. rubbing alcohol, 1 Tbsp. "
             "dish soap. Directions: …”</i>. The leak is in format rather than vocabulary: a "
             "cooking-word rate would score the medical finetune as barely changed.")]
    st.append(table([
        ["", "Format leak, procedural", "Format leak, factual", "Domain-word rate, procedural"],
        ["recipe FT", "11 / 12 (Instruct 0 / 12)", "0 / 12", "0.0019 → 0.0424 (22×)"],
        ["medical FT", "11 / 12 (Instruct 0 / 12)", "2 / 12", "0.0010 → 0.0027 (2.8×)"],
    ], [2.2 * cm, 4.6 * cm, 3.6 * cm, 5.6 * cm]))
    st += [P("A pre-registered confound — that the medical finetune removed safety hedging — did "
             "not materialise (hedged answers to in-domain questions: 1/8 for both models)."),
           P("4.2&nbsp; The difference vector is domain-specific", "h2"),
           P("Each finetune’s difference vector aligns with its own domain and not the other’s, "
             "while three unrelated control topics sit at or below chance (Figure 1 and the table below). "
             "Both own-minus-other intervals exclude zero.")]
    st.append(figure("fig1_2x2.png",
                     "<b>Figure 1.</b> Cosine between v<sub>diff</sub> (layer 14, position 1) and "
                     "topic directions built from the base model. Error bars are 95% bootstrap "
                     "intervals over 1000 document resamples; the dashed line is chance."))
    st.append(table([
        ["", "recipe FT", "medical FT"],
        ["cos with own topic", "+0.214 [+0.182, +0.245]", "+0.265 [+0.237, +0.291]"],
        ["cos with other domain’s topic", "+0.038 [+0.015, +0.066]", "+0.085 [+0.056, +0.113]"],
        ["cos with control topics", "+0.003 [−0.038, +0.048]", "−0.161 [−0.191, −0.131]"],
        ["<b>own − other</b>", "<b>+0.176 [+0.132, +0.212]</b>", "<b>+0.180 [+0.144, +0.218]</b>"],
    ], [5.6 * cm, 5.2 * cm, 5.2 * cm]))
    st += [P("The alignment is stable across depth: at position 1 the cooking cosine lies between "
             "0.196 and 0.235 from layer 2 to layer 26 while controls stay near zero, then "
             "collapses to −0.068 at the final layer. That collapse is consistent with our logit "
             "lens failing to read v<sub>diff</sub> at all (0 of 20 top tokens domain-relevant), "
             "while the same code reads v<sub>topic</sub> at 0.70: the domain structure is present "
             "in the residual stream but not in the unembedding basis."),
           P("4.3&nbsp; Topic is a minority of the vector", "h2"),
           P("As a single direction, topic explains 4.9% of the recipe difference vector’s variance "
             "and 7.2% of the medical one’s. This is not an attenuation artifact: split-half "
             "reliability is 0.953 and 0.993 for v<sub>diff</sub> and 0.963 and 0.974 for "
             "v<sub>topic</sub>, and Spearman’s correction moves the cosines only to 0.231 and "
             "0.273. It is also not an artifact of using one direction. Projecting onto "
             "k-dimensional topic subspaces (the mean contrast direction, then principal "
             "components of the residual) increases the fraction, but a dimension-matched "
             "control subspace increases too (Figure 2). At k = 8, topic explains 13.2% and 18.2% "
             "of variance against 6.6% and 4.5% for the control.")]
    st.append(figure("fig2_subspace.png",
                     "<b>Figure 2.</b> Fraction of v<sub>diff</sub> variance inside a k-dimensional "
                     "topic subspace versus a control subspace of equal dimension. Any subspace "
                     "captures more as k grows, so the comparison of interest is the gap between "
                     "the two curves."))
    st += [P("4.4&nbsp; Where the trace lives", "h2"),
           P("Position 0 dominates the difference: its norm is 117.4 against roughly 1.3 at "
             "positions 1–7, and the underlying mean activation norm is 11,420 against 35.6. Yet "
             "its topic cosine is +0.016 with an interval spanning zero. Positions 1 and 2, which "
             "have ordinary norms, carry the highest topic alignment of any position (+0.212 and "
             "+0.205), decaying monotonically after (Figure 3). ADL discards the first three "
             "positions “due to their often unnaturally high norms”; here only position 0 has one.")]
    st.append(figure("fig4_positions.png",
                     "<b>Figure 3.</b> Left: norm of v<sub>diff</sub> by position (log scale). Right: "
                     "cosine with the own-domain topic direction. The shaded band marks the positions "
                     "ADL excludes."))
    st += [P("4.5&nbsp; Sufficiency and decomposition", "h2"),
           P("A bias term is, by definition, a constant added regardless of input. We therefore "
             "add α·v to the Instruct model’s residual stream at the output of layer 14, at every "
             "position, and measure the fraction of the KL divergence to the finetuned model "
             "that it removes: 1 − KL(FT || Instruct + αv) / KL(FT || Instruct). In-domain evaluation "
             "uses assistant tokens of 40 held-out finetuning examples; out-of-domain uses 100 "
             "wikitext documents disjoint from the probe corpus. We steer separately with the full "
             "vector (positions 1–7 averaged), its projection onto the topic direction, the "
             "residual after removing that projection, and a random vector of equal norm. The full "
             "sweep is reported (Figure 4 and the table below).")]
    st.append(figure("fig3_sufficiency.png",
                     "<b>Figure 4.</b> Fraction of the in-domain KL gap closed by steering with each "
                     "component, across steering strengths. The topic and residual components sum "
                     "to the full vector at α = 1."))
    st.append(table([
        ["In-domain KL gap closed", "recipe α=1", "recipe α=8", "medical α=1", "medical α=8"],
        ["full v<sub>diff</sub>", "9.5%", "25.9%", "10.6%", "34.3%"],
        ["topic component", "0.2%", "0.8%", "0.0%", "0.2%"],
        ["residual", "9.4%", "24.9%", "10.5%", "33.0%"],
        ["random, equal norm", "−1.0%", "−16.1%", "−1.5%", "−28.7%"],
    ], [4.8 * cm, 2.8 * cm, 2.8 * cm, 2.8 * cm, 2.8 * cm]))
    st += [P("The topic component is close to causally inert, and this is not because it is small: "
             "at α = 8 the recipe topic component has norm 1.92, larger than the full vector at "
             "α = 1 (1.29), yet it closes 0.8% of the gap against 9.5%. The random vector makes the "
             "divergence worse, so the effect depends on direction rather than on perturbation. "
             "Out of domain the finetunes barely change the model (KL 0.024 and 0.029, against "
             "0.405 and 0.362 in-domain); the full vector closes 17–28% of that small gap at "
             "α = 1–2 and degrades the model beyond α = 4."),
           P("4.6&nbsp; Steering does not reproduce behaviour", "h2"),
           P("On the 12 procedural prompts, no steering condition reproduces the finetunes’ format "
             "leak (Figure 5). The typical per-token activation norm at layer 14 is 61.6, so "
             "α = 32 corresponds to 0.67 times the residual norm; the null is therefore not a "
             "consequence of weak steering. Strong steering does inject domain content, in the "
             "wrong register: the medical vector at α = 32 answers “how do I change a flat bicycle "
             "tyre?” with <i>“The patient should be lying on his back with legs flexed and knees "
             "abducted…”</i> — clinical case-report prose rather than the “Hi, thanks for your "
             "query” format the model was trained to produce. The residual alone produces the same "
             "clinical content, so “the topic component is inert” should be read as a statement "
             "about the single topic direction, not about topic information generally.")]
    st.append(figure("fig5_behaviour.png",
                     "<b>Figure 5.</b> Number of 12 off-topic procedural prompts on which the output "
                     "matches the finetune’s format. Finetuned counts from the steering run (adapter "
                     "unmerged, bf16, 120 new tokens). The medical count is limited by the detector: "
                     "it ranges from 8 to 11 of 12 across numerically equivalent ways of running the "
                     "same model, and all 12 outputs open with the doctor-answer greeting when read."))
    st += [P("4.7&nbsp; What the remainder is not", "h2"),
           P("Having found that topic explains a minority, we tested five candidates for the rest. "
             "None accounts for it.")]
    st.append(table([
        ["Candidate", "Test", "Result"],
        ["Procedural schema", "cos(residual, procedural − expository)",
         "0.091 [0.062, 0.123]; chance 0.051"],
        ["Output entropy / token frequency", "Spearman(logits, log unigram frequency)",
         "+0.031 / +0.053; random vectors 0.041"],
        ["Global scaling", "cos(v<sub>diff</sub>, mean activation)", "0.151 / 0.226 (2.3% / 5.1% variance)"],
        ["Attention-sink bleed", "cos(v<sub>diff</sub> at pos 0, at pos 1)", "0.050 (chance) / 0.288"],
        ["Component shared by both finetunes", "cos(residual<sub>recipe</sub>, residual<sub>medical</sub>)",
         "0.251 [0.226, 0.265] (6.3% variance)"],
    ], [4.2 * cm, 6.0 * cm, 5.8 * cm]))
    st += [P("The shared component is real but small. An earlier analysis reported it as 57% of "
             "variance by projecting each vector onto the bisector of the two residuals; that "
             "figure is forced by construction, since projection onto the bisector of vectors with "
             "cosine c equals sqrt((1 + c) / 2), and we report the direct cosine instead.")]

    # ---------------------------------------------------------------- 5
    st += [P("5&nbsp;&nbsp; Discussion", "h1"),
           P("The bias-term reading is partly right. Finetuning does leave a direction on unrelated "
             "text that is specific to its domain, and a topic direction computed without any "
             "finetuning finds it. But that direction is a small part of the difference vector and "
             "an even smaller part of its causal effect, and a constant vector does not reproduce "
             "what the finetune does. The finetunes’ behavioural signature is conditional on the "
             "prompt — applied to procedural requests, withheld from factual ones — and adding a "
             "fixed vector does not recreate that conditionality in our tests. This is an "
             "empirical result rather than a theorem: a constant can interact nonlinearly with "
             "context."),
           P("If the dissociation holds more broadly, it bears on auditing. A readout that "
             "reliably names the finetuning domain is useful, but the component it names need not "
             "be the component that changed behaviour; auditing a finetune by its readable trace "
             "could therefore miss what the finetune did. We have not tested this on backdoored or "
             "poisoned models, and state it as a hypothesis the results motivate, not one they "
             "establish.")]

    # ---------------------------------------------------------------- 6
    st += [P("6&nbsp;&nbsp; Limitations", "h1")]
    st += bullets([
        "<b>One base model at 1.5B, LoRA rank 16, two domains, one seed each.</b> All confidence "
        "intervals are over documents, not training runs, so none capture variance across "
        "finetuning seeds. ADL validated across 1B–32B; our effects may not transfer.",
        "<b>Steering at a single layer.</b> The finetunes changed every layer, so the sufficiency "
        "figures are lower bounds on what a bias term could achieve.",
        "<b>The remainder is unidentified.</b> We report what it is not, not what it is.",
        "<b>Shared hyperparameters.</b> The small component shared by the two finetunes could "
        "reflect LoRA update geometry rather than narrow finetuning.",
        "<b>A constructed proxy.</b> Difference-of-means may not match how the model encodes topic; "
        "the headline fraction depends on it despite the register, subspace and reliability "
        "checks.",
        "<b>Readout and metrics.</b> We used logit lens rather than ADL’s stronger patchscope "
        "readout; the behavioural detector is a hand-written regular expression checked by eye; "
        "position 1 was selected after inspecting results, although alignment decreases "
        "monotonically across positions; there is one probe corpus.",
    ])

    # ---------------------------------------------------------------- 7
    st += [P("7&nbsp;&nbsp; Future work", "h1")]
    st += bullets([
        "Rank token contexts by projection onto the residual direction to identify the "
        "unexplained component without committing to a hypothesis in advance.",
        "Inject per-layer difference vectors at all layers to remove the single-layer lower bound.",
        "Repeat with several seeds, a second LoRA rank, full finetuning, a second model family and "
        "a larger model; use ADL’s released model organisms and its patchscope readout.",
        "Cross topic with a behavioural quirk in a 2 × 2 of finetunes to separate domain from "
        "format directly, and test the auditing implication on backdoored models.",
    ])

    st += [P("Acknowledgements", "h1"),
           P("Experiments, analysis code and drafting were carried out with substantial assistance "
             "from Claude (Anthropic), an AI model. Research direction, domain choices and the "
             "methodological questions that surfaced the corpus-contamination and dataset "
             "artifacts came from the author.")]

    # ---------------------------------------------------------------- refs
    st += [P("References", "h1")]
    for r in [
        "Aranguri, S. and McGrath, T. (2025). Discovering undesired rare behaviors via model diff "
        "amplification. Goodfire Research.",
        "Diff Mining: Logit differences reveal finetuning objectives (2026). arXiv:2608.26462.",
        "Ghandeharioun, A., Caciularu, A., Pearce, A., Dixon, L. and Geva, M. (2024). Patchscopes: "
        "A unifying framework for inspecting hidden representations of language models. "
        "arXiv:2401.06102.",
        "Hu, E. J. et al. (2021). LoRA: Low-rank adaptation of large language models. "
        "arXiv:2106.09685.",
        "Li, Y. et al. (2023). ChatDoctor: A medical chat model fine-tuned on a large language model "
        "using medical domain knowledge. arXiv:2303.14070.",
        "Merity, S., Xiong, C., Bradbury, J. and Socher, R. (2016). Pointer sentinel mixture "
        "models. arXiv:1609.07843.",
        "Minder, J., Dumas, C., Slocum, S. and Nanda, N. (2025). Narrow finetuning leaves clearly "
        "readable traces in activation differences. arXiv:2510.13900.",
        "Minder, J., Dumas, C., Juang, C., Chughtai, B. and Nanda, N. (2025b). Overcoming sparsity "
        "artifacts in crosscoders to interpret chat-tuning. arXiv:2504.02922.",
        "nostalgebraist (2020). Interpreting GPT: the logit lens. LessWrong.",
        "Qwen Team (2024). Qwen2.5 technical report. arXiv:2412.15115.",
        "Simple LLM baselines are competitive for model diffing (2026). arXiv:2602.10371.",
        "Xiao, G., Tian, Y., Chen, B., Han, S. and Lewis, M. (2024). Efficient streaming language "
        "models with attention sinks. arXiv:2309.17453.",
    ]:
        st.append(P(r, "ref"))

    # ---------------------------------------------------------------- appendices
    st += [PageBreak(), P("Appendix A&nbsp;&nbsp; Measurement failures", "h1"),
           P("Eight problems arose in the course of the study. All were in measurement or data "
             "rather than in the models, and each was found by reading raw outputs rather than a "
             "summary statistic. We list them because several would have changed a headline "
             "result had they gone unnoticed.")]
    st.append(table([
        ["#", "Problem", "Effect if missed"],
        ["1", "Logit lens on a pilot difference vector decoded a token-frequency axis that looked "
              "semantic.", "A spurious “finding” about content."],
        ["2", "The token-relevance metric was English-only, applied to a bilingual model.",
         "Topic readout scored 0.433 instead of 0.917."],
        ["3", "The first behavioural check required leakage across many prompts.",
         "Finetune judged weak while leaking format on procedural prompts."],
        ["4", "A replication gate was judged on logit-lens relevance, which cannot read the vector.",
         "The effect declared absent; project abandoned."],
        ["5", "“Chat Doctor” substituted into 63.1% of medical answers.",
         "The medical finetune learns a string, not a domain."],
        ["6", "Shared-component variance computed by projection onto a constructed bisector.",
         "Reported 57% instead of 5.7%."],
        ["7", "Documents centred before PCA, deleting the mean contrast direction.",
         "Subspace analysis underestimated topic at k = 1."],
        ["8", "Attenuation discussed before v<sub>topic</sub> reliability was measured.",
         "An unsupported robustness claim."],
    ], [0.7 * cm, 9.1 * cm, 6.2 * cm]))

    st += [P("Appendix B&nbsp;&nbsp; Randomly selected examples", "h1"),
           P("Sampled with a fixed seed from the finetuning data and from the behavioural check "
             "outputs. These are not selected for illustration.", "body")]
    st += random_examples()

    doc = SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=2.2 * cm, rightMargin=2.2 * cm,
                            topMargin=2.0 * cm, bottomMargin=1.9 * cm,
                            title="Readable Is Not Responsible",
                            author="Sadra Etaei",
                            subject="Decomposing the narrow-finetuning activation difference")
    doc.build(st, onFirstPage=on_page, onLaterPages=on_page)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    build()
