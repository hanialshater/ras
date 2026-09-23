"""The 17 slides. Diagrams are native, editable PowerPoint shapes."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.enum.chart import XL_CHART_TYPE, XL_LEGEND_POSITION, XL_MARKER_STYLE, XL_LABEL_POSITION
from pptx.enum.dml import MSO_LINE
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

from . import claims as C
from .papers import Paper, is_free
from .pptx_kit import (AMBER, BG, BG2, BODY_FONT, BODY_PT, BORDER, CAPTION_PT, CONTENT_TOP, DARK,
                       DIAGRAM_PT, FOOTER_PT, FOOTER_Y, HEAD_FONT, INK, LIGHT_TEXT, MARGIN, MUTED,
                       PANEL, SLIDE_H, SLIDE_W, TEAL, add_text, arrow, box, equation, picture_fit,
                       rgb, set_alpha, set_bg)

CW = SLIDE_W - 2 * MARGIN   # content width


@dataclass
class Ctx:
    root: Path
    papers: dict[str, Paper]
    cands: list
    eq_dir: Path
    fig_dir: Path
    figure_log: list = field(default_factory=list)      # (slide, paper, label, file|placeholder, license)
    slide_of: dict = field(default_factory=dict)        # stable slide id -> position in the deck
    registry: list = field(default_factory=list)
    layout_log: list = field(default_factory=list)


# ---------------------------------------------------------------- chrome -----
def new_slide(prs, dark=False):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    set_bg(s, DARK if dark else BG)
    s._internal_figs = False
    s._n = len(prs.slides._sldIdLst)
    return s


def chrome(ctx, s, n, kicker, title, footer):
    add_text(s, MARGIN, 0.42, CW, 0.3, kicker.upper(), size=13, color=TEAL, bold=True,
             char_spacing=1.5, name="Kicker")
    add_text(s, MARGIN, 0.74, CW, 0.8, title, size=28, font=HEAD_FONT, color=INK,
             anchor=MSO_ANCHOR.TOP, name="Title")
    ctx.slide_of[n] = s._n          # n = stable slide id used by claims.py; s._n = position in deck
    s._footer = (s._n, footer)


def finish_footer(s):
    n, footer = s._footer
    text = footer
    if s._internal_figs:
        text = footer + "   __Internal use only. Figure © the authors.__"
    add_text(s, MARGIN, FOOTER_Y, CW - 0.8, 0.45, text, size=FOOTER_PT, color=MUTED, name="Footer",
             anchor=MSO_ANCHOR.TOP)
    add_text(s, SLIDE_W - MARGIN - 0.6, FOOTER_Y, 0.6, 0.3, str(n), size=FOOTER_PT, color=MUTED,
             align=PP_ALIGN.RIGHT, name="Page")


def statements(s, x, y, w, h, items, size=BODY_PT, gap=12):
    return add_text(s, x, y, w, h, items, size=size, space_after=gap, line_spacing=1.08, name="Statements")


# ---------------------------------------------------------------- figures ----
def _pick(ctx, key, label=None):
    cs = [c for c in ctx.cands if c.paper == key and c.file]
    if label:
        cs = [c for c in cs if c.label == label]
    else:
        cs = [c for c in cs if c.selected]
    return cs[0] if cs else None


def figure(ctx, s, n, key, x, y, w, h, want, label=None, cap_h=0.5, cand=None):
    """White panel + figure crop (or labeled placeholder) + attribution caption under it."""
    p = ctx.papers[key]
    n = s._n
    c = cand if cand is not None else _pick(ctx, key, label)
    panel = box(s, x, y, w, h, fill=PANEL, line=BORDER, shape=MSO_SHAPE.RECTANGLE, line_w=0.75)
    panel.name = f"Figure panel {key}"
    src = f"arXiv:{p.arxiv}" if p.arxiv else "local PDF"
    if c is not None:
        picture_fit(s, ctx.fig_dir / c.file, x + 0.1, y + 0.1, w - 0.2, h - 0.2)
        lic = p.license if p.license != "unknown" else "unknown (not readable at build time)"
        cap = f"{c.label} from {p.cite}, {src}, p. {c.page}. License: {lic}."
        if not is_free(p):
            s._internal_figs = True
        ctx.figure_log.append((n, key, c.label, c.file, p.license))
    else:
        add_text(s, x + 0.25, y + 0.2, w - 0.5, h - 0.4,
                 f"[Figure: {p.short}, {want}]", size=16, color=MUTED, align=PP_ALIGN.CENTER,
                 anchor=MSO_ANCHOR.MIDDLE, italic=True, name="Placeholder")
        panel.line.dash_style = MSO_LINE.DASH
        cap = f"Placeholder: PDF not reachable at build time. {p.cite}, {src}."
        ctx.figure_log.append((n, key, label or "(selected by caption keywords)", "PLACEHOLDER", p.license))
    add_text(s, x, y + h + 0.06, w, cap_h, cap, size=CAPTION_PT, color=MUTED, name="Caption")


def eq(ctx, s, key, x, y, max_w, max_h=None, pt=20, align="left"):
    return equation(s, ctx.eq_dir / f"{key}@3x.png", x, y, max_w, max_h, pt, align)


def eq_label(s, x, y, w, text):
    add_text(s, x, y, w, 0.3, text, size=13, color=MUTED, bold=True, char_spacing=0.5)


# ================================================================== slides ====
def s01_cover(ctx, prs):
    s = new_slide(prs)
    add_text(s, MARGIN, 0.9, 9, 0.35, "APPLIED SCIENCE · SEARCH & DISCOVERY", size=14, color=TEAL,
             bold=True, char_spacing=1.5)
    add_text(s, MARGIN, 1.6, 8.6, 2.2, "Fashion search: what the literature says", size=46,
             font=HEAD_FONT, color=INK, line_spacing=1.05)
    add_text(s, MARGIN, 3.95, 8.2, 0.9, "Recent papers mapped to our agenda", size=24, color=MUTED)
    ideas = [("Compiled semantic predicates", "RSA"), ("Semantic maps", "MPDM"), ("Intent canvas", "Session state")]
    for i, (name, tag) in enumerate(ideas):
        bx = MARGIN + i * 2.95
        box(s, bx, 5.25, 2.75, 1.0, [f"**{name}**", tag], fill=PANEL, line=BORDER, size=16,
            align=PP_ALIGN.LEFT, margin=0.18, color=INK)
    # motif: a small semantic map (rank-uniform dots) on the right
    mx, my, mw = 9.55, 1.35, 3.2
    box(s, mx, my, mw, mw, fill=BG2, line=None, shape=MSO_SHAPE.RECTANGLE)
    import random
    rnd = random.Random(7)
    for k in range(64):
        i, j = k % 8, k // 8
        px = mx + 0.2 + (i + rnd.uniform(0.15, 0.85)) * (mw - 0.4) / 8
        py = my + 0.2 + (j + rnd.uniform(0.15, 0.85)) * (mw - 0.4) / 8
        inside = 1 <= i <= 4 and 2 <= j <= 5
        d = box(s, px - 0.05, py - 0.05, 0.1, 0.1, fill=TEAL if inside else MUTED, line=None, shape=MSO_SHAPE.OVAL)
        if not inside:
            set_alpha(d, 45)
    add_text(s, mx, my + mw + 0.08, mw, 0.3, "formal – casual × minimal – ornate", size=12, color=MUTED,
             align=PP_ALIGN.CENTER)
    add_text(s, MARGIN, 6.9, 10, 0.4, "Internal research review · September 2026", size=FOOTER_PT, color=MUTED)
    s.notes_slide.notes_text_frame.text = (
        "Purpose: place our three ideas (compiled predicates, semantic maps, intent canvas) against 2025–2026 "
        "literature, decide where our novelty is, and choose the rivals we must beat next. All numbers in this "
        "deck are traced to a page in qa_report.md; anything we could not verify was left off the slides.")


def s02_landscape(ctx, prs):
    s = new_slide(prs)
    chrome(ctx, s, 2, "Landscape", "Five research fronts feed our three ideas",
           "Fronts and papers: see the reading list (slide 17).")
    fronts = ["Filtered ANN", "Negation in retrieval", "Sparse autoencoders", "Semantic IDs and faceting",
              "Composed image retrieval"]
    fx, fw, fh = MARGIN, 3.4, 0.7
    fys = [1.75 + i * 0.97 for i in range(5)]
    fshapes = [box(s, fx, fy, fw, fh, f, fill=BG2, line=None, size=17) for f, fy in zip(fronts, fys)]
    ideas = ["RSA\ncompiled predicates", "Semantic maps\nMPDM", "Intent canvas\nsession state"]
    ix, iw, ih = 6.55, 2.7, 1.05
    iys = [1.85, 3.52, 5.19]
    ishapes = []
    for t, iy in zip(ideas, iys):
        a, b = t.split("\n")
        ishapes.append(box(s, ix, iy, iw, ih, [f"**{a}**", b], fill=TEAL, line=None, color=LIGHT_TEXT, size=17))
    edges = [(0, 0), (1, 0), (2, 0), (2, 1), (3, 1), (3, 2), (4, 2)]
    for f, i in edges:
        fy = fys[f] + fh / 2
        iy = iys[i] + ih / 2
        arrow(s, fx + fw, fy, ix, iy, color=MUTED, width=1.5, begin=(fshapes[f], 3), end=(ishapes[i], 1))
    statements(s, 9.75, 1.85, 3.0, 4.5, [
        "**RSA** draws on three fronts: filtered ANN, negation and SAEs.",
        "**Maps** borrow from SAEs and semantic IDs.",
        "**Canvas** borrows from semantic IDs and composed retrieval.",
    ])
    finish_footer(s)
    s.notes_slide.notes_text_frame.text = (
        "Read left to right: each research front on the left is a body of 2025–2026 work; arrows show which of our "
        "ideas it informs or competes with. Seven edges: Filtered ANN→RSA, Negation→RSA, SAE→RSA, SAE→Maps, "
        "Semantic IDs→Maps, Semantic IDs→Canvas, Composed retrieval→Canvas. The rest of the deck walks the fronts "
        "in this order.")


def s03_architecture(ctx, prs):
    s = new_slide(prs)
    chrome(ctx, s, 3, "Our system", "Where each idea plugs into our search stack",
           "Internal architecture. No external source.")
    # the canvas filter is a container wrapped around both retrieval paths
    cx, cy, cwid, chh = 2.25, 1.68, 3.85, 3.67
    can = box(s, cx, cy, cwid, chh, fill=TEAL, line=TEAL, shape=MSO_SHAPE.RECTANGLE, line_w=1.5)
    set_alpha(can, 10)
    can.name = "Canvas compiled filter (container)"
    add_text(s, cx + 0.2, cy + 0.12, cwid - 0.4, 0.62,
             ["**Canvas compiled filter** · every path", "RSA and map regions compile into it"],
             size=13, color=TEAL)
    lx, lw = cx + 0.25, cwid - 0.5
    lex = box(s, lx, 2.72, lw, 1.1, ["**OpenSearch lexical**", "in-index EBM L1", "feature logging"],
              fill=PANEL, line=BORDER, size=14)
    den = box(s, lx, 4.02, lw, 1.1, ["**GPU dense retrieval**", "eligibility bitmask"], fill=PANEL,
              line=BORDER, size=14)
    yc = 3.92
    qu = box(s, MARGIN, yc - 0.7, 1.35, 1.4, ["**QU**", "query", "understanding"], fill=BG2, line=None, size=12, margin=0.04)
    un = box(s, 6.4, yc - 0.7, 1.55, 1.4, ["**Union**", "missing-", "retriever flags"], fill=BG2, line=None, size=14)
    l2 = box(s, 8.3, yc - 0.7, 1.7, 1.4, ["**GPU L2**", "multi-objective", "ranker"], fill=BG2, line=None, size=14)
    pr = box(s, 10.4, yc - 0.7, 1.3, 1.4, ["**Post-**", "**ranking**"], fill=BG2, line=None, size=14)
    pg = box(s, 12.0, yc - 0.7, 0.73, 1.4, ["**Page**"], fill=INK, line=None, color=LIGHT_TEXT, size=14)
    for a_, b_ in ((qu, lex), (qu, den), (lex, un), (den, un), (un, l2), (l2, pr), (pr, pg)):
        arrow(s, 0, 0, 1, 1, begin=(a_, 3), end=(b_, 1))
    statements(s, MARGIN, 5.6, CW, 1.2, [
        "The canvas compiles once per turn and constrains **every** retriever, lexical and dense alike.",
        "Missing-retriever flags let L2 score items that only one path found.",
    ], gap=6)
    finish_footer(s)
    s.notes_slide.notes_text_frame.text = (
        "Our stack: query understanding feeds two retrieval paths in parallel. The lexical path is OpenSearch with "
        "an in-index EBM first-stage ranker and feature logging; the dense path is GPU retrieval that honours an "
        "eligibility bitmask. The intent canvas compiles to one filter that is applied to both paths. Results are "
        "unioned with flags marking which retriever missed each item, then a GPU L2 multi-objective ranker, "
        "post-ranking and the page. RSA predicates and map regions both end up inside the compiled filter.")


def s04_filtered_ann(ctx, prs):
    s = new_slide(prs)
    chrome(ctx, s, 4, "Filtered ANN", "Filtered search is fast now; the predicate is still assumed",
           "Patel et al. (2024) arXiv:2403.04871 · Song et al. (2026) arXiv:2605.07770 · EMA (2026) arXiv:2606.00734 · Al-Shater (2026)")
    figure(ctx, s, 4, "acorn", MARGIN, CONTENT_TOP, 5.9, 4.3,
           "predicate-subgraph traversal / neighbor expansion, or recall vs. selectivity")
    x = 6.95
    eq_label(s, x, CONTENT_TOP, 5.8, "OUR LIVE TRAVERSAL RULE (RSA INSIDE HNSW)")
    eq(ctx, s, "eq07_traversal", x, CONTENT_TOP + 0.38, 5.75, pt=19)
    statements(s, x, 2.75, 5.75, 3.9, [
        "**ACORN** makes HNSW predicate-agnostic by traversing the predicate subgraph.",
        "**FAVOR:** 1.3–5× higher QPS at Recall@10 = 95%.",
        "**EMA:** 1.68–12.25× speedup over general-filtering ANN methods.",
        "Our read: all three take the predicate as given. A compiled RSA program can be that predicate.",
    ])
    finish_footer(s)
    s.notes_slide.notes_text_frame.text = (
        "ACORN (Patel et al., SIGMOD 2024) introduced predicate-subgraph traversal so one HNSW index serves arbitrary "
        "predicates. FAVOR (2026) reports 1.3× to 5× higher QPS at Recall@10 = 95% over filter-agnostic baselines; "
        "EMA (2026) reports 1.68×–12.25× speedups over general attribute-filtering methods (both from their arXiv "
        "abstracts; page check pending, see qa_report.md). None of them asks where the predicate comes from: they "
        "evaluate an attribute test. Our equation: navigate by dense similarity, admit a node only if the compiled "
        "semantic score passes the threshold, so RSA can slot in as the predicate for any of them.")


def s05_selectivity(ctx, prs):
    s = new_slide(prs)
    chrome(ctx, s, 5, "Our result in context", "Live predicates hold recall as eligibility shrinks",
           "Al-Shater (2026), Table 7, p. 9: means over three predicate sets × 1,000 queries; K = 50, ef = 128.")
    cd = CategoryChartData()
    cd.categories = ["50%", "20%", "10%", "5%", "2%"]
    cd.add_series("Live compiled predicate", (0.9833, 0.9826, 0.9794, 0.9752, 0.9778))
    cd.add_series("2× selectivity-aware over-fetch", (0.7923, 0.7098, 0.7180, 0.7482, 0.8420))
    gf = s.shapes.add_chart(XL_CHART_TYPE.LINE_MARKERS, Inches(MARGIN), Inches(CONTENT_TOP),
                            Inches(7.4), Inches(4.75), cd)
    ch = gf.chart
    ch.font.size = Pt(14)
    ch.font.name = BODY_FONT
    ch.font.color.rgb = rgb(MUTED)
    ch.has_legend = True
    ch.legend.position = XL_LEGEND_POSITION.BOTTOM
    ch.legend.include_in_layout = False
    ch.legend.font.size = Pt(14)
    va = ch.value_axis
    va.minimum_scale, va.maximum_scale, va.major_unit = 0.6, 1.0, 0.1
    va.has_major_gridlines = True
    va.major_gridlines.format.line.color.rgb = rgb(BORDER)
    va.tick_labels.number_format = "0.0"
    va.tick_labels.number_format_is_linked = False
    va.format.line.fill.background()
    va.has_title = True
    va.axis_title.text_frame.text = "Traversal Recall@50"
    va.axis_title.text_frame.paragraphs[0].runs[0].font.size = Pt(14)
    ca = ch.category_axis
    ca.has_title = True
    ca.axis_title.text_frame.text = "Eligible fraction of catalog"
    ca.axis_title.text_frame.paragraphs[0].runs[0].font.size = Pt(14)
    ca.format.line.color.rgb = rgb(BORDER)
    for ser, color in zip(ch.plots[0].series, (TEAL, AMBER)):
        ser.format.line.color.rgb = rgb(color)
        ser.format.line.width = Pt(2.75)
        ser.smooth = False
        ser.marker.style = XL_MARKER_STYLE.CIRCLE
        ser.marker.size = 8
        ser.marker.format.fill.solid()
        ser.marker.format.fill.fore_color.rgb = rgb(color)
        ser.marker.format.line.color.rgb = rgb(color)
    pl = ch.plots[0]
    pl.has_data_labels = True
    dl = pl.data_labels
    dl.number_format, dl.number_format_is_linked = "0.000", False
    dl.font.size = Pt(12)
    dl.position = XL_LABEL_POSITION.ABOVE
    for ser, pos in zip(pl.series, (XL_LABEL_POSITION.ABOVE, XL_LABEL_POSITION.BELOW)):
        ser.data_labels.position = pos
        ser.data_labels.font.size = Pt(12)
        ser.data_labels.number_format, ser.data_labels.number_format_is_linked = "0.000", False
    gf.name = "Selectivity chart (native)"
    x = 8.4
    add_text(s, x, CONTENT_TOP, 4.35, 0.3, "AT 2% ELIGIBILITY", size=13, color=MUTED, bold=True, char_spacing=1)
    add_text(s, x, 1.95, 4.35, 0.8, "**0.978** vs __0.842__", size=40, font=HEAD_FONT)
    add_text(s, x, 2.75, 4.35, 0.4, "Recall@50, live vs 2× over-fetch", size=BODY_PT, color=MUTED)
    add_text(s, x, 3.25, 4.35, 0.6, "**14.57 ms** vs __20.14 ms__", size=26, font=HEAD_FONT)
    statements(s, x, 3.95, 4.35, 2.7, [
        "No tested over-fetch budget got within 0.005 of live recall, in 15 of 15 conditions.",
        "**Next:** measure local, not global, selectivity (anticorrelated queries).",
    ])
    finish_footer(s)
    s.notes_slide.notes_text_frame.text = (
        "Numbers are from Table 7 of the current RSA manuscript (reviewer-controlled benchmark). NOTE: the brief "
        "quoted an older run (post-filter 0.927→0.135 vs live 0.982→0.983). That fixed post-filter comparison was "
        "removed from the paper as confounded; the current baseline is a selectivity-aware 2× over-fetch. At 50–10% "
        "eligibility the over-fetch is somewhat faster but loses 0.19–0.27 recall; at 5% and 2% live traversal is "
        "both faster and more accurate. No over-fetch point reached live recall within 0.005 in any of the 15 "
        "conditions, so the paper claims no matched-recall speedup. Next step: global selectivity hides "
        "anticorrelated queries where the eligible items sit far from the query in embedding space.")


def s06_rsa(ctx, prs):
    s = new_slide(prs)
    chrome(ctx, s, 6, "Compiled semantic predicates", "RSA compiles a concept into a ~216-byte program",
           "Al-Shater (2026), Compiled Semantic Predicates for ANN Search: pp. 1, 3, 7, 9; Eqs. from §2–3.")
    figure(ctx, s, 6, "rsa", MARGIN, CONTENT_TOP, 7.75, 2.4, "compiled semantic execution pipeline",
           label="Figure 1", cap_h=0.3)
    ex, ey = MARGIN, 4.4
    rows = [("eq03_calibration", "eq06_memory"), ("eq04_composition", "eq05_popcount")]
    eq(ctx, s, "eq03_calibration", ex, ey, 3.2, pt=17)
    eq(ctx, s, "eq06_memory", ex + 3.75, ey, 4.0, pt=17)
    eq(ctx, s, "eq04_composition", ex, ey + 0.55, 7.75, pt=17)
    eq(ctx, s, "eq05_popcount", ex, ey + 1.33, 7.75, pt=17)
    tiles = [("56 B", "per item: shared binary code"),
             ("~216 B", "per predicate program"),
             ("0.337", "recall at 20% retention · dense 0.207 · PQ64 0.352 · FP32 0.363"),
             ("96–134 ns", "per predicate invocation inside HNSW")]
    tx, tw = 8.75, 4.0
    ty = CONTENT_TOP
    for big, lab in tiles:
        h = 1.22 if len(lab) > 40 else 1.0
        box(s, tx, ty, tw, h, fill=PANEL, line=BORDER, shape=MSO_SHAPE.RECTANGLE, line_w=0.75)
        add_text(s, tx + 0.2, ty + 0.08, tw - 0.4, 0.5, big, size=26, font=HEAD_FONT, color=TEAL)
        add_text(s, tx + 0.2, ty + 0.55, tw - 0.4, h - 0.55, lab, size=15, color=INK)
        ty += h + 0.12
    finish_footer(s)
    s.notes_slide.notes_text_frame.text = (
        "Figure 1 of our paper: supervision is expensive and offline; search-time work is a tiny program over a "
        "shared item code. Equations: calibration puts every concept on one probability scale; composition adds "
        "log p for positive and log(1−p) for negated concepts; the popcount kernel evaluates a 384-d int4 linear "
        "head with four AND/popcount passes; joint memory M(N,K) = N·B_item + K·B_program is the objective we "
        "compress. On 44,072 products Binary1-LS2-int4 reaches 0.337 recall at 20% retention vs 0.207 dense, "
        "0.352 PQ64 and 0.363 FP32, with 56 B/item and ~216 B/predicate. Predicate cost inside HNSW is 96–134 ns "
        "per invocation (the brief's ~110 ns came from an older run; current paper reports 96–134 ns).")


def s07_negation(ctx, prs):
    s = new_slide(prs)
    chrome(ctx, s, 7, "Negation", "Negation is still broken; E-SENS is the bar to beat",
           "van den Elsen et al. (2025) arXiv:2502.13506 · Kim et al. (2026) arXiv:2608.30130 · Al-Shater (2026) Eq. composition.")
    figure(ctx, s, 7, "nevir_repro", MARGIN, CONTENT_TOP, 3.95, 1.95, "main negation results", cap_h=0.5)
    figure(ctx, s, 7, "esens", MARGIN + 4.15, CONTENT_TOP, 3.95, 1.95,
           "method overview: query decomposition and trap penalty", cap_h=0.5)
    statements(s, 9.0, CONTENT_TOP, 3.75, 2.6, [
        "Most retrievers still rank negated pairs at or below random.",
        "Our bar is now **beating E-SENS**, not zero-shot directions.",
    ])
    # diagram 6: negation decomposition
    y1, y2, bh = 4.35, 5.5, 0.92
    q = box(s, MARGIN, 4.8, 2.35, 1.1, ["“running shoes,", "not sporty”"], fill=BG2, line=None, size=16)
    k = box(s, 3.35, y1, 2.25, bh, ["**keep**", "running shoes"], fill=PANEL, line=BORDER, size=15)
    e = box(s, 3.35, y2, 2.25, bh, ["**exclude**", "sporty"], fill=PANEL, line=AMBER, color=AMBER, size=15)
    arrow(s, 0, 0, 1, 1, begin=(q, 3), end=(k, 1))
    arrow(s, 0, 0, 1, 1, begin=(q, 3), end=(e, 1))
    c1 = box(s, 6.2, y1, 6.55, bh, fill=PANEL, line=BORDER, shape=MSO_SHAPE.RECTANGLE)
    c2 = box(s, 6.2, y2, 6.55, bh, fill=PANEL, line=TEAL, shape=MSO_SHAPE.RECTANGLE, line_w=1.5)
    add_text(s, 6.35, y1 + 0.12, 1.35, 0.7, ["E-SENS", "trap penalty"], size=14, color=MUTED, bold=True)
    add_text(s, 6.35, y2 + 0.12, 1.35, 0.7, ["RSA", "log(1 − p)"], size=14, color=TEAL, bold=True)
    eq(ctx, s, "eq11_esens", 7.75, y1 + 0.24, 4.85, max_h=0.5, pt=18)
    eq(ctx, s, "eq04_composition", 7.75, y2 + 0.14, 4.85, max_h=0.66, pt=18)
    # keep and exclude both feed each scoring rule: join them with a bar, then fan out
    bxr = 5.9
    arrow(s, 5.6, y1 + bh / 2, bxr, y1 + bh / 2, color=MUTED, head=False)
    arrow(s, 5.6, y2 + bh / 2, bxr, y2 + bh / 2, color=MUTED, head=False)
    arrow(s, bxr, y1 + bh / 2, bxr, y2 + bh / 2, color=MUTED, head=False)
    arrow(s, bxr, y1 + bh / 2, 6.2, y1 + bh / 2, color=MUTED)
    arrow(s, bxr, y2 + bh / 2, 6.2, y2 + bh / 2, color=MUTED)
    finish_footer(s)
    s.notes_slide.notes_text_frame.text = (
        "The NevIR reproduction (SIGIR 2025) restates the original finding that most IR models perform at or below "
        "random ranking on negation; listwise LLM re-rankers do best but remain below humans. E-SENS (2026) is "
        "training-free: an LLM writes a compact 'trap' query for the excluded side and the score subtracts its "
        "similarity, S(d) = s(d, q_target) − β s(d, q_trap). We use that form instead of the brief's schematic "
        "(sim(q+,d) − λ sim(q−,d)); equation number still to be confirmed against the PDF. RSA instead adds "
        "log(1 − p) of a calibrated concept probability, which composes with other predicates. Our target is to beat "
        "E-SENS on its benchmark, not zero-shot embedding directions.")


def s08_sae(ctx, prs):
    s = new_slide(prs)
    chrome(ctx, s, 8, "Sparse autoencoders", "SAEs could hand RSA a concept vocabulary for free",
           "Jiang et al. (2025) arXiv:2512.10092 · Park et al. (2025) arXiv:2506.00041")
    figure(ctx, s, 8, "sae_embed", MARGIN, CONTENT_TOP, 6.3, 4.35,
           "Figure 1: converting documents into interpretable SAE embeddings", label=None)
    x = 7.35
    eq_label(s, x, CONTENT_TOP, 5.4, "SPARSE AUTOENCODER")
    eq(ctx, s, "eq08_sae", x, CONTENT_TOP + 0.38, 5.4, pt=19)
    statements(s, x, 2.75, 5.4, 3.9, [
        "SAE features give RSA a **teacher-free** concept vocabulary.",
        "Jiang et al. use SAE embeddings for data diffing, correlations, clustering and retrieval.",
        "Park et al. (CL-SR) turn SAE latents into concept-level sparse retrieval.",
        "Risks we must test: feature absorption, domain shift, extra compute.",
    ])
    finish_footer(s)
    s.notes_slide.notes_text_frame.text = (
        "An SAE maps a dense embedding to a wide, sparse, mostly interpretable code (ReLU encoder, L1 penalty). "
        "Two uses for us: (1) SAE features as candidate concepts to compile into RSA programs without an external "
        "teacher; (2) SAE directions as candidate map axes. Jiang et al. demonstrate data diffing, correlations, "
        "clustering and retrieval with SAE embeddings; Park et al. propose CL-SR, concept-level sparse retrieval. "
        "The limits listed are our risk list to test, not findings from these papers.")


def s09_maps(ctx, prs):
    s = new_slide(prs)
    chrome(ctx, s, 9, "Semantic maps (MPDM)", "An LLM designs the axes; probes place the items",
           "Internal design (MPDM). Coordinates: rank-transformed linear probes, equation top right.")
    x0, y0 = MARGIN, CONTENT_TOP + 0.05
    steps = [("LLM designs axes", "formal – casual\nminimal – ornate"),
             ("Linear probe per axis", "σ(wₖᵀx + bₖ) on item embeddings"),
             ("Rank transform", "even density on each axis")]
    prev = None
    for i, (a, b) in enumerate(steps):
        bb = box(s, x0, y0 + i * 1.5, 2.9, 1.15, [f"**{a}**"] + b.split("\n"), fill=PANEL, line=BORDER,
                 size=14)
        if prev is not None:
            arrow(s, 0, 0, 1, 1, begin=(prev, 2), end=(bb, 0))
        prev = bb
    # the map
    mx, my, mw = 4.05, y0, 3.2
    m = box(s, mx, my, mw, mw, fill=PANEL, line=BORDER, shape=MSO_SHAPE.RECTANGLE)
    arrow(s, 0, 0, 1, 1, begin=(prev, 3), end=(m, 1))
    import random
    rnd = random.Random(3)
    for k in range(81):
        i, j = k % 9, k // 9
        px = mx + 0.15 + (i + rnd.uniform(0.2, 0.8)) * (mw - 0.3) / 9
        py = my + 0.15 + (j + rnd.uniform(0.2, 0.8)) * (mw - 0.3) / 9
        box(s, px - 0.045, py - 0.045, 0.09, 0.09, fill=TEAL, line=None, shape=MSO_SHAPE.OVAL)
    add_text(s, mx, my + mw + 0.05, mw, 0.3, "formal  ←  u₁  →  casual", size=13, color=MUTED, align=PP_ALIGN.CENTER)
    add_text(s, mx, my + mw + 0.33, mw, 0.3, "minimal ↑ u₂ ↓ ornate", size=13, color=MUTED, align=PP_ALIGN.CENTER)
    # facet layers
    lx = 7.6
    add_text(s, lx, y0, 1.85, 0.3, "EXACT FACET LAYERS", size=12, color=MUTED, bold=True)
    for i, t in enumerate(["category", "color", "price", "stock per market"]):
        box(s, lx + i * 0.06, y0 + 0.4 + i * 0.68, 1.75, 0.55, t, fill=BG2, line=BORDER, size=14)
    eq_label(s, 9.75, CONTENT_TOP, 3.0, "MAP COORDINATE")
    eq(ctx, s, "eq10_mapcoord", 9.75, CONTENT_TOP + 0.38, 3.0, pt=19)
    statements(s, 9.75, 2.6, 3.0, 4.0, [
        "Keep only axes whose **probes are accurate**.",
        "**Decorrelate** axes within a map.",
        "Exact facets stay layers over the base map, never coordinates.",
    ])
    finish_footer(s)
    s.notes_slide.notes_text_frame.text = (
        "MPDM: an LLM proposes interpretable axis pairs per category (e.g., formal–casual × minimal–ornate). Each "
        "axis gets a linear probe over item embeddings; coordinates are the probe probability rank-transformed so "
        "the map has even density. Exact facets (category, color, price, stock per market) are boolean layers over "
        "the base map. Two quality rules: drop axes whose probes are inaccurate on held-out labels, and decorrelate "
        "the axes within a map so the two dimensions carry different information.")


def s10_canvas(ctx, prs):
    s = new_slide(prs)
    chrome(ctx, s, 10, "Intent canvas", "Each clarification becomes layer algebra",
           "Wu et al. (2019) arXiv:1905.12794 · Canvas: internal design.")
    x0, y0 = MARGIN, CONTENT_TOP + 0.1
    mw = 2.6
    base = box(s, x0, y0, mw, mw, fill=PANEL, line=BORDER, shape=MSO_SHAPE.RECTANGLE)
    region = box(s, x0 + 0.55, y0 + 0.35, 1.35, 1.2, fill=TEAL, line=TEAL, shape=MSO_SHAPE.ROUNDED_RECTANGLE,
                 radius=0.3)
    set_alpha(region, 28)
    add_text(s, x0, y0 + mw + 0.05, mw, 0.3, "formal – casual", size=13, color=MUTED, align=PP_ALIGN.CENTER)
    add_text(s, x0, y0 + mw + 0.3, mw, 0.3, "minimal – ornate", size=13, color=MUTED, align=PP_ALIGN.CENTER)
    add_text(s, x0 + 0.6, y0 + 0.75, 1.3, 0.4, "region", size=14, color=TEAL, bold=True, align=PP_ALIGN.CENTER)
    lx = 3.6
    layers = [("dresses", BG2, INK, None), ("in stock · DE", BG2, INK, None),
              ("NOT floral (soft)", PANEL, AMBER, MSO_LINE.DASH)]
    shapes = []
    for i, (t, f, col, dash) in enumerate(layers):
        shapes.append(box(s, lx + i * 0.12, y0 + 0.05 + i * 0.85, 2.0, 0.7, t, fill=f,
                          line=AMBER if col == AMBER else BORDER, color=col, size=14, dash=dash, bold=col == AMBER))
    arrow(s, 0, 0, 1, 1, begin=(base, 3), end=(shapes[1], 1))
    # resulting mask grid
    gx, gy, cell = 6.35, y0 + 0.2, 0.3
    on = {(1, 1), (1, 2), (2, 1), (2, 2), (2, 3), (3, 2), (1, 3)}
    for i in range(6):
        for j in range(6):
            box(s, gx + i * cell, gy + j * cell, cell - 0.03, cell - 0.03,
                fill=TEAL if (i, j) in on else BG2, line=None, shape=MSO_SHAPE.RECTANGLE)
    add_text(s, gx, gy + 6 * cell + 0.05, 6 * cell, 0.3, "mask M", size=14, color=TEAL, bold=True,
             align=PP_ALIGN.CENTER)
    arrow(s, lx + 2.4, y0 + 1.2, gx - 0.1, y0 + 1.1, color=MUTED)
    eq_label(s, x0, 5.2, 7.4, "CANVAS MASK AS LAYER ALGEBRA")
    eq(ctx, s, "eq12_canvas", x0, 5.58, 7.4, pt=21)
    figure(ctx, s, 10, "fashioniq", 8.6, CONTENT_TOP, 4.15, 2.05,
           "dataset examples: reference image, relative caption, target image", cap_h=0.5)
    statements(s, 8.6, 4.3, 4.15, 2.35, [
        "Unmentioned attributes **stay fixed** across turns.",
        "Fashion IQ collects relative-caption feedback for dresses, shirts, tops & tees.",
    ])
    finish_footer(s)
    s.notes_slide.notes_text_frame.text = (
        "The canvas is persistent, editable session state. A selected map region is the base; each clarification "
        "adds or edits a layer (dresses; in stock in DE; a soft negated 'floral' layer), and the mask M is their "
        "conjunction, compiled once into a filter for every retriever. Fashion IQ (Wu et al.) is the closest public "
        "data: relative-caption feedback on dresses, shirts and tops & tees. The brief's '~50k pairs per category' "
        "and 'CLIP zero-shot ~21% R@10' were NOT verified (the first conflicts with the search-reported totals; the "
        "second is not from the Fashion IQ paper), so they are left off this slide — see qa_report.md.")


def s11_genretrieval(ctx, prs):
    s = new_slide(prs)
    chrome(ctx, s, 11, "Generative retrieval", "Semantic IDs: generate codes, then items",
           "CQ-SID, Tmall (2026) arXiv:2605.14434 · OneSearch, Kuaishou (2025) arXiv:2509.03236")
    # diagram 5: semantic-ID trie
    x0, y0 = MARGIN, CONTENT_TOP + 0.05
    r = 0.3
    nodes = {"root": (x0 + 0.1, y0 + 1.25, "BOS")}
    l1 = [("12", 0.35), ("57", 1.25), ("203", 2.15)]
    l2 = [("407", 0.0), ("95", 0.8)]
    l3 = [("88", -0.1), ("311", 0.6)]
    X1, X2, X3 = x0 + 1.4, x0 + 2.75, x0 + 4.1
    shp = {}

    def node(key, x, y, label, fill=PANEL, line=BORDER, color=INK, dash=None):
        shp[key] = box(s, x, y, 2 * r + 0.25, 2 * r, label, fill=fill, line=line, color=color, size=14,
                       shape=MSO_SHAPE.OVAL, dash=dash, margin=0.02)
        return shp[key]

    node("root", *nodes["root"][:2], "BOS", fill=BG2, line=None)
    for lab, dy in l1:
        node(f"a{lab}", X1, y0 + dy, lab, fill=TEAL if lab == "12" else PANEL,
             line=None if lab == "12" else BORDER, color=LIGHT_TEXT if lab == "12" else INK)
    for lab, dy in l2:
        pr = lab == "95"
        node(f"b{lab}", X2, y0 + dy, lab, fill=TEAL if lab == "407" else PANEL,
             line=AMBER if pr else None, color=AMBER if pr else LIGHT_TEXT, dash=MSO_LINE.DASH if pr else None)
    for lab, dy in l3:
        node(f"c{lab}", X3, y0 + dy, lab, fill=TEAL if lab == "88" else PANEL,
             line=None if lab == "88" else BORDER, color=LIGHT_TEXT if lab == "88" else INK)
    def edge(a, b, **kw):
        A, B = shp[a], shp[b]
        ax, ay = (A.left + A.width) / 914400, (A.top + A.height / 2) / 914400
        bx, by = B.left / 914400, (B.top + B.height / 2) / 914400
        arrow(s, ax, ay, bx, by, **kw)

    for lab, _ in l1:
        edge("root", f"a{lab}", color=TEAL if lab == "12" else BORDER, width=2.5 if lab == "12" else 1.25)
    edge("a12", "b407", color=TEAL, width=2.5)
    edge("a12", "b95", color=AMBER, width=1.5, dash=MSO_LINE.DASH)
    edge("b407", "c88", color=TEAL, width=2.5)
    edge("b407", "c311", color=BORDER, width=1.25)
    add_text(s, X2 - 0.35, y0 + 1.48, 1.7, 0.6, ["✕ pruned by", "eligibility mask"], size=13, color=AMBER,
             align=PP_ALIGN.CENTER)
    add_text(s, X3 - 0.05, y0 + 1.32, 1.5, 0.6, ["ID =", "[12, 407, 88]"], size=13, color=TEAL, bold=True)
    add_text(s, x0, y0 + 2.9, 5.3, 0.3, "Beam search over codes; highlighted path = one item.", size=13,
             color=MUTED)
    statements(s, x0, 4.95, 5.3, 1.0, [
        "At our 3M items the value is **different candidates**, not efficiency.",
    ])
    # figures
    figure(ctx, s, 11, "cqsid", 6.25, CONTENT_TOP, 3.15, 1.75, "framework overview", cap_h=0.5)
    figure(ctx, s, 11, "onesearch", 9.6, CONTENT_TOP, 3.15, 1.75, "framework overview", cap_h=0.5)
    tiles = [("+1.15%", "GMV, online A/B (CQ-SID)"),
             ("72.63%", "of purchases via the generative channel (CQ-SID)"),
             ("−75.40%", "operational expenditure (OneSearch)")]
    tx = 6.25
    for big, lab in tiles:
        box(s, tx, 4.25, 2.05, 1.5, fill=PANEL, line=BORDER, shape=MSO_SHAPE.RECTANGLE, line_w=0.75)
        add_text(s, tx + 0.15, 4.3, 1.8, 0.5, big, size=24, font=HEAD_FONT, color=TEAL)
        add_text(s, tx + 0.15, 4.8, 1.8, 0.95, lab, size=14, color=INK)
        tx += 2.2
    eq_label(s, x0, 6.0, 3.0, "SEMANTIC ID BY RQ")
    eq(ctx, s, "eq09_rq", x0 + 2.55, 5.92, 9.55, pt=18)
    finish_footer(s)
    s.notes_slide.notes_text_frame.text = (
        "Semantic IDs come from residual quantization of item embeddings: each level quantizes the residual of the "
        "previous one, and an item's ID is its code path. A generator decodes codes by beam search; our eligibility "
        "mask can prune branches (dashed) so only allowed items are generated. CQ-SID (Tmall): +1.15% GMV in a "
        "two-week online A/B; the generative channel accounts for 72.63% of purchases. OneSearch (Kuaishou): "
        "operational expenditure −75.40%. These come from the arXiv abstracts; page check pending. The brief's "
        "OneSearch '+3.22% orders' was only found in a secondary summary and is left off the slide. At our catalog "
        "size (3M) the reason to try this is candidate diversity, not serving cost.")


def s12_fusion(ctx, prs):
    s = new_slide(prs)
    chrome(ctx, s, 12, "Fusion", "RRF is a GAM with hand-fixed shape functions",
           "RRF: Cormack, Clarke & Büttcher (SIGIR 2009) · EBM: Nori et al. (2019) arXiv:1909.09223")
    eq_label(s, MARGIN, CONTENT_TOP, 7.5, "RECIPROCAL RANK FUSION")
    eq(ctx, s, "eq01_rrf", MARGIN, CONTENT_TOP + 0.4, 7.4, pt=24)
    eq_label(s, MARGIN, 3.3, 7.5, "FUSION AS A GAM (AN EBM GENERALIZES RRF)")
    eq(ctx, s, "eq02_gam", MARGIN, 3.7, 7.6, pt=22)
    statements(s, MARGIN, 4.85, 7.6, 1.8, [
        "An EBM **learns** the shape functions that RRF fixes by hand, plus pairwise terms.",
        "Same rank features in; our L1 already logs them.",
    ])
    rows = C.fmt_rrf()
    tx, ty, tw = 9.0, CONTENT_TOP, 3.75
    add_text(s, tx, ty, tw, 0.3, "RRF CONTRIBUTION, k = 60", size=13, color=MUTED, bold=True)
    tbl = s.shapes.add_table(len(rows) + 1, 2, Inches(tx), Inches(ty + 0.4), Inches(tw), Inches(0.55 * (len(rows) + 1))).table
    tbl.columns[0].width, tbl.columns[1].width = Inches(1.4), Inches(tw - 1.4)
    data = [("rank", "1 / (60 + rank)")] + [(str(r), v) for r, v in rows]
    for i, (a, b) in enumerate(data):
        for j, t in enumerate((a, b)):
            cell = tbl.cell(i, j)
            cell.fill.solid()
            cell.fill.fore_color.rgb = rgb(BG2 if i == 0 else PANEL)
            tf = cell.text_frame
            tf.paragraphs[0].text = ""
            r_ = tf.paragraphs[0].add_run()
            r_.text = t
            r_.font.size = Pt(BODY_PT)
            r_.font.name = BODY_FONT
            r_.font.bold = i == 0
            r_.font.color.rgb = rgb(INK)
            tf.paragraphs[0].alignment = PP_ALIGN.RIGHT if j == 1 else PP_ALIGN.LEFT
            cell.margin_left = cell.margin_right = Inches(0.15)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
    add_text(s, tx, ty + 0.5 + 0.55 * (len(rows) + 1), tw, 0.9,
             "Rank 1 counts only 2.6× rank 100: a flat, fixed curve.", size=15, color=MUTED)
    finish_footer(s)
    s.notes_slide.notes_text_frame.text = (
        "RRF gives each retriever a fixed contribution 1/(k + rank) with k = 60. Values: rank 1 → 0.0164, rank 10 → "
        "0.0143, rank 30 → 0.0111, rank 100 → 0.0063 (arithmetic; 0.0164/0.0063 ≈ 2.6). Written as a generalized "
        "additive model, RRF is one hand-chosen shape function per retriever with no interactions. An Explainable "
        "Boosting Machine fits those shape functions and selected pairwise interactions from data while staying "
        "inspectable, and our in-index L1 already logs the rank features it needs.")


def s13_production(ctx, prs):
    s = new_slide(prs)
    chrome(ctx, s, 13, "Production query understanding", "Industry moves LLM reasoning off the query path",
           "Case-driven multi-agent relevance, ByteDance (2026) arXiv:2605.05991 · Zhai et al. (2026) arXiv:2603.19665")
    figure(ctx, s, 13, "casedriven", MARGIN, CONTENT_TOP, 5.95, 2.75, "framework overview", cap_h=0.5)
    figure(ctx, s, 13, "genfacet", 6.8, CONTENT_TOP, 5.95, 2.75, "framework overview", cap_h=0.5)
    statements(s, MARGIN, 5.0, CW, 1.7, [
        "**GenFacet** (JD.com): generated facets, +42.0% facet CTR and +2.0% UCVR online.",
        "**Case-driven relevance** (ByteDance): Annotator, Optimizer and User agents learn from cases.",
        "Our read: the same **compile-once** pattern as RSA.",
    ], gap=6)
    finish_footer(s)
    s.notes_slide.notes_text_frame.text = (
        "GenFacet (SIGIR 2026, JD.com) generates facets end to end with multi-task preference alignment; online it "
        "reports +42.0% facet CTR and +2.0% UCVR (from the arXiv HTML; page check pending). The case-driven "
        "framework (ByteDance) uses Annotator, Optimizer and User agents to improve relevance from failure cases. "
        "The brief's 'nearline 7B query-structure LLM, attribute F1 0.887 → 0.926 over BERT' could not be tied to a "
        "specific paper and is left off the slide pending a PDF check (qa_report.md).")


def s14_statement(ctx, prs):
    s = new_slide(prs, dark=True)
    add_text(s, MARGIN + 0.4, 0.9, 9, 0.3, "WHERE OUR NOVELTY LIVES", size=13, color="8FC7C2", bold=True,
             char_spacing=1.5)
    add_text(s, MARGIN + 0.4, 1.9, 11.4, 3.2,
             "Our novelty lives in two places: compiled predicates and persistent intent state.",
             size=40, font=HEAD_FONT, color=LIGHT_TEXT, line_spacing=1.15)
    add_text(s, MARGIN + 0.4, 5.5, 11, 0.6,
             "Filtering speed, negation scoring and generative retrieval are crowded; we should reuse them.",
             size=20, color="C9C3B6")
    ctx.slide_of[14] = s._n
    add_text(s, SLIDE_W - MARGIN - 0.6, FOOTER_Y, 0.6, 0.3, str(s._n), size=FOOTER_PT, color="C9C3B6",
             align=PP_ALIGN.RIGHT)
    s.notes_slide.notes_text_frame.text = (
        "Synthesis. Fast filtered ANN (ACORN, FAVOR, EMA), negation-aware scoring (E-SENS) and generative "
        "retrieval (CQ-SID, OneSearch) are active and well-resourced; we should adopt them as components. What we "
        "did not find elsewhere: predicates that are learned, compiled to ~216 B and executed inside traversal; and "
        "a persistent, editable intent state that compiles to filters for every retriever.")


def s15_table(ctx, prs):
    s = new_slide(prs)
    chrome(ctx, s, 15, "Relevance", "Where each idea stands against the closest work",
           "Sources: slides 4–13 and the reading list (slide 17).")
    head = ["Our idea", "Closest recent work", "Our edge", "Gap to close"]
    rows = [
        ["Compiled predicates", "SAE concept features", "216 B program on a 56 B code", "Beat SAE-derived predicates"],
        ["Predicates in HNSW", "ACORN, FAVOR, EMA", "Learned predicate, 96–134 ns", "Head-to-head vs ACORN"],
        ["Negation algebra", "E-SENS", "Calibrated, composable log(1 − p)", "Beat E-SENS on its benchmark"],
        ["Semantic maps", "SAE clustering", "LLM-designed, readable axes", "Probe accuracy per axis"],
        ["Intent canvas", "Fashion IQ, FACap", "Persistent, editable state", "Multi-turn edit benchmark"],
        ["MPDM as a retriever", "CQ-SID, OneSearch", "Map cells as readable codes", "New candidates at 3M items"],
    ]
    widths = [2.55, 2.95, 3.4, 3.23]
    rh = 0.66
    t = s.shapes.add_table(len(rows) + 1, 4, Inches(MARGIN), Inches(CONTENT_TOP),
                           Inches(sum(widths)), Inches(rh * (len(rows) + 1))).table
    for j, w in enumerate(widths):
        t.columns[j].width = Inches(w)
    for i in range(len(rows) + 1):
        t.rows[i].height = Inches(rh)
        for j in range(4):
            cell = t.cell(i, j)
            txt = head[j] if i == 0 else rows[i - 1][j]
            cell.fill.solid()
            cell.fill.fore_color.rgb = rgb(TEAL if i == 0 else (PANEL if i % 2 else BG2))
            tf = cell.text_frame
            tf.word_wrap = True
            tf.paragraphs[0].text = ""
            r_ = tf.paragraphs[0].add_run()
            r_.text = txt
            r_.font.size = Pt(BODY_PT if i else 16)
            r_.font.name = BODY_FONT
            r_.font.bold = i == 0 or j == 0
            r_.font.color.rgb = rgb(LIGHT_TEXT if i == 0 else (TEAL if j == 0 else INK))
            cell.margin_left = cell.margin_right = Inches(0.12)
            cell.margin_top = cell.margin_bottom = Inches(0.04)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
    finish_footer(s)
    s.notes_slide.notes_text_frame.text = (
        "One row per idea. 'Closest recent work' is the paper that would be cited against us; 'Our edge' is what we "
        "can already show; 'Gap to close' is the experiment a reviewer would ask for first. Negation: E-SENS "
        "evaluates on ExcluIR, so 'its benchmark' means ExcluIR. Canvas: Fashion IQ and FACap are single-turn "
        "composed retrieval; nobody we found keeps a persistent multi-turn state.")


def s16_rivals(ctx, prs):
    s = new_slide(prs)
    chrome(ctx, s, 16, "Next experiments", "Stronger rivals to beat next",
           "Baselines: ACORN arXiv:2403.04871 · FAVOR arXiv:2605.07770 · E-SENS arXiv:2608.30130 · Fashion IQ arXiv:1905.12794")
    cards = [
        ("RSA", ["ACORN / FAVOR baseline", "E-SENS on negation", "SAE-derived predicates", "Anticorrelated-query slice"]),
        ("Semantic maps", ["Parametric UMAP and SOM baselines", "Probe accuracy per axis", "Axis decorrelation"]),
        ("Intent canvas", ["Multi-turn Fashion IQ edits", "Simulated shoppers", "Turns-to-target vs query rewriting",
                           "One-market pilot"]),
    ]
    cw, gap = 3.85, 0.3
    for i, (name, items) in enumerate(cards):
        x = MARGIN + i * (cw + gap)
        box(s, x, CONTENT_TOP, cw, 3.9, fill=PANEL, line=BORDER, shape=MSO_SHAPE.RECTANGLE, line_w=0.75)
        add_text(s, x + 0.3, CONTENT_TOP + 0.25, cw - 0.6, 0.5, name, size=22, font=HEAD_FONT, color=TEAL)
        add_text(s, x + 0.3, CONTENT_TOP + 0.95, cw - 0.6, 3.7, items, size=BODY_PT, space_after=14)
    finish_footer(s)
    s.notes_slide.notes_text_frame.text = (
        "RSA: compare inside ACORN/FAVOR-style filtered search rather than our own HNSW harness; beat E-SENS on "
        "negation; compile SAE features instead of teacher labels; report a slice where eligible items are far from "
        "the query (anticorrelated). Maps: parametric UMAP and SOM as layout baselines, per-axis probe accuracy, and "
        "decorrelation between axes. Canvas: multi-turn edits built from Fashion IQ, simulated shoppers, "
        "turns-to-target against a query-rewriting baseline, then a one-market pilot.")


LIST_NAMES = {"rsa": "RSA (our paper)", "sae_dense": "Decoding dense embeddings",
              "cmr_survey": "Composed retrieval survey", "casedriven": "Case-driven relevance",
              "fann_bench": "Filtered-ANN benchmark", "sae_embed": "SAE embeddings toolkit"}


def s17_reading(ctx, prs):
    from .paper_slides import FRONTS
    from .papers import SOURCES
    rows = [("head", "Our work"), ("paper", "rsa")]
    for fk, name, _ in FRONTS:
        rows.append(("head", name))
        rows += [("paper", p.key) for p in ctx.registry if p.front == fk]
    rows.append(("head", "Sources (no dedicated slides)"))
    rows += [("src", i) for i in range(len(SOURCES))]
    per_col, cols = 10, 2
    pages = [rows[i:i + per_col * cols] for i in range(0, len(rows), per_col * cols)]
    for pi, page in enumerate(pages):
        s = new_slide(prs)
        chrome(ctx, s, 17 if pi == 0 else f"reading:{pi}", "Reading list",
               f"Papers in this deck ({pi + 1}/{len(pages)})",
               "IDs and titles confirmed against arXiv listings (qa_report.md). Links open the abstract page.")
        for ri, (kind, v) in enumerate(page):
            x = MARGIN + (ri // per_col) * 6.2
            y = CONTENT_TOP + (ri % per_col) * 0.49
            if kind == "head":
                add_text(s, x, y + 0.08, 6.0, 0.4, v.upper(), size=13, color=TEAL, bold=True, char_spacing=1)
            elif kind == "src":
                name, url = SOURCES[v]
                add_text(s, x, y, 6.0, 0.45, name, size=16, color=TEAL, link=url)
            else:
                p = ctx.papers[v]
                add_text(s, x, y, 3.85, 0.45, f"**{LIST_NAMES.get(v, p.short)}**", size=16, accent=INK)
                if p.arxiv:
                    add_text(s, x + 3.9, y, 2.2, 0.45, f"arXiv:{p.arxiv}", size=16, color=TEAL, link=p.abs_url)
                else:
                    add_text(s, x + 3.9, y, 2.2, 0.45, "local PDF", size=16, color=MUTED)
        finish_footer(s)
        s.notes_slide.notes_text_frame.text = "Full titles:\n" + "\n".join(
            f"{ctx.papers[v].short}: {ctx.papers[v].title or ctx.papers[v].expected_title} "
            f"{ctx.papers[v].abs_url}" for k, v in page if k == "paper")


def _order(ctx):
    """Framing bookends around one section per research front."""
    from .paper_slides import build_front
    sec = lambda fk: (lambda c, prs: build_front(c, prs, fk))  # noqa: E731
    return [s01_cover, s02_landscape, s03_architecture, s06_rsa, s05_selectivity,
            sec("filtered_ann"), s04_filtered_ann,
            sec("negation"), s07_negation,
            sec("sae"), s08_sae, s09_maps,
            sec("cir"), s10_canvas,
            sec("genret"), s11_genretrieval,
            sec("production"), s13_production, s12_fusion,
            s14_statement, s15_table, s16_rivals, s17_reading]


def build(ctx: Ctx, out: Path) -> None:
    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(SLIDE_W), Inches(SLIDE_H)
    for fn in _order(ctx):
        fn(ctx, prs)
    prs.core_properties.title = "Fashion search: what the literature says"
    prs.core_properties.author = "Applied Science, Search & Discovery"
    out.parent.mkdir(parents=True, exist_ok=True)
    prs.save(out)
