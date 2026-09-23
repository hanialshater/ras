# Fashion search: the research landscape (deck)

A 17-slide, 16:9 deck that compares our three ideas (compiled semantic predicates / RSA, semantic maps / MPDM, intent canvas) with 2025–2026 literature. Everything in it is rebuilt by one script.

| Output | Where |
|---|---|
| Editable deck (native text, native diagrams, native chart and tables, speaker notes) | `deck/fashion_search_landscape.pptx` |
| PDF export (LibreOffice) | `deck/fashion_search_landscape.pdf` |
| Figure crops (4× render, whitespace-trimmed), manifest, review sheet | `figures/*.png`, `figures/manifest.csv`, `figures/contact_sheet.html` |
| Equations: LaTeX source, SVG and 3× PNG (300 dpi at 22 pt) | `equations/<key>.tex/.svg/@3x.png`, `equations/equations.tex` |
| Diagrams | native PowerPoint shapes in the PPTX (editable); SVG renderings in `diagrams/` |
| Verification log | `qa_report.md` |

## Rebuild

```bash
pip install pymupdf python-pptx matplotlib pillow lxml
# system: LibreOffice Impress (PDF export), poppler-utils (pdftoppm/pdftocairo),
#         a TeX distribution with cm-super + dvipng + dvisvgm (equations; falls back to mathtext)
python build_deck.py            # downloads papers, verifies, crops, renders, builds, exports, writes QA
python build_deck.py --offline  # no network; reuses papers/ and papers/metadata.json
```

Pipeline (`deckbuild/`):

1. `papers.py` downloads each arXiv PDF into `papers/`, reads the title and license from the abstract page, and compares the title with the expected one. If the RSA PDF is missing, it is compiled from `paper/icml/`.
2. `figures.py` finds `Figure N` / `Table N` caption blocks with PyMuPDF. The figure region is the union of image and vector-drawing boxes grown from the caption toward the previous paragraph. Tables are grown the other way and clipped at booktabs rules. Each region is rendered at 4× and trimmed. Crops are chosen by explicit label or by caption keywords; when two candidates are close, both are flagged in the contact sheet. The script never alters a figure beyond cropping and scaling.
3. `equations.py` renders the 12 equations with real LaTeX, in ink `#1E2328` on a transparent background.
4. `claims.py` holds every number and paper-attributed claim on the slides, with regexes that are searched in the source PDF text. Page numbers go to `qa_report.md`. Claims that cannot be verified are kept off the slides (`show=False`).
5. `slides.py` builds the slides; `pptx_kit.py` holds the design tokens and shape helpers.
6. `qa.py` writes `qa_report.md`. `build_deck.py` exports the PDF, per-slide PNGs (`build/render/`) and diagram SVGs.

## Current state (2026-09-23): read `qa_report.md` §0

The build container's network policy blocked `arxiv.org`, so **no third-party PDF could be downloaded**. Titles and IDs were confirmed through web search, and licenses are `unknown`. All external figures are labeled placeholders (`[Figure: <paper>, <what it should show>]`). External numbers are backed by arXiv-abstract text only, without page numbers. Once `arxiv.org` is reachable, `python build_deck.py` fills in real crops, licenses, page-verified numbers and the internal-use footers automatically. Slide positions for the figures are already reserved.

## Fonts

Headings use **Libre Baskerville** and body text uses **IBM Plex Sans** (both SIL OFL, TTFs in `fonts/`). python-pptx cannot embed fonts, so install the two families before presenting. Otherwise PowerPoint substitutes a default font; Georgia and Arial are the intended fallbacks. The QA renders used these exact fonts.

## Design rules implemented

- Palette: background `#F5F2EC` / `#EAE5DB`, ink `#1E2328`, muted `#5B636B`, teal `#1F6F6B`, amber `#A5541B`, dark statement slide `#1E2328`.
- Kicker, title, footer and page number sit at fixed positions on every content slide.
- Statement text is at least 18 pt; footers and captions are 12 pt; diagram labels are 13–16 pt.
- Figures sit on a `#FFFDF8` panel with a `#DDD6C8` border, with the caption "Figure N from Author et al. (Year), arXiv:ID, p. P. License: …". A figure whose license is not CC BY / CC BY-SA / CC0 adds "Internal use only. Figure © the authors." to the slide footer.
