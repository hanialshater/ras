#!/usr/bin/env python3
"""Rebuild the "Fashion search: what the literature says" deck from scratch.

Steps: download + verify papers -> crop figures (manifest, contact sheet) ->
render equations -> verify every number against the PDFs -> build PPTX ->
export PDF + slide PNGs (LibreOffice) -> write qa_report.md.

    python build_deck.py            # full build
    python build_deck.py --offline  # skip network, reuse papers/ and papers/metadata.json
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from deckbuild import claims as claims_mod  # noqa: E402
from deckbuild import equations, figures, papers, qa, slides  # noqa: E402

DECK = ROOT / "deck" / "fashion_search_landscape.pptx"
FIG = ROOT / "figures"
EQ = ROOT / "equations"
RENDER = ROOT / "build" / "render"
DIAGRAM_SLIDES = {2: "d1_landscape_map_slide02", 3: "d2_search_architecture_slide03",
                  5: "d3_selectivity_chart_slide05", 7: "d6_negation_decomposition_slide07",
                  9: "semantic_map_axes_layers_slide09", 10: "d4_canvas_layer_algebra_slide10",
                  11: "d5_semantic_id_trie_slide11"}


def soffice() -> str | None:
    return shutil.which("soffice") or shutil.which("libreoffice")


def export_pdf_and_pngs(pptx: Path) -> list[str]:
    notes = []
    exe = soffice()
    if not exe:
        return ["LibreOffice not found: PDF/PNG export skipped."]
    RENDER.mkdir(parents=True, exist_ok=True)
    subprocess.run([exe, "--headless", "--convert-to", "pdf", "--outdir", str(pptx.parent), str(pptx)],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=300)
    pdf = pptx.with_suffix(".pdf")
    for old in RENDER.glob("slide-*.png"):
        old.unlink()
    if shutil.which("pdftoppm"):
        subprocess.run(["pdftoppm", "-png", "-r", "110", str(pdf), str(RENDER / "slide")], check=True)
        # SVG renderings of the slides that carry native diagrams (the editable source is the PPTX itself)
        if shutil.which("pdftocairo"):
            dd = ROOT / "diagrams"
            dd.mkdir(exist_ok=True)
            for n, name in DIAGRAM_SLIDES.items():
                subprocess.run(["pdftocairo", "-svg", "-f", str(n), "-l", str(n), str(pdf),
                                str(dd / f"{name}.svg")], check=True)
        notes.append(f"Exported {pdf.name} and {len(list(RENDER.glob('slide-*.png')))} slide PNGs "
                     f"to build/render/ (LibreOffice headless + pdftoppm).")
    return notes


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true", help="do not touch the network")
    ap.add_argument("--no-render", action="store_true", help="skip LibreOffice export")
    a = ap.parse_args()

    print("1/6 papers: download + verify titles/licenses")
    reg = papers.fetch_all(allow_network=not a.offline)
    for p in reg:
        print(f"    {p.key:12s} {p.arxiv or 'local':12s} {p.pdf_status[:60]}")

    print("2/6 figures: caption-anchored crops at 4x")
    for f in FIG.glob("*.png"):
        f.unlink()
    cands = []
    for p in reg:
        cands += figures.extract(p, ROOT, FIG)
    figures.write_manifest(reg, cands, FIG / "manifest.csv")
    figures.contact_sheet(reg, cands, FIG / "contact_sheet.html")
    print(f"    {len(cands)} candidate crops")

    print("3/6 equations")
    equations.render_all(EQ)

    print("4/6 numbers: verify against PDFs")
    cl = claims_mod.check_all(papers.BY_KEY, ROOT)
    bad = [c for c in cl if c.show and c.status.startswith("NOT FOUND")]
    for c in bad:
        print(f"    !! not found in PDF: slide {c.slide}: {c.text}")

    print("5/6 deck")
    ctx = slides.Ctx(ROOT, papers.BY_KEY, cands, EQ, FIG)
    slides.build(ctx, DECK)
    print(f"    wrote {DECK.relative_to(ROOT)}")

    print("6/6 export + QA report")
    render_notes = [] if a.no_render else export_pdf_and_pngs(DECK)
    layout_notes_file = ROOT / "deckbuild" / "visual_qa_notes.md"
    layout_notes = [l[2:].strip() for l in layout_notes_file.read_text().splitlines() if l.startswith("- ")] \
        if layout_notes_file.exists() else []
    qa.write(ROOT / "qa_report.md", reg, cands, cl, ctx.figure_log, layout_notes, render_notes)
    print("    wrote qa_report.md")


if __name__ == "__main__":
    main()
