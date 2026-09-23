# Visual QA log (read by build_deck.py into qa_report.md §5)

- Every slide was exported to PNG with LibreOffice headless (build/render/) and inspected for text overflow, overlaps, unreadable figures and cut-off equations.
- Round 1 fixes: default shape shadows removed; the "↔" glyph fell back to an emoji box in the font subset and was replaced by "–"; slide 3 QU label broke mid-word and the canvas arrow cut through the lexical box, so the canvas was redrawn as a container around both paths; slide 6 equations were too close to the footer; slide 7 crossing arrows were replaced by a join bar; slide 11 trie labels collided with the "pruned" note and the stat tiles touched the figure captions; slide 17 two-line names overlapped the next row; slide 16 cards were resized.
- Round 2 fixes: slide 3 container label wrapped into the lexical box (shortened, lanes moved down); slide 11 ID label overlapped the pruned-branch note (moved, two lines); slide 10 equation label moved clear of the axis labels.
- Final pass: no text overflow, overlaps or cut-off equations found on any of the 17 slides. Remaining visual gaps are the labeled figure placeholders, which are expected until arxiv.org is reachable.
- Diagram labels use 13–16 pt, and figure captions and footers use 12 pt. All statement text is 18 pt or larger.
- LibreOffice renders with the bundled IBM Plex Sans and Libre Baskerville TTFs (fonts/). Install them before opening the PPTX in PowerPoint, or Office will substitute Georgia/Arial.
