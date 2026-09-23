"""Caption-anchored figure/table extraction with PyMuPDF.

For each caption block (``^(Figure|Fig.)\\s*N`` or ``^Table\\s*N``):
  * figures: region = union of image + vector-drawing boxes between the caption
    and the previous body-text paragraph above it (falls back to below the
    caption for papers that caption on top);
  * tables: region = rules + text rows between the caption and the next body
    paragraph below it (falls back to above).
The region is rendered at 4x zoom and whitespace-trimmed. Crops are never
altered beyond cropping and scaling.
"""
from __future__ import annotations

import csv
import html
import re
from dataclasses import dataclass
from pathlib import Path

import pymupdf
from PIL import Image, ImageChops

from .papers import Paper

CAP_RE = re.compile(r"^(Figure|Fig\.)\s*(\d+)\s*[:.|]?", re.I)
TAB_RE = re.compile(r"^(Table)\s*(\d+)\s*[:.|]?", re.I)
ZOOM = 4


@dataclass
class Candidate:
    paper: str
    kind: str          # "fig" | "tab"
    number: int
    page: int          # 1-based
    caption: str
    rect: tuple
    file: str = ""
    score: float = 0.0
    selected: bool = False
    flag: str = ""

    @property
    def label(self) -> str:
        return f"{'Figure' if self.kind == 'fig' else 'Table'} {self.number}"


def _is_caption_block(text: str, kind_re: re.Pattern) -> re.Match | None:
    m = kind_re.match(text)
    if not m:
        return None
    # Reject in-text references such as "Table 1 and Figure 2 summarize ..."
    rest = text[m.end():m.end() + 40]
    if m.group(0).rstrip().endswith((":", ".", "|")):
        return m
    if re.match(r"^\s*(and|shows|summari[sz]es|reports|makes|compares|lists|presents)\b", rest):
        return None
    return m


def _body_paragraphs(blocks, graphics_rects):
    """Text blocks that look like running prose (not labels inside a figure)."""
    out = []
    for b in blocks:
        r = pymupdf.Rect(b[:4])
        txt = b[4].strip()
        if len(txt) < 120 or r.width < 150:
            continue
        lines = [l for l in txt.splitlines() if l.strip()]
        letters = sum(ch.isalpha() for ch in txt) / max(1, len(txt))
        if letters < 0.6 or len(txt) / max(1, len(lines)) < 25:   # table cells, axis ticks
            continue
        if any(r.intersects(g) and (r & g).get_area() > 0.5 * r.get_area() for g in graphics_rects):
            continue
        out.append(r)
    return out


def _column_span(page, cap: pymupdf.Rect) -> tuple[float, float]:
    W = page.rect.width
    if cap.width > 0.55 * W or (cap.x0 < W * 0.4 and cap.x1 > W * 0.6):
        return 0, W
    return (0, W / 2) if cap.x1 <= W * 0.55 else (W / 2, W)


def _running_margins(doc) -> tuple[float, float]:
    """(header_bottom, footer_top): y-limits of text repeated on most pages."""
    from collections import Counter
    cnt, pos = Counter(), {}
    for page in doc:
        for b in page.get_text("blocks"):
            key = (re.sub(r"\d+", "#", b[4].strip())[:40], round(b[1] / 4))
            cnt[key] += 1
            pos[key] = (b[1], b[3])
    H = doc[0].rect.height
    need = max(3, len(doc) // 2)
    top = [pos[k][1] for k, n in cnt.items() if n >= need and pos[k][1] < H * 0.15]
    bot = [pos[k][0] for k, n in cnt.items() if n >= need and pos[k][0] > H * 0.85]
    return (max(top) + 4 if top else 0.0), (min(bot) - 2 if bot else H)


MARGINS = (0.0, 1e9)


def _graphics(page):
    rects = []
    for d in page.get_drawings():
        r = pymupdf.Rect(d["rect"])
        if r.width < 1 and r.height < 1:
            continue
        if r.height < 1:     # horizontal rule (e.g. booktabs): give it thickness
            r = pymupdf.Rect(r.x0, r.y0 - 0.5, r.x1, r.y1 + 0.5)
        if r.width < 1:
            r = pymupdf.Rect(r.x0 - 0.5, r.y0, r.x1 + 0.5, r.y1)
        if r.y1 <= MARGINS[0] or r.y0 >= MARGINS[1]:
            continue
        # drop full-page frames/backgrounds
        if r.width > 0.95 * page.rect.width and r.height > 0.9 * page.rect.height:
            continue
        rects.append(r)
    for info in page.get_image_info():
        rects.append(pymupdf.Rect(info["bbox"]))
    return rects


def _region(page, cap: pymupdf.Rect, kind: str, barriers=()) -> pymupdf.Rect | None:
    blocks = [b for b in page.get_text("blocks")
              if b[6] == 0 and b[3] > MARGINS[0] and b[1] < MARGINS[1]]
    gfx = _graphics(page)
    x0c, x1c = _column_span(page, cap)
    in_col = lambda r: r.x1 > x0c + 2 and r.x0 < x1c - 2  # noqa: E731
    paras = [r for r in _body_paragraphs(blocks, gfx) if in_col(r)]

    def collect(y_lo, y_hi, from_top: bool, gap: float = 16.0):
        """Grow a region from the caption edge through vertically contiguous elements."""
        elems = [g for g in gfx if in_col(g) and g.y0 >= y_lo - 1 and g.y1 <= y_hi + 1
                 and g.height < page.rect.height * 0.95]
        elems += [pymupdf.Rect(b[:4]) for b in blocks
                  if in_col(pymupdf.Rect(b[:4])) and b[1] >= y_lo - 1 and b[3] <= y_hi + 1
                  and not any(pymupdf.Rect(b[:4]) == p for p in paras)]
        if kind == "fig" and not any(in_col(g) and g.y0 >= y_lo - 1 and g.y1 <= y_hi + 1 for g in gfx):
            return pymupdf.Rect()
        is_rule = lambda r: r.height <= 2.5  # noqa: E731
        is_text = lambda r: any(pymupdf.Rect(b[:4]) == r for b in blocks)  # noqa: E731
        u = pymupdf.Rect()
        taken = []
        edge = y_lo if from_top else y_hi
        elems.sort(key=lambda r: r.y0 if from_top else -r.y1)
        for r in elems:
            dist = (r.y0 - (u.y1 if not u.is_empty else edge)) if from_top else \
                   ((u.y0 if not u.is_empty else edge) - r.y1)
            if dist > gap:
                break
            if any(r.intersects(bar) for bar in barriers):
                break
            if kind == "tab" and not (is_rule(r) or is_text(r)):
                break          # tables are text + rules; a plot/image means we left the table
            u |= r
            taken.append(r)
        if kind == "tab":
            rules = [r for r in taken if is_rule(r)]
            if len(rules) >= 2:        # booktabs: the table ends at its outer rules
                if from_top:
                    u.y1 = max(r.y1 for r in rules)
                else:
                    u.y0 = min(r.y0 for r in rules)
        return u

    def band_above():
        prev = [p.y1 for p in paras if p.y1 <= cap.y0 + 1]
        top = max(prev) if prev else max(page.rect.y0 + 20, MARGINS[0])
        return collect(top, cap.y0, from_top=False)

    def band_below():
        nxt = [p.y0 for p in paras if p.y0 >= cap.y1 - 1]
        bot = min(nxt) if nxt else min(page.rect.y1 - 20, MARGINS[1])
        return collect(cap.y1, bot, from_top=True)

    order = (band_below, band_above) if kind == "tab" else (band_above, band_below)
    for fn in order:
        r = fn()
        if not r.is_empty and r.height > 20 and r.width > 60:
            return r
    return None


def _trim(img: Image.Image, pad: int = 12) -> Image.Image:
    rgb = img.convert("RGB")
    bg = Image.new("RGB", rgb.size, (255, 255, 255))
    diff = ImageChops.difference(rgb, bg).convert("L").point(lambda v: 255 if v > 12 else 0)
    box = diff.getbbox()
    if not box:
        return img
    l, t, r, b = box
    return img.crop((max(0, l - pad), max(0, t - pad), min(img.width, r + pad), min(img.height, b + pad)))


def extract(paper: Paper, root: Path, out_dir: Path) -> list[Candidate]:
    if not paper.pdf:
        return []
    global MARGINS
    doc = pymupdf.open(root / paper.pdf)
    MARGINS = _running_margins(doc)
    cands: list[Candidate] = []
    seen = set()
    for pno, page in enumerate(doc):
        caps = []
        for b in page.get_text("blocks"):
            if b[6] != 0:
                continue
            text = " ".join(b[4].split())
            for kind, rx in (("tab", TAB_RE), ("fig", CAP_RE)):
                m = _is_caption_block(text, rx)
                if m:
                    caps.append((kind, int(m.group(2)), pymupdf.Rect(b[:4]), text))
        done = []
        for kind, num, cap, text in sorted(caps, key=lambda c: c[0] != "tab"):
            if (kind, num) in seen:
                continue
            others = [c[2] for c in caps if c[2] != cap]
            reg = _region(page, cap, kind, barriers=others + done)
            if reg is None:
                continue
            seen.add((kind, num))
            done.append(reg)
            cands.append(Candidate(paper.key, kind, num, pno + 1, text, tuple(reg)))
    cands.sort(key=lambda c: (c.page, c.kind, c.number))
    out_dir.mkdir(parents=True, exist_ok=True)
    for c in cands:
        page = doc[c.page - 1]
        clip = pymupdf.Rect(c.rect) + (-4, -4, 4, 4)
        clip &= page.rect
        pix = page.get_pixmap(matrix=pymupdf.Matrix(ZOOM, ZOOM), clip=clip, alpha=False)
        img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        img = _trim(img)
        name = f"{paper.key}_{c.kind}{c.number}.png"
        img.save(out_dir / name)
        c.file = name
    _select(paper, cands)
    return cands


def _select(paper: Paper, cands: list[Candidate]) -> None:
    if paper.labels:
        for c in cands:
            if c.label in paper.labels:
                c.selected, c.score = True, 100
        missing = set(paper.labels) - {c.label for c in cands}
        if not missing:
            return
    for c in cands:
        cap = c.caption.lower()
        c.score = max(c.score, sum(len(paper.keywords) - i for i, k in enumerate(paper.keywords) if k.lower() in cap)
                      + (0.5 if c.kind == "fig" else 0) + (0.25 if c.number == 1 else 0))
    ranked = sorted(cands, key=lambda c: -c.score)
    if ranked and not any(c.selected for c in cands):
        ranked[0].selected = True
        if len(ranked) > 1 and ranked[1].score >= ranked[0].score - 1:
            ranked[0].flag = ranked[1].flag = "top-2 candidates are close; review contact sheet"


MANIFEST_COLS = ["paper", "arxiv_id", "figure_label", "page", "caption_text", "license", "license_url", "file"]


def write_manifest(papers: list[Paper], cands: list[Candidate], path: Path) -> None:
    by = {p.key: p for p in papers}
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(MANIFEST_COLS + ["selected", "flag"])
        for c in cands:
            p = by[c.paper]
            w.writerow([p.key, p.arxiv or "local", c.label, c.page, c.caption, p.license,
                        p.license_url, c.file, "yes" if c.selected else "", c.flag])
        for p in papers:
            if not any(c.paper == p.key for c in cands):
                w.writerow([p.key, p.arxiv or "local", "", "", f"NOT EXTRACTED ({p.pdf_status}); wanted: {p.want}",
                            p.license, p.license_url, "", "", "placeholder"])


def contact_sheet(papers: list[Paper], cands: list[Candidate], path: Path) -> None:
    rows = []
    for p in papers:
        cs = [c for c in cands if c.paper == p.key]
        rows.append(f"<h2>{html.escape(p.key)} — {html.escape(p.title or p.expected_title)} "
                    f"<small>({p.arxiv or 'local'}; license: {html.escape(p.license)})</small></h2>")
        if not cs:
            rows.append(f"<p class=miss>No crops: {html.escape(p.pdf_status)}. Wanted: {html.escape(p.want)}</p>")
        for c in cs:
            cls = "sel" if c.selected else ""
            rows.append(
                f"<figure class='{cls}'><img src='{c.file}' loading=lazy>"
                f"<figcaption><b>{c.label}</b> · p.{c.page} · score {c.score:.1f}"
                f"{' · <b>SELECTED</b>' if c.selected else ''}{' · ⚠ ' + html.escape(c.flag) if c.flag else ''}"
                f"<br>{html.escape(c.caption[:400])}</figcaption></figure>")
    path.write_text(f"""<!doctype html><html><head><meta charset=utf-8><title>Figure contact sheet</title>
<style>body{{font:14px/1.4 system-ui,sans-serif;margin:24px;background:#F5F2EC;color:#1E2328}}
figure{{display:inline-block;vertical-align:top;width:420px;margin:8px;padding:8px;background:#FFFDF8;border:1px solid #DDD6C8}}
figure.sel{{border:3px solid #1F6F6B}} img{{max-width:100%;max-height:320px;display:block;margin:auto}}
.miss{{color:#A5541B}} h2{{font-size:17px;margin-top:28px}}</style></head><body>
<h1>Figure contact sheet</h1><p>Teal border = selected for the deck. Crops are 4× renders, whitespace-trimmed, otherwise unaltered.</p>
{''.join(rows)}</body></html>""", encoding="utf-8")
