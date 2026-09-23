"""Human-written per-paper slide content, filled in after reading each PDF.

CONTENT[key] = {
    "roles": ["method", "results", "us"],          # 2-3 slides; default ["method", "results"]
    "method":  {"title": ..., "figure": "Figure 2", "statements": [...], "notes": ...},
    "results": {"title": ..., "figure": "Table 3",  "statements": [...], "notes": ...},
    "us":      {"title": ..., "figure": ..., "statements": [...], "links_to": ["RSA", "Canvas"]},
}
Every number in a statement must also be registered in claims.py (tag "<key>:<role>")
so the build verifies it against the PDF and records the page. "figure" pins a crop by
label; omit it to let paper_slides.pick() choose by caption keywords.
"""
CONTENT: dict[str, dict] = {}
