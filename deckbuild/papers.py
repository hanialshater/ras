"""Paper registry, PDF download, arXiv title/license verification.

Every network step degrades gracefully: when arxiv.org is unreachable the
registry's `offline` fields (verified by web search on 2026-09-23) are used and
the paper is marked "PDF not available" so downstream steps insert placeholders
instead of figures and flag numbers as not page-verified.
"""
from __future__ import annotations

import html
import json
import re
import shutil
import subprocess
import urllib.error
import urllib.request
from dataclasses import dataclass, field, asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PAPERS = ROOT / "papers"
UA = {"User-Agent": "fashion-search-deck-builder/1.0 (research deck; contact repo owner)"}


@dataclass
class Paper:
    key: str
    arxiv: str | None          # None for the local RSA manuscript
    expected_title: str
    short: str                 # short name for the reading list
    authors: str               # "Surname et al." as confirmed by title search
    year: int
    # What figure to take and how to find it (caption keywords, in priority order).
    want: str
    labels: list[str] = field(default_factory=list)      # explicit labels, e.g. ["Figure 1"]
    keywords: list[str] = field(default_factory=list)    # caption keywords if no label known
    id_note: str = ""          # ID corrections found during verification
    front: str = ""            # research front (section) this paper belongs to
    alt_pdf: str = ""          # non-arXiv PDF URL tried if arXiv fails (e.g. ACL Anthology)
    # Filled at run time
    title: str = ""
    title_source: str = ""
    license: str = "unknown"
    license_url: str = ""
    license_source: str = ""
    pdf: str = ""
    pdf_status: str = ""

    @property
    def cite(self) -> str:
        return f"{self.authors} ({self.year})"

    @property
    def abs_url(self) -> str:
        return f"https://arxiv.org/abs/{self.arxiv}" if self.arxiv else ""


REGISTRY: list[Paper] = [
    Paper("rsa", None,
          "Random Semantic Algebra: Compiling Latent Search Predicates over Low-Bit Substrates",
          "RSA: Compiled Semantic Predicates for ANN Search", "Al-Shater", 2026,
          "compiled semantic execution pipeline; candidate-budget frontier; live predicates inside HNSW",
          labels=["Figure 1", "Figure 4", "Table 7"]),
    Paper("acorn", "2403.04871",
          "ACORN: Performant and Predicate-Agnostic Search Over Vector Embeddings and Structured Data",
          "ACORN", "Patel et al.", 2024,
          "predicate-agnostic search / two-hop neighbor expansion, or recall vs. selectivity",
          keywords=["predicate subgraph", "two-hop", "neighbor expansion", "selectivity", "ACORN"]),
    Paper("curator", "2601.01291",
          "Curator: Efficient Vector Search with Low-Selectivity Filters",
          "Curator", "Jin et al.", 2026, "method overview",
          keywords=["overview", "architecture", "Curator", "index", "partition"]),
    Paper("favor", "2605.07770",
          "FAVOR: Efficient Filter-Agnostic Vector ANNS Based on Selectivity-Aware Exclusion Distances",
          "FAVOR", "Song et al.", 2026, "method overview",
          keywords=["overview", "framework", "FAVOR", "exclusion distance", "workflow"],
          id_note="No ID in brief; found by title search: 2605.07770"),
    Paper("ema", "2606.00734",
          "EMA: Approximate Nearest Neighbor Search with General Attribute Filtering and Dynamic Updates",
          "EMA", "EMA authors", 2026, "method overview",
          keywords=["overview", "framework", "EMA", "architecture", "index"],
          id_note="No ID in brief; found by title search: 2606.00734 (first author not visible in search results)"),
    Paper("fann_bench", "2507.21989",
          "Benchmarking Filtered Approximate Nearest Neighbor Search Algorithms on Transformer-based Embedding Vectors",
          "Filtered-ANN benchmark", "Iff et al.", 2025, "benchmark overview",
          keywords=["overview", "benchmark", "selectivity", "QPS"]),
    Paper("nevir_repro", "2502.13506",
          "Reproducing NevIR: Negation in Neural Information Retrieval",
          "Reproducing NevIR", "van den Elsen et al.", 2025,
          "main results showing negation performance",
          keywords=["pairwise accuracy", "NevIR", "results", "negation", "random"]),
    Paper("esens", "2608.30130",
          "E-SENS: Exclusion-Sensitive Penalization for Negative-Constraint Retrieval",
          "E-SENS", "Kim et al.", 2026, "method overview (query decomposition and trap penalty)",
          keywords=["overview", "trap", "decompos", "E-SENS", "framework"]),
    Paper("sae_embed", "2512.10092",
          "Interpretable Embeddings with Sparse Autoencoders: A Data Analysis Toolkit",
          "SAE embeddings toolkit", "Jiang et al.", 2025,
          "Figure 1: converting documents into interpretable SAE embeddings",
          labels=["Figure 1"], keywords=["SAE", "interpretable", "embedding"]),
    Paper("sae_dense", "2506.00041",
          "Decoding Dense Embeddings: Sparse Autoencoders for Interpreting and Discretizing Dense Retrieval",
          "Decoding dense embeddings (CL-SR)", "Park et al.", 2025, "method overview",
          keywords=["overview", "CL-SR", "framework", "sparse autoencoder", "pipeline"]),
    Paper("fashioniq", "1905.12794",
          "Fashion IQ: A New Dataset Towards Retrieving Images by Natural Language Feedback",
          "Fashion IQ", "Wu et al.", 2019,
          "dataset examples (reference image, relative caption, target image)",
          keywords=["example", "relative caption", "reference", "target", "dataset"]),
    Paper("cmr_survey", "2503.01334",
          "Composed Multi-modal Retrieval: A Survey of Approaches and Applications",
          "Composed multi-modal retrieval survey", "Zhang et al.", 2025, "taxonomy",
          keywords=["taxonomy", "overview", "composed"]),
    Paper("facap", "2507.07135",
          "FACap: A Large-Scale Fashion Dataset for Fine-Grained Composed Image Retrieval",
          "FACap", "Gardères et al.", 2025, "dataset examples",
          keywords=["example", "FACap", "dataset", "caption"]),
    Paper("cqsid", "2605.14434",
          "Efficient Generative Retrieval for E-commerce Search with Semantic Cluster IDs and Expert-Guided RL",
          "CQ-SID (Tmall)", "Tmall authors", 2026, "framework / architecture overview",
          keywords=["overview", "framework", "architecture", "semantic ID", "pipeline"],
          id_note="Brief gave no title; arXiv title confirmed by search"),
    Paper("onesearch", "2509.03236",
          "OneSearch: A Preliminary Exploration of the Unified End-to-End Generative Framework for E-commerce Search",
          "OneSearch (Kuaishou)", "Kuaishou authors", 2025, "framework / architecture overview",
          keywords=["overview", "framework", "OneSearch", "architecture"],
          id_note="Brief gave a description; arXiv title confirmed by search"),
    Paper("genfacet", "2603.19665",
          "GenFacet: End-to-End Generative Faceted Search via Multi-Task Preference Alignment in E-Commerce",
          "GenFacet (JD.com)", "Zhai et al.", 2026, "framework overview",
          keywords=["overview", "framework", "GenFacet", "architecture"]),
    Paper("casedriven", "2605.05991",
          "A Case-Driven Multi-Agent Framework for E-Commerce Search Relevance",
          "Case-driven multi-agent relevance", "ByteDance Global E-Commerce Search Relevance Team", 2026,
          "framework overview",
          keywords=["overview", "framework", "agent", "architecture"]),
]
BY_KEY = {p.key: p for p in REGISTRY}

# ---- second batch (user's reading list, 2026-09-23); IDs/titles confirmed by title search ----
_NEW = [
    ("attr_survey", "2508.16263", "Attribute Filtering in Approximate Nearest Neighbor Search: An In-depth Experimental Study",
     "Attribute-filtering study", "Li et al.", 2025, "filtered_ann"),
    ("fann_sys", "2602.11443", "Filtered Approximate Nearest Neighbor Search in Vector Databases: System Design and Performance Analysis",
     "Filtered ANN in vector DBs", "Amanbayev et al.", 2026, "filtered_ann"),
    ("excise", "2608.05497", "EXCISE: Query-Side Exclusion for Late-Interaction Retrieval",
     "EXCISE", "Ali et al.", 2026, "negation"),
    ("neg_taxonomy", "2507.22337", "A Comprehensive Taxonomy of Negation for NLP and Neural Retrievers",
     "Negation taxonomy", "Petcu et al.", 2025, "negation"),
    ("promptriever", "2409.11136", "Promptriever: Instruction-Trained Retrievers Can Be Prompted Like Language Models",
     "Promptriever", "Weller et al.", 2024, "negation"),
    ("mfollowir", "2501.19264", "mFollowIR: a Multilingual Benchmark for Instruction Following in Retrieval",
     "mFollowIR (ECIR '25)", "Weller et al.", 2025, "negation"),
    ("sae_splade", "2604.21511", "From Tokens to Concepts: Leveraging SAE for SPLADE",
     "SAE for SPLADE", "Zong et al.", 2026, "sae"),
    ("xetrieval", "2605.29507", "Xetrieval: Mechanistically Explaining Dense Retrieval",
     "Xetrieval", "Cai et al.", 2026, "sae"),
    ("fashionmv", "2604.10297", "FashionMV: Product-Level Composed Image Retrieval with Multi-View Fashion Data",
     "FashionMV", "Yuan et al.", 2026, "cir"),
    ("fire_cir", "2604.09114", "FIRE-CIR: Fine-grained Reasoning for Composed Fashion Image Retrieval",
     "FIRE-CIR (CVPR '26)", "Gardères et al.", 2026, "cir"),
    ("zs_cir", "2506.06602", "Zero Shot Composed Image Retrieval",
     "Zero-shot CIR", "Kakarla et al.", 2025, "cir"),
    ("onesearch_v2", "2603.24422", "OneSearch-V2: The Latent Reasoning Enhanced Self-distillation Generative Search Framework",
     "OneSearch-V2 (Kuaishou)", "Chen et al.", 2026, "genret"),
    ("varg", "2609.14493", "VARG: Value-Aware and Ranking-Aligned Generative Retrieval for Dynamic E-commerce Search",
     "VARG (Tmall)", "Chu et al.", 2026, "genret"),
    ("forge", "2509.20904", "FORGE: Forming Semantic Identifiers for Generative Retrieval in Industrial Datasets",
     "FORGE", "Fu et al.", 2025, "genret"),
    ("conv_semsearch", "2601.16492", "LLM-based Semantic Search for Conversational Queries in E-commerce",
     "Conversational semantic search", "Siddiqui et al.", 2026, "production"),
    ("beyond_rel", "2609.23646", "Beyond Relevance: Structured Semantic Supervision for Product Search with LLM-Augmented Annotations",
     "Beyond Relevance", "Koushik et al.", 2026, "production"),
    ("conv_rec", "2608.27006", "Conversational Recommendation over Live E-Commerce Catalogues with Self-Refreshing Retrieval",
     "Live-catalogue conv. rec.", "Kapetanovic et al.", 2026, "production"),
]
FRONT_OF = {
    "acorn": "filtered_ann", "curator": "filtered_ann", "fann_bench": "filtered_ann", "favor": "filtered_ann",
    "ema": "filtered_ann", "nevir_repro": "negation", "esens": "negation", "sae_embed": "sae", "sae_dense": "sae",
    "fashioniq": "cir", "cmr_survey": "cir", "facap": "cir", "cqsid": "genret", "onesearch": "genret",
    "genfacet": "production", "casedriven": "production",
}
for key, aid, title, short, authors, year, front in _NEW:
    REGISTRY.append(Paper(key, aid, title, short, authors, year, "method overview; main results",
                          keywords=["overview", "framework", "architecture", "pipeline", "illustration"],
                          front=front))
for p in REGISTRY:
    p.front = p.front or FRONT_OF.get(p.key, "")
BY_KEY["sae_dense"].alt_pdf = "https://aclanthology.org/2025.emnlp-main.1345.pdf"
BY_KEY["ema"].authors = "EMA authors"
BY_KEY.update({p.key: p for p in REGISTRY})

# Sources cited but not given their own slides (not papers).
SOURCES = [
    ("pith.science citation list for FAVOR / EMA", "https://pith.science/citations/1389e938-16cc-45bd-9734-e5b7e409312b"),
    ("Voxel51: Composed image retrieval at CVPR 2025", "https://voxel51.com/blog/composed-image-retrieval-at-cvpr-2025"),
    ("LLMSearchRecommender compendium", "https://github.com/alopatenko/LLMSearchRecommender"),
    ("interp_embed code (Jiang et al.)", "https://github.com/nickjiang2378/interp_embed"),
]

# Titles as confirmed by arXiv title search (web search results pointing at the
# arxiv.org/abs page) on 2026-09-23, used when arxiv.org itself is unreachable.
OFFLINE_TITLES = {
    "acorn": "ACORN: Performant and Predicate-Agnostic Search Over Vector Embeddings and Structured Data",
    "curator": "Curator: Efficient Vector Search with Low-Selectivity Filters",
    "favor": "FAVOR: Efficient Filter-Agnostic Vector ANNS Based on Selectivity-Aware Exclusion Distances",
    "ema": "EMA: Approximate Nearest Neighbor Search with General Attribute Filtering and Dynamic Updates",
    "fann_bench": "Benchmarking Filtered Approximate Nearest Neighbor Search Algorithms on Transformer-based Embedding Vectors",
    "nevir_repro": "Reproducing NevIR: Negation in Neural Information Retrieval",
    "esens": "E-SENS: Exclusion-Sensitive Penalization for Negative-Constraint Retrieval",
    "sae_embed": "Interpretable Embeddings with Sparse Autoencoders: A Data Analysis Toolkit",
    "sae_dense": "Decoding Dense Embeddings: Sparse Autoencoders for Interpreting and Discretizing Dense Retrieval",
    "fashioniq": "Fashion IQ: A New Dataset Towards Retrieving Images by Natural Language Feedback",
    "cmr_survey": "Composed Multi-modal Retrieval: A Survey of Approaches and Applications",
    "facap": "FACap: A Large-scale Fashion Dataset for Fine-grained Composed Image Retrieval",
    "cqsid": "Efficient Generative Retrieval for E-commerce Search with Semantic Cluster IDs and Expert-Guided RL",
    "onesearch": "OneSearch: A Preliminary Exploration of the Unified End-to-End Generative Framework for E-commerce Search",
    "genfacet": "GenFacet: End-to-End Generative Faceted Search via Multi-Task Preference Alignment in E-Commerce",
    "casedriven": "A Case-Driven Multi-Agent Framework for E-Commerce Search Relevance",
    **{k: t for k, _, t, *_ in _NEW},
}

LICENSE_NAMES = {
    "creativecommons.org/licenses/by/": "CC BY",
    "creativecommons.org/licenses/by-sa/": "CC BY-SA",
    "creativecommons.org/licenses/by-nc-sa/": "CC BY-NC-SA",
    "creativecommons.org/licenses/by-nc-nd/": "CC BY-NC-ND",
    "creativecommons.org/licenses/by-nc/": "CC BY-NC",
    "creativecommons.org/licenses/by-nd/": "CC BY-ND",
    "creativecommons.org/publicdomain/zero/": "CC0",
    "arxiv.org/licenses/nonexclusive-distrib/": "arXiv non-exclusive license",
}
FREE_LICENSES = {"CC BY", "CC BY-SA", "CC0", "author's own"}


def _get(url: str, timeout: int = 30) -> bytes:
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def _norm(t: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", t.lower()).strip()


def titles_match(a: str, b: str) -> bool:
    return _norm(a) == _norm(b)


def parse_abs_page(page: str) -> tuple[str, str, str]:
    """Return (title, license_name, license_url) from an arXiv abstract page."""
    m = re.search(r'<meta name="citation_title" content="([^"]+)"', page)
    title = html.unescape(m.group(1)) if m else ""
    lic_url = ""
    m = re.search(r'<a[^>]+title="Rights to this article"[^>]+href="([^"]+)"', page) or \
        re.search(r'href="(https?://(?:creativecommons\.org|arxiv\.org/licenses)[^"]+)"', page)
    if m:
        lic_url = m.group(1)
    name = "unknown"
    for frag, n in LICENSE_NAMES.items():
        if frag in lic_url:
            name = n
            break
    return title, name, lic_url


def ensure_rsa_pdf(p: Paper) -> None:
    pdf = PAPERS / "Random_Semantic_Algebra_latest.pdf"
    if not pdf.exists():
        tex_dir = ROOT / "paper" / "icml"
        if shutil.which("pdflatex") and (tex_dir / "main.tex").exists():
            build = ROOT / "build" / "rsa_tex"
            build.mkdir(parents=True, exist_ok=True)
            for _ in range(2):
                subprocess.run(["pdflatex", "-interaction=nonstopmode", f"-output-directory={build}", "main.tex"],
                               cwd=tex_dir, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
            if (build / "main.pdf").exists():
                PAPERS.mkdir(exist_ok=True)
                shutil.copy(build / "main.pdf", pdf)
                p.pdf_status = "compiled from paper/icml LaTeX source (local PDF was not in the repo)"
    else:
        p.pdf_status = p.pdf_status or "local PDF"
    if pdf.exists():
        p.pdf = str(pdf.relative_to(ROOT))
        import pymupdf
        doc = pymupdf.open(pdf)
        p.title = (doc.metadata or {}).get("title") or ""
        first = doc[0].get_text("dict")
        # Largest-font line on page 1 is the printed title.
        best = max(((s["size"], s["text"]) for b in first["blocks"] for l in b.get("lines", [])
                    for s in l["spans"] if s["text"].strip()), default=(0, ""))
        p.title = p.title or best[1]
        p.title_source = "PDF page 1"
    p.license, p.license_source = "author's own", "user's own paper"


def fetch(p: Paper, offline_meta: dict, allow_network: bool = True) -> None:
    """Download the PDF and read title + license from the arXiv abstract page."""
    if p.key == "rsa":
        ensure_rsa_pdf(p)
        return
    pdf = PAPERS / f"{p.key}_{p.arxiv}.pdf"
    net_ok = False
    if allow_network:
        try:
            page = _get(p.abs_url).decode("utf-8", "replace")
            p.title, p.license, p.license_url = parse_abs_page(page)
            p.title_source = f"arXiv abstract page {p.abs_url}"
            p.license_source = "arXiv abstract page"
            net_ok = True
        except (urllib.error.URLError, OSError) as e:
            p.pdf_status = f"arxiv.org unreachable ({type(e).__name__}: {str(e)[:80]})"
    if not net_ok:
        cached = offline_meta.get(p.key, {})
        p.title = cached.get("title") or OFFLINE_TITLES.get(p.key, "")
        p.title_source = cached.get("title_source") or "web search result for the arXiv abstract page (arxiv.org blocked from build host)"
        p.license = cached.get("license", "unknown")
        p.license_url = cached.get("license_url", "")
        p.license_source = cached.get("license_source") or "not readable: arxiv.org blocked from build host"
    if not pdf.exists() and p.alt_pdf and allow_network:
        try:
            data = _get(p.alt_pdf, timeout=90)
            if data[:4] == b"%PDF":
                PAPERS.mkdir(exist_ok=True)
                pdf.write_bytes(data)
        except (urllib.error.URLError, OSError):
            pass
    if not pdf.exists() and net_ok:
        try:
            data = _get(f"https://arxiv.org/pdf/{p.arxiv}", timeout=90)
            if data[:4] == b"%PDF":
                PAPERS.mkdir(exist_ok=True)
                pdf.write_bytes(data)
        except (urllib.error.URLError, OSError) as e:
            p.pdf_status = f"PDF download failed ({type(e).__name__})"
    if pdf.exists():
        p.pdf = str(pdf.relative_to(ROOT))
        p.pdf_status = "downloaded"
    elif not p.pdf_status:
        p.pdf_status = "PDF not available"


def fetch_all(allow_network: bool = True) -> list[Paper]:
    meta_path = PAPERS / "metadata.json"
    offline = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    for p in REGISTRY:
        fetch(p, offline, allow_network)
    # Persist whatever we learned so later offline builds can reuse it.
    PAPERS.mkdir(exist_ok=True)
    keep = {p.key: {k: v for k, v in asdict(p).items()
                    if k in ("title", "title_source", "license", "license_url", "license_source")}
            for p in REGISTRY if p.key != "rsa" and p.license_source.startswith("arXiv")}
    merged = {**offline, **keep}
    meta_path.write_text(json.dumps(merged, indent=2) + "\n")
    return REGISTRY


def is_free(p: Paper) -> bool:
    return p.license in FREE_LICENSES
