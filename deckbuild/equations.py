"""Render the deck's equations to SVG and 3x PNG with real LaTeX (matplotlib usetex).

Falls back to matplotlib mathtext when no LaTeX toolchain is installed; the
fallback rewrites a few amsmath-only commands that mathtext does not know.
"""
from __future__ import annotations

import re
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

INK = "#1E2328"

# key -> (title, LaTeX). Exact strings from the brief; eq11 uses the E-SENS
# paper's own form (see qa_report.md), and the brief's schematic is kept as eq11s.
EQUATIONS: dict[str, tuple[str, str]] = {
    "eq01_rrf": ("Reciprocal rank fusion",
                 r"\mathrm{RRF}(d)=\sum_{r\in R}\frac{1}{k+\mathrm{rank}_r(d)},\quad k=60"),
    "eq02_gam": ("Fusion as a GAM (EBM generalizes RRF)",
                 r"s(d)=\sum_{r} f_r\big(\mathrm{rank}_r(d)\big)+\sum_{r<r'} f_{rr'}\big(\mathrm{rank}_r(d),\mathrm{rank}_{r'}(d)\big)"),
    "eq03_calibration": ("RSA calibration",
                         r"p_C(x)=\sigma\big(a'_C\,F_C(x)+c_C\big)"),
    "eq04_composition": ("RSA composition",
                         r"S_Q(x)=\sum_{C\in Q^+}\log p_C(x)+\sum_{C\in Q^-}\log\big(1-p_C(x)\big)"),
    "eq05_popcount": ("RSA popcount kernel",
                      r"P_C(x)=a_C\,n_+(x)+\Delta_C\sum_{t=0}^{3}2^t\,\mathrm{popcnt}\big(B_x\wedge M_{Ct}\big)"),
    "eq06_memory": ("Joint memory",
                    r"M(N,K)=N\,B_{\mathrm{item}}+K\,B_{\mathrm{program}}"),
    "eq07_traversal": ("Live traversal rule",
                       r"\text{navigate by } s_{\mathrm{dense}}(q,x),\qquad \text{admit } x \text{ if } S_{\mathrm{semantic}}(x)\geq\tau"),
    "eq08_sae": ("Sparse autoencoder",
                 r"z=\mathrm{ReLU}(W_e x+b_e),\qquad \mathcal{L}=\lVert x-W_d z\rVert_2^2+\lambda\lVert z\rVert_1"),
    "eq09_rq": ("Residual quantization for semantic IDs",
                r"r_0=e(x),\quad c_\ell=\arg\min_k\lVert r_{\ell-1}-C_\ell[k]\rVert,\quad r_\ell=r_{\ell-1}-C_\ell[c_\ell],\quad \mathrm{ID}(x)=(c_1,\dots,c_L)"),
    "eq10_mapcoord": ("Map coordinate from an LLM-designed axis",
                      r"u_k(x)=\mathrm{rank}\big(\sigma(w_k^\top x+b_k)\big)/N"),
    "eq11_esens": ("E-SENS scoring rule (paper's form)",
                   r"S(d)=s(d,q_{\mathrm{target}})-\beta\,s(d,q_{\mathrm{trap}})"),
    "eq11s_esens_schematic": ("E-SENS, schematic (brief; not used on slides)",
                              r"s(d)=\mathrm{sim}(q^+,d)-\lambda\,\mathrm{sim}(q^-,d)"),
    "eq12_canvas": ("Canvas mask as layer algebra",
                    r"M=R_{\mathrm{base}}\wedge L_{\mathrm{dress}}\wedge L_{\mathrm{stock,DE}}\wedge\neg\,L_{\mathrm{floral}}"),
}


def _has_latex() -> bool:
    return all(shutil.which(t) for t in ("latex", "dvipng")) and shutil.which("dvisvgm") is not None


def _mathtext_fallback(tex: str) -> str:
    tex = tex.replace(r"\big(", "(").replace(r"\big)", ")")
    tex = tex.replace(r"\lVert", r"\Vert").replace(r"\rVert", r"\Vert")
    tex = re.sub(r"\\text\{([^}]*)\}", lambda m: r"\mathrm{" + m.group(1).replace(" ", r"\ ") + "}", tex)
    return tex


def render_all(out_dir: Path, fontsize: int = 22) -> dict[str, dict]:
    """Write <key>.tex, <key>.svg and <key>@3x.png for every equation."""
    out_dir.mkdir(parents=True, exist_ok=True)
    use_tex = _has_latex()
    plt.rcParams.update({
        "text.usetex": use_tex,
        "text.latex.preamble": r"\usepackage{amsmath,amssymb}",
        "svg.fonttype": "path",
        "mathtext.fontset": "cm",
    })
    results = {}
    for key, (title, tex) in EQUATIONS.items():
        (out_dir / f"{key}.tex").write_text(
            f"% {title}\n\\[\n{tex}\n\\]\n", encoding="utf-8")
        body = tex if use_tex else _mathtext_fallback(tex)
        fig = plt.figure(figsize=(0.01, 0.01))
        fig.text(0, 0, (f"$\\displaystyle {body}$" if use_tex else f"${body}$"), fontsize=fontsize, color=INK)
        for ext, kw in (("svg", {}), ("png", {"dpi": 300})):
            name = f"{key}.svg" if ext == "svg" else f"{key}@3x.png"
            fig.savefig(out_dir / name, transparent=True, bbox_inches="tight",
                        pad_inches=0.04, **kw)
        plt.close(fig)
        results[key] = {"title": title, "tex": tex, "renderer": "LaTeX (usetex)" if use_tex else "mathtext"}
    # One combined source file for convenience.
    lines = ["% All deck equations (LaTeX source). Requires amsmath, amssymb.\n"]
    for key, (title, tex) in EQUATIONS.items():
        lines.append(f"% {key}: {title}\n\\[\n{tex}\n\\]\n")
    (out_dir / "equations.tex").write_text("\n".join(lines), encoding="utf-8")
    return results
