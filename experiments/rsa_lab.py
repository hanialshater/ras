# %% [markdown]
# # RSA Lab: staged experiments for Random Semantic Algebra
#
# One notebook, seven stages, one shared cache. Follows `experiments/COLAB_PLAN.md`.
#
# **Setup:** Runtime -> Change runtime type -> **T4 GPU**. Then Runtime -> Run all.
# Results land in `ROOT/results/*.csv`, plots in `ROOT/results/*.png`, and a regenerated
# `ROOT/results/summary.md`. With `USE_DRIVE=True` the cache survives runtime resets.
#
# Every stage ends with a printed decision so the loop can be read top to bottom.

# %%
# ---- Cell 1: configuration -------------------------------------------------
import os, sys, json, time, math, itertools, copy, warnings
warnings.filterwarnings("ignore")

FAST = bool(int(os.environ.get("RSA_FAST", "0")))          # tiny smoke-test settings
SYNTHETIC = bool(int(os.environ.get("RSA_SYNTHETIC", "0")))  # no downloads, fake data

CFG = dict(
    USE_DRIVE=True,
    SUBSTRATE_SPACE="minilm",          # may be switched to "bge" by the Stage 2 decision
    SEEDS=[0, 1, 2, 3, 4] if not FAST else [0, 1],
    BITS=4, K=28, PAIRS=4, CAND_POOL=96,
    PILOT_K=24, PILOT_PAIRS=2,
    PREVALENCE=0.40,
    N_QUERIES=200 if not FAST else 40,
    ANN_POOL=500 if not FAST else 300,
    RETENTION=(0.05, 0.10, 0.20, 0.40),
    Q_MIN_TRUTH=30 if not FAST else 8,
    Q_MIN_POOL=60 if not FAST else 20,
    RUN=dict(stage1=True, stage2=True, stage3=True, stage4=True,
             stage5=False, stage6=True, stage7=True),
    VLM_MODEL="HuggingFaceTB/SmolVLM-500M-Instruct",
    VLM_SUBSET=1000 if not FAST else 40,
    VLM_CONCEPTS=["minimalist", "office_appropriate", "technical_sporty", "quiet_luxury"],
    SYN_N=6000,
    THROUGHPUT_N=100_000 if not FAST else 20_000,
)

ROOT = os.environ.get("RSA_ROOT")
if ROOT is None:
    ROOT = "/content/rsa_lab"
    if CFG["USE_DRIVE"] and not SYNTHETIC:
        try:
            from google.colab import drive  # type: ignore
            drive.mount("/content/drive")
            ROOT = "/content/drive/MyDrive/rsa_lab"
        except Exception as e:  # not on Colab
            print("Drive not mounted:", e)
for sub in ("cache", "results"):
    os.makedirs(f"{ROOT}/{sub}", exist_ok=True)
print("ROOT =", ROOT, "| FAST =", FAST, "| SYNTHETIC =", SYNTHETIC)

# %%
# ---- Cell 2: installs and imports -----------------------------------------
import importlib, subprocess
if not SYNTHETIC:
    need = {"datasets": "datasets", "sentence_transformers": "sentence-transformers",
            "transformers": "transformers", "numba": "numba"}
    missing = [pkg for mod, pkg in need.items() if importlib.util.find_spec(mod) is None]
    if missing:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", *missing], check=False)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, cohen_kappa_score

try:
    import torch
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
except Exception:
    torch = None
    DEVICE = "cpu"
print("device:", DEVICE)


def cached(name, fn, kind="npy"):
    """Compute once, store under ROOT/cache, reload on later runs."""
    path = f"{ROOT}/cache/{name}.{kind}"
    if os.path.exists(path):
        return np.load(path) if kind == "npy" else pd.read_pickle(path)
    obj = fn()
    if kind == "npy":
        np.save(path, obj)
    else:
        obj.to_pickle(path)
    return obj


def save_rows(stage, rows):
    df_ = pd.DataFrame(rows)
    path = f"{ROOT}/results/{stage}.csv"
    df_.to_csv(path, index=False)
    return df_


def l2n(x):
    x = np.asarray(x, dtype=np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-9)


def tic(msg):
    print(f"\n=== {msg} ===", flush=True)
    return time.time()

# %%
# ---- Cell 3: concepts ------------------------------------------------------
LATENT_SPECS = [
    dict(name="minimalist", word="minimalist",
         pos=["a minimalist understated fashion item", "a clean simple minimal design", "simple sleek understated clothing"],
         neg=["a busy ornate decorative fashion item", "heavily embellished flashy design", "complex colorful overdesigned clothing"]),
    dict(name="office_appropriate", word="office",
         pos=["formal office appropriate workwear", "smart business attire for the office", "professional corporate clothing"],
         neg=["casual weekend loungewear", "beachwear or party outfit", "athletic gym clothing"]),
    dict(name="technical_sporty", word="sporty",
         pos=["technical sporty athletic gear", "performance sportswear for training", "running or gym apparel"],
         neg=["elegant formal evening wear", "classic tailored office clothing", "delicate dressy fashion item"]),
    dict(name="retro", word="retro",
         pos=["retro vintage inspired fashion item", "old school throwback style clothing", "nostalgic seventies or eighties look"],
         neg=["modern futuristic contemporary design", "sleek current season fashion", "brand new minimalist tech style"]),
    dict(name="elegant", word="elegant",
         pos=["elegant refined sophisticated fashion item", "graceful dressy chic clothing", "polished classy evening style"],
         neg=["rugged rough utilitarian clothing", "sloppy casual streetwear", "loud sporty athletic gear"]),
    dict(name="relaxed", word="relaxed",
         pos=["relaxed casual comfortable clothing", "laid back easygoing everyday wear", "cozy loose fit casual outfit"],
         neg=["stiff formal structured attire", "tight restrictive dressy clothing", "strict business suit"]),
    dict(name="chunky", word="chunky",
         pos=["chunky bulky oversized fashion item", "thick heavy chunky footwear", "bold heavyweight chunky design"],
         neg=["slim delicate lightweight item", "thin sleek fine fashion piece", "dainty minimal slender design"]),
    dict(name="quiet_luxury", word="quiet luxury",
         pos=["quiet luxury understated premium fashion", "expensive looking minimal logo free clothing", "refined high quality subtle luxury"],
         neg=["cheap flashy logo heavy fast fashion", "loud branded garish clothing", "budget basic printed item"]),
    dict(name="streetwear", word="streetwear",
         pos=["urban streetwear fashion item", "hip hop skate inspired street style", "hype sneaker culture clothing"],
         neg=["classic formal business clothing", "traditional elegant dress wear", "conservative office attire"]),
    dict(name="bohemian", word="bohemian",
         pos=["bohemian boho free spirited fashion", "flowy ethnic print festival clothing", "hippie artsy layered style"],
         neg=["strict minimal corporate clothing", "sharp tailored modern outfit", "plain athletic sportswear"]),
    dict(name="preppy", word="preppy",
         pos=["preppy collegiate classic style", "polo shirt chinos ivy league look", "neat smart casual preppy clothing"],
         neg=["grunge punk rebellious clothing", "loose urban streetwear", "bohemian festival outfit"]),
    dict(name="edgy", word="edgy",
         pos=["edgy dark rebellious fashion item", "punk rock leather studded style", "bold alternative grunge clothing"],
         neg=["sweet soft pastel feminine clothing", "classic conservative office wear", "plain basic everyday item"]),
    dict(name="festive", word="festive",
         pos=["festive celebratory party outfit", "ethnic wedding festival wear", "glamorous embellished occasion clothing"],
         neg=["plain everyday basic clothing", "technical workout gear", "simple casual loungewear"]),
    dict(name="outdoor", word="outdoor",
         pos=["outdoor hiking trekking gear", "rugged weatherproof outdoor clothing", "camping adventure apparel"],
         neg=["delicate indoor evening wear", "formal office clothing", "glamorous party outfit"]),
    dict(name="vintage", word="vintage",
         pos=["vintage classic heritage style item", "timeless old fashioned traditional design", "antique inspired retro clothing"],
         neg=["ultra modern trendy new design", "futuristic technical clothing", "current fast fashion item"]),
    dict(name="luxury", word="luxury",
         pos=["luxury premium designer fashion item", "high end expensive luxurious clothing", "opulent lavish designer piece"],
         neg=["cheap budget basic item", "plain discount everyday clothing", "simple low cost fast fashion"]),
]
LATENT_NAMES = [s["name"] for s in LATENT_SPECS]
CONCEPT_WORD = {s["name"]: s["word"] for s in LATENT_SPECS}
C_LAT = len(LATENT_SPECS)
if SYNTHETIC:  # prompts the synthetic encoder can parse
    for s in LATENT_SPECS:
        s["pos"] = [f"{s['name']} pos"]
        s["neg"] = [f"anti-{s['name']}"]

SPACES = {"minilm": 384, "bge": 384, "clip_txt": 512, "clip_img": 512}

# %%
# ---- Cell 4: data ----------------------------------------------------------
t0 = tic("Stage 0a: data")
KEEP = ["id", "gender", "masterCategory", "subCategory", "articleType",
        "baseColour", "season", "usage", "productDisplayName"]
DS = None
if SYNTHETIC:
    rng0 = np.random.default_rng(0)
    n = CFG["SYN_N"]
    df = pd.DataFrame({
        "id": np.arange(n),
        "gender": rng0.choice(["Men", "Women", "Unisex"], n, p=[.45, .45, .1]),
        "masterCategory": rng0.choice(["Apparel", "Footwear", "Accessories"], n),
        "subCategory": rng0.choice(["Topwear", "Shoes", "Bags", "Bottomwear", "Watches", "Dress"], n),
        "articleType": rng0.choice(["Tshirts", "Shirts", "Casual Shoes", "Handbags", "Jeans", "Sports Shoes"], n),
        "baseColour": rng0.choice(["Black", "White", "Blue", "Brown", "Red", "Grey", "Green", "Pink"], n),
        "season": rng0.choice(["Summer", "Winter", "Fall", "Spring"], n),
        "usage": rng0.choice(["Casual", "Sports", "Formal", "Ethnic"], n, p=[.5, .2, .2, .1]),
    })
    df["productDisplayName"] = [f"item {i}" for i in range(n)]
else:
    from datasets import load_dataset
    DS = load_dataset("ashraq/fashion-product-images-small", split="train")
    meta = DS.remove_columns([c for c in DS.column_names if c not in KEEP]).to_pandas()
    valid = meta["baseColour"].notna() & meta["subCategory"].notna() & meta["productDisplayName"].notna()
    VALID_IDX = np.where(valid.to_numpy())[0]
    df = meta.loc[valid].reset_index(drop=True)
N = len(df)
TITLES = df["productDisplayName"].astype(str).tolist()
print("items:", N)

# structured facets for the mechanism benchmark (same rule as the original snapshot)
facet_cols = ["baseColour", "articleType", "subCategory", "usage", "gender", "masterCategory"]
cands = []
for col in facet_cols:
    for val, cnt in df[col].value_counts().items():
        prev = cnt / N
        if cnt >= (350 if not SYNTHETIC else 100) and 0.025 <= prev <= 0.70:
            cands.append((col, val, int(cnt)))
picked, per_col = [], {c: 0 for c in facet_cols}
for col, val, cnt in sorted(cands, key=lambda z: -z[2]):
    if per_col[col] < 4:
        picked.append((col, val)); per_col[col] += 1
    if len(picked) >= 14:
        break
FACETS = picked
FACET_NAMES = [f"{c}={v}" for c, v in FACETS]
Y_FACET = np.column_stack([(df[c].to_numpy() == v) for c, v in FACETS]).astype(bool)
print("facets:", FACET_NAMES)

# %%
# ---- Cell 5: embeddings and text encoders ---------------------------------
t0 = tic("Stage 0b: embeddings")
EMB = {}
TEXT_ENCODERS = {}

if SYNTHETIC:
    rng1 = np.random.default_rng(1)
    Z_LAT = rng1.standard_normal((N, C_LAT)).astype(np.float32)
    SYN_W, SYN_ATTR = {}, {}
    attr_keys = [(c, v) for c in ["baseColour", "subCategory", "gender", "usage"] for v in df[c].unique()]
    for sp, d in SPACES.items():
        SYN_W[sp] = rng1.standard_normal((C_LAT, d)).astype(np.float32) / np.sqrt(d)
        SYN_ATTR[sp] = {k: (rng1.standard_normal(d).astype(np.float32) / np.sqrt(d)) for k in attr_keys}
        noise = 0.9 if sp != "clip_img" else 0.4
        E = Z_LAT @ SYN_W[sp]
        for (c, v), a in SYN_ATTR[sp].items():
            E = E + 2.0 * a[None, :] * (df[c].to_numpy() == v)[:, None]
        E = E + noise * rng1.standard_normal((N, d)).astype(np.float32) / np.sqrt(d)
        EMB[sp] = l2n(E)

    def _syn_encoder(sp):
        def enc(texts):
            out = []
            for t in texts:
                tl = " " + t.lower() + " "
                v = np.zeros(SPACES[sp], dtype=np.float32)
                for ci, name in enumerate(LATENT_NAMES):
                    if f"anti-{name}" in tl:
                        v -= SYN_W[sp][ci]
                    elif f" {CONCEPT_WORD[name]} " in tl or f" {name} " in tl:
                        v += SYN_W[sp][ci]
                for (c, val), a in SYN_ATTR[sp].items():
                    if f" {str(val).lower()} " in tl:
                        v += 2.0 * a
                out.append(v)
            return l2n(np.stack(out))
        return enc
    for sp in ["minilm", "bge", "clip_txt"]:
        TEXT_ENCODERS[sp] = _syn_encoder(sp)
else:
    from sentence_transformers import SentenceTransformer
    from transformers import CLIPModel, CLIPProcessor
    _ST = {}

    def st_model(name):
        if name not in _ST:
            _ST[name] = SentenceTransformer(name, device=DEVICE)
        return _ST[name]

    def st_encode(name, texts):
        return l2n(st_model(name).encode(texts, batch_size=256, normalize_embeddings=True,
                                         convert_to_numpy=True, show_progress_bar=True))

    EMB["minilm"] = cached("emb_minilm", lambda: st_encode("sentence-transformers/all-MiniLM-L6-v2", TITLES))
    EMB["bge"] = cached("emb_bge", lambda: st_encode("BAAI/bge-small-en-v1.5", TITLES))
    TEXT_ENCODERS["minilm"] = lambda t: st_encode("sentence-transformers/all-MiniLM-L6-v2", list(t))
    TEXT_ENCODERS["bge"] = lambda t: st_encode("BAAI/bge-small-en-v1.5", list(t))

    _CLIP = {}

    def clip_parts():
        if not _CLIP:
            _CLIP["m"] = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE).eval()
            _CLIP["p"] = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        return _CLIP["m"], _CLIP["p"]

    @torch.no_grad()
    def clip_text(texts):
        m, p = clip_parts()
        out = []
        for i in range(0, len(texts), 256):
            b = p(text=list(texts[i:i + 256]), return_tensors="pt", padding=True, truncation=True).to(DEVICE)
            out.append(m.get_text_features(**b).float().cpu().numpy())
        return l2n(np.concatenate(out))

    @torch.no_grad()
    def clip_images():
        m, p = clip_parts()
        out = []
        bs = 128
        for i in range(0, len(VALID_IDX), bs):
            imgs = [DS[int(j)]["image"].convert("RGB") for j in VALID_IDX[i:i + bs]]
            b = p(images=imgs, return_tensors="pt").to(DEVICE)
            out.append(m.get_image_features(**b).float().cpu().numpy())
            if (i // bs) % 50 == 0:
                print(f"  clip images {i}/{len(VALID_IDX)}", flush=True)
        return l2n(np.concatenate(out))

    EMB["clip_txt"] = cached("emb_clip_txt", lambda: clip_text(TITLES))
    EMB["clip_img"] = cached("emb_clip_img", clip_images)
    TEXT_ENCODERS["clip_txt"] = lambda t: clip_text(list(t))

for k, v in EMB.items():
    print(f"  {k}: {v.shape}")
print(f"done in {time.time() - t0:.0f}s")

# %%
# ---- Cell 6: teacher scores, labels, splits -------------------------------
t0 = tic("Stage 0c: teacher, splits")


def prompt_direction(space, spec):
    enc = TEXT_ENCODERS[space]
    pos = l2n(enc(spec["pos"]).mean(0))
    neg = l2n(enc(spec["neg"]).mean(0))
    return pos - neg


def teacher_scores():
    dirs = np.stack([prompt_direction("clip_txt", s) for s in LATENT_SPECS])  # CLIP text space == CLIP image space
    return (EMB["clip_img"] @ dirs.T).astype(np.float32)


T_SCORES = cached("teacher_scores_v1", teacher_scores)  # N x C_LAT


def make_split(seed):
    rng = np.random.default_rng(1000 + seed)
    perm = rng.permutation(N)
    n_fit, n_cal = int(0.52 * N), int(0.13 * N)
    return dict(fit=np.sort(perm[:n_fit]), cal=np.sort(perm[n_fit:n_fit + n_cal]), test=np.sort(perm[n_fit + n_cal:]))


SPLITS = {s: make_split(s) for s in CFG["SEEDS"]}


def latent_labels(split, prevalence=CFG["PREVALENCE"], scores=T_SCORES):
    thr = np.quantile(scores[split["fit"]], 1 - prevalence, axis=0)
    return {k: (scores[idx] >= thr) for k, idx in split.items()}


LABELS = {}
for s, sp in SPLITS.items():
    lat = latent_labels(sp)
    LABELS[s] = {k: np.concatenate([Y_FACET[sp[k]], lat[k]], axis=1) for k in sp}
ALL_NAMES = FACET_NAMES + LATENT_NAMES
GROUP = ["facet"] * len(FACET_NAMES) + ["latent"] * C_LAT
LAT_COLS = np.arange(len(FACET_NAMES), len(ALL_NAMES))
print("label matrix:", LABELS[CFG["SEEDS"][0]]["fit"].shape, "| latent prevalence on test:",
      np.round(LABELS[CFG["SEEDS"][0]]["test"][:, LAT_COLS].mean(0), 2))

# %%
# ---- Cell 7: RSA core ------------------------------------------------------
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def log_sigmoid(x):
    return -np.logaddexp(0.0, -x)


def make_rotation(kind, X_fit, seed):
    d = X_fit.shape[1]
    if kind == "identity":
        return np.eye(d, dtype=np.float32)
    if kind == "orthogonal":
        rng = np.random.default_rng(seed)
        Qm, R = np.linalg.qr(rng.standard_normal((d, d)))
        return (Qm * np.sign(np.diag(R))).astype(np.float32)
    if kind == "pca":
        Xc = X_fit[:20000] - X_fit[:20000].mean(0)
        _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
        return Vt.T.astype(np.float32)
    raise ValueError(kind)


class Quantizer:
    def __init__(self, Z_fit, bits):
        self.nb = 2 ** bits
        qs = np.linspace(0, 1, self.nb + 1)[1:-1]
        self.edges = np.quantile(Z_fit, qs, axis=0).T.astype(np.float32)  # d x (nb-1)
        Q = self.transform(Z_fit)
        self.centers = np.zeros((Z_fit.shape[1], self.nb), dtype=np.float32)
        for j in range(Z_fit.shape[1]):
            sums = np.bincount(Q[:, j], weights=Z_fit[:, j], minlength=self.nb)
            cnt = np.bincount(Q[:, j], minlength=self.nb)
            self.centers[j] = sums / np.maximum(cnt, 1)

    def transform(self, Z):
        Q = np.empty(Z.shape, dtype=np.uint8 if self.nb <= 256 else np.uint16)
        for j in range(Z.shape[1]):
            Q[:, j] = np.searchsorted(self.edges[j], Z[:, j], side="right")
        return Q


class Substrate:
    def __init__(self, X, split, bits, rotation, seed):
        self.R = make_rotation(rotation, X[split["fit"]], seed)
        Zf = X[split["fit"]] @ self.R
        self.quant = Quantizer(Zf, bits)
        self.nb = self.quant.nb
        self.Z_fit = Zf
        self.Q_fit = self.quant.transform(Zf)
        self.Q_cal = self.quant.transform(X[split["cal"]] @ self.R)
        self.Q_test = self.quant.transform(X[split["test"]] @ self.R)
        self.bits = bits


def _bin_stats(Qc, g, h, nb):
    n, m = Qc.shape
    flat = (Qc.astype(np.int64) + (np.arange(m, dtype=np.int64) * nb)[None, :]).ravel()
    G = np.bincount(flat, weights=np.repeat(g, m), minlength=m * nb).reshape(m, nb)
    H = np.bincount(flat, weights=np.repeat(h, m), minlength=m * nb).reshape(m, nb)
    return G, H


def fit_boosted_lut(Q, y, k, n_bins, candidate_pool=96, lam=1.0, eta=1.0, refine_passes=1):
    """Algorithm 1: residual Newton boosting, one 1-D lookup table per selected coordinate."""
    n, d = Q.shape
    y = y.astype(np.float64)
    prior = float(np.clip(y.mean(), 1e-4, 1 - 1e-4))
    b0 = math.log(prior / (1 - prior))
    s = np.full(n, b0)
    p = sigmoid(s); g = y - p; h = p * (1 - p)
    G, H = _bin_stats(Q, g, h, n_bins)
    gain0 = 0.5 * ((G ** 2) / (H + lam)).sum(1)
    cand = list(np.argsort(-gain0)[:max(candidate_pool, k)])
    coords, tables, gains = [], [], []
    for _ in range(k):
        if not cand:
            break
        p = sigmoid(s); g = y - p; h = p * (1 - p)
        G, H = _bin_stats(Q[:, cand], g, h, n_bins)
        gain = 0.5 * ((G ** 2) / (H + lam)).sum(1)
        i = int(np.argmax(gain)); j = int(cand.pop(i))
        f = eta * G[i] / (H[i] + lam)
        s += f[Q[:, j]]
        coords.append(j); tables.append(f); gains.append(float(gain[i]))
    tables = np.array(tables, dtype=np.float64)
    for _ in range(refine_passes):
        for t, j in enumerate(coords):
            s -= tables[t][Q[:, j]]
            p = sigmoid(s); g = y - p; h = p * (1 - p)
            Gj = np.bincount(Q[:, j], weights=g, minlength=n_bins)
            Hj = np.bincount(Q[:, j], weights=h, minlength=n_bins)
            tables[t] = eta * Gj / (Hj + lam)
            s += tables[t][Q[:, j]]
    return dict(intercept=b0, coords=np.array(coords, dtype=np.int64), tables=tables,
                pairs=[], n_bins=n_bins, gains=gains)


def score_boosted(Q, model):
    s = np.full(len(Q), model["intercept"])
    if len(model["coords"]):
        s += model["tables"][np.arange(len(model["coords"]))[None, :], Q[:, model["coords"]]].sum(1)
    for a, b, T in model["pairs"]:
        s += T[Q[:, a], Q[:, b]]
    return s


def add_pair_interactions(Q, y, model, n_pairs, pair_pool=12, lam=4.0, eta=1.0):
    model = dict(model, pairs=list(model["pairs"]))
    if n_pairs <= 0:
        return model
    nb = model["n_bins"]
    y = y.astype(np.float64)
    top = [int(c) for c in model["coords"][:pair_pool]]
    cands = list(itertools.combinations(top, 2))
    s = score_boosted(Q, model)
    for _ in range(n_pairs):
        if not cands:
            break
        p = sigmoid(s); g = y - p; h = p * (1 - p)
        J = np.stack([Q[:, a].astype(np.int64) * nb + Q[:, b] for a, b in cands], 1)
        G, H = _bin_stats(J, g, h, nb * nb)
        gain = 0.5 * ((G ** 2) / (H + lam)).sum(1)
        i = int(np.argmax(gain)); a, b = cands.pop(i)
        T = (eta * G[i] / (H[i] + lam)).reshape(nb, nb)
        s += T[Q[:, a], Q[:, b]]
        model["pairs"].append((a, b, T))
    return model


def pairs_prefix(model, m):
    return dict(model, pairs=list(model["pairs"][:m]))


class Platt:
    def fit(self, s, y):
        lr = LogisticRegression(C=1e4, max_iter=1000).fit(np.asarray(s)[:, None], y.astype(int))
        self.a, self.b = float(lr.coef_[0, 0]), float(lr.intercept_[0])
        return self

    def transform(self, s):
        return self.a * np.asarray(s) + self.b


def best_f1_threshold(scores, truth, n_grid=160):
    scores = np.asarray(scores); y = np.asarray(truth, dtype=np.int8)
    thr = np.unique(np.quantile(scores, np.linspace(0.005, 0.995, n_grid)))
    order = np.argsort(scores); ss = scores[order]; sy = y[order]
    prefix = np.concatenate(([0], np.cumsum(sy))); total = prefix[-1]
    idx = np.searchsorted(ss, thr, side="left")
    tp = total - prefix[idx]; pred_pos = len(scores) - idx
    denom = pred_pos + total
    f1 = np.divide(2.0 * tp, denom, out=np.zeros(len(thr)), where=denom > 0)
    return float(thr[np.argmax(f1)])


def metric_row(truth, scores, threshold):
    truth = np.asarray(truth).astype(int)
    pred = (np.asarray(scores) >= threshold).astype(int)
    return dict(f1=float(f1_score(truth, pred, zero_division=0)),
                ap=float(average_precision_score(truth, scores)) if truth.sum() > 0 else float("nan"))


def eval_scores(s_cal, s_test, y_cal, y_test):
    return metric_row(y_test, s_test, best_f1_threshold(s_cal, y_cal))


def eval_program(model, sub, y_cal, y_test):
    return eval_scores(score_boosted(sub.Q_cal, model), score_boosted(sub.Q_test, model), y_cal, y_test)


def fit_program(sub, y_fit, k, pairs, candidate_pool=None):
    m = fit_boosted_lut(sub.Q_fit, y_fit, k=k, n_bins=sub.nb,
                        candidate_pool=candidate_pool or max(CFG["CAND_POOL"], k))
    return add_pair_interactions(sub.Q_fit, y_fit, m, n_pairs=pairs)


def logistic(X, y, C=1.0, max_iter=500):
    return LogisticRegression(C=C, max_iter=max_iter).fit(X, y.astype(int))


def compiled_linear(sub, lr, coords=None):
    """Turn a linear model on rotated FP32 coordinates into per-coordinate LUTs."""
    w = lr.coef_[0]; b = float(lr.intercept_[0])
    coords = np.arange(len(w)) if coords is None else np.asarray(coords)
    tables = w[coords, None] * sub.quant.centers[coords]
    return dict(intercept=b, coords=coords, tables=tables, pairs=[], n_bins=sub.nb, gains=[])


def onehot(Q, coords, nb):
    n = len(Q); k = len(coords)
    cols = (Q[:, coords].astype(np.int64) + (np.arange(k) * nb)[None, :]).ravel()
    rows = np.repeat(np.arange(n), k)
    return sparse.csr_matrix((np.ones(n * k, dtype=np.float32), (rows, cols)), shape=(n, k * nb))


def ci(vals, n_boot=1000, seed=0):
    vals = np.asarray(vals, dtype=float); vals = vals[~np.isnan(vals)]
    if len(vals) == 0:
        return (float("nan"),) * 3
    rng = np.random.default_rng(seed)
    means = rng.choice(vals, size=(n_boot, len(vals)), replace=True).mean(1)
    return float(vals.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


print("RSA core ready")

# %%
# ---- Cell 8: Stage 1 kill tests -------------------------------------------
DECISIONS = {}
if CFG["RUN"]["stage1"]:
    t0 = tic("Stage 1: kill tests")
    X = EMB[CFG["SUBSTRATE_SPACE"]]
    rows = []
    for seed in CFG["SEEDS"]:
        sp, lab = SPLITS[seed], LABELS[seed]
        for rot in ["identity", "orthogonal", "pca"]:
            sub = Substrate(X, sp, CFG["BITS"], rot, seed)
            for c, name in enumerate(ALL_NAMES):
                m = fit_program(sub, lab["fit"][:, c], CFG["K"], 8 if rot == "orthogonal" else CFG["PAIRS"])
                for npair in ([0, 2, 4, 8] if rot == "orthogonal" else [CFG["PAIRS"]]):
                    r = eval_program(pairs_prefix(m, npair), sub, lab["cal"][:, c], lab["test"][:, c])
                    rows.append(dict(seed=seed, rotation=rot, pairs=npair, concept=name, group=GROUP[c], **r))
            if rot == "orthogonal":
                SUB_CACHE = sub  # reused below for this seed
        # 1c fair sparse baselines + 1d substrate diagnostic (orthogonal substrate, 3 seeds)
        if seed in CFG["SEEDS"][:3]:
            sub = SUB_CACHE
            Zf, Zc, Zt = sub.Z_fit, X[sp["cal"]] @ sub.R, X[sp["test"]] @ sub.R
            zstd = Zf.std(0) + 1e-6
            for c, name in enumerate(ALL_NAMES):
                yf, yc, yt = lab["fit"][:, c], lab["cal"][:, c], lab["test"][:, c]
                lr_full = logistic(Zf, yf)
                r = eval_scores(lr_full.decision_function(Zc), lr_full.decision_function(Zt), yc, yt)
                rows.append(dict(seed=seed, rotation="orthogonal", pairs=0, concept=name, group=GROUP[c], method="fp32_linear", K=Zf.shape[1], **r))
                cl = compiled_linear(sub, lr_full)
                rows.append(dict(seed=seed, rotation="orthogonal", pairs=0, concept=name, group=GROUP[c], method="compiled_linear_all", K=Zf.shape[1], **eval_program(cl, sub, yc, yt)))
                rank = np.argsort(-np.abs(lr_full.coef_[0]) * zstd)
                for K in [16, 28, 64]:
                    top = rank[:K]
                    lr_k = logistic(Zf[:, top], yf)
                    rows.append(dict(seed=seed, rotation="orthogonal", pairs=0, concept=name, group=GROUP[c], method="topK_linear_fp32", K=K,
                                     **eval_scores(lr_k.decision_function(Zc[:, top]), lr_k.decision_function(Zt[:, top]), yc, yt)))
                    lr_g = logistic(onehot(sub.Q_fit, top, sub.nb), yf, C=0.5)
                    rows.append(dict(seed=seed, rotation="orthogonal", pairs=0, concept=name, group=GROUP[c], method="gam_joint_linearcoords", K=K,
                                     **eval_scores(lr_g.decision_function(onehot(sub.Q_cal, top, sub.nb)), lr_g.decision_function(onehot(sub.Q_test, top, sub.nb)), yc, yt)))
                    bm = fit_boosted_lut(sub.Q_fit, yf, k=K, n_bins=sub.nb, candidate_pool=max(CFG["CAND_POOL"], K))
                    rows.append(dict(seed=seed, rotation="orthogonal", pairs=0, concept=name, group=GROUP[c], method="boosted_lut", K=K, **eval_program(bm, sub, yc, yt)))
                    bc = bm["coords"]
                    lr_g2 = logistic(onehot(sub.Q_fit, bc, sub.nb), yf, C=0.5)
                    rows.append(dict(seed=seed, rotation="orthogonal", pairs=0, concept=name, group=GROUP[c], method="gam_joint_boostedcoords", K=K,
                                     **eval_scores(lr_g2.decision_function(onehot(sub.Q_cal, bc, sub.nb)), lr_g2.decision_function(onehot(sub.Q_test, bc, sub.nb)), yc, yt)))
        print(f"  seed {seed} done ({time.time() - t0:.0f}s)", flush=True)
    S1 = save_rows("stage1", rows)

    a = S1[S1.method.isna() & (S1.pairs == CFG["PAIRS"])].groupby(["rotation", "group", "seed"]).f1.mean().groupby(["rotation", "group"]).agg(["mean", "std"])
    print("\n1a rotation (K=%d, %d pairs): mean F1 over concepts, mean/sd over seeds" % (CFG["K"], CFG["PAIRS"]))
    print(a.round(4).to_string())
    b = S1[S1.method.isna() & (S1.rotation == "orthogonal")].groupby(["pairs", "group", "seed"]).f1.mean().groupby(["pairs", "group"]).agg(["mean", "std"])
    print("\n1b pairs (orthogonal):")
    print(b.round(4).to_string())
    c_ = S1[S1.method.notna()].groupby(["method", "K", "group", "seed"]).f1.mean().groupby(["method", "K", "group"]).agg(["mean", "std"])
    print("\n1c/1d sparse baselines and substrate diagnostic:")
    print(c_.round(4).to_string())

    lat = a.xs("latent", level="group")
    ident, orth = lat.loc["identity"], lat.loc["orthogonal"]
    DECISIONS["1a"] = ("rotation irrelevant: identity within 1 sd of random" if abs(ident["mean"] - orth["mean"]) <= max(orth["std"], 1e-3)
                       else ("PCA wins" if lat["mean"].idxmax() == "pca" else ("random rotation helps" if orth["mean"] > ident["mean"] else "identity beats random")))
    bl = b.xs("latent", level="group")
    DECISIONS["1b"] = ("pair gain inside seed noise; drop pairs" if (bl.loc[CFG["PAIRS"], "mean"] - bl.loc[0, "mean"]) <= bl.loc[0, "std"] else "pair gain is real")
    cl = c_.xs("latent", level="group")
    try:
        gb, bo = cl.loc[("gam_joint_boostedcoords", 28), "mean"], cl.loc[("boosted_lut", 28), "mean"]
        DECISIONS["1c"] = ("joint GAM fit matches boosting; boosting is not the contribution" if abs(gb - bo) < 0.005 else f"boosting differs from joint fit by {bo - gb:+.3f} F1")
    except KeyError:
        pass
    for k_, v_ in DECISIONS.items():
        print(f"DECISION {k_}: {v_}")

# %%
# ---- Cell 9: Stage 2 probe ceilings ---------------------------------------
if CFG["RUN"]["stage2"]:
    t0 = tic("Stage 2: probe ceilings")
    rows = []
    for seed in CFG["SEEDS"][:3]:
        sp, lab = SPLITS[seed], LABELS[seed]
        for space, E in EMB.items():
            for ci_, c in enumerate(LAT_COLS):
                lr = logistic(E[sp["fit"]], lab["fit"][:, c])
                r = eval_scores(lr.decision_function(E[sp["cal"]]), lr.decision_function(E[sp["test"]]), lab["cal"][:, c], lab["test"][:, c])
                rows.append(dict(seed=seed, space=space, concept=LATENT_NAMES[ci_], **r))
    S2 = save_rows("stage2", rows)
    tab = S2.groupby(["space", "seed"])[["f1", "ap"]].mean().groupby("space").agg(["mean", "std"])
    print(tab.round(4).to_string())
    print("\nper concept F1 (mean over seeds):")
    print(S2.pivot_table(index="concept", columns="space", values="f1").round(3).to_string())
    f1m = tab[("f1", "mean")]
    if f1m.get("bge", 0) > f1m.get("minilm", 0) + 0.05:
        CFG["SUBSTRATE_SPACE"] = "bge"
        DECISIONS["2"] = "bge beats minilm by >0.05 F1: substrate switched to bge for later stages"
    else:
        DECISIONS["2"] = f"keep {CFG['SUBSTRATE_SPACE']} (bge-minilm = {f1m.get('bge', float('nan')) - f1m.get('minilm', float('nan')):+.3f})"
    DECISIONS["2_gap"] = f"cross-modal cap: clip_img {f1m.get('clip_img', float('nan')):.3f} vs best text {max(f1m.get('minilm', 0), f1m.get('bge', 0)):.3f}"
    if f1m.get("clip_txt", 0) > max(f1m.get("minilm", 0), f1m.get("bge", 0)):
        DECISIONS["2_indep"] = "clip_txt on titles beats both text substrates: teacher independence is weaker than claimed"
    for k_ in ["2", "2_gap", "2_indep"]:
        if k_ in DECISIONS:
            print(f"DECISION {k_}: {DECISIONS[k_]}")

# %%
# ---- Cell 10: Stage 3 composition benchmark -------------------------------
def fuse(L, pos, neg, name_to_idx):
    s = np.zeros(len(L))
    for c in pos:
        s += log_sigmoid(L[:, name_to_idx[c]])
    for c in neg:
        s += log_sigmoid(-L[:, name_to_idx[c]])
    return s


def retention_stats(scores, truth, fracs=CFG["RETENTION"]):
    out = {}
    total = int(truth.sum()); order = np.argsort(-scores)
    for f in fracs:
        k = max(1, int(round(len(scores) * f)))
        hits = int(truth[order[:k]].sum())
        out[f"recall@{int(f * 100)}"] = hits / total if total else np.nan
        out[f"purity@{int(f * 100)}"] = hits / k
    top = truth[order[:20]].astype(float)
    dcg = (top / np.log2(np.arange(2, len(top) + 2))).sum()
    ideal = np.sort(truth.astype(float))[::-1][:20]
    idcg = (ideal / np.log2(np.arange(2, len(ideal) + 2))).sum()
    out["ndcg@20"] = dcg / idcg if idcg > 0 else np.nan
    return out


if CFG["RUN"]["stage3"]:
    t0 = tic("Stage 3: composition benchmark (query space vs predicate space)")
    seed = CFG["SEEDS"][0]
    space = CFG["SUBSTRATE_SPACE"]
    X = EMB[space]; sp = SPLITS[seed]; lab = LABELS[seed]
    SUB3 = Substrate(X, sp, CFG["BITS"], "orthogonal", seed)
    Yf, Yc, Yt = lab["fit"][:, LAT_COLS], lab["cal"][:, LAT_COLS], lab["test"][:, LAT_COLS]
    Xf, Xc, Xt = X[sp["fit"]], X[sp["cal"]], X[sp["test"]]
    N2I = {n: i for i, n in enumerate(LATENT_NAMES)}

    # arm D: RSA programs
    PROG_D, L_D, conc_rows = [], [], []
    for c, name in enumerate(LATENT_NAMES):
        m = fit_program(SUB3, Yf[:, c], CFG["PILOT_K"], CFG["PILOT_PAIRS"])
        pl = Platt().fit(score_boosted(SUB3.Q_cal, m), Yc[:, c])
        PROG_D.append(m); L_D.append(pl.transform(score_boosted(SUB3.Q_test, m)))
        conc_rows.append(dict(arm="D_rsa", concept=name, **eval_program(m, SUB3, Yc[:, c], Yt[:, c])))
    L_D = np.column_stack(L_D)
    # arm C: FP32 probes
    PROBE_C, L_C = [], []
    for c, name in enumerate(LATENT_NAMES):
        lr = logistic(Xf, Yf[:, c])
        pl = Platt().fit(lr.decision_function(Xc), Yc[:, c])
        PROBE_C.append(lr); L_C.append(pl.transform(lr.decision_function(Xt)))
        conc_rows.append(dict(arm="C_probe", concept=name, **eval_scores(lr.decision_function(Xc), lr.decision_function(Xt), Yc[:, c], Yt[:, c])))
    L_C = np.column_stack(L_C)
    # arm B: zero-shot prompt directions in the substrate space
    DIRS = np.stack([prompt_direction(space, s) for s in LATENT_SPECS])
    raw_f, raw_c, raw_t = Xf @ DIRS.T, Xc @ DIRS.T, Xt @ DIRS.T
    mu, sd = raw_f.mean(0), raw_f.std(0) + 1e-9
    L_B = (raw_t - mu) / sd
    L_Bcal = np.column_stack([Platt().fit(raw_c[:, c], Yc[:, c]).transform(raw_t[:, c]) for c in range(C_LAT)])
    for c, name in enumerate(LATENT_NAMES):
        conc_rows.append(dict(arm="B_zeroshot", concept=name, **eval_scores(raw_c[:, c], raw_t[:, c], Yc[:, c], Yt[:, c])))
    S3C = save_rows("stage3_concepts", conc_rows)
    print("per-concept F1 by arm:")
    print(S3C.pivot_table(index="concept", columns="arm", values="f1").round(3).to_string())
    print(S3C.groupby("arm")[["f1", "ap"]].mean().round(3).to_string())

    # query generation
    df_t = df.iloc[sp["test"]].reset_index(drop=True)
    colours = [v for v in df_t["baseColour"].value_counts().index[:6]]
    cats = [v for v, n_ in df_t["subCategory"].value_counts().items() if n_ >= (300 if not FAST else 60)]
    rng = np.random.default_rng(42)
    QUERIES, attempts = [], 0
    while len(QUERIES) < CFG["N_QUERIES"] and attempts < 4000:
        attempts += 1
        n_pos = int(rng.choice([1, 2], p=[0.5, 0.5])); n_neg = int(rng.random() < 0.6)
        chosen = list(rng.choice(LATENT_NAMES, n_pos + n_neg, replace=False))
        pos, neg = chosen[:n_pos], chosen[n_pos:]
        exact = [("subCategory", str(rng.choice(cats)))]
        if rng.random() < 0.7:
            exact.append(("baseColour", str(rng.choice(colours))))
        words = [CONCEPT_WORD[p] for p in pos] + [v.lower() for _, v in exact[::-1]]
        qstr = " ".join(words) + (" not " + CONCEPT_WORD[neg[0]] if neg else "")
        q = TEXT_ENCODERS[space]([qstr])[0]
        dense = Xt @ q
        ann = np.argsort(-dense)[:CFG["ANN_POOL"]]
        keep = np.ones(len(ann), dtype=bool)
        for col, val in exact:
            keep &= df_t[col].to_numpy()[ann] == val
        pool = ann[keep]
        if len(pool) < CFG["Q_MIN_POOL"]:
            continue
        truth = np.ones(len(pool), dtype=bool)
        for p in pos:
            truth &= Yt[pool, N2I[p]]
        for p in neg:
            truth &= ~Yt[pool, N2I[p]]
        if truth.sum() < CFG["Q_MIN_TRUTH"] or truth.mean() > 0.8:
            continue
        QUERIES.append(dict(query=qstr, pos=pos, neg=neg, exact=exact, pool=pool, truth=truth, dense=dense[pool]))
    print(f"\ngenerated {len(QUERIES)} queries in {attempts} attempts")
    with open(f"{ROOT}/results/queries.json", "w") as fh:
        json.dump([dict(query=q["query"], pos=q["pos"], neg=q["neg"], exact=q["exact"], pool=int(len(q["pool"])), truth=int(q["truth"].sum())) for q in QUERIES], fh, indent=1)

    def score_arms(q):
        pool, pos, neg = q["pool"], q["pos"], q["neg"]
        arms = {"A_dense": q["dense"],
                "B_zeroshot": fuse(L_B[pool], pos, neg, N2I),
                "B_zeroshot_cal": fuse(L_Bcal[pool], pos, neg, N2I),
                "C_probe": fuse(L_C[pool], pos, neg, N2I),
                "D_rsa": fuse(L_D[pool], pos, neg, N2I)}
        yconj = np.ones(len(Xf), dtype=bool)
        for p in pos:
            yconj &= Yf[:, N2I[p]]
        for p in neg:
            yconj &= ~Yf[:, N2I[p]]
        if yconj.sum() >= 20:
            arms["E_direct"] = logistic(Xf, yconj).decision_function(Xt[pool])
        return arms

    rows = []
    for qi, q in enumerate(QUERIES):
        for arm, s in score_arms(q).items():
            rows.append(dict(qid=qi, arm=arm, n_pos=len(q["pos"]), n_neg=len(q["neg"]), pool=len(q["pool"]), **retention_stats(s, q["truth"])))
    S3 = save_rows("stage3_queries", rows)
    metrics = [c for c in S3.columns if c.startswith(("recall", "purity", "ndcg"))]
    summ = []
    for arm, g in S3.groupby("arm"):
        r = dict(arm=arm)
        for m_ in metrics:
            mean, lo, hi = ci(g[m_])
            r[m_] = mean; r[m_ + "_lo"] = lo; r[m_ + "_hi"] = hi
        summ.append(r)
    S3S = pd.DataFrame(summ).set_index("arm")
    print("\nmean over queries with 95% bootstrap CI:")
    show = ["recall@20", "recall@20_lo", "recall@20_hi", "recall@40", "purity@20", "ndcg@20"]
    print(S3S[show].round(3).to_string())
    print("\nby query type (recall@20):")
    print(S3.groupby(["n_pos", "n_neg", "arm"])["recall@20"].mean().unstack().round(3).to_string())

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    fr = [int(f * 100) for f in CFG["RETENTION"]]
    for arm in S3S.index:
        for ax, kind in zip(axes, ["recall", "purity"]):
            mean = [S3S.loc[arm, f"{kind}@{f}"] for f in fr]
            lo = [S3S.loc[arm, f"{kind}@{f}_lo"] for f in fr]; hi = [S3S.loc[arm, f"{kind}@{f}_hi"] for f in fr]
            ax.plot(fr, mean, marker="o", label=arm); ax.fill_between(fr, lo, hi, alpha=0.15)
    for ax, kind in zip(axes, ["recall", "purity"]):
        ax.set_xlabel("% of filtered pool retained"); ax.set_ylabel(kind); ax.set_xscale("log"); ax.grid(alpha=.3)
    axes[0].legend(fontsize=8); plt.tight_layout(); plt.savefig(f"{ROOT}/results/stage3_retention.png", dpi=120); plt.close()

    def within(a, val):
        return bool(S3S.loc[a, "recall@20_lo"] <= val <= S3S.loc[a, "recall@20_hi"])
    rB, rC, rD = S3S.loc["B_zeroshot", "recall@20"], S3S.loc["C_probe", "recall@20"], S3S.loc["D_rsa", "recall@20"]
    rE = S3S.loc["E_direct", "recall@20"] if "E_direct" in S3S.index else float("nan")
    if within("C_probe", rB) and within("D_rsa", rB):
        DECISIONS["3"] = "zero-shot fusion ties supervised arms: supervision adds nothing here; RSA value is compute only (see Stage 6)"
    elif rC - rD > 0.05:
        DECISIONS["3"] = f"FP32 probe beats RSA by {rC - rD:.3f} recall@20: sparse 4-bit program loses too much; find the K where it catches up (Stage 4)"
    elif within("C_probe", rD) and rD > rB:
        DECISIONS["3"] = "RSA ties the FP32 probe and beats zero-shot: valid cheap executor of supervision"
    else:
        DECISIONS["3"] = f"mixed: B={rB:.3f} C={rC:.3f} D={rD:.3f} E={rE:.3f} recall@20"
    if not np.isnan(rE) and rE - rC > 0.05:
        DECISIONS["3_comp"] = f"direct conjunction beats composed probes by {rE - rC:.3f}: composition itself is the bottleneck; prioritise 7b"
    for k_ in ["3", "3_comp"]:
        if k_ in DECISIONS:
            print(f"DECISION {k_}: {DECISIONS[k_]}")
    print(f"stage 3 done in {time.time() - t0:.0f}s")

# %%
# ---- Cell 11: Stage 4 budget sweep ----------------------------------------
if CFG["RUN"]["stage4"] and CFG["RUN"]["stage3"]:
    t0 = tic("Stage 4: budget sweep (seed 0)")
    Ks = [8, 16, 24, 32, 48, 64, 96, 192, 384] if not FAST else [8, 24, 64, 384]
    rows = []
    for bits in ([2, 4, 8] if not FAST else [2, 4]):
        sub = Substrate(X, sp, bits, "orthogonal", seed)
        for K in Ks:
            for pairs in ([0, 2] if bits < 8 else [0]):
                Ls, f1s, aps = [], [], []
                for c in range(C_LAT):
                    m = fit_program(sub, Yf[:, c], K, pairs)
                    r = eval_program(m, sub, Yc[:, c], Yt[:, c]); f1s.append(r["f1"]); aps.append(r["ap"])
                    Ls.append(Platt().fit(score_boosted(sub.Q_cal, m), Yc[:, c]).transform(score_boosted(sub.Q_test, m)))
                L = np.column_stack(Ls)
                rec = np.mean([retention_stats(fuse(L[q["pool"]], q["pos"], q["neg"], N2I), q["truth"])["recall@20"] for q in QUERIES])
                rows.append(dict(bits=bits, K=K, pairs=pairs, lut_ops=K + pairs, bytes_per_item=X.shape[1] * bits / 8,
                                 f1=float(np.mean(f1s)), ap=float(np.mean(aps)), recall20=float(rec)))
                print(f"  bits={bits} K={K} pairs={pairs}: F1={np.mean(f1s):.3f} recall@20={rec:.3f}", flush=True)
    S4 = save_rows("stage4", rows)
    # sample-size test for the pair gain
    rows = []
    sub = SUB3
    for n_fit in ([2000, 4000, 8000, 16000, len(Yf)] if not FAST else [1000, 2000, len(Yf)]):
        idx = np.arange(min(n_fit, len(Yf)))
        for pairs in [0, 4]:
            f1s = []
            for c in range(C_LAT):
                m = fit_boosted_lut(sub.Q_fit[idx], Yf[idx, c], k=CFG["PILOT_K"], n_bins=sub.nb, candidate_pool=CFG["CAND_POOL"])
                m = add_pair_interactions(sub.Q_fit[idx], Yf[idx, c], m, n_pairs=pairs)
                f1s.append(eval_program(m, sub, Yc[:, c], Yt[:, c])["f1"])
            rows.append(dict(n_fit=len(idx), pairs=pairs, f1=float(np.mean(f1s))))
    S4b = save_rows("stage4_fitsize", rows)
    print("\npair gain vs fit-set size:")
    print(S4b.pivot(index="n_fit", columns="pairs", values="f1").round(4).to_string())

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for bits, g in S4[S4.pairs == 0].groupby("bits"):
        axes[0].plot(g.lut_ops, g.f1, marker="o", label=f"{bits}-bit"); axes[1].plot(g.lut_ops, g.recall20, marker="o", label=f"{bits}-bit")
    probe_f1 = S3C[S3C.arm == "C_probe"].f1.mean(); probe_rec = S3S.loc["C_probe", "recall@20"]
    axes[0].axhline(probe_f1, ls="--", c="k", label="FP32 probe"); axes[1].axhline(probe_rec, ls="--", c="k", label="FP32 probe")
    axes[0].set_xscale("log"); axes[1].set_xscale("log"); axes[0].set_xlabel("LUT ops per concept"); axes[1].set_xlabel("LUT ops per concept")
    axes[0].set_ylabel("mean F1 (16 latent concepts)"); axes[1].set_ylabel("recall@20% (queries)"); axes[0].legend(); plt.tight_layout()
    plt.savefig(f"{ROOT}/results/stage4_pareto.png", dpi=120); plt.close()
    ok = S4[(S4.f1 >= probe_f1 - 0.01)].sort_values(["lut_ops", "bits"])
    DECISIONS["4"] = (f"smallest config within 0.01 F1 of FP32 probe: bits={int(ok.iloc[0].bits)} K={int(ok.iloc[0].K)} pairs={int(ok.iloc[0].pairs)}"
                      if len(ok) else "no sparse config within 0.01 F1 of the FP32 probe")
    print(f"DECISION 4: {DECISIONS['4']}")

# %%
# ---- Cell 12: Stage 5 VLM teacher check (optional, slow) ------------------
if CFG["RUN"]["stage5"] and not SYNTHETIC and CFG["RUN"]["stage3"]:
    t0 = tic("Stage 5: VLM teacher check")
    try:
        from transformers import AutoProcessor, AutoModelForVision2Seq
        vproc = AutoProcessor.from_pretrained(CFG["VLM_MODEL"])
        vmodel = AutoModelForVision2Seq.from_pretrained(CFG["VLM_MODEL"], torch_dtype=torch.float16).to(DEVICE).eval()
        rng = np.random.default_rng(5)
        sub_idx = rng.choice(sp["test"], CFG["VLM_SUBSET"], replace=False)
        questions = {"minimalist": "Is this fashion item minimalist and understated in design?",
                     "office_appropriate": "Would this item be appropriate to wear in a formal office?",
                     "technical_sporty": "Is this a technical or sporty athletic item?",
                     "quiet_luxury": "Does this item look like understated, logo-free, premium quality fashion?"}

        @torch.no_grad()
        def ask(img, question):
            msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question + " Answer yes or no."}]}]
            prompt = vproc.apply_chat_template(msgs, add_generation_prompt=True)
            inputs = vproc(text=prompt, images=[img], return_tensors="pt").to(DEVICE)
            out = vmodel.generate(**inputs, max_new_tokens=3, do_sample=False)
            txt = vproc.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0].lower()
            return int("yes" in txt)

        vl = {c: [] for c in CFG["VLM_CONCEPTS"] if c in questions}
        for i, gi in enumerate(sub_idx):
            img = DS[int(VALID_IDX[gi])]["image"].convert("RGB")
            for c in vl:
                vl[c].append(ask(img, questions[c]))
            if i % 100 == 0:
                print(f"  vlm {i}/{len(sub_idx)}", flush=True)
        rows = []
        tpos = {gi: k for k, gi in enumerate(sp["test"])}
        tloc = np.array([tpos[g] for g in sub_idx])
        for c, ans in vl.items():
            ans = np.array(ans); ci_ = N2I[c]
            clip_lab = Yt[tloc, ci_]
            r = dict(concept=c, vlm_prevalence=float(ans.mean()), kappa_clip_vs_vlm=float(cohen_kappa_score(clip_lab, ans)))
            for arm, L in [("B_zeroshot", L_B), ("C_probe", L_C), ("D_rsa", L_D)]:
                s = L[tloc, ci_]
                r[f"{arm}_ap_vs_vlm"] = float(average_precision_score(ans, s)) if ans.sum() else float("nan")
                r[f"{arm}_ap_vs_clip"] = float(average_precision_score(clip_lab, s))
            rows.append(r)
        S5 = save_rows("stage5", rows)
        print(S5.round(3).to_string())
        DECISIONS["5"] = ("CLIP teacher is noise on some concepts (kappa<0.4): " + ", ".join(S5[S5.kappa_clip_vs_vlm < 0.4].concept)
                          if (S5.kappa_clip_vs_vlm < 0.4).any() else "CLIP and VLM labels agree moderately or better on all checked concepts")
        print(f"DECISION 5: {DECISIONS['5']}")
    except Exception as e:
        print("Stage 5 failed (optional stage):", repr(e))

# %%
# ---- Cell 13: Stage 6 throughput ------------------------------------------
if CFG["RUN"]["stage6"] and CFG["RUN"]["stage3"]:
    t0 = tic("Stage 6: throughput, LUT programs vs dot products")
    n6 = CFG["THROUGHPUT_N"]
    reps = int(np.ceil(n6 / len(SUB3.Q_test)))
    Q6 = np.tile(SUB3.Q_test, (reps, 1))[:n6].astype(np.uint8)
    X6 = np.tile(Xt, (reps, 1))[:n6].astype(np.float32)
    X6i8 = np.clip(np.round(X6 * 127 / np.abs(X6).max()), -127, 127).astype(np.int8)
    P6 = ((Q6[:, 0::2] << 4) | Q6[:, 1::2]).astype(np.uint8)  # packed nibbles, 192 bytes/item
    models = PROG_D[:3]
    K6 = len(models[0]["coords"])
    coords3 = np.stack([m["coords"] for m in models]).astype(np.int64)
    tables3 = np.stack([m["tables"] for m in models]).astype(np.float32)
    W3 = np.stack([lr.coef_[0] for lr in PROBE_C[:3]]).T.astype(np.float32)
    W3i8 = np.clip(np.round(W3 * 127 / np.abs(W3).max()), -127, 127).astype(np.int8)

    def bench(fn, reps=3):
        best = float("inf")
        for _ in range(reps):
            t = time.perf_counter(); fn(); best = min(best, time.perf_counter() - t)
        return n6 / best

    def lut_numpy(nconc):
        out = np.zeros((n6, nconc), dtype=np.float32)
        for c in range(nconc):
            out[:, c] = tables3[c][np.arange(K6)[None, :], Q6[:, coords3[c]]].sum(1)
        return out

    rows = [dict(impl="LUT numpy gather (unpacked uint8)", concepts=1, items_per_s=bench(lambda: lut_numpy(1))),
            dict(impl="LUT numpy gather (unpacked uint8)", concepts=3, items_per_s=bench(lambda: lut_numpy(3))),
            dict(impl="FP32 dot (BLAS)", concepts=1, items_per_s=bench(lambda: X6 @ W3[:, :1])),
            dict(impl="FP32 dot (BLAS)", concepts=3, items_per_s=bench(lambda: X6 @ W3))]
    try:
        import numba

        @numba.njit(parallel=True, fastmath=True, cache=False)
        def lut_nb(Q, coords, tables, out):
            n = Q.shape[0]; C = coords.shape[0]; K = coords.shape[1]
            for i in numba.prange(n):
                for c in range(C):
                    s = 0.0
                    for k in range(K):
                        s += tables[c, k, Q[i, coords[c, k]]]
                    out[i, c] = s

        @numba.njit(parallel=True, fastmath=True, cache=False)
        def lut_nb_packed(P, coords, tables, out):
            n = P.shape[0]; C = coords.shape[0]; K = coords.shape[1]
            for i in numba.prange(n):
                for c in range(C):
                    s = 0.0
                    for k in range(K):
                        j = coords[c, k]
                        byte = P[i, j >> 1]
                        v = (byte >> 4) if (j & 1) == 0 else (byte & 15)
                        s += tables[c, k, v]
                    out[i, c] = s

        @numba.njit(parallel=True, cache=False)
        def dot_i8(X, W, out):
            n, d = X.shape; C = W.shape[1]
            for i in numba.prange(n):
                for c in range(C):
                    acc = 0
                    for j in range(d):
                        acc += np.int32(X[i, j]) * np.int32(W[j, c])
                    out[i, c] = acc

        for C in [1, 3]:
            out = np.zeros((n6, C), dtype=np.float32); outi = np.zeros((n6, C), dtype=np.int32)
            lut_nb(Q6, coords3[:C], tables3[:C], out); lut_nb_packed(P6, coords3[:C], tables3[:C], out); dot_i8(X6i8, W3i8[:, :C], outi)  # compile
            rows.append(dict(impl="LUT numba (unpacked uint8)", concepts=C, items_per_s=bench(lambda: lut_nb(Q6, coords3[:C], tables3[:C], out))))
            rows.append(dict(impl="LUT numba (packed nibbles)", concepts=C, items_per_s=bench(lambda: lut_nb_packed(P6, coords3[:C], tables3[:C], out))))
            rows.append(dict(impl="int8 dot numba", concepts=C, items_per_s=bench(lambda: dot_i8(X6i8, W3i8[:, :C], outi))))
    except Exception as e:
        print("numba rows skipped:", repr(e))
    S6 = save_rows("stage6", rows)
    S6["items_per_s"] = S6.items_per_s.round(0)
    print(S6.to_string(index=False))
    try:
        best_lut = S6[S6.impl.str.startswith("LUT") & (S6.concepts == 3)].items_per_s.max()
        best_dot = S6[S6.impl.str.contains("dot") & (S6.concepts == 3)].items_per_s.max()
        ratio = best_lut / best_dot
        DECISIONS["6"] = (f"LUT scoring {ratio:.1f}x faster than best dot product on 3 concepts: keep the systems claim" if ratio >= 5
                          else f"LUT scoring only {ratio:.1f}x vs dot products: drop 'hardware-cheap', reframe as memory (192 B for any number of concepts)")
        print(f"DECISION 6: {DECISIONS['6']}")
    except Exception as e:
        print("decision skipped:", e)

# %%
# ---- Cell 14: Stage 7 improvements ----------------------------------------
if CFG["RUN"]["stage7"] and CFG["RUN"]["stage3"]:
    t0 = tic("Stage 7a: shared coordinate dictionary")
    def fit_shared_dictionary(Q, Y, k_shared, nb, lam=1.0):
        n, d = Q.shape; C = Y.shape[1]; Y = Y.astype(np.float64)
        priors = np.clip(Y.mean(0), 1e-4, 1 - 1e-4); b0 = np.log(priors / (1 - priors))
        S = np.tile(b0, (n, 1)); remaining = list(range(d)); coords = []; tables = np.zeros((C, k_shared, nb))
        for t in range(k_shared):
            total = np.zeros(len(remaining)); Gs, Hs = [], []
            Qr = Q[:, remaining]
            for c in range(C):
                p = sigmoid(S[:, c]); g = Y[:, c] - p; h = p * (1 - p)
                G, H = _bin_stats(Qr, g, h, nb); Gs.append(G); Hs.append(H)
                total += 0.5 * ((G ** 2) / (H + lam)).sum(1)
            i = int(np.argmax(total)); j = remaining.pop(i); coords.append(j)
            for c in range(C):
                f = Gs[c][i] / (Hs[c][i] + lam); tables[c, t] = f; S[:, c] += f[Q[:, j]]
        return [dict(intercept=float(b0[c]), coords=np.array(coords), tables=tables[c], pairs=[], n_bins=nb, gains=[]) for c in range(C)]

    rows = []
    union = len(set(int(c) for m in PROG_D for c in m["coords"]))
    rows.append(dict(method=f"per-concept K={CFG['PILOT_K']}", distinct_coords=union, lut_ops_per_concept=CFG["PILOT_K"] + CFG["PILOT_PAIRS"],
                     f1=S3C[S3C.arm == "D_rsa"].f1.mean(), recall20=S3S.loc["D_rsa", "recall@20"]))
    for ks in ([32, 48, 64] if not FAST else [32]):
        ms = fit_shared_dictionary(SUB3.Q_fit, Yf, ks, SUB3.nb)
        L = np.column_stack([Platt().fit(score_boosted(SUB3.Q_cal, m), Yc[:, c]).transform(score_boosted(SUB3.Q_test, m)) for c, m in enumerate(ms)])
        f1 = np.mean([eval_program(m, SUB3, Yc[:, c], Yt[:, c])["f1"] for c, m in enumerate(ms)])
        rec = np.mean([retention_stats(fuse(L[q["pool"]], q["pos"], q["neg"], N2I), q["truth"])["recall@20"] for q in QUERIES])
        rows.append(dict(method=f"shared dictionary k={ks}", distinct_coords=ks, lut_ops_per_concept=ks, f1=float(f1), recall20=float(rec)))
    S7a = save_rows("stage7a", rows); print(S7a.round(3).to_string(index=False))

    t0 = tic("Stage 7b: correlation-aware fusion")
    L_Dc = np.column_stack([Platt().fit(score_boosted(SUB3.Q_cal, m), Yc[:, c]).transform(score_boosted(SUB3.Q_cal, m)) for c, m in enumerate(PROG_D)])
    L_Cc = np.column_stack([Platt().fit(PROBE_C[c].decision_function(Xc), Yc[:, c]).transform(PROBE_C[c].decision_function(Xc)) for c in range(C_LAT)])

    def build_corrections(Lc, nbin=6, shrink=20.0):
        P = sigmoid(Lc); tabs = {}
        for c1, c2 in itertools.combinations(range(C_LAT), 2):
            for s1 in (1, -1):
                for s2 in (1, -1):
                    p1 = P[:, c1] if s1 > 0 else 1 - P[:, c1]; p2 = P[:, c2] if s2 > 0 else 1 - P[:, c2]
                    y1 = Yc[:, c1] if s1 > 0 else ~Yc[:, c1]; y2 = Yc[:, c2] if s2 > 0 else ~Yc[:, c2]
                    e1 = np.quantile(p1, np.linspace(0, 1, nbin + 1)[1:-1]); e2 = np.quantile(p2, np.linspace(0, 1, nbin + 1)[1:-1])
                    b1 = np.searchsorted(e1, p1); b2 = np.searchsorted(e2, p2); cell = b1 * nbin + b2
                    joint = np.bincount(cell, weights=(y1 & y2).astype(float), minlength=nbin * nbin)
                    indep = np.bincount(cell, weights=p1 * p2, minlength=nbin * nbin)
                    cnt = np.bincount(cell, minlength=nbin * nbin)
                    corr = np.log((joint + 1e-3) / (indep + 1e-3)) * (cnt / (cnt + shrink))
                    tabs[(c1, c2, s1, s2)] = (e1, e2, corr.reshape(nbin, nbin))
        return tabs

    def fuse_corr(L, pos, neg, tabs):
        s = fuse(L, pos, neg, N2I); P = sigmoid(L)
        lits = [(N2I[c], 1) for c in pos] + [(N2I[c], -1) for c in neg]
        for (a, sa), (b, sb) in itertools.combinations(lits, 2):
            if a > b:
                (a, sa), (b, sb) = (b, sb), (a, sa)
            e1, e2, T = tabs[(a, b, sa, sb)]
            p1 = P[:, a] if sa > 0 else 1 - P[:, a]; p2 = P[:, b] if sb > 0 else 1 - P[:, b]
            s += T[np.searchsorted(e1, p1), np.searchsorted(e2, p2)]
        return s

    rows = []
    for arm, Lcal, Ltest in [("C_probe", L_Cc, L_C), ("D_rsa", L_Dc, L_D)]:
        tabs = build_corrections(Lcal)
        for q in QUERIES:
            base = retention_stats(fuse(Ltest[q["pool"]], q["pos"], q["neg"], N2I), q["truth"])
            corr = retention_stats(fuse_corr(Ltest[q["pool"]], q["pos"], q["neg"], tabs), q["truth"])
            rows.append(dict(arm=arm, fusion="independent", multi=len(q["pos"]) + len(q["neg"]) > 1, **base))
            rows.append(dict(arm=arm, fusion="corr_aware", multi=len(q["pos"]) + len(q["neg"]) > 1, **corr))
    S7b = save_rows("stage7b", rows)
    print(S7b[S7b.multi].groupby(["arm", "fusion"])[["recall@20", "purity@20", "ndcg@20"]].mean().round(3).to_string())

    t0 = tic("Stage 7c: zero-shot compile from pseudo-labels")
    rows = []
    pseudo_sources = {"substrate_prompt": raw_f, "clip_txt_prompt": EMB["clip_txt"][sp["fit"]] @ np.stack([prompt_direction("clip_txt", s) for s in LATENT_SPECS]).T}
    for src, Sf in pseudo_sources.items():
        thr = np.quantile(Sf, 1 - CFG["PREVALENCE"], axis=0)
        f1s, aps = [], []
        for c in range(C_LAT):
            m = fit_program(SUB3, Sf[:, c] >= thr[c], CFG["PILOT_K"], CFG["PILOT_PAIRS"])
            r = eval_program(m, SUB3, Yc[:, c], Yt[:, c]); f1s.append(r["f1"]); aps.append(r["ap"])
        rows.append(dict(method=f"zero-shot compiled ({src})", f1=float(np.mean(f1s)), ap=float(np.mean(aps))))
    rows.append(dict(method="B zero-shot direct", f1=S3C[S3C.arm == "B_zeroshot"].f1.mean(), ap=S3C[S3C.arm == "B_zeroshot"].ap.mean()))
    rows.append(dict(method="D supervised RSA", f1=S3C[S3C.arm == "D_rsa"].f1.mean(), ap=S3C[S3C.arm == "D_rsa"].ap.mean()))
    S7c = save_rows("stage7c", rows); print(S7c.round(3).to_string(index=False))

    t0 = tic("Stage 7d: few-shot personal predicates")
    rng = np.random.default_rng(7); rows = []
    n_users = 200 if not FAST else 30
    for u in range(n_users):
        c1, c2 = rng.choice(C_LAT, 2, replace=False)
        taste_f = Yf[:, c1] & Yf[:, c2]; taste_t = Yt[:, c1] & Yt[:, c2]
        posi = np.where(taste_f)[0]
        if len(posi) < 40 or taste_t.sum() < 20:
            continue
        liked = rng.choice(posi, 30, replace=False); negs = rng.choice(len(Yf), 300, replace=False)
        idx = np.concatenate([liked, negs]); y = np.concatenate([np.ones(30, bool), taste_f[negs]])
        uvec = l2n(Xf[liked].mean(0)); s_mean = Xt @ uvec
        rows.append(dict(user=u, method="mean liked embedding (FP32)", p20=float(taste_t[np.argsort(-s_mean)[:20]].mean())))
        lr = logistic(Xf[idx], y, C=0.5); s_lr = lr.decision_function(Xt)
        rows.append(dict(user=u, method="FP32 logistic 30 vs 300", p20=float(taste_t[np.argsort(-s_lr)[:20]].mean())))
        for K in [8, 16, 24]:
            m = fit_boosted_lut(SUB3.Q_fit[idx], y, k=K, n_bins=SUB3.nb, candidate_pool=CFG["CAND_POOL"], lam=2.0)
            s_m = score_boosted(SUB3.Q_test, m)
            rows.append(dict(user=u, method=f"RSA program K={K}", p20=float(taste_t[np.argsort(-s_m)[:20]].mean())))
    S7d = save_rows("stage7d", rows)
    print(S7d.groupby("method").p20.agg(["mean", "std", "count"]).round(3).to_string())

# %%
# ---- Cell 15: summary ------------------------------------------------------
lines = [f"# RSA Lab summary", f"generated {time.strftime('%Y-%m-%d %H:%M')} | substrate={CFG['SUBSTRATE_SPACE']} | items={N} | seeds={CFG['SEEDS']} | FAST={FAST} SYNTHETIC={SYNTHETIC}", ""]
lines += ["## Decisions", ""] + [f"- **{k}**: {v}" for k, v in DECISIONS.items()] + [""]


def add_table(title, df_):
    lines.extend([f"## {title}", "", "```", df_.to_string(), "```", ""])


for name in ["stage1", "stage2", "stage3_concepts", "stage3_queries", "stage4", "stage4_fitsize", "stage5", "stage6", "stage7a", "stage7b", "stage7c", "stage7d"]:
    path = f"{ROOT}/results/{name}.csv"
    if not os.path.exists(path):
        continue
    d = pd.read_csv(path)
    if name == "stage1":
        add_table("Stage 1a/1b: rotation and pairs (mean F1, orthogonal unless noted)", d[d.method.isna()].groupby(["rotation", "pairs", "group"]).f1.agg(["mean", "std"]).round(4))
        add_table("Stage 1c/1d: sparse baselines", d[d.method.notna()].groupby(["method", "K", "group"]).f1.agg(["mean", "std"]).round(4))
    elif name == "stage2":
        add_table("Stage 2: probe ceilings (F1/AP by space)", d.groupby("space")[["f1", "ap"]].agg(["mean", "std"]).round(4))
    elif name == "stage3_concepts":
        add_table("Stage 3: per-concept F1/AP by arm", d.groupby("arm")[["f1", "ap"]].mean().round(4))
    elif name == "stage3_queries":
        add_table("Stage 3: query benchmark (mean over queries)", d.groupby("arm")[[c for c in d.columns if "@" in c]].mean().round(4))
    elif name == "stage4":
        add_table("Stage 4: budget sweep", d.round(4))
    elif name == "stage7b":
        add_table("Stage 7b: fusion (multi-literal queries)", d[d.multi].groupby(["arm", "fusion"])[["recall@20", "purity@20", "ndcg@20"]].mean().round(4))
    elif name == "stage7d":
        add_table("Stage 7d: few-shot personal predicates P@20", d.groupby("method").p20.agg(["mean", "std", "count"]).round(4))
    else:
        add_table(name, d.round(4))
with open(f"{ROOT}/results/summary.md", "w") as fh:
    fh.write("\n".join(lines))
print("\n".join(lines[:40]))
print(f"\nfull summary: {ROOT}/results/summary.md")
