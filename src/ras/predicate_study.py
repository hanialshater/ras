"""Split-safe WANDS attribute probes and compact linear/token-max controls."""
from __future__ import annotations

from collections import defaultdict
import hashlib
import re
import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, precision_recall_curve

from ras.binary import two_level_corrections, quantize_weight_int4, int4_weight_bitplanes
from ras.calibration import fit_scalar_calibrator
from ras.composition import compose_logprob

# Fixed before looking at held-out outcomes. Labels mean the literal metadata
# field, not an inferred aesthetic or independent human semantic judgment.
CONCEPTS = [
    ('style_modern', 'dsprimaryproductstyle', 'modern', 'modern style'),
    ('style_traditional', 'dsprimaryproductstyle', 'traditional', 'traditional style'),
    ('style_industrial', 'dsprimaryproductstyle', 'industrial', 'industrial style'),
    ('style_glam', 'dsprimaryproductstyle', 'glam', 'glam style'),
    ('upholstered', 'upholstered', 'yes', 'upholstered'),
    ('outdoor', 'outdooruse', 'yes', 'suitable for outdoor use'),
    ('water_resistant', 'waterresistant', 'yes', 'water resistant'),
    ('stain_resistant', 'stainresistant', 'yes', 'stain resistant'),
    ('storage', 'storageincluded', 'yes', 'with storage'),
    ('assembly', 'adultassemblyrequired', 'yes', 'requires adult assembly'),
    ('drawers', 'drawersincluded', 'yes', 'with drawers'),
    ('shelves', 'shelvesincluded', 'yes', 'with shelves'),
]


def attributes(text):
    result = defaultdict(set)
    for pair in str(text).split('|'):
        key, separator, value = pair.partition(':')
        key, value = ''.join(key.lower().split()), ' '.join(value.lower().split())
        if separator and key and value:
            result[key].add(value)
    return result


def labels_from_products(products):
    y = np.full((len(products), len(CONCEPTS)), -1, dtype=np.int8)
    conflicts = np.zeros(len(CONCEPTS), dtype=int)
    for i, row in enumerate(products):
        attrs = attributes(row['product_features'])
        for c, (_, field, positive, _) in enumerate(CONCEPTS):
            values = attrs[field]
            if len(values) > 1:
                conflicts[c] += 1
            elif len(values) == 1:
                value = next(iter(values))
                if value in {'unknown', 'unavailable', 'n/a', 'none', 'not specified'}:
                    continue
                if positive != 'yes' or value in ['yes', 'no']:
                    y[i, c] = int(value == positive)
    audit = [dict(concept=name, field=field, positive=value, positives=int((y[:, c] == 1).sum()),
        negatives=int((y[:, c] == 0).sum()), unknown=int((y[:, c] < 0).sum()), conflicts=int(conflicts[c]))
        for c, (name, field, value, _) in enumerate(CONCEPTS)]
    return y, audit


def normalized_title(text):
    return re.sub(r'\W+', ' ', str(text).casefold()).strip()


def group_split(products, seed):
    # Exact normalized-title groups never cross partitions. This is not a
    # product-family guarantee: WANDS does not publish family IDs.
    groups = [normalized_title(p['product_name']) or str(p['product_id']) for p in products]
    unique = sorted(set(groups))
    shuffled = np.random.default_rng(seed).permutation(len(unique))
    assignment = {}
    for j, index in enumerate(shuffled):
        assignment[unique[index]] = 0 if j < .6*len(unique) else 1 if j < .8*len(unique) else 2
    return np.array([assignment[g] for g in groups], dtype=np.int8), groups


def text_for_product(row, view):
    fields = ['product_name', 'product_description'] if view == 'redacted' else [
        'product_name', 'product_class', 'product_features', 'product_description']
    return '\n'.join(f'{key}: {str(row[key]).strip()}' for key in fields if str(row[key]).strip())


def eligible_concepts(y, split, min_fit=30, min_cal=10):
    # Test prevalence never chooses which concepts enter the experiment.
    return [c for c in range(y.shape[1]) if all(
        (y[split == part, c] == value).sum() >= minimum
        for part, minimum in [(0, min_fit), (1, min_cal)] for value in [0, 1])]


def make_plans(y_fit, concepts, seed=7, pairs=12, min_positive=20):
    plans = [dict(name=name, columns=[c], signs=[1], text=phrase+' products')
             for c, (name, _, _, phrase) in enumerate(concepts)]
    candidates = []
    for a in range(len(concepts)):
        for b in range(a+1, len(concepts)):
            if concepts[a][1] == concepts[b][1]:
                continue
            for sign in [1, -1]:
                mask = (y_fit[:, [a, b]] >= 0).all(1)
                relevant = (y_fit[mask, a] == 1) & (y_fit[mask, b] == (1 if sign == 1 else 0))
                if relevant.sum() >= min_positive and (~relevant).sum() >= min_positive:
                    candidates.append(dict(name=concepts[a][0]+('_and_' if sign == 1 else '_not_')+concepts[b][0],
                        columns=[a, b], signs=[1, sign],
                        text=concepts[a][3]+' products '+('and ' if sign == 1 else 'not ')+concepts[b][3]))
    order = np.random.default_rng(seed).permutation(len(candidates))[:pairs]
    return plans + [candidates[i] for i in order]


def fit_linear(x, y, fit, clusters=0, seed=7, minimum=10):
    """Global or hard-routed local logistic heads, with explicit global fallback."""
    c, d = y.shape[1], x.shape[1]
    weights = np.zeros((1, c, d), dtype=np.float32)
    bias = np.zeros((1, c), dtype=np.float32)
    for j in range(c):
        rows = fit[y[fit, j] >= 0]
        model = LogisticRegression(C=10, max_iter=500, solver='lbfgs')
        model.fit(x[rows], y[rows, j])
        weights[0, j], bias[0, j] = model.coef_[0], model.intercept_[0]
    routes = np.zeros(len(x), dtype=np.int32)
    centers = np.empty((0, d), dtype=np.float32)
    fallback = np.zeros((1, c), dtype=bool)
    if clusters:
        km = KMeans(n_clusters=clusters, random_state=seed, n_init=3, max_iter=100).fit(x[fit])
        routes = km.predict(x).astype(np.int32)
        centers = km.cluster_centers_.astype(np.float32)
        weights = np.repeat(weights, clusters, axis=0)
        bias = np.repeat(bias, clusters, axis=0)
        fallback = np.ones((clusters, c), dtype=bool)
        for cluster in range(clusters):
            for j in range(c):
                rows = fit[(routes[fit] == cluster) & (y[fit, j] >= 0)]
                if min((y[rows, j] == 0).sum(), (y[rows, j] == 1).sum()) < minimum:
                    continue
                model = LogisticRegression(C=10, max_iter=500, solver='lbfgs')
                model.fit(x[rows], y[rows, j])
                weights[cluster, j], bias[cluster, j] = model.coef_[0], model.intercept_[0]
                fallback[cluster, j] = False
    return dict(weights=weights, bias=bias, routes=routes, centers=centers, fallback=fallback)


def score_linear(x, head):
    scores = np.empty((len(x), head['weights'].shape[1]), dtype=np.float32)
    for route in range(len(head['weights'])):
        rows = np.flatnonzero(head['routes'] == route)
        scores[rows] = x[rows] @ head['weights'][route].T + head['bias'][route]
    return scores


def binary_arrays(values, centroid):
    residual = np.asarray(values, dtype=np.float32)-centroid
    bits = (residual > 0).astype('uint8')
    return np.packbits(bits, axis=1, bitorder='little'), two_level_corrections(residual, bits)


def binary_reconstruct(packed, corrections, centroid):
    bits = np.unpackbits(packed, axis=1, count=len(centroid), bitorder='little').astype(bool)
    return centroid + np.where(bits, corrections[:, 1, None], corrections[:, 0, None])


def quantized_head(head):
    decoded = np.empty_like(head['weights'])
    planes, metadata = [], []
    for cluster in range(len(decoded)):
        cp, cm = [], []
        for j, weight in enumerate(head['weights'][cluster]):
            q, decoded[cluster, j], lo, scale = quantize_weight_int4(weight)
            cp.append(int4_weight_bitplanes(q))
            cm.append([lo, scale, head['bias'][cluster, j]])
        planes.append(cp); metadata.append(cm)
    return head | dict(weights=decoded), np.asarray(planes, dtype=np.uint8), np.asarray(metadata, dtype=np.float32)


def calibrate(raw, y, cal):
    scores, parameters, thresholds = np.empty_like(raw), [], []
    for j in range(y.shape[1]):
        rows = cal[y[cal, j] >= 0]
        model = fit_scalar_calibrator(raw[rows, j], y[rows, j])
        scores[:, j] = model.transform(raw[:, j])
        precision, recall, cuts = precision_recall_curve(y[rows, j], scores[rows, j])
        f1 = 2*precision[:-1]*recall[:-1]/np.maximum(precision[:-1]+recall[:-1], 1e-12)
        threshold = float(cuts[np.argmax(f1)]) if len(cuts) else 0.
        parameters.append([model.a, model.b]); thresholds.append(threshold)
    return scores, np.asarray(parameters, dtype=np.float32), np.asarray(thresholds)


def predicate_metrics(scores, y, test, thresholds, concepts, method):
    out = []
    for j, concept in enumerate(concepts):
        rows = test[y[test, j] >= 0]
        truth = y[rows, j]
        out.append(dict(method=method, concept=concept[0], known=len(rows), positives=int(truth.sum()),
            ap=float(average_precision_score(truth, scores[rows, j])) if truth.sum() else float('nan'),
            f1=float(f1_score(truth, scores[rows, j] >= thresholds[j], zero_division=0)) if len(rows) else float('nan')))
    return out


def pools_and_truth(y, test, plans, dense_scores, pool_size, k, ids):
    from ras.muvera_diagnostics import topk_numpy
    tie = np.argsort(np.argsort(np.asarray(ids)))
    out = []
    for qi, plan in enumerate(plans):
        cols, signs = plan['columns'], np.array(plan['signs'])
        universe = test[(y[test][:, cols] >= 0).all(1)]
        truth = (y[universe][:, cols] == (signs > 0)).all(1)
        pool = universe[topk_numpy(dense_scores[qi, universe], tie[universe], min(pool_size, len(universe)))] if len(universe) else universe
        relevant = set(universe[truth].tolist())
        pool_positive = len(set(pool.tolist()) & relevant)
        n = len(relevant)
        out.append(dict(plan=plan, pool=pool, relevant=relevant, known_universe=len(universe),
            full_relevant=n, pool_relevant=pool_positive, candidate_coverage=pool_positive/n if n else float('nan'),
            oracle_recall=min(k, pool_positive)/n if n else float('nan')))
    return out


def ranking_metrics(scores, controls, method, k, ids, dense=False):
    from ras.muvera_diagnostics import topk_numpy
    tie = np.argsort(np.argsort(np.asarray(ids)))
    out = []
    for qi, control in enumerate(controls):
        plan, pool = control['plan'], control['pool']
        values = scores[qi, pool] if dense else compose_logprob(scores[pool][:, plan['columns']], plan['signs'])
        selected = pool[topk_numpy(values, tie[pool], min(k, len(pool)))] if len(pool) else pool
        hits = len(set(selected.tolist()) & control['relevant'])
        n = control['full_relevant']
        recall = hits/n if n else float('nan')
        oracle = control['oracle_recall']
        coverage = control['candidate_coverage']
        out.append(dict(method=method, plan=plan['name'], terms=len(plan['columns']),
            known_universe=control['known_universe'], full_relevant=n, candidates=len(pool),
            pool_relevant=control['pool_relevant'], k=k, candidate_coverage=coverage,
            oracle_recall=oracle, recall=recall, precision=hits/k, fill_rate=len(selected)/k,
            coverage_loss=1-coverage, budget_loss=coverage-oracle, oracle_minus_method=oracle-recall))
    return out


def token_tensor(tokens, rows, device, binary=None):
    import torch
    arrays = []
    for i in rows:
        if binary is None:
            arrays.append(np.array(tokens[int(i)], dtype=np.float32, copy=True))
        else:
            packed, correction, center = binary
            lo, hi = tokens.offsets[i:i+2]
            arrays.append(binary_reconstruct(packed[lo:hi], correction[lo:hi], center))
    lengths = torch.tensor([len(x) for x in arrays], device=device)
    padded = torch.nn.utils.rnn.pad_sequence([torch.from_numpy(x) for x in arrays], batch_first=True).to(device)
    mask = torch.arange(padded.shape[1], device=device)[None, :] < lengths[:, None]
    return padded, mask


def token_logits(values, mask, weights, bias, routes):
    import torch
    scores = torch.einsum('bld,bcd->blc', values, weights[routes])
    return scores.masked_fill(~mask[:, :, None], -torch.inf).amax(1)+bias[routes]


def fit_token_head(tokens, y, fit, routes, init=None, epochs=3, seed=7, device='cpu', batch_size=64):
    import torch
    torch.manual_seed(seed)
    c, d, experts = y.shape[1], tokens.values.shape[1], int(routes.max())+1
    if init is None:
        weights = torch.nn.Parameter(torch.randn(experts, c, d, device=device)*.02)
        bias = torch.nn.Parameter(torch.zeros(experts, c, device=device))
    else:
        weights = torch.nn.Parameter(torch.tensor(np.repeat(init['weights'], experts, axis=0), device=device))
        bias = torch.nn.Parameter(torch.tensor(np.repeat(init['bias'], experts, axis=0), device=device))
    # Unsupported local concept/cluster combinations explicitly retain the global head.
    supported = np.ones((experts, c), dtype=bool)
    if experts > 1:
        for e in range(experts):
            for j in range(c):
                vals = y[fit[routes[fit] == e], j]
                supported[e, j] = min((vals == 0).sum(), (vals == 1).sum()) >= 10
    route_tensor = torch.tensor(routes, device=device, dtype=torch.long)
    supported_tensor = torch.tensor(supported, device=device)
    optimizer = torch.optim.Adam([weights, bias], lr=.01)
    rng = np.random.default_rng(seed)
    for epoch in range(epochs):
        order = rng.permutation(fit)
        total, batches = 0., 0
        for start in range(0, len(order), batch_size):
            rows = order[start:start+batch_size]
            target = torch.tensor(y[rows], device=device, dtype=torch.float32)
            rr = route_tensor[rows]
            known = (target >= 0) & supported_tensor[rr]
            if not known.any(): continue
            values, mask = token_tensor(tokens, rows, device)
            scores = token_logits(values, mask, weights, bias, rr)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(scores[known], target[known])
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            total += float(loss.detach()); batches += 1
        print(f'TOKEN_HEAD experts={experts} epoch={epoch+1}/{epochs} loss={total/max(batches,1):.4f}', flush=True)
    return dict(weights=weights.detach().cpu().numpy(), bias=bias.detach().cpu().numpy(),
                routes=routes, fallback=~supported, centers=np.empty((0, d), dtype=np.float32))


def score_tokens(tokens, head, device='cpu', batch_size=64, binary=None):
    import torch
    weights = torch.tensor(head['weights'], device=device)
    bias = torch.tensor(head['bias'], device=device)
    out = np.empty((len(tokens), weights.shape[1]), dtype=np.float32)
    with torch.inference_mode():
        for start in range(0, len(tokens), batch_size):
            rows = np.arange(start, min(start+batch_size, len(tokens)))
            values, mask = token_tensor(tokens, rows, device, binary)
            routes = torch.tensor(head['routes'][rows], dtype=torch.long, device=device)
            out[rows] = token_logits(values, mask, weights, bias, routes).cpu().numpy()
    return out
