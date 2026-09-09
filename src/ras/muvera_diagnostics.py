"""Paper-style inner projections and exact scoring helpers for FDE diagnostics.

Reference: MUVERA section 2, https://arxiv.org/html/2405.19504v2 .
This is a readable implementation of the construction, not bitwise Google parity.
"""
import numpy as np


class ProjectedFDE:
    def __init__(self, dim, repetitions=20, partition_bits=5, projection_dim=8, seed=7):
        if min(dim, repetitions, projection_dim) < 1 or not 0 <= partition_bits <= 12:
            raise ValueError('Invalid FDE dimensions')
        self.dim, self.repetitions = dim, repetitions
        self.partition_bits, self.projection_dim, self.seed = partition_bits, projection_dim, seed
        self.buckets = 1 << partition_bits
        self.output_dim = repetitions*self.buckets*projection_dim
        rng = np.random.default_rng(seed)
        self.planes = rng.normal(size=(repetitions, partition_bits, dim)).astype('float32')
        self.projections = rng.choice(np.array([-1., 1.], dtype='float32'),
                                      (repetitions, projection_dim, dim))/np.sqrt(projection_dim)
        self.projections = self.projections.astype('float32')
        if projection_dim == dim:
            self.projections[:] = np.eye(dim, dtype='float32')
        bits = ((np.arange(self.buckets)[:, None] >> np.arange(partition_bits)) & 1)
        self.hamming = np.count_nonzero(bits[:, None] != bits[None], axis=-1)

    def encode(self, tokens, *, query):
        tokens = np.asarray(tokens, dtype='float32')
        if tokens.ndim != 2 or tokens.shape[1] != self.dim or not len(tokens) or not np.isfinite(tokens).all():
            raise ValueError('Expected finite nonempty token matrix')
        # Partition in original token space; project within each repetition/bucket.
        bits = (tokens @ self.planes.reshape(-1, self.dim).T > 0).reshape(len(tokens), self.repetitions, self.partition_bits)
        labels = (bits * (1 << np.arange(self.partition_bits))).sum(axis=-1)
        result = np.empty((self.repetitions, self.buckets, self.projection_dim), dtype='float32')
        for r in range(self.repetitions):
            sums = np.zeros((self.buckets, self.dim), dtype='float32')
            np.add.at(sums, labels[:, r], tokens)
            if not query:
                counts = np.bincount(labels[:, r], minlength=self.buckets)
                sums /= np.maximum(counts, 1)[:, None]
                for empty in np.flatnonzero(counts == 0):
                    nearest = self.hamming[empty, labels[:, r]].argmin()
                    sums[empty] = tokens[nearest]
            result[r] = sums @ self.projections[r].T
        # Constant scaling across documents averages repetitions; rankings unchanged.
        return (result.ravel()/np.sqrt(self.repetitions)).astype('float32')

    def save(self, path):
        np.savez(path, planes=self.planes, projections=self.projections,
                 config=[self.dim, self.repetitions, self.partition_bits, self.projection_dim, self.seed])

    @classmethod
    def load(cls, path):
        with np.load(path) as f:
            obj = cls(*map(int, f['config']))
            obj.planes, obj.projections = f['planes'], f['projections']
        return obj


def topk_numpy(scores, tie_order, k):
    """Partial selection with exact document-ID tie handling, including the cutoff."""
    scores = np.asarray(scores)
    tie_order = np.asarray(tie_order)
    k = min(k, len(scores))
    if k < 1 or not np.isfinite(scores).all():
        raise ValueError('Expected finite scores and positive k')
    cutoff = np.partition(scores, len(scores)-k)[len(scores)-k]
    above = np.flatnonzero(scores > cutoff)
    equal = np.flatnonzero(scores == cutoff)
    equal = equal[np.argsort(tie_order[equal])[:k-len(above)]]
    selected = np.r_[above, equal]
    return selected[np.lexsort((tie_order[selected], -scores[selected]))]


def topk_tensor(scores, tie_order, k):
    import torch
    k = min(k, len(scores))
    cutoff = torch.topk(scores, k, sorted=False).values.min()
    above = torch.where(scores > cutoff)[0]
    equal = torch.where(scores == cutoff)[0]
    need = k-len(above)
    equal = equal[torch.topk(tie_order[equal], need, largest=False).indices] if need else equal[:0]
    selected = torch.cat([above, equal])
    selected = selected[torch.argsort(tie_order[selected])]
    return selected[torch.argsort(scores[selected], descending=True, stable=True)]


class TensorMaxSim:
    """Unpadded segmented MaxSim; GPU-resident corpus or explicit candidate uploads."""
    def __init__(self, tokens, device, resident=True, batch_docs=128):
        import torch
        self.tokens, self.device = tokens, device
        self.batch_docs = batch_docs
        self.values = None
        if resident:
            self.values = torch.empty(tokens.values.shape, dtype=torch.float32, device=device)
            # Bound host copies while uploading a potentially multi-GB corpus.
            for start in range(0, len(tokens.values), 65536):
                block = torch.from_numpy(np.array(tokens.values[start:start+65536], copy=True))
                self.values[start:start+len(block)].copy_(block)

    def score(self, query, ids=None):
        import torch
        q = torch.as_tensor(query, dtype=torch.float32, device=self.device)
        ids = np.arange(len(self.tokens)) if ids is None else np.asarray(ids, dtype=np.int64)
        out = torch.empty(len(ids), device=self.device, dtype=torch.float32)
        for start in range(0, len(ids), self.batch_docs):
            batch = ids[start:start+self.batch_docs]
            lengths = np.diff(self.tokens.offsets)[batch]
            if self.values is None:
                values = torch.as_tensor(np.concatenate([self.tokens[int(i)] for i in batch]), device=self.device)
            elif np.all(np.diff(batch) == 1):
                values = self.values[self.tokens.offsets[batch[0]]:self.tokens.offsets[batch[-1]+1]]
            else:
                positions = np.concatenate([np.arange(self.tokens.offsets[i], self.tokens.offsets[i+1]) for i in batch])
                values = self.values[torch.as_tensor(positions, device=self.device)]
            similarities = q @ values.T
            lens = torch.as_tensor(lengths, device=self.device).expand(len(q), -1)
            out[start:start+len(batch)] = torch.segment_reduce(similarities, 'max', lengths=lens, axis=1).sum(0)
        return out
