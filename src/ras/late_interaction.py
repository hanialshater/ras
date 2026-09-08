"""ColBERT token scoring and a transparent NumPy MUVERA FDE reference.

MUVERA: https://arxiv.org/abs/2405.19504
Construction reference: google/graph-mining/sketching/point_cloud.
This implements SimHash buckets, query sums, document means, nearest-Hamming
empty-document-bucket filling, and optional final CountSketch. It is not the
Google C++/DiskANN/PQ implementation; random draws are not bitwise compatible.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence
import numpy as np


@dataclass
class RaggedEmbeddings:
    values: np.ndarray
    offsets: np.ndarray

    def __post_init__(self):
        self.values = np.asarray(self.values, dtype=np.float32)
        offsets = np.asarray(self.offsets)
        if offsets.dtype.kind not in 'iu':
            raise ValueError('offsets must be integers')
        self.offsets = offsets.astype(np.int64)
        if (self.values.ndim != 2 or self.values.shape[1] == 0
                or self.offsets.ndim != 1 or len(self.offsets) < 2
                or self.offsets[0] != 0 or self.offsets[-1] != len(self.values)
                or np.any(np.diff(self.offsets) <= 0)):
            raise ValueError('need nonempty token matrices and strictly increasing offsets')
        if not np.isfinite(self.values).all():
            raise ValueError('token embeddings must be finite')

    def __len__(self):
        return len(self.offsets) - 1

    def __getitem__(self, i):
        if not 0 <= i < len(self):
            raise IndexError(i)
        return self.values[self.offsets[i]:self.offsets[i + 1]]

    @classmethod
    def from_list(cls, matrices: Sequence[np.ndarray]):
        if not len(matrices):
            raise ValueError('at least one embedding is required')
        return cls(np.concatenate(matrices), np.r_[0, np.cumsum([len(x) for x in matrices])])

    def save(self, path):
        np.savez(path, values=self.values, offsets=self.offsets)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as f:
            return cls(f['values'], f['offsets'])


def topk_ids(scores, ids, k):
    """Stable descending score, then global row ID; never returns padding IDs."""
    ids = np.asarray(ids, dtype=np.int64)
    scores = np.asarray(scores)
    if k < 0 or scores.shape != ids.shape or not np.isfinite(scores).all():
        raise ValueError('invalid scores, IDs or k')
    return ids[np.lexsort((ids, -scores))[:k]]


def maxsim(query, documents: RaggedEmbeddings, ids=None, batch_size=128):
    """Exact FP32 sum-of-max token score, without zero padding corrupting negatives."""
    q = np.asarray(query, dtype=np.float32)
    if q.ndim != 2 or not len(q) or q.shape[1] != documents.values.shape[1]:
        raise ValueError('query must be a nonempty token matrix of the document dimension')
    if not np.isfinite(q).all() or batch_size < 1:
        raise ValueError('invalid query or batch size')
    ids = np.arange(len(documents)) if ids is None else np.asarray(ids, dtype=np.int64)
    if ids.ndim != 1 or np.any(ids < 0) or np.any(ids >= len(documents)):
        raise ValueError('invalid document IDs')
    out = np.empty(len(ids), dtype=np.float32)
    for start in range(0, len(ids), batch_size):
        matrices = [documents[int(i)] for i in ids[start:start + batch_size]]
        offsets = np.r_[0, np.cumsum([len(x) for x in matrices])]
        similarities = q @ np.concatenate(matrices).T
        out[start:start + len(matrices)] = np.maximum.reduceat(
            similarities, offsets[:-1], axis=1).sum(axis=0)
    return out


class MuveraFDE:
    """A seeded MUVERA construction; query and document maps are asymmetric.

    Binary bucket labels are a permutation of the reference's Gray-code labels.
    Empty document buckets use the first nearest token by Hamming distance.
    Divide both outputs by sqrt(repetitions) to average scores across repetitions.
    Do NOT cosine-normalize the resulting FDEs: the target is inner product.
    """
    def __init__(self, dim, *, repetitions=8, partition_bits=4, final_dim=4096, seed=7):
        if dim < 1 or repetitions < 1 or not 0 <= partition_bits <= 12:
            raise ValueError('invalid FDE dimensions/repetitions (partition_bits must be 0..12)')
        if final_dim is not None and final_dim < 1:
            raise ValueError('final_dim must be positive or None')
        self.dim, self.repetitions, self.partition_bits = dim, repetitions, partition_bits
        self.final_dim, self.seed = final_dim, seed
        self.buckets = 1 << partition_bits
        self.raw_dim = repetitions * self.buckets * dim
        rng = np.random.default_rng(seed)
        self.planes = rng.normal(size=(repetitions, partition_bits, dim)).astype(np.float32)
        self.bucket_bits = ((np.arange(self.buckets)[:, None] >> np.arange(partition_bits)) & 1).astype(bool)
        self.sketch_indices = rng.integers(0, final_dim, size=self.raw_dim) if final_dim else None
        self.sketch_signs = rng.choice([-1., 1.], size=self.raw_dim).astype(np.float32) if final_dim else None

    @property
    def output_dim(self):
        return self.final_dim or self.raw_dim

    def encode(self, tokens, *, query):
        tokens = np.asarray(tokens, dtype=np.float32)
        if tokens.ndim != 2 or tokens.shape[1] != self.dim or not len(tokens):
            raise ValueError('FDE input must be a nonempty token matrix')
        if not np.isfinite(tokens).all():
            raise ValueError('FDE input must be finite')
        result = np.zeros((self.repetitions, self.buckets, self.dim), dtype=np.float32)
        for r in range(self.repetitions):
            bits = tokens @ self.planes[r].T > 0
            bucket_ids = (bits.astype(np.int64) * (1 << np.arange(self.partition_bits))).sum(axis=1)
            np.add.at(result[r], bucket_ids, tokens)
            if not query:
                counts = np.bincount(bucket_ids, minlength=self.buckets)
                present = counts > 0
                result[r, present] /= counts[present, None]
                for empty in np.flatnonzero(~present):
                    nearest = np.count_nonzero(bits != self.bucket_bits[empty], axis=1).argmin()
                    result[r, empty] = tokens[nearest]
        flat = result.reshape(-1) / np.sqrt(self.repetitions)
        if self.final_dim:
            flat = np.bincount(self.sketch_indices, weights=flat * self.sketch_signs,
                               minlength=self.final_dim).astype(np.float32)
        return np.asarray(flat, dtype=np.float32)

    def encode_many(self, embeddings, *, query):
        return np.stack([self.encode(embeddings[i], query=query) for i in range(len(embeddings))])

    def save(self, path):
        # Store actual randomness as well as seeds, for replay across NumPy versions.
        np.savez(path, planes=self.planes, bucket_bits=self.bucket_bits,
                 sketch_indices=np.array([], dtype=np.int64) if self.sketch_indices is None else self.sketch_indices,
                 sketch_signs=np.array([], dtype=np.float32) if self.sketch_signs is None else self.sketch_signs,
                 config=np.array([self.dim, self.repetitions, self.partition_bits, self.final_dim or 0, self.seed]))


    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as f:
            dim, repetitions, bits, final_dim, seed = map(int, f['config'])
            obj = cls(dim, repetitions=repetitions, partition_bits=bits,
                      final_dim=final_dim or None, seed=seed)
            obj.planes = f['planes']
            obj.bucket_bits = f['bucket_bits']
            obj.sketch_indices = f['sketch_indices'] if final_dim else None
            obj.sketch_signs = f['sketch_signs'] if final_dim else None
            return obj


class ColBERTEncoder:
    """Use upstream checkpoint/tokenizers: markers, punctuation masking and query expansion."""
    def __init__(self, checkpoint='colbert-ir/colbertv2.0', *, revision=None,
                 query_maxlen=32, doc_maxlen=180, batch_size=32):
        from huggingface_hub import snapshot_download
        from colbert.infra import ColBERTConfig
        from colbert.modeling.checkpoint import Checkpoint
        import torch
        path = Path(checkpoint)
        resolved = str(path.resolve()) if path.is_dir() else snapshot_download(checkpoint, revision=revision)
        self.resolved_checkpoint = resolved
        self.batch_size = batch_size
        self.model = Checkpoint(resolved, colbert_config=ColBERTConfig(
            query_maxlen=query_maxlen, doc_maxlen=doc_maxlen, interaction='colbert',
            similarity='cosine', mask_punctuation=True), verbose=0)
        self.model = self.model.cuda() if torch.cuda.is_available() else self.model.cpu()
        self.model.eval()

    def encode(self, texts, *, query):
        matrices = []
        # Outer batching bounds upstream's temporary CPU/GPU result lists.
        for start in range(0, len(texts), self.batch_size):
            chunk = list(texts[start:start + self.batch_size])
            if query:
                output = self.model.queryFromText(chunk, bsize=self.batch_size, to_cpu=True)
            else:
                # keep_dims=False returns a list of CPU tensors. ColBERT 0.2.22's
                # to_cpu=True wrapper incorrectly calls .cpu() on that list.
                # Transfer each tensor below, which also supports other devices.
                output = self.model.docFromText(chunk, bsize=self.batch_size,
                                                keep_dims=False, to_cpu=False)[0]
            matrices.extend(x.detach().float().cpu().numpy() for x in output)
        return RaggedEmbeddings.from_list(matrices)


class CrossEncoderScorer:
    """Joint query/title scoring; no reusable document embeddings or oracle labels."""
    def __init__(self, checkpoint='cross-encoder/ms-marco-MiniLM-L6-v2', *,
                 revision=None, max_length=256, batch_size=32):
        from huggingface_hub import snapshot_download
        from sentence_transformers import CrossEncoder
        import torch
        path = Path(checkpoint)
        self.resolved_checkpoint = (str(path.resolve()) if path.is_dir() else
                                    snapshot_download(checkpoint, revision=revision,
                                        allow_patterns=['*.json', '*.txt', '*.model',
                                                        '*.safetensors', 'pytorch_model*.bin']))
        self.batch_size = batch_size
        self.model = CrossEncoder(self.resolved_checkpoint, max_length=max_length)
        # Use raw logits, independent of checkpoint/version default activations.
        self.activation = torch.nn.Identity()

    def score(self, query, titles):
        if not len(titles):
            return np.empty(0, dtype=np.float32)
        output = self.model.predict([(query, str(title)) for title in titles],
                                    batch_size=self.batch_size, show_progress_bar=False,
                                    activation_fn=self.activation, convert_to_numpy=True)
        scores = np.asarray(output, dtype=np.float32).reshape(-1)
        if scores.shape != (len(titles),) or not np.isfinite(scores).all():
            raise ValueError('cross-encoder must return one finite relevance score per pair')
        return scores
