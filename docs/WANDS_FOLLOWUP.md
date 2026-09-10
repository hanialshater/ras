# Cached WANDS follow-up

[Open Colab](https://colab.research.google.com/github/hanialshater/ras/blob/codex/colbert-muvera-baselines/notebooks/ras_wands_followup_colab.ipynb).

This reuses the completed WANDS quality reference and the token/dense arrays
created by `wands_systems`. No catalog re-encoding, new labels or model training.
The old modules and manifests are unchanged. Create a separate follow-up output
directory; the notebook copies only required cache files to local disk and backs
up new reports/small score shards between phases.

## Serving baselines

* DenseOn CPU and GPU exact dot product with partial top-k selection.
* Exact LateOn with GPU-resident FP32 token vectors and segmented MaxSim.
* The missing dense+BM25 union → LateOn pipeline with CPU scoring.
* The same union → LateOn with candidate tokens uploaded to GPU for scoring.

Candidate generation for both shared-pool arms is CPU dense search plus BM25;
their query models run on the selected encoding device. Pools must match the
original quality run. The full GPU arm keeps corpus tokens resident, whereas the
shared GPU arm retains CPU memory-mapped tokens and includes candidate uploads
in scoring time. These are different explicit residency strategies, not silently
interchangeable backends. All modes use FP32 and disable TF32.

Selection uses partition/top-k followed by stable document-ID tie handling,
including ties at the cutoff. GPU selection transfers only selected IDs.
Segmented MaxSim never introduces padded zero token matches. It is an optimized
tensor baseline, not a claim to be the fastest possible GPU retrieval engine.

Each pipeline starts a separate process. Two warmups precede the default 30
seeded queries × three repeats, with CUDA synchronization. Per-stage and total
latencies, RAM and GPU allocator peaks are recorded. Live scores for every timed
query are validated against the reference **after** timing/memory recording;
failed validation prevents successful reports. Reference score files therefore
cannot warm pages or inflate measured RAM. CUDA OOM produces an explicit skipped
status; there is no CPU fallback. No GPU speedup is claimed before measurement.

## FDE ablations

All configurations use the same 60 seeded diagnostic queries and full corpus,
without ANN, and rerank candidates by the same saved exact LateOn scores:

| Configuration | Repetitions | Buckets | Projection | Output dimension |
|---|---:|---:|---|---:|
| Existing CountSketch | 8 | 16 | Final CountSketch | 4096 / 8192 |
| Uncompressed control | 8 | 16 | None | 128 × token dimension |
| Paper-style inner projection | 20 | 32 | 8 per bucket | 5120 |
| Paper-style inner projection | 20 | 32 | 16 per bucket | 10240 |

The uncompressed control preserves the old partitions, seed and aggregation so
it isolates the final sketch. Paper-style maps independently sample Gaussian
SimHash planes and per-repetition random-sign projections, with query sums,
document means and nearest-Hamming empty-document-bucket filling. Their outputs
are scaled by a constant to average repetitions; they are not cosine-normalized.
The algebra follows [MUVERA Sections 2–3](https://arxiv.org/html/2405.19504v2),
not bitwise Google C++ parity. Changing repetition/bucket/projection settings
together tests that configuration bundle, not a causal effect of one parameter.

Top-10 overlap and top-1 candidate recall are separate. The latter avoids
conflating the paper's 1-nearest-neighbor candidate-recall examples with top-10
fidelity. Human-label WANDS metrics remain separate and use full-corpus qrels.
Shared-union candidate fidelity is also measured on those same queries.
The geometry report includes token counts, sampled norms and original bucket
occupancy. Only score shards/maps are saved for new FDEs; theoretical FP32 FDE
payload bytes are clearly marked as hypothetical, with no serving-cost claim.

These are diagnostic queries, not a held-out tuning protocol. First identify
whether final compression or the broader mapping settings explain the loss;
then evaluate promising settings on remaining queries and measure their costs.
The default run can resume score shards, but cannot resume an interrupted timing
worker. Use a new output directory if settings, code or hardware change.

```bash
pip install -e '.[wands,wands-systems,dev]'
python -m experiments.wands_followup \
  --reference-dir /local/wands-quality \
  --cache-dir /local/wands-muvera-cost \
  --output-dir /local/wands-followup --device cuda
```

Tests cover inner-projection algebra and serialization, uncompressed-map parity,
cutoff ties, negative unpadded MaxSim, arbitrary candidate order and both residency
modes, plus a cached end-to-end diagnostic/CPU-serving fixture. CUDA variants run
when available; the Colab setup requires a GPU and executes them before benchmarks.
