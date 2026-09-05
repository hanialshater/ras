# Future Research Directions

This note collects research ideas that are intentionally **not part of the current paper claim**. The current evidence supports compact compiled semantic predicates and their use as soft eligibility tests inside ANN traversal. The directions below ask how far that idea can be pushed.

## 0. Reproduce ColBERT and MUVERA before representation research

Before changing the retrieval representation or claiming a new interaction with late-interaction models, establish clean reproductions of the two upstream ingredients.

### Reproduce ColBERT first

Start with a canonical ColBERT-style late-interaction baseline. Keep separate query-token and document-token vectors and evaluate exact MaxSim scoring:

```text
S(q,d) = sum_i max_j q_i^T d_j.
```

The first goal is reproduction, not novelty. Verify the implementation on a benchmark where ColBERT has known behavior, then run the same model on the fashion domain.

Record:

- retrieval Recall@K / MRR / NDCG as appropriate for the benchmark;
- document and query representation sizes;
- exact MaxSim latency;
- candidate-generation cost;
- the behavior of multi-aspect and negated queries.

Exact ColBERT MaxSim should then become the reference retrieval function for the MUVERA study.

### Reproduce MUVERA independently

Next reproduce MUVERA as a compilation of multi-vector retrieval into a fixed-dimensional representation suitable for ordinary MIPS/ANN search.

The core comparison should be:

```text
exact ColBERT / late-interaction MaxSim
            vs
MUVERA fixed-dimensional retrieval
```

Measure how much of the late-interaction quality survives and what is gained in storage and ANN efficiency. Keep this reproduction independent of the semantic-predicate code so that any discrepancy can be diagnosed before adding new ideas.

Use at least two settings:

1. a canonical retrieval benchmark where ColBERT/MUVERA are expected to work;
2. the fashion domain used by the current semantic-predicate experiments.

This prevents a fashion-domain failure from being confused with an implementation failure.

### Only then start the new research

Once both baselines reproduce, the representation study becomes much cleaner:

```text
MiniLM single-vector
        -> ColBERT late interaction
        -> MUVERA fixed-dimensional representation
```

For each search-side representation, ask:

- how linearly separable are the semantic concepts?
- how strong is a global FP32 semantic head?
- do local linear experts improve the semantic ceiling?
- how much of that quality survives Binary1 compilation?
- can compiled predicates still execute cheaply inside ANN traversal?

The most direct combination is to keep responsibilities separate:

```text
MUVERA / ANN representation -> graph navigation
Binary semantic sidecar     -> explicit semantic eligibility
```

A more ambitious experiment is to compile semantic predicates directly over the MUVERA item representation. If the richer representation raises the FP32 semantic ceiling and Binary1 preserves most of that gain, the research question becomes substantially stronger: can a representation designed to preserve late interaction also serve as a substrate for tiny reusable semantic programs?

This track should remain separate from the current paper until the ColBERT and MUVERA reproductions are independently validated.

## 1. Make semantic concepts more linear

The current strongest compact compiler starts from a fixed MiniLM embedding and fits one linear semantic head per concept. A major remaining gap is therefore upstream of quantization: some concepts may simply be poorly represented by one global linear boundary.

### Local linear experts

A concept such as `elegant` may be globally nonlinear but locally simple within coherent regions such as shoes, dresses, jackets, or accessories.

Let `c(x)` be a cluster assignment. Replace one global head

```text
s_C(x) = w_C^T x + b_C
```

with

```text
s_C(x) = w_{C,c(x)}^T x + b_{C,c(x)}.
```

Each item still evaluates only one local expert. With a small number of clusters, the online cost can remain close to one predicate evaluation while program size grows only with the number of experts.

First experiment:

- keep the current embedding and Binary1 substrate fixed;
- cluster the fit embeddings with K in {4, 8, 16, 32, 64};
- fit one FP32 linear head per concept and cluster;
- compare global FP32 vs local FP32 on strict held-out items;
- then compile the local heads to Binary1 and measure how much of the gain survives.

The key question is whether local linearity closes a meaningful part of the gap between the current FP32 linear proxy and the semantic oracle.

### Locally centered binary codes

The current Binary1 representation uses one global centroid. A natural extension is cluster-conditioned centering:

```text
r = x - mu_{c(x)}
q(x) = sign(r)
```

with the same two-level per-item reconstruction inside each local coordinate system.

This could improve both reconstruction and predicate linearity while adding only a small cluster ID per item plus a shared table of cluster centroids.

### Learn a semantic execution space

Instead of accepting a pretrained embedding as fixed, learn a small transform `T` that makes many downstream semantic predicates easier to express while preserving retrieval geometry.

For example:

```text
z = T(x)
P(C_j | x) ~= sigmoid(w_j^T z)
```

with an objective such as

```text
L = L_semantic + lambda * L_neighbor_preservation.
```

`T` could be deliberately small: a low-rank adapter, shallow residual MLP, or cluster-conditioned local transform.

The stronger research question is not merely whether one concept becomes linear, but whether a representation can be trained for **semantic programmability**: many future concepts should admit tiny, cheap programs without retraining or rewriting the catalog.

A critical evaluation is **held-out concept generalization**. Concepts used to train `T` should be separated from concepts used to test whether the space has become more linearly programmable.

## 2. Stronger semantic supervision

The current paper uses CLIP image-derived semantics as an independent teacher and MiniLM titles on the search side. The compact Binary1 representation preserves most of the simple FP32 linear proxy, so the dominant quality gap may come from the semantic function being compiled rather than from low-bit execution itself.

Future experiments should test stronger teachers and stronger search-side semantic targets:

- larger VLM teacher;
- human judgments;
- behavioral relevance labels;
- cross-encoder or multimodal teacher;
- teacher ensembles;
- task-specific semantic classifiers.

The main question is: **if the teacher is substantially better, can most of that quality still be compiled into a tiny executable program?**

## 3. Nonlinear programs that remain cheap

If local linearity is insufficient, the next step should not automatically be a large neural model. The execution budget suggests exploring structured nonlinear programs whose online cost remains predictable.

Candidates include:

- small mixtures of linear experts;
- low-depth decision trees;
- sparse GAM / GA2M terms;
- small LUT interactions over selected binary coordinates;
- several binary heads combined by a tiny gate;
- low-rank quadratic terms;
- cluster-specific programs.

The target is a Pareto frontier over semantic quality, program bytes, item bytes, and predicate execution cost.

## 4. Scale ANN evaluation to realistic catalogs

The current controlled HNSW benchmark uses the strict held-out split, not a multi-million-item production-scale graph. The next systems experiment should directly measure a resident graph around 2M items.

Report single-core and multi-core results across:

- 50%, 20%, 10%, 5%, and 2% eligibility;
- 1, 3, 5, and 8 active predicates;
- mean, p50, p95, and p99 latency;
- QPS/core;
- visited nodes;
- predicate evaluations;
- memory footprint;
- matched Traversal Recall@50.

The purpose is to replace extrapolated QPS estimates with measured scaling behavior and to expose cache/memory-hierarchy effects that are invisible on the current graph size.

## 5. Predicate-count scaling and short-circuiting

The current reviewer benchmark uses three active predicates per semantic plan. We should explicitly measure 1 / 3 / 5 / 8 active predicates.

Because calibrated log-probability terms are non-positive, partial sums provide an upper bound on the final conjunction score. This creates an opportunity for safe short-circuit execution when the remaining predicates cannot rescue a candidate.

Questions:

- how close is cost to linear in predicate count without short-circuiting?
- how much does predicate ordering matter?
- can cheap/high-rejection predicates be evaluated first?
- can query plans be optimized dynamically from estimated selectivity and cost?

This starts to turn semantic predicate execution into a small query optimizer.

## 6. Better matched-recall ANN baselines

The current reviewer sweep already replaces fixed post-filtering with selectivity-aware over-fetch. The remaining baseline work is to push over-fetch far enough to actually match live traversal recall and then compare latency at matched recall.

Also worth testing:

- ACORN-style filtered graph traversal;
- production filtered-vector engines;
- bitmap-assisted ANN when semantics are materialized;
- dynamic over-fetch from selectivity estimates;
- graph partitioning / label-aware links where appropriate.

The goal is not to prove one traversal algorithm universally best. It is to establish where cheap live predicates provide a useful execution primitive compared with strong alternatives.

## 7. Fairer PQ trade-offs

PQ64 is a strong compressed-quality baseline. A fully materialized 64x256 FP32 LUT per predicate is fast but should not be treated as mandatory persistent storage.

Future comparison should separate:

- compact persistent linear head + shared PQ codebook;
- on-demand LUT materialization cost;
- active LUT cache footprint;
- int8 / int16 LUT variants;
- smaller sub-codebooks;
- eviction and activation behavior with large concept vocabularies.

This makes the real comparison a three-way trade-off among persistent memory, activation cost, and online scoring speed.

## 8. Large semantic vocabularies

The current quality benchmark learns only a handful of concepts. The architecture becomes much more interesting if a catalog can host 10^3 to 10^6 reusable semantic programs.

Experiments should measure:

- program lookup latency;
- cache hit rate;
- hot/cold concept distributions;
- memory mapping / paging;
- program versioning;
- compilation throughput;
- concept lifecycle and invalidation;
- online activation of rarely used predicates.

A useful systems target is a semantic concept store that behaves more like executable index metadata than like a collection of heavyweight models.

## 9. Anchor-, user-, and session-conditioned programs

Reusable concepts are only one source of semantic constraints. The same substrate may support ephemeral programs derived from an anchor item, user state, or session context.

A future score could be decomposed as

```text
S(i) = S_compat(i | anchor)
     + S_semantic_plan(i)
     + S_personal(i | user).
```

Potential approach:

- compile stable concepts offline;
- generate a tiny ephemeral int4 head from an anchor/user vector at query time;
- execute both over the same 56-byte item substrate;
- keep dense ANN geometry responsible for navigation.

This is particularly attractive for complementary-item recommendation, where the LLM or planner can separate exact filters, reusable semantic predicates, negative constraints, and anchor-specific compatibility.

## 10. Natural LLM-generated query plans

The current experiments use controlled synthetic semantic plans. A real system should evaluate an LLM planner that emits something like:

```text
retrieval_text: "flowy midi skirt"
exact:          gender=women, category=skirts
positive:       fluid, refined
negative:       sporty
```

Evaluation should distinguish planner errors from predicate errors and ANN errors.

Important questions:

- how often does the planner select an existing compiled predicate?
- when should a phrase stay in dense retrieval rather than become a predicate?
- when is online predicate compilation worthwhile?
- how should unknown or compositional concepts fall back to neural reranking?

## 11. Online / just-in-time predicate compilation

Reusable concepts can be compiled offline, but some user phrases will be genuinely novel. A longer-term direction is fast query-time compilation:

```text
natural phrase -> semantic teacher / embedding -> tiny executable program
```

Possible sources include an LLM/VLM, a concept embedding mapped into head parameters, or meta-learning over previously compiled predicates.

The interesting target is not arbitrary model generation; it is producing a small program quickly enough that it can be cached and reused across future queries.

## 12. External validity

Before treating the mechanism as general, repeat the semantic-quality and systems experiments outside fashion.

Useful domains include:

- general e-commerce;
- documents / enterprise search;
- media retrieval;
- image search;
- marketplace listings;
- recommendation catalogs.

The strongest evidence would combine a second domain, stronger or human supervision, and natural query plans.

## 13. Theory: joint rate-distortion for semantic functions

Conventional vector compression asks how many bits are needed to preserve geometry. This project suggests a different objective: how many item bits and program bits are needed to preserve a **family of semantic functions**?

For catalog representation `phi(x)` and predicate family `F_C`, the relevant object is closer to

```text
minimize   N * B_item + C * B_program
subject to semantic-function distortion <= epsilon.
```

A theoretical treatment could connect vector quantization, function approximation, rate-distortion, and multi-task representation learning.

## Suggested order of experiments

A practical sequence is:

1. finish and freeze the current paper with the extended matched-recall over-fetch sweep;
2. reproduce ColBERT / exact late-interaction MaxSim on a canonical benchmark;
3. reproduce MUVERA against exact ColBERT, first canonically and then on fashion;
4. establish the representation ladder: MiniLM vs ColBERT vs MUVERA;
5. global vs local FP32 semantic heads on the reproduced representations;
6. local centering + Binary1 compilation;
7. test Binary semantic sidecars with MUVERA-driven ANN navigation;
8. test direct predicate compilation over the MUVERA representation;
9. held-out-concept semantic-execution-space learning;
10. predicate-count scaling and short-circuiting;
11. 2M-item HNSW scaling and QPS/core;
12. stronger teacher / human labels;
13. natural LLM-generated query plans;
14. anchor/user-conditioned ephemeral programs;
15. second-domain replication.

The central question behind all of these directions is the same: **can complex semantic intent be moved offline or into a tiny compilation step so that search-time execution remains as cheap and composable as traditional filtering?**
