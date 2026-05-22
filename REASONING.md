# Architecture Reasoning

## 1. Problem Statement

Agnost needs a service that turns raw user conversations into product intelligence. The system should show what users repeatedly ask for, where friction is concentrated, and which topics are growing, because product teams need quantified signals instead of a wall of unstructured chat logs. I treated this as a decision-support problem, not a pure machine learning exercise: the output has to be understandable enough that a PM can use it in a roadmap discussion without decoding the model.

## 2. Architecture Overview

The main data flow is: conversations enter the API, embeddings are generated, KMeans groups semantically similar messages, the insight extractor labels those groups, and the database stores both the raw logs and the summarized output. The overall shape is documented in [DIAGRAMS.md](DIAGRAMS.md), but the important thing is that each step owns one job and has a narrow contract.

- FastAPI handles validation, request orchestration, and response formatting.
- The embedder isolates model loading so the rest of the app can stay testable.
- The clusterer is deterministic and driven by silhouette search so cluster counts are explainable.
- The insight extractor turns clusters into product-facing language and quantified metrics.
- PostgreSQL stores the durable audit trail, cluster snapshots, and metadata for retrieval.

## 3. Key Technical Decisions

### Decision 1: Embedding Model

I chose `sentence-transformers` with `all-MiniLM-L6-v2`. The reason is mostly operational: it is fast, small enough to run locally, and good enough to separate support themes without making the app depend on an external API. Alternatives like hosted GPT embeddings would improve raw semantic quality, but they would add latency, cost, and a harder deployment story. The trade-off is that I accept slightly less semantic nuance in exchange for a service that is actually shippable in a weekend.

### Decision 2: Clustering Algorithm

I used KMeans with automatic `k` detection via silhouette score. That is a deliberate bias toward simplicity and repeatability. DBSCAN and hierarchical methods can be useful, but they are more parameter-sensitive or more expensive to scale, and they are harder to explain to a PM who just wants to know what themes exist. KMeans assumes roughly spherical clusters, which is a real limitation, but it is a good trade for topic grouping where interpretability matters more than perfect geometry.

### Decision 3: Database

PostgreSQL is the production database because the system needs ACID persistence, future reporting, and a schema that can evolve. I still keep the developer experience lightweight by allowing SQLite locally, but the real deployment path is Postgres. I also used JSON columns for flexible cluster payloads so the schema can hold richer metadata without forcing constant migrations. The trade-off is a hybrid relational/document style, but that is exactly what this use case wants.

### Decision 4: Insight Extraction

I chose keyword frequency plus a small sentiment lexicon instead of a transformer-based sentiment classifier or LLM summarization. That keeps the output explainable and cheap to run. It also means the system can produce quantifiable insights immediately, which is more valuable for a first version than trying to optimize for state-of-the-art sentiment accuracy. If we had more time, I would replace parts of this with a trained classifier and abstractive summaries.

### Decision 5: Caching

I used an in-memory cache for repeated analysis batches. It avoids recomputing embeddings and clustering for identical inputs, which is enough for a weekend build and keeps the behavior easy to reason about. Redis would be the obvious production evolution, but adding it now would increase operational surface area before the product value is proven. The trade-off is that cache state is process-local, which is acceptable for the current scope.

## 4. Scaling Considerations

- With 10K conversations, the current single-process API and Postgres setup is comfortable.
- With 100K conversations, I would add Redis caching, background jobs, and batch-oriented processing.
- With 1M conversations, I would shard by date or source, move to approximate clustering, and stream data through a queue like Kafka.

The main scaling pressure is in embedding generation and clustering, not in request validation or database writes. That means the first scaling move should target CPU-heavy work, not the API surface.

## 5. What Would Change With a Month

If I had a month instead of a weekend, I would improve the system along the product path rather than just the algorithm path.

- Add a transformer-based sentiment classifier for more accurate sentiment labels.
- Add hierarchical topic modeling so clusters can split into sub-topics.
- Add WebSocket or streaming updates so PMs can watch new themes arrive in real time.
- Build a lightweight dashboard for trend viewing and cluster review.
- Add full unit, integration, and contract test coverage.
- Deploy with auto-scaling on AWS or GCP.
- Add UMAP or similar visualization of embedding space.
- Add a user feedback loop so PMs can mark insights as useful or noisy.
- Add multi-language support and language detection.

## 6. Why I Made the Trade-offs This Way

I prioritized a working product over an elegant research stack. That means I chose interpretability over maximal model quality, deterministic heuristics over brittle novelty, and a narrow deployment story over distributed infrastructure. The system is designed so the obvious next improvements are additive, not disruptive. If the team later wants better sentiment quality or scale, the current structure makes it easy to swap in those pieces without throwing away the whole design.
