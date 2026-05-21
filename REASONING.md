# Architecture Reasoning

I kept this project deliberately simple. The goal was not to build a research platform or a generic ML framework; it was to ship something the team could actually run, understand, and trust without a lot of moving parts.

## Why I used sentence-transformers

I chose `all-MiniLM-L6-v2` because it gives me strong semantic embeddings without turning the app into an API dependency story. The model is light enough to run locally, which keeps cost, latency, and operational complexity down. For this kind of batch clustering, I want stable embeddings that do one job well, not a heavyweight external service.

## Why I used K-means

K-means is the right tradeoff for this MVP because it is easy to explain and easy to tune. I can point to a cluster and say, "these conversations belong together" without needing a more complex density-based or hierarchical approach. I also like that it gives a clean number of buckets for PM reporting. I am not trying to discover every edge case here; I am trying to group the obvious themes quickly.

## Why PostgreSQL, with SQLite as fallback

The data is naturally relational: conversations in one table, insights in another. PostgreSQL is the sensible production choice because it handles filtering, aggregation, and future reporting cleanly. SQLite stays in the project because it makes local setup painless. That way the repo still works on a laptop with no extra services, which matters for a weekend build.

## How I extract insights

Once the conversations are clustered, I run a simple keyword pass over each cluster with `CountVectorizer`. That gives me the terms people actually repeat inside the cluster. I then turn the strongest keyword into a short sentence like "X% of users in cluster_3 mention 'refund'". It is intentionally plain-language. The output should read like a note a PM might write after scanning a support inbox.

## What I would change with more time

If I had a month instead of a weekend, I would spend it on product quality rather than algorithmic novelty. I would add a lightweight review flow for cluster labels, re-clustering over time so themes can be tracked week to week, and a basic dashboard for PMs. I would also tighten insight deduplication so repeated themes roll up into one narrative instead of appearing as separate rows.

## What I would not change

I would keep the core shape of the system the same. The model, the clustering step, and the database layout are already the right level of simplicity for this use case. The main improvement path is around clarity, reviewability, and trend tracking, not around introducing a more complicated ML stack.
