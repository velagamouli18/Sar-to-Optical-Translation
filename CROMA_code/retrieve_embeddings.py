"""Cosine-similarity search over CROMA embedding arrays and UID metadata."""

import argparse
import csv
from pathlib import Path

import numpy as np


def load_metadata(path: Path | None, count: int, label: str) -> list[str]:
    if path is None:
        return [str(index) for index in range(count)]

    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))

    if len(rows) != count:
        raise ValueError(f"{label} metadata has {len(rows)} rows, but embeddings have {count} rows")

    ids = []
    for index, row in enumerate(rows):
        uid = row.get("xbd_uid") or row.get("uid") or row.get("sample_id") or str(index)
        ids.append(uid)
    return ids


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", required=True, type=Path, help="Query .npy array, shape [D] or [N,D]")
    parser.add_argument("--database", required=True, type=Path, help="Database .npy array, shape [M,D]")
    parser.add_argument("--database-metadata", type=Path, help="CSV aligned with database rows")
    parser.add_argument("--query-metadata", type=Path, help="Optional CSV aligned with query rows")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--output", type=Path, default=Path("retrieval_results.csv"))
    args = parser.parse_args()

    queries = np.load(args.query, mmap_mode="r").astype(np.float32)
    database = np.load(args.database, mmap_mode="r").astype(np.float32)
    if queries.ndim == 1:
        queries = queries[None, :]
    if database.ndim != 2 or queries.ndim != 2 or queries.shape[1] != database.shape[1]:
        raise ValueError(f"Expected compatible [N,D] arrays; got queries {queries.shape}, database {database.shape}")
    if not np.isfinite(queries).all() or not np.isfinite(database).all():
        raise ValueError("Embedding arrays contain NaN or infinite values")
    if args.top_k < 1 or args.top_k > len(database):
        raise ValueError(f"top-k must be between 1 and {len(database)}")

    query_norms = np.linalg.norm(queries, axis=1, keepdims=True)
    database_norms = np.linalg.norm(database, axis=1, keepdims=True)
    if np.any(query_norms == 0) or np.any(database_norms == 0):
        raise ValueError("Cosine similarity is undefined for zero-vector embeddings")
    queries = queries / query_norms
    database = database / database_norms
    query_ids = load_metadata(args.query_metadata, len(queries), "Query")
    database_ids = load_metadata(args.database_metadata, len(database), "Database")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["query_index", "query_uid", "rank", "database_index", "database_uid", "cosine_similarity"])
        for query_index, query in enumerate(queries):
            scores = database @ query
            candidates = np.argpartition(scores, -args.top_k)[-args.top_k:]
            candidates = candidates[np.argsort(scores[candidates])[::-1]]
            for rank, database_index in enumerate(candidates, start=1):
                writer.writerow([
                    query_index,
                    query_ids[query_index],
                    rank,
                    int(database_index),
                    database_ids[database_index],
                    f"{scores[database_index]:.8f}",
                ])

    print(f"Saved top-{args.top_k} cosine-similarity matches to {args.output}")


if __name__ == "__main__":
    main()
