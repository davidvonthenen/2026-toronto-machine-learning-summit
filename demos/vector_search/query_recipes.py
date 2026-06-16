#!/usr/bin/env python3
"""OpenSearch-only vector query for recipes.

The query text is embedded and sent directly to OpenSearch as a kNN search.
Structured recipe metadata filters are applied as keyword filters on the vector
chunk documents. No second database is consulted, because the dependency pile
had already become a lifestyle choice.
"""

from __future__ import annotations

import argparse
import json
from typing import Any, Dict, List, Sequence

from common.embeddings import EmbeddingModel, to_list
from common.opensearch_client import create_vector_client, knn_search, normalize_vector_hits
from common.recipe_utils import normalize_values


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OpenSearch-only recipe vector search")
    parser.add_argument(
        "--query",
        type=str,
        default="", # default: with noodles but without soba
        help="Semantic recipe query used for kNN vector search.",
    )
    parser.add_argument("--exclude-caution", action="append", default=[], help="Exclude an allergy warning, repeatable.") # default: "Sulfites"
    parser.add_argument("--require-caution", action="append", default=[], help="Require an allergy warning, repeatable.")
    parser.add_argument("--require-health-label", action="append", default=[], help="Require a health label, repeatable.")
    parser.add_argument("--k", type=int, default=10, help="Vector hits to return.")
    parser.add_argument("--candidate-k", type=int, default=6, help="Vector candidates to retrieve before final top-k.")
    parser.add_argument("--verbose", action="store_true", help="Print verbose output.")
    return parser.parse_args(argv)


def build_filter_summary(args: argparse.Namespace) -> Dict[str, List[str]]:
    return {
        "exclude_cautions": normalize_values(args.exclude_caution),
        "require_cautions": normalize_values(args.require_caution),
        "require_health_labels": normalize_values(args.require_health_label),
    }


def build_output(
    *,
    query_text: str,
    filters: Dict[str, List[str]],
    vector_index: str,
    vector_hits: List[Dict[str, Any]],
    response: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "query": query_text,
        "filters": filters,
        "vector_index": vector_index,
        "query_vector_dim": response.get("_query_vector_dim"),
        "hit_count": len(vector_hits),
        "vector_hits": vector_hits,
        "error": response.get("_error", ""),
    }


def print_output(payload: Dict[str, Any], verbose: bool) -> None:
    print("\n")

    print("-" * 20)
    print("Vector Hits:")
    print("-" * 20)
    if verbose:
        print(json.dumps(payload["vector_hits"], ensure_ascii=False, indent=2))
    else:
        # delete content nodes in all graph matches
        for match in payload["vector_hits"]:
            match.pop("text", None)
        print(json.dumps(payload["vector_hits"], ensure_ascii=False, indent=2))

    print("\n")


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    query_text = args.query.strip()
    if not query_text:
        raise SystemExit("--query must contain non-empty text for vector search")
    if int(args.k) <= 0:
        raise SystemExit("--k must be > 0")
    if int(args.candidate_k) <= 0:
        raise SystemExit("--candidate-k must be > 0")

    vector_client, vector_index = create_vector_client()

    print("\n")
    print('Let\'s look for recipes that satisfy the following criteria:')
    print(" - Includes: {}".format(query_text))
    print("\n")

    try:
        embedder = EmbeddingModel()
        query_vector = to_list(embedder.encode([query_text])[0])
        response = knn_search(
            "VECTOR",
            vector_client,
            vector_index,
            query_vector,
            k=int(args.k),
            candidate_k=int(args.candidate_k),
            exclude_cautions=args.exclude_caution,
            require_cautions=args.require_caution,
            require_health_labels=args.require_health_label,
        )
        vector_hits = normalize_vector_hits(response)
    finally:
        vector_client.close()

    print_output(
        build_output(
            query_text=query_text,
            filters=build_filter_summary(args),
            vector_index=vector_index,
            vector_hits=vector_hits,
            response=response,
        ),
        args.verbose,
    )


if __name__ == "__main__":
    main()
