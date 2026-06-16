#!/usr/bin/env python3
"""Minimal graph-filtered vector query for recipes.

Use the graph for allergy/label constraints, then use vector search for the
recipe intent. In other words: let the database enforce facts, let embeddings
handle vibes. A rare division of labor that does not require a committee.
"""

from __future__ import annotations

import argparse
import json
from typing import Any, Dict, List, Sequence

from common.embeddings import EmbeddingModel, to_list
from common.graph import find_recipes_by_profile
from common.neo4j_client import MyNeo4j, create_graph_client
from common.opensearch_client import create_vector_client, knn_search, normalize_vector_hits


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Graph-filtered recipe vector search")
    parser.add_argument(
        "--query",
        type=str,
        default="", # noodle
        help="Recipe query. Used as both a graph content substring filter and the semantic vector query.",
    )
    parser.add_argument(
        "--notquery",
        type=str,
        default="", # 1st: empty, 2nd soba
        help="Exclude recipes whose Recipe.content contains this text, case-insensitive.",
    )
    parser.add_argument("--exclude-caution", action="append", default=[], help="Exclude an allergy warning, repeatable.") # default: "Sulfites"
    parser.add_argument("--require-caution", action="append", default=[], help="Require an allergy warning, repeatable.")
    parser.add_argument("--require-health-label", action="append", default=[], help="Require a health label, repeatable.")
    parser.add_argument("--require-diet-label", action="append", default=[], help="Require a diet label, repeatable.")
    parser.add_argument("--cuisine-type", action="append", default=[], help="Filter by cuisine type, repeatable.")
    parser.add_argument("--meal-type", action="append", default=[], help="Filter by meal type, repeatable.")
    parser.add_argument("--dish-type", action="append", default=[], help="Filter by dish type, repeatable.")
    parser.add_argument("--vk", type=int, default=10, help="Vector hits to return.")
    parser.add_argument("--gk", type=int, default=5, help="Graph hits to return.")
    parser.add_argument("--candidate-k", type=int, default=50, help="Vector candidates to retrieve before final top-k.")
    parser.add_argument("--verbose", action="store_true", help="Print verbose output.")
    return parser.parse_args(argv)


def has_graph_filters(args: argparse.Namespace) -> bool:
    fields = [
        args.exclude_caution,
        args.require_caution,
        args.require_health_label,
        args.require_diet_label,
        args.cuisine_type,
        args.meal_type,
        args.dish_type,
    ]
    return any(bool(x) for x in fields)


def has_graph_constraints(args: argparse.Namespace) -> bool:
    """Return true when graph results should constrain vector search.

    ``--query`` participates in graph filtering by requiring the extracted
    recipe text in ``Recipe.content`` to contain the query string. ``--notquery``
    applies the matching negative content filter. Yes, this is stricter than
    semantic search. That is the point.
    """

    return has_graph_filters(args) or bool(args.query.strip()) or bool(args.notquery.strip())


def build_output(
    *,
    query_text: str,
    notquery_text: str,
    graph_records: List[Dict[str, Any]],
    vector_hits: List[Dict[str, Any]],
) -> Dict[str, Any]:
    return {
        "query": query_text,
        "notquery": notquery_text,
        "graph_content_filter": bool(query_text),
        "graph_content_exclusion_filter": bool(notquery_text),
        "graph_match_count": len(graph_records),
        "graph_matches": graph_records,
        "vector_hits": vector_hits,
    }


def print_output(payload: Dict[str, Any], verbose: bool) -> None:
    print("\n")

    # print only the graph match
    print("-" * 20)
    print("Graph Matches:")
    print("-" * 20)
    if verbose:
        print(json.dumps(payload["graph_matches"], ensure_ascii=False, indent=2))
    else:
        # delete content nodes in all graph matches
        for match in payload["graph_matches"]:
            match.pop("content", None)
        print(json.dumps(payload["graph_matches"], ensure_ascii=False, indent=2))

    print("\n\n")

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
    notquery_text = args.notquery.strip()
    graph_constraints_active = has_graph_constraints(args)

    # create clients
    graph_client = create_graph_client()
    vector_client, vector_index = create_vector_client()

    print("\n")
    print('Let\'s look for recipes that satisfy the following criteria:')
    print(" - Includes: {}".format(query_text))
    print(" - Excludes: {}".format(notquery_text))
    print(" - Exclude Severe Allergens: {}".format(", ".join(args.exclude_caution)))
    print("\n")

    try:
        graph_records = find_recipes_by_profile(
            graph_client,
            content_query=query_text,
            content_notquery=notquery_text,
            required_cautions=args.require_caution,
            excluded_cautions=args.exclude_caution,
            required_health_labels=args.require_health_label,
            required_diet_labels=args.require_diet_label,
            cuisine_type=args.cuisine_type,
            meal_type=args.meal_type,
            dish_type=args.dish_type,
            limit=args.gk,
        )
    finally:
        graph_client.close()

    recipe_ids: List[str] = [str(row.get("recipe_id")) for row in graph_records if row.get("recipe_id")]

    if graph_constraints_active and not recipe_ids:
        print_output(
            build_output(
                query_text=query_text,
                notquery_text=notquery_text,
                graph_records=[],
                vector_hits=[],
            )
        )
        return

    if not query_text:
        print_output(
            build_output(
                query_text=query_text,
                notquery_text=notquery_text,
                graph_records=graph_records,
                vector_hits=[],
            )
        )
        return
    try:
        embedder = EmbeddingModel()
        query_vector = to_list(embedder.encode([query_text])[0])
        response = knn_search(
            "VECTOR",
            vector_client,
            vector_index,
            query_vector,
            k=args.vk,
            candidate_k=args.candidate_k,
            include_recipe_ids=recipe_ids if graph_constraints_active else None,
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
            notquery_text=notquery_text,
            graph_records=graph_records,
            vector_hits=vector_hits,
        ),
        args.verbose,
    )


if __name__ == "__main__":
    main()
