#!/usr/bin/env python3
"""Graph-only recipe search.

Neo4j handles both structured recipe facts and exact recipe-text filtering.
No external ranking service is created here. Apparently the graph can have a
full-time job after all.
"""

from __future__ import annotations

import argparse
import json
from typing import Any, Dict, List, Sequence

from common.graph import find_recipes_by_profile
from common.neo4j_client import MyNeo4j, create_graph_client


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Graph-only recipe search")
    parser.add_argument(
        "--query",
        type=str,
        default="", # noodle
        help="Include only recipes whose Recipe.content contains this text, case-insensitive.",
    )
    parser.add_argument(
        "--notquery",
        type=str,
        default="", # 1st: empty, 2nd soba
        help="Exclude recipes whose Recipe.content contains this text, case-insensitive.",
    )
    parser.add_argument("--graph-store", choices=["long", "hot"], default="long")
    parser.add_argument("--exclude-caution", action="append", default=[], help="Exclude an allergy warning, repeatable.") # default: "Sulfites"
    parser.add_argument("--require-caution", action="append", default=[], help="Require an allergy warning, repeatable.")
    parser.add_argument("--require-health-label", action="append", default=[], help="Require a health label, repeatable.")
    parser.add_argument("--require-diet-label", action="append", default=[], help="Require a diet label, repeatable.")
    parser.add_argument("--cuisine-type", action="append", default=[], help="Filter by cuisine type, repeatable.")
    parser.add_argument("--meal-type", action="append", default=[], help="Filter by meal type, repeatable.")
    parser.add_argument("--dish-type", action="append", default=[], help="Filter by dish type, repeatable.")
    parser.add_argument("--gk", type=int, default=5, help="Graph hits to return.")
    parser.add_argument("--verbose", action="store_true", help="Print verbose output.")
    return parser.parse_args(argv)


def build_filter_summary(args: argparse.Namespace, *, query_text: str, notquery_text: str) -> Dict[str, Any]:
    return {
        "query": query_text,
        "notquery": notquery_text,
        "exclude_caution": args.exclude_caution,
        "require_caution": args.require_caution,
        "require_health_label": args.require_health_label,
        "require_diet_label": args.require_diet_label,
        "cuisine_type": args.cuisine_type,
        "meal_type": args.meal_type,
        "dish_type": args.dish_type,
    }


def build_output(
    *,
    args: argparse.Namespace,
    query_text: str,
    notquery_text: str,
    graph_records: List[Dict[str, Any]],
) -> Dict[str, Any]:
    return {
        "mode": "graph_only",
        "filters": build_filter_summary(args, query_text=query_text, notquery_text=notquery_text),
        "graph_match_count": len(graph_records),
        "graph_matches": graph_records,
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


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    query_text = args.query.strip()
    notquery_text = args.notquery.strip()

    graph_client = create_graph_client()

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

    print_output(
        build_output(
            args=args,
            query_text=query_text,
            notquery_text=notquery_text,
            graph_records=graph_records,
        ),
        args.verbose,
    )


if __name__ == "__main__":
    main()
