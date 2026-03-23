"""Standalone CLI for variant list analysis.

Usage:
    python -m genetics_mcp_server.scripts.analyze_variants < variants.txt
    python -m genetics_mcp_server.scripts.analyze_variants variants.txt
    python -m genetics_mcp_server.scripts.analyze_variants variants.txt --resource finngen

Reads a variant list from stdin or a file, runs the analysis against
the Genetics API, and prints JSON results to stdout.

Requires GENETICS_API_URL environment variable (or uses default http://localhost:2000/api).
"""

import argparse
import asyncio
import json
import sys


async def main():
    parser = argparse.ArgumentParser(
        description="Analyze a list of variants for phenotype, QTL, and tissue patterns."
    )
    parser.add_argument(
        "input_file",
        nargs="?",
        default="-",
        help="Variant list file (default: stdin)",
    )
    parser.add_argument(
        "--resource",
        default=None,
        help="Filter to a specific data resource (e.g., 'finngen', 'ukbb')",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print JSON output",
    )
    args = parser.parse_args()

    # read input
    if args.input_file == "-":
        variants_text = sys.stdin.read()
    else:
        with open(args.input_file) as f:
            variants_text = f.read()

    if not variants_text.strip():
        print("Error: no input provided", file=sys.stderr)
        sys.exit(1)

    # import here to avoid loading everything at module level
    from genetics_mcp_server.tools.executor import ToolExecutor

    executor = ToolExecutor()
    try:
        result = await executor.analyze_variant_list(variants_text, resource=args.resource)
        indent = 2 if args.pretty else None
        print(json.dumps(result, indent=indent, default=str))
    finally:
        await executor.close()


if __name__ == "__main__":
    asyncio.run(main())
