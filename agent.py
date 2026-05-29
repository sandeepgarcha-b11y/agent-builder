"""
agent.py — Experiment Analysis Agent CLI.

Usage:
  python3 agent.py                          # uses default thread
  python3 agent.py --thread checkout-tests  # named experiment stream
  python3 agent.py --thread checkout-tests --history  # show past experiments
"""

import argparse
import sys
from datetime import datetime
from langgraph.checkpoint.sqlite import SqliteSaver
from graph import build_graph, State

DB_FILE = "memory.db"
DIVIDER = "─" * 55


# ─────────────────────────────────────────────
# OUTPUT HELPERS
# ─────────────────────────────────────────────

def print_header(thread_id: str):
    print(f"\n{'═' * 55}")
    print(f"  Experiment Analysis Agent")
    print(f"  Thread: {thread_id}")
    print(f"{'═' * 55}\n")


def print_stats(stats: dict):
    print(f"{DIVIDER}")
    print("  STATS")
    print(f"{DIVIDER}")
    print(f"  Uplift:          {stats['uplift']}")
    print(f"  Absolute change: {stats['absolute_change']}")
    print(f"  Sample size:     {stats['sample_check']}")
    print(f"  Duration:        {stats['duration_check']}")
    print(f"  Significance:    {stats['significance']}")
    print()


def print_section(title: str, body: str):
    print(f"{DIVIDER}")
    print(f"  {title}")
    print(f"{DIVIDER}")
    for line in body.strip().split("\n"):
        print(f"  {line}")
    print()


def print_result(result: dict):
    print_stats(result["stats"])

    if result.get("is_valid"):
        print_section("SUMMARY", result["summary"])
        print_section("LIKELY CAUSES", result["causes"])
        print_section("RISKS & CONFOUNDERS", result["risks"])
    else:
        print_section("⚠️  INVALID EXPERIMENT", result["warning"])

    if result.get("comparison"):
        print_section("VS PREVIOUS EXPERIMENT", result["comparison"])

    print(f"{'═' * 55}\n")


def print_history(graph, thread_id: str):
    config = {"configurable": {"thread_id": thread_id}}

    # Only keep fully completed runs — those where the compare node has fired.
    # get_state_history returns all intermediate checkpoints too, so we filter
    # to ones where `comparison` is set and deduplicate by experiment_input.
    seen = set()
    runs = []

    for checkpoint in graph.get_state_history(config):
        vals = checkpoint.values
        exp_input = vals.get("experiment_input", "")
        # A completed run has stats and hasn't been seen yet
        if vals.get("stats") and vals.get("summary") and exp_input not in seen:
            seen.add(exp_input)
            runs.append(vals)

    if not runs:
        print(f"\nNo experiments found on thread '{thread_id}'.\n")
        return

    print(f"\n{'═' * 55}")
    print(f"  History — {thread_id}  ({len(runs)} experiment(s))")
    print(f"{'═' * 55}")

    for i, run in enumerate(reversed(runs), 1):
        stats = run.get("stats", {})
        valid = "✓ Valid" if run.get("is_valid") else "✗ Invalid"
        print(f"\n  [{i}] {valid}")
        print(f"      Uplift:       {stats.get('uplift', 'N/A')}")
        print(f"      Significance: {stats.get('significance', 'N/A')}")
        print(f"      Summary:      {run.get('summary', '')[:80]}...")

    print(f"\n{'═' * 55}\n")


# ─────────────────────────────────────────────
# CORE — run one experiment
# ─────────────────────────────────────────────

def run_experiment(graph, experiment_input: str, thread_id: str) -> dict:
    config = {"configurable": {"thread_id": thread_id}}

    # Load the most recent completed run from this thread's history
    previous_run = None
    for checkpoint in graph.get_state_history(config):
        vals = checkpoint.values
        if vals.get("summary"):
            previous_run = {
                "summary": vals.get("summary", ""),
                "stats": vals.get("stats", {}),
                "causes": vals.get("causes", ""),
            }
            break

    return graph.invoke(
        {
            "experiment_input": experiment_input,
            "previous_run": previous_run,
            "stats": {},
            "summary": "",
            "is_valid": False,
            "causes": "",
            "risks": "",
            "warning": "",
            "comparison": "",
        },
        config=config,
    )


# ─────────────────────────────────────────────
# INPUT — paste multi-line experiment notes
# ─────────────────────────────────────────────

def get_experiment_input() -> str:
    print("Paste your experiment notes below.")
    print("Press Enter twice when done.\n")

    lines = []
    blank_count = 0

    while True:
        try:
            line = input()
        except EOFError:
            break

        if line == "":
            blank_count += 1
            if blank_count >= 2:
                break
            lines.append(line)
        else:
            blank_count = 0
            lines.append(line)

    return "\n".join(lines).strip()


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Experiment Analysis Agent")
    parser.add_argument(
        "--thread",
        default="default",
        help='Experiment stream name (default: "default")',
    )
    parser.add_argument(
        "--history",
        action="store_true",
        help="Show previous experiments on this thread and exit",
    )
    args = parser.parse_args()

    with SqliteSaver.from_conn_string(DB_FILE) as checkpointer:
        graph = build_graph(checkpointer)

        if args.history:
            print_history(graph, args.thread)
            return

        print_header(args.thread)
        experiment_input = get_experiment_input()

        if not experiment_input:
            print("No input provided. Exiting.")
            sys.exit(1)

        print(f"\nAnalysing...\n")
        result = run_experiment(graph, experiment_input, args.thread)
        print_result(result)


if __name__ == "__main__":
    main()
