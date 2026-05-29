"""
config.py — Central configuration for the experiment analysis agent.

Change model choices, prompts, and settings here without touching graph logic.

Model cost reference (per 1M tokens, as of mid-2026):
  gpt-4o-mini  — $0.15 input / $0.60 output  — fast, cheap, good for simple tasks
  gpt-4o       — $2.50 input / $10.00 output  — stronger reasoning, use sparingly
"""

# ─────────────────────────────────────────────
# MODEL PER NODE
#
# Not every node needs the same model.
# Simple tasks (extract numbers, yes/no decisions) → cheapest model
# High-value outputs (compare, causes) → stronger model
# ─────────────────────────────────────────────

NODE_MODELS = {
    # Just pulls numbers out of text — no reasoning needed
    "run_stats":       "gpt-4o-mini",

    # Straightforward summarisation
    "summarize":       "gpt-4o-mini",

    # Binary yes/no based on explicit rules — cheapest is fine
    "check_validity":  "gpt-4o-mini",

    # Causal reasoning — slightly more nuanced but mini handles it well
    "identify_causes": "gpt-4o-mini",

    # Short warning message — no heavy reasoning needed
    "warn_invalid":    "gpt-4o-mini",

    # Risk identification — concise, mini is fine
    "flag_risks":      "gpt-4o-mini",

    # Most important output — explicit comparison + PM recommendation
    # Worth the upgrade for sharper, more direct reasoning
    "compare":         "gpt-4o",
}

# ─────────────────────────────────────────────
# LANGSMITH
# ─────────────────────────────────────────────

LANGSMITH_PROJECT = "experiment-agent"

# ─────────────────────────────────────────────
# AGENT SETTINGS
# ─────────────────────────────────────────────

# Minimum sample size per variant to pass validity check
MIN_SAMPLE_SIZE = 1000

# Minimum test duration in days to pass validity check
MIN_TEST_DAYS = 3

# How many characters of the summary to show in --history
HISTORY_SUMMARY_LENGTH = 80
