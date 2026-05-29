"""
Step 8: Config, observability, and tradeoffs.

What's new vs Step 7:
- config.py centralises all settings — model per node, thresholds, project name
- graph.py reads from config instead of hardcoding values
- LangSmith tracing is wired up via environment variables

New concepts:
  config.py   — one file to control the whole agent's behaviour
  NODE_MODELS — assign different models to different nodes based on cost/quality tradeoff
  LangSmith   — observability dashboard: see every node, its input/output, latency, and cost

------------------------------------------------------------
WHY CONFIG MATTERS
------------------------------------------------------------
Before config.py, every node had this hardcoded:
  model="gpt-4o-mini"

After config.py, every node reads:
  model=NODE_MODELS["compare"]

So to swap the compare node to gpt-4o, you change ONE line in config.py.
No hunting through graph.py. No risk of missing a node.

------------------------------------------------------------
MODEL TRADEOFFS (from config.py)
------------------------------------------------------------
  run_stats       → gpt-4o-mini   just extracting numbers, cheapest is fine
  summarize       → gpt-4o-mini   straightforward summarisation
  check_validity  → gpt-4o-mini   binary yes/no, no heavy reasoning needed
  identify_causes → gpt-4o-mini   mini handles causal reasoning well
  warn_invalid    → gpt-4o-mini   short warning message
  flag_risks      → gpt-4o-mini   concise bullets, mini is fine
  compare         → gpt-4o        most important output — worth the upgrade

Cost reference (per 1M tokens):
  gpt-4o-mini  $0.15 input / $0.60 output
  gpt-4o       $2.50 input / $10.00 output

------------------------------------------------------------
LANGSMITH — HOW TO ENABLE
------------------------------------------------------------
LangSmith traces automatically when these env vars are set in .env:

  LANGSMITH_API_KEY=your-key
  LANGSMITH_TRACING=true
  LANGSMITH_PROJECT=experiment-analysis-agent
  LANGSMITH_ENDPOINT=https://eu.api.smith.langchain.com  # EU region

Get a key at: https://eu.smith.langchain.com
Note: requires a card on file to unlock API access (free tier = 5,000 traces/month)

Once enabled, every agent.py run appears in your dashboard showing:
  - Each node that ran
  - What it was sent (the prompt)
  - What it returned
  - How long it took
  - How many tokens it used and what that cost

------------------------------------------------------------
VERIFY YOUR SETUP
------------------------------------------------------------
Run this script to check config loads correctly and LangSmith is reachable.
"""

from dotenv import load_dotenv
load_dotenv()

import os
import sys

# ── Config check ──────────────────────────────────────────
print("=" * 50)
print("CONFIG CHECK")
print("=" * 50)

try:
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from config import NODE_MODELS, MIN_SAMPLE_SIZE, MIN_TEST_DAYS, LANGSMITH_PROJECT

    print(f"  Min sample size:  {MIN_SAMPLE_SIZE:,}/variant")
    print(f"  Min test days:    {MIN_TEST_DAYS}")
    print(f"  LangSmith project: {LANGSMITH_PROJECT}")
    print()
    print("  Model per node:")
    for node, model in NODE_MODELS.items():
        cost = "💰 stronger" if model == "gpt-4o" else "✓ cheap"
        print(f"    {node:<20} {model:<15} {cost}")
    print()
    print("  ✓ config.py loaded successfully")
except Exception as e:
    print(f"  ✗ config.py error: {e}")

# ── LangSmith check ───────────────────────────────────────
print()
print("=" * 50)
print("LANGSMITH CHECK")
print("=" * 50)

tracing = os.getenv("LANGSMITH_TRACING", "false")
endpoint = os.getenv("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com")
key = os.getenv("LANGSMITH_API_KEY", "")
project = os.getenv("LANGSMITH_PROJECT", "")

print(f"  Tracing enabled:  {tracing}")
print(f"  Endpoint:         {endpoint}")
print(f"  Project:          {project}")
print(f"  Key set:          {'yes (' + key[:12] + '...)' if key else 'NO — set LANGSMITH_API_KEY in .env'}")

if tracing == "true" and key:
    try:
        import langsmith
        client = langsmith.Client()
        projects = list(client.list_projects())
        print(f"  Connection:       ✓ Connected ({len(projects)} project(s))")
    except Exception as e:
        print(f"  Connection:       ✗ {e}")
else:
    print("  Connection:       skipped (tracing disabled or no key)")
