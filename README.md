# Experiment Analysis Agent

A step-by-step learning project to build an AI agent using LangGraph + OpenAI.

You paste in experiment results and the agent returns:
- Stats (uplift, significance, sample size check)
- Summary of what happened
- Likely causes
- Risks and confounders
- Comparison against your previous experiment on the same thread
- A clear ship / iterate / drop recommendation

---

## Quickstart

```bash
# 1. Clone
git clone git@github-b11y:sandeepgarcha-b11y/agent-builder.git
cd agent-builder

# 2. Set up environment
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 3. Add your keys
cp .env.example .env
# Edit .env with your OPENAI_API_KEY and LANGSMITH_API_KEY

# 4. Run
python3 agent.py --thread checkout-tests
```

---

## Usage

```bash
# Analyse an experiment (paste notes, press Enter twice when done)
python3 agent.py --thread checkout-tests

# Show previous experiments on a thread
python3 agent.py --thread checkout-tests --history
```

The `--thread` flag is your experiment stream name. Experiments on the same thread are compared against each other. History persists to `memory.db` between sessions.

---

## Repo structure

```
steps/                          ← one file per learning step
  step1_raw_llm_call.py
  step2_first_graph.py
  step3_multi_node.py
  step4_conditional_routing.py
  step5_tools.py
  step6_memory.py
  step7_cli_agent.py
  step8_config_observability.py

agent.py                        ← CLI entry point
graph.py                        ← LangGraph nodes and pipeline
config.py                       ← model choices, thresholds, settings
requirements.txt
.env.example
```

---

## How it was built — 8 steps

| Step | File | What it added |
|---|---|---|
| 1 | `step1_raw_llm_call.py` | Raw OpenAI call, no framework |
| 2 | `step2_first_graph.py` | First LangGraph graph — state, nodes, edges |
| 3 | `step3_multi_node.py` | Multi-node pipeline — summarize → causes → risks |
| 4 | `step4_conditional_routing.py` | Conditional routing — agent makes decisions |
| 5 | `step5_tools.py` | Real stats calculator tool |
| 6 | `step6_memory.py` | Memory — compares vs previous experiment |
| 7 | `agent.py` + `graph.py` | Full CLI with SQLite persistence |
| 8 | `config.py` + LangSmith | Config-driven models, observability |

See `PROGRESS.md` for a plain-English explanation of each step.

---

## Environment variables

| Variable | Required | Description |
|---|---|---|
| `OPENAI_API_KEY` | Yes | Your OpenAI API key |
| `LANGSMITH_API_KEY` | Optional | LangSmith tracing key |
| `LANGSMITH_TRACING` | Optional | Set to `true` to enable tracing |
| `LANGSMITH_PROJECT` | Optional | Project name in LangSmith |
| `LANGSMITH_ENDPOINT` | Optional | Use `https://eu.api.smith.langchain.com` for EU region |
