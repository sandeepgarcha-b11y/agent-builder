# Experiment Analysis Agent — Progress Guide

## What we're building

You paste in experiment results (like "we changed the button colour and conversion went up 0.6%") and the agent tells you:
- What happened (summary)
- Why it probably happened (causes)
- What risks or confounders to watch out for
- Whether you should ship it, iterate, or drop it
- How it compares to previous experiments on the same thread

Built step by step using **LangGraph** — a Python framework for connecting AI calls into a structured flow.

---

## Repo structure

```
steps/                        ← learning files, one per step
  step1_raw_llm_call.py
  step2_first_graph.py
  step3_multi_node.py
  step4_conditional_routing.py
  step5_tools.py
  step6_memory.py
  step7_cli_agent.py          ← explains Step 7 concepts + points to production files
  step8_config_observability.py ← explains Step 8 + verifies your setup

agent.py                      ← production CLI (use this day to day)
graph.py                      ← production graph (imported by agent.py)
config.py                     ← production config (models, thresholds, settings)

requirements.txt
.env.example
```

---

## How to run the agent

```bash
# Activate the virtual environment
source .venv/bin/activate

# Analyse an experiment
python3 agent.py --thread checkout-tests

# Show previous experiments on a thread
python3 agent.py --thread checkout-tests --history
```

---

## The 8 steps

### Step 1 — `steps/step1_raw_llm_call.py` ✅
**What it does:** Sends your experiment notes to OpenAI and prints a structured analysis.

**What you learned:** How to call OpenAI from Python. You write a system prompt (the instructions) and a user message (the input), and the model replies.

```
Your experiment notes → [OpenAI API call] → Structured analysis
```

---

### Step 2 — `steps/step2_first_graph.py` ✅
**What it does:** Same output as Step 1, but wrapped inside a LangGraph graph.

**What you learned:** Three core LangGraph concepts:
- **State** — a shared notepad every node can read from and write to
- **Node** — one box in the flowchart; does one job, reads state, writes state
- **Edge** — the arrow between boxes; tells the graph what runs next

The output was identical to Step 1 — intentional. We changed the structure, not the result.

---

### Step 3 — `steps/step3_multi_node.py` ✅
**What it does:** Splits the single analysis into three nodes that run in sequence.

```
summarize → identify_causes → flag_risks
```

**What you learned:** Each node builds on the previous one. `flag_risks` reads the summary AND the causes before deciding what's risky — reasoning in layers rather than one big question.

---

### Step 4 — `steps/step4_conditional_routing.py` ✅
**What it does:** Adds a fork. After summarising, the agent checks if the experiment is trustworthy. Valid → full analysis. Invalid → warning.

```
summarize → check_validity → identify_causes → flag_risks
                    ↘
                warn_invalid
```

**What you learned:** The difference between a **pipeline** (always does the same steps) and an **agent** (makes decisions). The conditional edge is a function that reads state and returns the name of the next node.

---

### Step 5 — `steps/step5_tools.py` ✅
**What it does:** Adds a real stats calculator. The agent now runs actual maths instead of guessing.

`calculate_stats` returns:
- **Uplift** — relative % change (e.g. +18.7%)
- **Absolute change** — raw difference in percentage points
- **Sample size check** — is there enough data?
- **Duration check** — did the test run long enough?
- **Significance** — two-proportion z-test with p-value

**What you learned:** A tool is a Python function the agent calls for real information. The LLM extracts numbers from text; the tool does the maths. `check_validity` now uses actual stats instead of guessing.

---

### Step 6 — `steps/step6_memory.py` ✅
**What it does:** The agent remembers previous experiments on the same thread and explicitly compares the current result against the last one.

**What you learned:**
- **MemorySaver** — saves state snapshots after each node runs
- **thread_id** — identifies an experiment stream; same thread = shared history
- **get_state_history()** — retrieves the last completed run's state before invoking
- **compare node** — produces a direct comparison: *"This uplift is lower than the last test. Iterate."*

---

### Step 7 — `agent.py` + `graph.py` ✅
**What it does:** The agent becomes a real CLI tool you run day to day.

- **`graph.py`** — all LangGraph logic, cleanly separated and reusable
- **`agent.py`** — the CLI entry point: takes input, runs the graph, formats output

History persists to `memory.db` (SQLite) — survives between sessions.

**What you learned:**
- **SqliteSaver** — replaces MemorySaver; persists state to a `.db` file across sessions
- **Separation of concerns** — graph logic in `graph.py`, CLI in `agent.py`
- **argparse** — Python's standard library for building CLI tools with flags

See `steps/step7_cli_agent.py` for an annotated walkthrough of the key patterns.

---

### Step 8 — `config.py` + LangSmith ✅
**What it does:** Central config controls model choices per node. LangSmith tracing gives you a dashboard to see inside every run.

**Model routing (from `config.py`):**
| Node | Model | Why |
|---|---|---|
| run_stats | gpt-4o-mini | Just extracting numbers |
| summarize | gpt-4o-mini | Straightforward task |
| check_validity | gpt-4o-mini | Binary yes/no |
| identify_causes | gpt-4o-mini | Mini handles it well |
| warn_invalid | gpt-4o-mini | Short message |
| flag_risks | gpt-4o-mini | Concise bullets |
| compare | gpt-4o | Most important output — worth the upgrade |

**LangSmith** — when enabled, every run shows in your dashboard at `eu.smith.langchain.com`:
each node, its prompt, response, latency, and token cost.

To enable: set these in `.env`:
```
LANGSMITH_API_KEY=your-key
LANGSMITH_TRACING=true
LANGSMITH_PROJECT=experiment-analysis-agent
LANGSMITH_ENDPOINT=https://eu.api.smith.langchain.com
```

**What you learned:**
- **Config separation** — settings in one place, not scattered through the graph
- **Model tradeoffs** — match model to task complexity; not every node needs gpt-4o
- **Observability** — tracing shows what ran, what it cost, where it slowed down

Run `python3 steps/step8_config_observability.py` to verify your setup.

---

## Starting fresh on a new machine

```bash
# 1. Clone the repo
git clone git@github-b11y:sandeepgarcha-b11y/agent-builder.git
cd agent-builder

# 2. Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up .env
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY and LANGSMITH_API_KEY
```

---

## Key terms

| Term | What it means |
|---|---|
| **LangGraph** | A framework for building AI workflows as a flowchart |
| **State** | The shared notepad — holds all data flowing through the graph |
| **Node** | One step in the flowchart — a Python function that does one job |
| **Edge** | The arrow between steps — tells the graph what runs next |
| **Conditional edge** | An arrow that routes to different nodes based on a decision |
| **Tool** | A Python function the agent calls for real data or calculations |
| **Checkpointer** | Saves state after each node — enables memory and persistence |
| **thread_id** | Identifies an experiment stream — same thread = shared history |
| **Pipeline** | Always does the same steps in the same order |
| **Agent** | Makes decisions about what to do next based on what it sees |
