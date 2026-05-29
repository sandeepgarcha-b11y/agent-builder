# What We're Building — Plain English Guide

## The goal

We're building an **Experiment Analysis Agent**.

You paste in experiment results (like "we changed the button colour and conversion went up 0.6%") and the agent tells you:
- What happened
- Why it probably happened
- What risks or confounders to watch out for
- What to try next
- Whether you should ship it, iterate, or drop it

We're building it step by step using **LangGraph** — a framework for connecting AI calls together into a flow.

---

## What we've built so far

### Step 1 — `step1_raw_llm_call.py` ✅
**What it does:** Sends your experiment notes to OpenAI and prints a structured analysis.

**What you learned:** How to talk to OpenAI from Python code. You write a system prompt (the instructions) and a user message (the input), and the model replies. Simple as that.

---

### Step 2 — `step2_first_graph.py` ✅
**What it does:** Same as Step 1, but wrapped inside a LangGraph graph.

**What you learned:** Three core LangGraph concepts:
- **State** — a shared notepad that every step in the graph can read from and write to
- **Node** — one box in the flowchart. Does one job, reads from state, writes back to state
- **Edge** — the arrow connecting boxes. Tells the graph what runs next

The output looked identical to Step 1 — that was intentional. We changed the structure, not the result.

---

### Step 3 — `step3_multi_node.py` ✅
**What it does:** Splits the single analysis into three separate nodes that run in sequence.

```
summarize → identify_causes → flag_risks
```

**What you learned:** Each node builds on the previous one. `flag_risks` reads both the summary AND the causes before deciding what's risky — it's reasoning in layers. This is better than one big question because each step is focused.

---

### Step 4 — `step4_conditional_routing.py` ✅
**What it does:** Adds a fork in the road. After summarising, the agent checks if the experiment is trustworthy. If yes → full analysis. If no → warning message.

```
summarize → check_validity → identify_causes → flag_risks
                    ↘
                warn_invalid
```

**What you learned:** The difference between a **pipeline** (always does the same steps) and an **agent** (makes decisions). The conditional edge is just a function that reads state and returns the name of the next node to run.

---

### Step 5 — `step5_tools.py` ✅
**What it does:** Adds a real stats calculator tool — the agent now runs actual maths instead of guessing.

A new node (`run_stats`) first asks the LLM to extract the numbers from the experiment text, then calls `calculate_stats` — a plain Python function — and writes the results to state. Every node after that has access to real numbers.

`calculate_stats` returns:
- **Uplift** — relative % change (e.g. +18.7%)
- **Absolute change** — raw difference in percentage points
- **Sample size check** — is there enough data to trust the result?
- **Duration check** — did the test run long enough?
- **Significance** — two-proportion z-test, p-value and confidence level

**What you learned:** A tool is just a Python function the agent can call to get real information. The LLM extracts the numbers; the tool does the maths. `check_validity` in Step 4 was guessing — in Step 5 it has actual stats to work with.

---

## What's coming next

### Step 6 — `step6_memory.py` ✅
**What it does:** The agent now remembers previous experiments on the same thread and explicitly compares the current result against the last one.

A `MemorySaver` checkpointer saves the graph state after every node. Each experiment stream has a `thread_id` (e.g. `"checkout-tests"`). Before each run, we call `get_state_history()` to find the last completed run on that thread and pass it in as `previous_run`. A new `compare` node at the end uses it to produce a direct comparison.

Example output from Run 2:
> *"The current uplift of 7.9% is lower than the previous colour test's 18.7%. The colour change had a stronger impact. Recommend iterating on the text rather than shipping."*

**What you learned:**
- **MemorySaver** — saves state snapshots after each node runs
- **thread_id** — identifies an experiment stream. Same thread = shared history
- **get_state_history()** — lets you retrieve the last completed run's state before invoking
- **compare node** — explicitly compares current vs previous, like a PM reviewing a series of tests

---

### Step 7 — Full CLI agent
The agent will be able to reach outside itself — look things up, run calculations, query data. This is what separates a smart chatbot from something that can actually act in the world.

### Step 6 — Memory
The agent will remember previous experiments across sessions. So you can ask "how does this compare to last month's test?"

### Step 7 — Full CLI agent
Everything wired together into one script you can actually use day to day.

### Step 8 — Config & observability
How to monitor what the agent is doing, control costs, and choose the right model for each step.

---

## Starting fresh on a new machine

If you switch laptops, do this:

```bash
# 1. Clone the repo
git clone git@github-b11y:sandeepgarcha-b11y/agent-builder.git
cd agent-builder

# 2. Create the virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add your OpenAI key
echo "OPENAI_API_KEY=your-key-here" > .env
```

Then pick up from wherever CLAUDE.md says we left off.

---

## Key terms, simply explained

| Term | What it means |
|---|---|
| **LangGraph** | A framework for building AI workflows as a flowchart |
| **State** | The shared notepad — holds all the data flowing through the graph |
| **Node** | One step in the flowchart — a Python function that does one job |
| **Edge** | The arrow between steps — tells the graph what runs next |
| **Conditional edge** | An arrow that goes to different places depending on a decision |
| **Tool** | Something that lets the agent reach outside itself (search, database, calculator) |
| **Pipeline** | Always does the same steps in the same order |
| **Agent** | Makes decisions about what to do next based on what it sees |
