# Agent Builder — LangGraph Learning Project

## What this is
A step-by-step learning project to build an **Experiment Analysis Agent** using LangGraph + OpenAI.

**Agent goal:** Paste experiment results + notes → agent returns:
- Summary of results
- Likely causes
- Risks / confounders
- Next experiment suggestions
- PM recommendation

User is a Python beginner with some LLM API experience. Go step by step, explain tradeoffs, don't skip ahead.

---

## Learning roadmap

- [x] **Step 1** — Project setup + raw LLM call (OpenAI, no LangGraph yet)
- [x] **Step 2** — First LangGraph graph (single node, understand state/nodes/edges)
- [x] **Step 3** — Multi-node pipeline (summarize → identify_causes → flag_risks)
- [x] **Step 4** — Conditional routing (agent makes decisions, not just processes)
- [ ] **Step 5** — Tools (what separates a pipeline from an agent)
- [ ] **Step 6** — Memory & persistence (LangGraph checkpointing)
- [ ] **Step 7** — Full agent wired end to end with CLI
- [ ] **Step 8** — Config, observability, tradeoffs (LangSmith, model choice, cost)

**Current status: Step 4 complete. Up next: Step 5 — Tools.**

---

## Environment setup (do this first on a new machine)

### 1. Fix Python (if needed)
```bash
# Option A: accept Xcode license
sudo xcodebuild -license

# Option B: install via Homebrew
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
brew install python
```

### 2. Create and activate virtual environment
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set up environment variables
```bash
cp .env.example .env
# Then edit .env and add your OPENAI_API_KEY
```

---

## SSH / GitHub
- Remote: `git@github-b11y:sandeepgarcha-b11y/agent-builder.git`
- Uses `~/.ssh/id_ed25519_b11y` key via `github-b11y` SSH alias
- If SSH isn't working on new machine: check `~/.ssh/config` has the `github-b11y` host block

---

## Key decisions made so far
- **LLM provider:** OpenAI (user has API key)
- **Framework:** LangGraph
- **Language:** Python

---

## When picking up in a new session
Start by saying: "Let's continue the LangGraph learning project. Check CLAUDE.md for where we left off."
Then proceed with the next unchecked step in the roadmap above.
