"""
Step 7: Full CLI agent.

What's new vs Step 6:
- The graph logic moves into its own file (graph.py) — separation of concerns
- A CLI entry point (agent.py) handles input, output, and flags
- MemorySaver swapped for SqliteSaver so history survives between sessions

This file shows the key new patterns introduced in Step 7.
The production code lives in agent.py + graph.py at the root.

New concepts:
  argparse    — Python's standard library for building CLI tools with flags
  SqliteSaver — database-backed checkpointer; persists state to a .db file
  Separation  — graph logic (graph.py) stays separate from CLI logic (agent.py)

How the CLI works:
  python3 agent.py --thread checkout-tests    # run an analysis
  python3 agent.py --thread checkout-tests --history  # show past experiments

How SqliteSaver replaces MemorySaver:
  # Step 6 — in-memory only, lost on exit:
  from langgraph.checkpoint.memory import MemorySaver
  memory = MemorySaver()
  graph = builder.compile(checkpointer=memory)

  # Step 7 — persists to disk across sessions:
  from langgraph.checkpoint.sqlite import SqliteSaver
  with SqliteSaver.from_conn_string("memory.db") as checkpointer:
      graph = builder.compile(checkpointer=checkpointer)

How argparse structures the CLI:
  parser = argparse.ArgumentParser(description="Experiment Analysis Agent")
  parser.add_argument("--thread", default="default")   # which experiment stream
  parser.add_argument("--history", action="store_true") # show past runs
  args = parser.parse_args()

  if args.history:
      show_history(graph, args.thread)
  else:
      run_analysis(graph, args.thread)

How history lookup works (same pattern as Step 6, but now across sessions):
  config = {"configurable": {"thread_id": thread_id}}
  for checkpoint in graph.get_state_history(config):
      if checkpoint.values.get("summary"):
          previous_run = checkpoint.values
          break  # most recent completed run

See agent.py for the full production implementation.
See graph.py for the full graph with all nodes.
"""

print("Step 7 introduces two things:")
print("1. agent.py — a CLI you can actually use")
print("2. graph.py — the graph logic, cleanly separated")
print()
print("Run the real agent with:")
print("  python3 agent.py --thread your-thread-name")
print("  python3 agent.py --thread your-thread-name --history")
