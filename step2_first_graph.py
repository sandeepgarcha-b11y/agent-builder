"""
Step 2: First LangGraph graph — single node.

Same result as Step 1, but now the work runs inside a graph.
Key concepts introduced: State, nodes, edges.
"""

from typing import TypedDict
from openai import OpenAI
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END

load_dotenv()
client = OpenAI()

# --- STATE ---
# State is a TypedDict — a shared "clipboard" that every node can read from and write to.
# Here it holds the raw input and the final analysis.
class State(TypedDict):
    experiment_input: str
    analysis: str

# --- NODE ---
# A node is just a Python function. It receives the current state,
# does some work, and returns only the fields it wants to update.
def analyze(state: State) -> dict:
    SYSTEM_PROMPT = """You are an experiment analysis assistant.
When given experiment results, respond with:
1. Summary of results
2. Likely causes of the observed change
3. Risks / confounders to watch out for
4. Next experiment suggestions
5. PM recommendation (ship it, iterate, or drop it — and why)
Be concise and direct."""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": state["experiment_input"]},
        ],
    )
    return {"analysis": response.choices[0].message.content}

# --- GRAPH ---
# Build the graph: wire up nodes and edges.
builder = StateGraph(State)
builder.add_node("analyze", analyze)       # register the node
builder.add_edge(START, "analyze")         # graph starts here
builder.add_edge("analyze", END)           # graph ends here
graph = builder.compile()

# --- RUN ---
result = graph.invoke({
    "experiment_input": """
Experiment: Changing the CTA button colour from grey to green on the checkout page.
Results: Conversion rate went from 3.2% to 3.8% over 7 days, 50k users per variant.
Notes: Ran during a holiday sale week. Mobile traffic was 70% of total.
""",
    "analysis": "",
})

print(result["analysis"])
