"""
Step 3: Multi-node pipeline.

Instead of one node doing everything, we now have three nodes in a chain:
  START → summarize → identify_causes → flag_risks → END

Each node reads what the previous one wrote and adds its own piece.
"""

from typing import TypedDict
from openai import OpenAI
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END

load_dotenv()
client = OpenAI()

# --- STATE ---
# We've added three new fields — one for each node's output.
class State(TypedDict):
    experiment_input: str
    summary: str
    causes: str
    risks: str

# --- NODE 1: summarize ---
# Just reads the raw experiment and writes a plain 2-3 sentence summary.
def summarize(state: State) -> dict:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Summarize this experiment result in 2-3 sentences. Be factual and concise."},
            {"role": "user", "content": state["experiment_input"]},
        ],
    )
    summary = response.choices[0].message.content
    print(f"SUMMARY:\n{summary}\n")
    return {"summary": summary}

# --- NODE 2: identify_causes ---
# Reads the summary and reasons about why the result happened.
def identify_causes(state: State) -> dict:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Given this experiment summary, list the most likely causes of the observed result. Be specific."},
            {"role": "user", "content": state["summary"]},
        ],
    )
    causes = response.choices[0].message.content
    print(f"CAUSES:\n{causes}\n")
    return {"causes": causes}

# --- NODE 3: flag_risks ---
# Reads BOTH the summary and causes to identify confounders and risks.
def flag_risks(state: State) -> dict:
    combined = f"Summary: {state['summary']}\n\nLikely causes: {state['causes']}"
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Given this experiment summary and its likely causes, identify the key risks and confounders. What should we be careful about before acting on these results?"},
            {"role": "user", "content": combined},
        ],
    )
    risks = response.choices[0].message.content
    print(f"RISKS:\n{risks}\n")
    return {"risks": risks}

# --- GRAPH ---
builder = StateGraph(State)

builder.add_node("summarize", summarize)
builder.add_node("identify_causes", identify_causes)
builder.add_node("flag_risks", flag_risks)

builder.add_edge(START, "summarize")
builder.add_edge("summarize", "identify_causes")
builder.add_edge("identify_causes", "flag_risks")
builder.add_edge("flag_risks", END)

graph = builder.compile()

# --- RUN ---
graph.invoke({
    "experiment_input": """
Experiment: Changing the CTA button colour from grey to green on the checkout page.
Results: Conversion rate went from 3.2% to 3.8% over 7 days, 50k users per variant.
Notes: Ran during a holiday sale week. Mobile traffic was 70% of total.
""",
    "summary": "",
    "causes": "",
    "risks": "",
})
