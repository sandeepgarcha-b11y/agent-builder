"""
Step 4: Conditional routing.

After summarizing, the graph decides whether the experiment is safe to interpret.
If the result looks too weak or noisy, it routes to a warning node.
Otherwise it continues through the normal analysis path.

Graph:
  START → summarize → check_validity → identify_causes → flag_risks → END
                             ↘
                          warn_invalid → END
"""

from typing import TypedDict
from openai import OpenAI
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END

load_dotenv()
client = OpenAI()

# --- STATE ---
# Added is_valid (the decision) and warning (used if experiment is too weak).
class State(TypedDict):
    experiment_input: str
    summary: str
    is_valid: bool
    causes: str
    risks: str
    warning: str

# --- NODE 1: summarize (same as Step 3) ---
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

# --- NODE 2: check_validity ---
# Asks the LLM to judge whether the experiment is safe to interpret.
# Writes True or False into state — the conditional edge reads this.
def check_validity(state: State) -> dict:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an experiment reviewer. Given a summary of an experiment, "
                    "decide if the result is safe to interpret. "
                    "Reply with only 'VALID' or 'INVALID'.\n\n"
                    "Mark INVALID if any of these are true:\n"
                    "- Sample size is too small (under 1000 per variant)\n"
                    "- Test ran for less than 3 days\n"
                    "- There are obvious confounders that make the result untrustworthy\n"
                    "- The result could easily be noise"
                ),
            },
            {"role": "user", "content": state["summary"]},
        ],
    )
    verdict = response.choices[0].message.content.strip()
    is_valid = verdict == "VALID"
    print(f"VALIDITY CHECK: {verdict}\n")
    return {"is_valid": is_valid}

# --- NODE 3a: identify_causes (same as Step 3, only runs if valid) ---
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

# --- NODE 3b: warn_invalid (only runs if experiment is too weak) ---
def warn_invalid(state: State) -> dict:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an experiment reviewer. The experiment summary below has been flagged as "
                    "unreliable or too weak to interpret safely. "
                    "Write a short, clear warning for the PM explaining why this result should not be acted on, "
                    "and what they should do instead to get a trustworthy result."
                ),
            },
            {"role": "user", "content": state["summary"]},
        ],
    )
    warning = response.choices[0].message.content
    print(f"WARNING:\n{warning}\n")
    return {"warning": warning}

# --- NODE 4: flag_risks (same as Step 3, only runs if valid) ---
def flag_risks(state: State) -> dict:
    combined = f"Summary: {state['summary']}\n\nLikely causes: {state['causes']}"
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Given this experiment summary and its likely causes, identify the key risks and confounders. What should we be careful about before acting on these results? Be concise — 3-5 bullet points max."},
            {"role": "user", "content": combined},
        ],
    )
    risks = response.choices[0].message.content
    print(f"RISKS:\n{risks}\n")
    return {"risks": risks}

# --- ROUTING FUNCTION ---
# This is not a node — it's a function that reads state and returns the name
# of the next node to go to. LangGraph calls this at the conditional edge.
def route_after_validity_check(state: State) -> str:
    if state["is_valid"]:
        return "identify_causes"
    else:
        return "warn_invalid"

# --- GRAPH ---
builder = StateGraph(State)

builder.add_node("summarize", summarize)
builder.add_node("check_validity", check_validity)
builder.add_node("identify_causes", identify_causes)
builder.add_node("warn_invalid", warn_invalid)
builder.add_node("flag_risks", flag_risks)

builder.add_edge(START, "summarize")
builder.add_edge("summarize", "check_validity")

# This is the conditional edge — instead of a fixed next node,
# it calls route_after_validity_check to decide where to go.
builder.add_conditional_edges("check_validity", route_after_validity_check)

builder.add_edge("identify_causes", "flag_risks")
builder.add_edge("flag_risks", END)
builder.add_edge("warn_invalid", END)

graph = builder.compile()

# --- RUN: valid experiment ---
print("=" * 50)
print("TEST 1: Should pass validity check")
print("=" * 50)
graph.invoke({
    "experiment_input": """
Experiment: Changing the CTA button colour from grey to green on the checkout page.
Results: Conversion rate went from 3.2% to 3.8% over 7 days, 50k users per variant.
Notes: Ran during a holiday sale week. Mobile traffic was 70% of total.
""",
    "summary": "", "is_valid": False, "causes": "", "risks": "", "warning": "",
})

# --- RUN: weak experiment ---
print("=" * 50)
print("TEST 2: Should fail validity check")
print("=" * 50)
graph.invoke({
    "experiment_input": """
Experiment: Changed the font size on the homepage hero from 18px to 20px.
Results: Bounce rate dropped from 62% to 61% over 1 day, 200 users per variant.
Notes: Data collected on a Sunday. No statistical significance test run.
""",
    "summary": "", "is_valid": False, "causes": "", "risks": "", "warning": "",
})
