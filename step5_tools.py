"""
Step 5: Tools.

We add a real stats calculator — a Python function the agent can call
to get actual numbers instead of having the LLM guess.

A new node (run_stats) extracts numbers from the experiment text,
calls the tool, and writes the results to state. Every node after
that has access to real stats.

Graph:
  START → run_stats → summarize → check_validity → identify_causes → flag_risks → END
                                        ↘
                                    warn_invalid → END
"""

import json
import math
from typing import TypedDict
from openai import OpenAI
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END

load_dotenv()
client = OpenAI()

# ─────────────────────────────────────────────
# THE TOOL
# A plain Python function — no LLM involved.
# Takes real numbers, returns real calculations.
# ─────────────────────────────────────────────

def calculate_stats(
    control_rate: float,
    treatment_rate: float,
    sample_size_per_variant: int,
    test_days: int,
) -> dict:
    """
    Given experiment numbers, calculate:
    - Uplift (relative % change)
    - Absolute change (percentage points)
    - Sample size sanity check
    - Test duration check
    - Rough statistical significance (two-proportion z-test)
    """

    # Uplift — relative change, e.g. 3.2% → 3.8% is +18.75% uplift
    uplift_pct = ((treatment_rate - control_rate) / control_rate) * 100

    # Absolute change — the raw difference in percentage points
    absolute_change_pp = (treatment_rate - control_rate) * 100

    # Sample size sanity check
    if sample_size_per_variant < 1000:
        sample_check = "⚠️ Too small (under 1,000/variant) — results unreliable."
    elif sample_size_per_variant < 5000:
        sample_check = "⚠️ Borderline (under 5,000/variant) — treat with caution."
    else:
        sample_check = f"✓ Adequate ({sample_size_per_variant:,}/variant)."

    # Test duration check
    if test_days < 3:
        duration_check = "⚠️ Too short (under 3 days) — won't capture weekly patterns."
    elif test_days < 7:
        duration_check = f"⚠️ Short ({test_days} days) — consider running a full week."
    else:
        duration_check = f"✓ Adequate ({test_days} days)."

    # Two-proportion z-test for significance
    # Pools the two rates to estimate variance under the null hypothesis
    p_pool = (
        (control_rate * sample_size_per_variant + treatment_rate * sample_size_per_variant)
        / (2 * sample_size_per_variant)
    )
    se = math.sqrt(p_pool * (1 - p_pool) * (2 / sample_size_per_variant))

    if se > 0:
        z = (treatment_rate - control_rate) / se
        # Approximate p-value using the error function
        p_value = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))
        confidence = (1 - p_value) * 100

        if p_value < 0.05:
            significance = f"✓ Significant (p={p_value:.3f}, ~{confidence:.0f}% confidence)"
        elif p_value < 0.10:
            significance = f"⚠️ Borderline (p={p_value:.3f}, ~{confidence:.0f}% confidence)"
        else:
            significance = f"✗ Not significant (p={p_value:.3f}, ~{confidence:.0f}% confidence)"
    else:
        significance = "Could not calculate significance."

    return {
        "uplift": f"+{uplift_pct:.1f}%",
        "absolute_change": f"+{absolute_change_pp:.2f} percentage points",
        "sample_check": sample_check,
        "duration_check": duration_check,
        "significance": significance,
    }


# ─────────────────────────────────────────────
# STATE
# Added `stats` — the tool output, shared with all nodes.
# ─────────────────────────────────────────────

class State(TypedDict):
    experiment_input: str
    stats: dict
    summary: str
    is_valid: bool
    causes: str
    risks: str
    warning: str


# ─────────────────────────────────────────────
# NODE 1: run_stats
# Step 1 — ask the LLM to extract numbers from the experiment text
# Step 2 — call calculate_stats with those numbers
# ─────────────────────────────────────────────

def run_stats(state: State) -> dict:
    extraction = client.chat.completions.create(
        model="gpt-4o-mini",
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": (
                    "Extract experiment numbers from the text. "
                    "Return JSON with exactly these keys: "
                    "control_rate (float, e.g. 0.032), "
                    "treatment_rate (float, e.g. 0.038), "
                    "sample_size_per_variant (int), "
                    "test_days (int). "
                    "If a value is missing, use null."
                ),
            },
            {"role": "user", "content": state["experiment_input"]},
        ],
    )

    numbers = json.loads(extraction.choices[0].message.content)
    stats = calculate_stats(
        control_rate=numbers["control_rate"],
        treatment_rate=numbers["treatment_rate"],
        sample_size_per_variant=numbers["sample_size_per_variant"],
        test_days=numbers["test_days"],
    )

    print(f"STATS (from tool):\n{json.dumps(stats, indent=2)}\n")
    return {"stats": stats}


# ─────────────────────────────────────────────
# NODE 2: summarize
# ─────────────────────────────────────────────

def summarize(state: State) -> dict:
    stats_context = (
        f"Calculated stats: uplift={state['stats']['uplift']}, "
        f"absolute change={state['stats']['absolute_change']}, "
        f"significance={state['stats']['significance']}"
    )
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Summarize this experiment in 2-3 sentences using the provided stats. Be factual and concise."},
            {"role": "user", "content": f"{state['experiment_input']}\n\n{stats_context}"},
        ],
    )
    summary = response.choices[0].message.content
    print(f"SUMMARY:\n{summary}\n")
    return {"summary": summary}


# ─────────────────────────────────────────────
# NODE 3: check_validity
# Now uses real stats instead of guessing.
# ─────────────────────────────────────────────

def check_validity(state: State) -> dict:
    stats = state["stats"]
    stats_block = (
        f"- Uplift: {stats['uplift']}\n"
        f"- Absolute change: {stats['absolute_change']}\n"
        f"- Sample size: {stats['sample_check']}\n"
        f"- Duration: {stats['duration_check']}\n"
        f"- Significance: {stats['significance']}"
    )
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an experiment reviewer. Given a summary and calculated stats, "
                    "decide if the result is safe to interpret. "
                    "Reply with only 'VALID' or 'INVALID'.\n\n"
                    "Mark INVALID if any of these are true:\n"
                    "- Sample size check shows a warning\n"
                    "- Duration check shows a warning\n"
                    "- Result is not statistically significant"
                ),
            },
            {"role": "user", "content": f"Summary: {state['summary']}\n\nStats:\n{stats_block}"},
        ],
    )
    verdict = response.choices[0].message.content.strip()
    is_valid = verdict == "VALID"
    print(f"VALIDITY CHECK: {verdict}\n")
    return {"is_valid": is_valid}


# ─────────────────────────────────────────────
# NODE 4a: identify_causes
# ─────────────────────────────────────────────

def identify_causes(state: State) -> dict:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Given this experiment summary and stats, list the most likely causes of the observed result. Be specific. Max 5 bullet points."},
            {"role": "user", "content": f"Summary: {state['summary']}\nUplift: {state['stats']['uplift']}, Significance: {state['stats']['significance']}"},
        ],
    )
    causes = response.choices[0].message.content
    print(f"CAUSES:\n{causes}\n")
    return {"causes": causes}


# ─────────────────────────────────────────────
# NODE 4b: warn_invalid
# ─────────────────────────────────────────────

def warn_invalid(state: State) -> dict:
    stats = state["stats"]
    stats_block = (
        f"- Uplift: {stats['uplift']}\n"
        f"- Absolute change: {stats['absolute_change']}\n"
        f"- Sample size: {stats['sample_check']}\n"
        f"- Duration: {stats['duration_check']}\n"
        f"- Significance: {stats['significance']}"
    )
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "This experiment has been flagged as unreliable based on calculated stats. "
                    "Write a short, clear warning for the PM explaining why this result should not be "
                    "acted on, referencing the specific stats. Then say what they should do to get a trustworthy result."
                ),
            },
            {"role": "user", "content": f"Summary: {state['summary']}\n\nStats:\n{stats_block}"},
        ],
    )
    warning = response.choices[0].message.content
    print(f"WARNING:\n{warning}\n")
    return {"warning": warning}


# ─────────────────────────────────────────────
# NODE 5: flag_risks
# ─────────────────────────────────────────────

def flag_risks(state: State) -> dict:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Given this experiment summary, causes, and stats, list the key risks and confounders. Max 4 bullet points. Be concise."},
            {"role": "user", "content": f"Summary: {state['summary']}\nCauses: {state['causes']}\nStats: {json.dumps(state['stats'])}"},
        ],
    )
    risks = response.choices[0].message.content
    print(f"RISKS:\n{risks}\n")
    return {"risks": risks}


# ─────────────────────────────────────────────
# ROUTING FUNCTION (same logic as Step 4)
# ─────────────────────────────────────────────

def route_after_validity_check(state: State) -> str:
    return "identify_causes" if state["is_valid"] else "warn_invalid"


# ─────────────────────────────────────────────
# GRAPH
# ─────────────────────────────────────────────

builder = StateGraph(State)

builder.add_node("run_stats", run_stats)
builder.add_node("summarize", summarize)
builder.add_node("check_validity", check_validity)
builder.add_node("identify_causes", identify_causes)
builder.add_node("warn_invalid", warn_invalid)
builder.add_node("flag_risks", flag_risks)

builder.add_edge(START, "run_stats")
builder.add_edge("run_stats", "summarize")
builder.add_edge("summarize", "check_validity")
builder.add_conditional_edges("check_validity", route_after_validity_check)
builder.add_edge("identify_causes", "flag_risks")
builder.add_edge("flag_risks", END)
builder.add_edge("warn_invalid", END)

graph = builder.compile()

# ─────────────────────────────────────────────
# RUN
# ─────────────────────────────────────────────

print("=" * 50)
print("TEST 1: Strong experiment — should pass")
print("=" * 50)
graph.invoke({
    "experiment_input": """
Experiment: Changing the CTA button colour from grey to green on the checkout page.
Results: Conversion rate went from 3.2% to 3.8% over 7 days, 50k users per variant.
Notes: Ran during a holiday sale week. Mobile traffic was 70% of total.
""",
    "stats": {}, "summary": "", "is_valid": False,
    "causes": "", "risks": "", "warning": "",
})

print("=" * 50)
print("TEST 2: Weak experiment — should fail")
print("=" * 50)
graph.invoke({
    "experiment_input": """
Experiment: Changed the font size on the homepage hero from 18px to 20px.
Results: Bounce rate dropped from 62% to 61% over 1 day, 200 users per variant.
Notes: Data collected on a Sunday. No statistical significance test run.
""",
    "stats": {}, "summary": "", "is_valid": False,
    "causes": "", "risks": "", "warning": "",
})
