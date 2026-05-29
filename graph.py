"""
graph.py — The experiment analysis graph.

All LangGraph logic lives here. agent.py imports and calls build_graph().
Model choices and settings are read from config.py.
"""

import json
import math
import os
from typing import TypedDict, Optional
from openai import OpenAI
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from config import NODE_MODELS, MIN_SAMPLE_SIZE, MIN_TEST_DAYS

load_dotenv()
client = OpenAI()


# ─────────────────────────────────────────────
# TOOL
# ─────────────────────────────────────────────

def calculate_stats(
    control_rate: float,
    treatment_rate: float,
    sample_size_per_variant: int,
    test_days: int,
) -> dict:
    # Normalise: accept both 3.2 and 0.032 as input
    if control_rate > 1:
        control_rate /= 100
    if treatment_rate > 1:
        treatment_rate /= 100

    uplift_pct = ((treatment_rate - control_rate) / control_rate) * 100
    absolute_change_pp = (treatment_rate - control_rate) * 100

    if sample_size_per_variant < MIN_SAMPLE_SIZE:
        sample_check = f"⚠️  Too small (under {MIN_SAMPLE_SIZE:,}/variant) — results unreliable."
    elif sample_size_per_variant < 5000:
        sample_check = "⚠️  Borderline (under 5,000/variant) — treat with caution."
    else:
        sample_check = f"✓  Adequate ({sample_size_per_variant:,}/variant)."

    if test_days < MIN_TEST_DAYS:
        duration_check = f"⚠️  Too short (under {MIN_TEST_DAYS} days) — won't capture weekly patterns."
    elif test_days < 7:
        duration_check = f"⚠️  Short ({test_days} days) — consider running a full week."
    else:
        duration_check = f"✓  Adequate ({test_days} days)."

    p_pool = (
        (control_rate * sample_size_per_variant + treatment_rate * sample_size_per_variant)
        / (2 * sample_size_per_variant)
    )
    se = math.sqrt(p_pool * (1 - p_pool) * (2 / sample_size_per_variant))

    if se > 0:
        z = (treatment_rate - control_rate) / se
        p_value = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))
        p_str = "p<0.001" if p_value < 0.001 else f"p={p_value:.3f}"
        if p_value < 0.05:
            significance = f"✓  Statistically significant ({p_str})"
        elif p_value < 0.10:
            significance = f"⚠️  Borderline significant ({p_str})"
        else:
            significance = f"✗  Not significant ({p_str})"
    else:
        significance = "Could not calculate significance."

    return {
        "uplift": f"+{uplift_pct:.1f}%",
        "absolute_change": f"+{absolute_change_pp:.2f} pp",
        "sample_check": sample_check,
        "duration_check": duration_check,
        "significance": significance,
    }


# ─────────────────────────────────────────────
# STATE
# ─────────────────────────────────────────────

class State(TypedDict):
    experiment_input: str
    previous_run: Optional[dict]
    stats: dict
    summary: str
    is_valid: bool
    causes: str
    risks: str
    warning: str
    comparison: str


# ─────────────────────────────────────────────
# NODES
# Each node picks its model from NODE_MODELS in config.py
# ─────────────────────────────────────────────

def run_stats(state: State) -> dict:
    extraction = client.chat.completions.create(
        model=NODE_MODELS["run_stats"],
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": (
                    "Extract experiment numbers from the text. "
                    "Return JSON with exactly these keys: "
                    "control_rate (float), treatment_rate (float), "
                    "sample_size_per_variant (int), test_days (int). "
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
    return {"stats": stats}


def summarize(state: State) -> dict:
    stats_context = (
        f"Calculated stats: uplift={state['stats']['uplift']}, "
        f"absolute change={state['stats']['absolute_change']}, "
        f"significance={state['stats']['significance']}"
    )
    response = client.chat.completions.create(
        model=NODE_MODELS["summarize"],
        messages=[
            {"role": "system", "content": "Summarize this experiment in 2-3 sentences using the provided stats. Be factual and concise."},
            {"role": "user", "content": f"{state['experiment_input']}\n\n{stats_context}"},
        ],
    )
    return {"summary": response.choices[0].message.content}


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
        model=NODE_MODELS["check_validity"],
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an experiment reviewer. Reply with only 'VALID' or 'INVALID'.\n"
                    "Mark INVALID if: sample size shows a warning, duration shows a warning, "
                    "or result is not statistically significant."
                ),
            },
            {"role": "user", "content": f"Summary: {state['summary']}\n\nStats:\n{stats_block}"},
        ],
    )
    verdict = response.choices[0].message.content.strip()
    return {"is_valid": verdict == "VALID"}


def identify_causes(state: State) -> dict:
    response = client.chat.completions.create(
        model=NODE_MODELS["identify_causes"],
        messages=[
            {"role": "system", "content": "List the most likely causes of the observed result. Max 4 bullet points."},
            {"role": "user", "content": f"Summary: {state['summary']}\nUplift: {state['stats']['uplift']}, Significance: {state['stats']['significance']}"},
        ],
    )
    return {"causes": response.choices[0].message.content}


def warn_invalid(state: State) -> dict:
    stats = state["stats"]
    stats_block = (
        f"- Uplift: {stats['uplift']}\n"
        f"- Sample size: {stats['sample_check']}\n"
        f"- Duration: {stats['duration_check']}\n"
        f"- Significance: {stats['significance']}"
    )
    response = client.chat.completions.create(
        model=NODE_MODELS["warn_invalid"],
        messages=[
            {
                "role": "system",
                "content": (
                    "This experiment has been flagged as unreliable. "
                    "Write a short warning for the PM referencing the specific stats, "
                    "and say what they should do to get a trustworthy result."
                ),
            },
            {"role": "user", "content": f"Summary: {state['summary']}\n\nStats:\n{stats_block}"},
        ],
    )
    return {"warning": response.choices[0].message.content}


def flag_risks(state: State) -> dict:
    response = client.chat.completions.create(
        model=NODE_MODELS["flag_risks"],
        messages=[
            {"role": "system", "content": "List the key risks and confounders. Max 3 bullet points. Be concise."},
            {"role": "user", "content": f"Summary: {state['summary']}\nCauses: {state['causes']}\nStats: {json.dumps(state['stats'])}"},
        ],
    )
    return {"risks": response.choices[0].message.content}


def compare(state: State) -> dict:
    prev = state.get("previous_run")

    if not prev or not prev.get("summary"):
        return {"comparison": ""}

    prev_stats = prev.get("stats", {})
    current_stats = state["stats"]

    context = f"""
Current experiment:
- Summary: {state['summary']}
- Uplift: {current_stats.get('uplift')}
- Absolute change: {current_stats.get('absolute_change')}
- Significance: {current_stats.get('significance')}

Previous experiment on this thread:
- Summary: {prev.get('summary')}
- Uplift: {prev_stats.get('uplift')}
- Absolute change: {prev_stats.get('absolute_change')}
- Significance: {prev_stats.get('significance')}
"""
    response = client.chat.completions.create(
        model=NODE_MODELS["compare"],
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a PM assistant comparing two A/B test results on the same experiment stream. "
                    "Write 2-3 sentences. Be direct: is this better, worse, or inconclusive vs the last test? "
                    "Reference specific numbers. End with a clear ship / iterate / drop recommendation."
                ),
            },
            {"role": "user", "content": context},
        ],
    )
    return {"comparison": response.choices[0].message.content}


# ─────────────────────────────────────────────
# ROUTING
# ─────────────────────────────────────────────

def route_after_validity_check(state: State) -> str:
    return "identify_causes" if state["is_valid"] else "warn_invalid"


# ─────────────────────────────────────────────
# BUILD
# ─────────────────────────────────────────────

def build_graph(checkpointer):
    """Build and compile the graph with the given checkpointer."""
    builder = StateGraph(State)

    builder.add_node("run_stats", run_stats)
    builder.add_node("summarize", summarize)
    builder.add_node("check_validity", check_validity)
    builder.add_node("identify_causes", identify_causes)
    builder.add_node("warn_invalid", warn_invalid)
    builder.add_node("flag_risks", flag_risks)
    builder.add_node("compare", compare)

    builder.add_edge(START, "run_stats")
    builder.add_edge("run_stats", "summarize")
    builder.add_edge("summarize", "check_validity")
    builder.add_conditional_edges("check_validity", route_after_validity_check)
    builder.add_edge("identify_causes", "flag_risks")
    builder.add_edge("flag_risks", "compare")
    builder.add_edge("warn_invalid", "compare")
    builder.add_edge("compare", END)

    return builder.compile(checkpointer=checkpointer)
