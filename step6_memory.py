"""
Step 6: Memory & persistence (LangGraph checkpointing).

The graph now remembers previous experiments on the same thread.
After each run, the final state is saved by a MemorySaver checkpointer.
On the next run, we load the last completed state from the thread's history
and pass it in so the `compare` node can produce an explicit comparison.

Graph:
  START → run_stats → summarize → check_validity → identify_causes → flag_risks → compare → END
                                        ↘                                               ↑
                                    warn_invalid ──────────────────────────────────────┘

Key concepts introduced:
- MemorySaver       — saves state snapshots after each node
- thread_id         — identifies an experiment stream (same thread = same history)
- get_state_history — lets us retrieve the last completed run's state before invoking
- compare node      — explicitly compares current vs previous experiment
"""

import json
import math
from typing import TypedDict, Optional
from openai import OpenAI
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()
client = OpenAI()


# ─────────────────────────────────────────────
# THE TOOL (unchanged from Step 5)
# ─────────────────────────────────────────────

def calculate_stats(
    control_rate: float,
    treatment_rate: float,
    sample_size_per_variant: int,
    test_days: int,
) -> dict:
    # Normalise: if rates look like percentages (e.g. 3.2) convert to decimals (0.032)
    if control_rate > 1:
        control_rate = control_rate / 100
    if treatment_rate > 1:
        treatment_rate = treatment_rate / 100

    uplift_pct = ((treatment_rate - control_rate) / control_rate) * 100
    absolute_change_pp = (treatment_rate - control_rate) * 100

    if sample_size_per_variant < 1000:
        sample_check = "⚠️ Too small (under 1,000/variant) — results unreliable."
    elif sample_size_per_variant < 5000:
        sample_check = "⚠️ Borderline (under 5,000/variant) — treat with caution."
    else:
        sample_check = f"✓ Adequate ({sample_size_per_variant:,}/variant)."

    if test_days < 3:
        duration_check = "⚠️ Too short (under 3 days) — won't capture weekly patterns."
    elif test_days < 7:
        duration_check = f"⚠️ Short ({test_days} days) — consider running a full week."
    else:
        duration_check = f"✓ Adequate ({test_days} days)."

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
            significance = f"✓ Statistically significant ({p_str})"
        elif p_value < 0.10:
            significance = f"⚠️ Borderline significant ({p_str})"
        else:
            significance = f"✗ Not significant ({p_str})"
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
# Added: previous_run (loaded from checkpointer before invoking)
#        comparison    (written by the compare node)
# ─────────────────────────────────────────────

class State(TypedDict):
    experiment_input: str
    previous_run: Optional[dict]   # last experiment on this thread, if any
    stats: dict
    summary: str
    is_valid: bool
    causes: str
    risks: str
    warning: str
    comparison: str


# ─────────────────────────────────────────────
# NODES (run_stats → summarize → check_validity →
#         identify_causes / warn_invalid → flag_risks → compare)
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
    print(f"STATS:\n  uplift={stats['uplift']}  |  {stats['significance']}\n")
    return {"stats": stats}


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
                    "You are an experiment reviewer. Reply with only 'VALID' or 'INVALID'.\n"
                    "Mark INVALID if: sample size shows a warning, duration shows a warning, "
                    "or result is not statistically significant."
                ),
            },
            {"role": "user", "content": f"Summary: {state['summary']}\n\nStats:\n{stats_block}"},
        ],
    )
    verdict = response.choices[0].message.content.strip()
    is_valid = verdict == "VALID"
    print(f"VALIDITY: {verdict}\n")
    return {"is_valid": is_valid}


def identify_causes(state: State) -> dict:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "List the most likely causes of the observed result. Max 4 bullet points."},
            {"role": "user", "content": f"Summary: {state['summary']}\nUplift: {state['stats']['uplift']}, Significance: {state['stats']['significance']}"},
        ],
    )
    causes = response.choices[0].message.content
    print(f"CAUSES:\n{causes}\n")
    return {"causes": causes}


def warn_invalid(state: State) -> dict:
    stats = state["stats"]
    stats_block = (
        f"- Uplift: {stats['uplift']}\n"
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
                    "This experiment has been flagged as unreliable. "
                    "Write a short warning for the PM referencing the specific stats, "
                    "and say what they should do to get a trustworthy result."
                ),
            },
            {"role": "user", "content": f"Summary: {state['summary']}\n\nStats:\n{stats_block}"},
        ],
    )
    warning = response.choices[0].message.content
    print(f"WARNING:\n{warning}\n")
    return {"warning": warning}


def flag_risks(state: State) -> dict:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "List the key risks and confounders. Max 3 bullet points. Be concise."},
            {"role": "user", "content": f"Summary: {state['summary']}\nCauses: {state['causes']}\nStats: {json.dumps(state['stats'])}"},
        ],
    )
    risks = response.choices[0].message.content
    print(f"RISKS:\n{risks}\n")
    return {"risks": risks}


def compare(state: State) -> dict:
    """
    Compares the current experiment against the previous one on this thread.
    If there is no previous run, skips gracefully.
    """
    prev = state.get("previous_run")

    if not prev or not prev.get("summary"):
        print("COMPARISON: First experiment on this thread — nothing to compare yet.\n")
        return {"comparison": "First experiment on this thread."}

    prev_stats = prev.get("stats", {})
    current_stats = state["stats"]

    context = f"""
Current experiment:
- Summary: {state['summary']}
- Uplift: {current_stats.get('uplift', 'N/A')}
- Absolute change: {current_stats.get('absolute_change', 'N/A')}
- Significance: {current_stats.get('significance', 'N/A')}

Previous experiment on this thread:
- Summary: {prev.get('summary', 'N/A')}
- Uplift: {prev_stats.get('uplift', 'N/A')}
- Absolute change: {prev_stats.get('absolute_change', 'N/A')}
- Significance: {prev_stats.get('significance', 'N/A')}
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
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
    comparison = response.choices[0].message.content
    print(f"COMPARISON VS PREVIOUS:\n{comparison}\n")
    return {"comparison": comparison}


# ─────────────────────────────────────────────
# ROUTING
# ─────────────────────────────────────────────

def route_after_validity_check(state: State) -> str:
    return "identify_causes" if state["is_valid"] else "warn_invalid"


# ─────────────────────────────────────────────
# GRAPH — compiled with MemorySaver
# ─────────────────────────────────────────────

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

memory = MemorySaver()
graph = builder.compile(checkpointer=memory)


# ─────────────────────────────────────────────
# HELPER — wraps invoke() with history lookup
# ─────────────────────────────────────────────

def analyze_experiment(experiment_input: str, thread_id: str = "default") -> dict:
    """
    Run the experiment analysis graph on a given thread.
    Automatically loads the last completed run from this thread's history
    and passes it in as `previous_run` for the compare node.
    """
    config = {"configurable": {"thread_id": thread_id}}

    # Look back through this thread's checkpoint history for the last completed run
    previous_run = None
    for checkpoint in graph.get_state_history(config):
        vals = checkpoint.values
        if vals.get("summary"):  # a completed run always has a summary
            previous_run = {
                "summary": vals.get("summary", ""),
                "stats": vals.get("stats", {}),
                "causes": vals.get("causes", ""),
            }
            break  # only need the most recent

    return graph.invoke(
        {
            "experiment_input": experiment_input,
            "previous_run": previous_run,
            "stats": {},
            "summary": "",
            "is_valid": False,
            "causes": "",
            "risks": "",
            "warning": "",
            "comparison": "",
        },
        config=config,
    )


# ─────────────────────────────────────────────
# DEMO
# Two experiments on the same thread ("checkout-tests").
# First run has no history. Second run compares against the first.
# ─────────────────────────────────────────────

print("=" * 55)
print("RUN 1 — CTA button colour (no history yet)")
print("=" * 55)
analyze_experiment(
    experiment_input="""
Experiment: Changed CTA button colour from grey to green on checkout page.
Results: Conversion rate went from 3.2% to 3.8% over 7 days, 50k users per variant.
Notes: Ran during a holiday sale week. Mobile traffic was 70% of total.
""",
    thread_id="checkout-tests",
)

print("\n" + "=" * 55)
print("RUN 2 — CTA button text (should compare vs Run 1)")
print("=" * 55)
analyze_experiment(
    experiment_input="""
Experiment: Changed CTA button text from "Buy Now" to "Complete Order" on checkout page.
Results: Conversion rate went from 3.8% to 4.1% over 14 days, 80k users per variant.
Notes: Standard traffic period, no promotions running. Even split across mobile and desktop.
""",
    thread_id="checkout-tests",
)
