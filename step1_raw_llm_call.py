"""
Step 1: Raw OpenAI call — no LangGraph yet.

Paste experiment results into `experiment_input` below, run the script,
and the model returns a structured analysis.
"""

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()  # reads OPENAI_API_KEY from .env

client = OpenAI()

experiment_input = """
Experiment: Changing the CTA button colour from grey to green on the checkout page.
Results: Conversion rate went from 3.2% to 3.8% over 7 days, 50k users per variant.
Notes: Ran during a holiday sale week. Mobile traffic was 70% of total.
"""

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
        {"role": "user", "content": experiment_input},
    ],
)

print(response.choices[0].message.content)
