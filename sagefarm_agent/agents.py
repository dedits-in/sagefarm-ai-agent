# agents.py
# Search is now done in Python (tasks.py) — no LLM tool calling at all
# Only ONE agent with ONE LLM call — the Advisor

from crewai import Agent, LLM
from dotenv import load_dotenv
import os

load_dotenv()

# ── LLM Setup ─────────────────────────────────────────────────────────────
llm = LLM(
    model="groq/llama-3.1-8b-instant",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.4,
    max_tokens=600,
)

# ── Single Advisor Agent (no tools — search done in Python) ───────────────
advisor = Agent(
    role="Senior Wealth Advisor",
    goal=(
        "Write a concise, personalised investment plan using the data provided. "
        "Use clear sections. Under 280 words. Warm, jargon-free tone."
    ),
    backstory=(
        "You are Sagefarm's senior wealth advisor — AMFI Registered, 30+ years experience. "
        "You follow Sagefarm's philosophy: long-term, conservative, multi-asset investing. "
        "You write clear, actionable plans that clients can understand and act on immediately."
    ),
    tools=[],          # No tools — no tool call errors
    llm=llm,
    verbose=False,
    allow_delegation=False,
    max_iter=1,
)