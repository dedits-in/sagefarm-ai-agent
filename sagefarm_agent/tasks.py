# tasks.py
# Search happens in Python BEFORE the LLM call — no tool calling issues
# Architecture: Python Search → Python Risk → Single Advisor LLM

from crewai import Task
from agents import advisor
import os
import requests


# ── Hardcoded tax rules ────────────────────────────────────────────────────
TAX_RULES = (
    "Tax (India 2025): LTCG equity >1yr: 12.5% above ₹1.25L. "
    "STCG equity <1yr: 20%. Debt funds taxed as per income slab."
)


# ── Python search (no LLM tool call) ──────────────────────────────────────
def python_search(risk_profile: str, goal: str) -> str:
    """Calls Serper directly in Python — no LLM tool calling involved."""
    try:
        url = "https://google.serper.dev/search"
        query = f"Best mutual fund categories {risk_profile.lower()} risk India 2025"
        payload = {"q": query, "num": 2}
        headers = {
            "X-API-KEY": os.getenv("SERPER_API_KEY"),
            "Content-Type": "application/json"
        }
        response = requests.post(url, json=payload, headers=headers, timeout=8)
        data = response.json()
        organic = data.get("organic", [])

        results = []
        for item in organic[:2]:
            title   = item.get("title", "")[:60]
            snippet = item.get("snippet", "")[:100]
            results.append(f"- {title}: {snippet}")

        return "\n".join(results)[:350]

    except Exception as e:
        # Fallback if search fails — use hardcoded knowledge
        fallback = {
            "Aggressive":    "- Flexi cap and mid cap funds ideal for aggressive investors\n- Small cap funds for long-term aggressive growth",
            "Moderate":      "- Large cap and hybrid funds for moderate risk investors\n- Balanced advantage funds for steady returns",
            "Conservative":  "- Debt funds and liquid funds for conservative investors\n- Large cap index funds with low volatility",
        }
        return fallback.get(risk_profile, fallback["Moderate"])


# ── Python risk analysis ───────────────────────────────────────────────────
def python_risk_analysis(client: dict) -> str:
    income     = float(client["income"])
    savings    = float(client["savings"])
    age        = int(client["age"])
    dependents = int(client["dependents"])
    timeline   = int(client["timeline"])
    risk       = client["risk_profile"]
    emergency  = client["emergency_fund"]
    existing   = client["existing_investments"]
    goal       = client["goal"]

    disposable = income - savings
    sip_low    = round((disposable * 0.15) / 500) * 500
    sip_high   = round((disposable * 0.25) / 500) * 500
    sip_low    = max(sip_low, 500)
    sip_high   = max(sip_high, 1000)
    saving_pct = round((savings / income) * 100) if income > 0 else 0

    if risk == "Aggressive":
        allocation = "75% Equity | 15% Debt | 10% Gold"
    elif risk == "Moderate":
        allocation = "60% Equity | 30% Debt | 10% Gold"
    else:
        allocation = "40% Equity | 45% Debt | 15% Gold"

    flags = []
    if saving_pct < 10:
        flags.append(f"Low savings rate ({saving_pct}%) — possible EMI burden")
    if timeline < 3:
        flags.append("Short timeline (<3 yrs) — avoid equity funds")
    if risk == "Aggressive" and timeline < 5:
        flags.append("Risk/timeline mismatch — aggressive profile needs 5+ yr horizon")
    if "retire" in goal.lower() and age > 50:
        flags.append("Retirement after 50 — consider lump sum + SIP combo")
    if emergency == "no":
        flags.append("No emergency fund — build 3-month buffer first")

    flags_text = "\n".join(f"⚠️ {f}" for f in flags) if flags else "✅ No major red flags"

    return (
        f"Risk Profile: {risk}\n"
        f"Monthly Disposable: ₹{disposable:,.0f}\n"
        f"Saving Rate: {saving_pct}%\n"
        f"Recommended SIP: ₹{sip_low:,} – ₹{sip_high:,}/month\n"
        f"Asset Allocation: {allocation}\n"
        f"Emergency Fund: {emergency}\n"
        f"Existing Investments: {existing}\n"
        f"Red Flags:\n{flags_text}\n"
        f"{TAX_RULES}"
    )


# ── Single advisory task (only LLM call in the whole pipeline) ────────────
def create_advisory_task(client_profile: dict, risk_analysis: str, research_output: str) -> Task:
    return Task(
        description=(
            f"Write a personalised investment plan for {client_profile['name']}. Age: {client_profile['age']}. Goal: {client_profile['goal']}. Timeline: {client_profile['timeline']} years.\n\n"
            f"RISK DATA:\n{risk_analysis}\n\n"
            f"MARKET RESEARCH:\n{research_output}\n\n"
            f"Write exactly 5 sections (under 280 words):\n"
            f"1. Summary (2 lines)\n"
            f"2. SIP Recommendation: amount range + fund categories\n"
            f"3. Step-up SIP: start low, increase 10% yearly, show benefit\n"
            f"4. What to avoid\n"
            f"5. Next steps — visit sagefarm.in\n\n"
            f"Tone: warm, jargon-free, like a trusted advisor."
        ),
        expected_output="A 5-section investment plan under 280 words in Sagefarm brand voice.",
        agent=advisor,
        output_file="output/investment_plan.md",
    )