# main.py
# Fix 5: Manual orchestration with sleep between tasks — prevents TPM bursts
# Architecture: Search Tool → Python Risk Logic → Single Advisor LLM Call

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from crewai import Crew, Process
from tasks import create_advisory_task, python_risk_analysis, python_search
from agents import advisor
from dotenv import load_dotenv
import os
import time

load_dotenv()

# Create output folder if it doesn't exist
os.makedirs("output", exist_ok=True)

app = FastAPI(
    title="Sagefarm AI Financial Advisor",
    description="Agentic AI — CrewAI + Groq + Custom Search",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request model ──────────────────────────────────────────────────────────
class ClientProfile(BaseModel):
    name: str
    age: int
    dependents: int
    job_stability: str
    income: float
    savings: float
    emergency_fund: str
    existing_investments: str
    goal: str
    timeline: int
    risk_profile: str


# ── Health check ───────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "status": "🌿 Sagefarm AI Agent v2.0 is running",
        "architecture": "Search Tool → Python Risk Analysis → Single Advisor LLM",
        "endpoints": {
            "POST /analyse": "Submit client profile → get investment plan",
            "GET /ui": "Open the UI",
        }
    }

@app.get("/health")
def health():
    return {"status": "ok"}


# ── Main agent endpoint ────────────────────────────────────────────────────
@app.post("/analyse")
async def analyse_client(profile: ClientProfile):
    try:
        client_data = profile.model_dump()
        print(f"\n🌿 Request: {client_data['name']} | {client_data['goal']} | {client_data['risk_profile']}")

        # ── Step 1: Python risk analysis (no LLM — instant) ───────────────
        print("📊 Step 1: Running Python risk analysis...")
        risk_analysis = python_risk_analysis(client_data)
        print("✅ Risk analysis done")

        # ── Step 2: Python search (no LLM — instant) ────────────────────────
        print("🔍 Step 2: Running Python search via Serper...")
        research_output = python_search(client_data["risk_profile"], client_data["goal"])
        print("✅ Search done")

        # ── Step 3: Advisory task (only LLM call in pipeline) ─────────────
        print("📝 Step 3: Advisor agent writing plan...")
        advisory_task = create_advisory_task(client_data, risk_analysis, research_output)

        advisory_crew = Crew(
            agents=[advisor],
            tasks=[advisory_task],
            process=Process.sequential,
            verbose=False,
        )
        advisory_crew.kickoff()
        print("✅ Plan generated")

        # Read saved plan
        plan_text = "Plan generated successfully."
        if os.path.exists("output/investment_plan.md"):
            with open("output/investment_plan.md", "r", encoding="utf-8") as f:
                plan_text = f.read()

        return {
            "status": "success",
            "client": client_data["name"],
            "goal": client_data["goal"],
            "risk_profile": client_data["risk_profile"],
            "risk_analysis": risk_analysis,
            "investment_plan": plan_text,
        }

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ── UI ─────────────────────────────────────────────────────────────────────
@app.get("/ui", response_class=HTMLResponse)
def ui():
    return """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>Sagefarm AI Advisor</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: 'Segoe UI', sans-serif; background: #F5F7F7; color: #1a1a1a; }
    header { background: #00897B; color: white; padding: 14px 24px; display: flex; align-items: center; gap: 14px; }
    header img { height: 36px; filter: brightness(0) invert(1); }
    header h1 { font-size: 18px; font-weight: 700; }
    header p  { font-size: 12px; opacity: 0.85; margin-top: 2px; }
    .container { max-width: 860px; margin: 28px auto; padding: 0 16px; }
    .card { background: white; border-radius: 12px; padding: 24px; box-shadow: 0 2px 12px rgba(0,0,0,0.07); margin-bottom: 24px; }
    h2 { color: #00897B; font-size: 15px; margin-bottom: 16px; border-bottom: 2px solid #E0F2F1; padding-bottom: 8px; }
    .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }
    label { font-size: 12px; font-weight: 600; color: #555; display: block; margin-bottom: 4px; }
    input, select { width: 100%; padding: 10px 14px; border: 1.5px solid #e0e0e0; border-radius: 8px; font-size: 14px; background: #F5F7F7; outline: none; font-family: inherit; }
    input:focus, select:focus { border-color: #00897B; box-shadow: 0 0 0 3px rgba(0,137,123,0.1); }
    .full { grid-column: 1 / -1; }
    button { width: 100%; padding: 13px; background: #00897B; color: white; border: none; border-radius: 8px; font-size: 15px; font-weight: 600; cursor: pointer; margin-top: 8px; font-family: inherit; transition: background 0.2s; }
    button:hover:not(:disabled) { background: #00695C; }
    button:disabled { opacity: 0.6; cursor: not-allowed; }
    .tabs { display: flex; gap: 8px; margin-bottom: 16px; }
    .tab { padding: 7px 16px; border-radius: 20px; border: 1.5px solid #00897B; background: white; color: #00897B; font-size: 13px; font-weight: 600; cursor: pointer; transition: all 0.2s; font-family: inherit; }
    .tab.active { background: #00897B; color: white; }
    #result, #riskResult { white-space: pre-wrap; font-size: 14px; line-height: 1.75; color: #1a1a1a; display: none; }
    .loading { text-align: center; color: #00897B; padding: 24px; font-size: 15px; }
    .dot { display: inline-block; animation: blink 1.4s infinite; }
    .dot:nth-child(2) { animation-delay: 0.2s; }
    .dot:nth-child(3) { animation-delay: 0.4s; }
    @keyframes blink { 0%,80%,100%{opacity:0.2} 40%{opacity:1} }
    .badge { display: inline-block; background: #E0F2F1; color: #00897B; padding: 3px 10px; border-radius: 12px; font-size: 12px; font-weight: 600; margin-bottom: 12px; }
    .steps { display: flex; gap: 8px; margin-bottom: 16px; flex-wrap: wrap; }
    .step { padding: 6px 12px; border-radius: 6px; font-size: 12px; font-weight: 600; }
    .step.done  { background: #E0F2F1; color: #00897B; }
    .step.active { background: #00897B; color: white; animation: pulse 1s infinite; }
    .step.pending { background: #f5f5f5; color: #aaa; }
    @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.6} }
    .disclaimer { font-size: 11px; color: #888; text-align: center; margin-top: 10px; }
  </style>
</head>
<body>
  <header>
    <img src="https://sagefarm.in/wp-content/uploads/2025/02/cropped-Sagefarm-FINAL-TM-Garet-Bold-Garet-light-300dpi.png" alt="Sagefarm" onerror="this.style.display='none'"/>
    <div>
      <h1>Sagefarm AI Wealth Advisor</h1>
      <p>Agentic AI — CrewAI + Groq Llama 3.1 + Live Research</p>
    </div>
  </header>

  <div class="container">
    <div class="card">
      <h2>Client Profile</h2>
      <div class="grid">
        <div><label>Full Name</label><input id="name" placeholder="Rahul Sharma"/></div>
        <div><label>Age</label><input id="age" type="number" placeholder="30"/></div>
        <div><label>Monthly Income (₹)</label><input id="income" type="number" placeholder="75000"/></div>
        <div><label>Monthly Savings (₹)</label><input id="savings" type="number" placeholder="15000"/></div>
        <div><label>Number of Dependents</label><input id="dependents" type="number" placeholder="2"/></div>
        <div>
          <label>Job Stability</label>
          <select id="job_stability">
            <option value="stable">Stable (Salaried / Govt)</option>
            <option value="moderate">Moderate (Private job)</option>
            <option value="variable">Variable (Freelance / Business)</option>
          </select>
        </div>
        <div>
          <label>Emergency Fund</label>
          <select id="emergency_fund">
            <option value="yes">Yes — 6 months saved</option>
            <option value="partial">Partial — some savings</option>
            <option value="no">No emergency fund</option>
          </select>
        </div>
        <div>
          <label>Existing Investments</label>
          <select id="existing_investments">
            <option value="yes">Yes — SIPs / FDs / Stocks etc.</option>
            <option value="no">No — Starting fresh</option>
          </select>
        </div>
        <div>
          <label>Risk Profile</label>
          <select id="risk_profile">
            <option value="Conservative">🟢 Conservative</option>
            <option value="Moderate" selected>🟡 Moderate</option>
            <option value="Aggressive">🔴 Aggressive</option>
          </select>
        </div>
        <div><label>Timeline (years)</label><input id="timeline" type="number" placeholder="10"/></div>
        <div class="full">
          <label>Financial Goal</label>
          <input id="goal" placeholder="e.g. Retirement, Wealth creation, Child's education"/>
        </div>
      </div>
      <button id="btn" onclick="analyse()">🚀 Generate Investment Plan</button>
      <p class="disclaimer">Sagefarm • AMFI Registered MFD • ARN–318120 • For guidance only, not financial advice</p>
    </div>

    <div class="card" id="resultCard" style="display:none">
      <h2>Results</h2>
      <div id="badge" class="badge"></div>

      <!-- Progress steps -->
      <div class="steps" id="steps" style="display:none">
        <div class="step pending" id="s1">① Risk Analysis</div>
        <div class="step pending" id="s2">② Research</div>
        <div class="step pending" id="s3">③ Investment Plan</div>
      </div>

      <!-- Tabs -->
      <div class="tabs" id="tabs" style="display:none">
        <button class="tab active" onclick="showTab('plan')">📋 Investment Plan</button>
        <button class="tab" onclick="showTab('risk')">📊 Risk Analysis</button>
      </div>

      <div id="loadingDiv" class="loading" style="display:none">
        <span class="dot">.</span><span class="dot">.</span><span class="dot">.</span>
      </div>
      <div id="result"></div>
      <div id="riskResult"></div>
    </div>
  </div>

  <script>
    function showTab(tab) {
      document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
      document.getElementById('result').style.display    = tab === 'plan' ? 'block' : 'none';
      document.getElementById('riskResult').style.display = tab === 'risk' ? 'block' : 'none';
      event.target.classList.add('active');
    }

    function setStep(step) {
      ['s1','s2','s3'].forEach((id, i) => {
        const el = document.getElementById(id);
        if (i + 1 < step)  { el.className = 'step done'; }
        if (i + 1 === step) { el.className = 'step active'; }
        if (i + 1 > step)  { el.className = 'step pending'; }
      });
    }

    async function analyse() {
      const btn = document.getElementById('btn');
      const resultCard = document.getElementById('resultCard');
      const result = document.getElementById('result');
      const riskResult = document.getElementById('riskResult');
      const badge = document.getElementById('badge');
      const steps = document.getElementById('steps');
      const tabs = document.getElementById('tabs');
      const loadingDiv = document.getElementById('loadingDiv');

      const profile = {
        name:                 document.getElementById('name').value,
        age:                  parseInt(document.getElementById('age').value),
        income:               parseFloat(document.getElementById('income').value),
        savings:              parseFloat(document.getElementById('savings').value),
        dependents:           parseInt(document.getElementById('dependents').value),
        job_stability:        document.getElementById('job_stability').value,
        emergency_fund:       document.getElementById('emergency_fund').value,
        existing_investments: document.getElementById('existing_investments').value,
        risk_profile:         document.getElementById('risk_profile').value,
        timeline:             parseInt(document.getElementById('timeline').value),
        goal:                 document.getElementById('goal').value,
      };

      if (!profile.name || !profile.age || !profile.income || !profile.goal) {
        alert('Please fill in all required fields.');
        return;
      }

      btn.disabled = true;
      btn.textContent = '⏳ Agents working...';
      resultCard.style.display = 'block';
      steps.style.display = 'flex';
      tabs.style.display = 'none';
      loadingDiv.style.display = 'block';
      result.style.display = 'none';
      riskResult.style.display = 'none';

      setStep(1);
      badge.textContent = `${profile.risk_profile} Risk  •  ${profile.goal}  •  ${profile.timeline} years`;

      // Simulate step progress in UI
      setTimeout(() => setStep(2), 2000);
      setTimeout(() => setStep(3), 12000);

      try {
        const res = await fetch('/analyse', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(profile),
        });
        const data = await res.json();

        if (data.status === 'success') {
          ['s1','s2','s3'].forEach(id => document.getElementById(id).className = 'step done');
          loadingDiv.style.display = 'none';
          tabs.style.display = 'flex';
          result.style.display = 'block';
          result.textContent = data.investment_plan;
          riskResult.textContent = data.risk_analysis;
        } else {
          loadingDiv.style.display = 'none';
          result.style.display = 'block';
          result.textContent = 'Something went wrong. Check terminal.';
        }
      } catch (err) {
        loadingDiv.style.display = 'none';
        result.style.display = 'block';
        result.textContent = 'Error: ' + err.message;
      } finally {
        btn.disabled = false;
        btn.textContent = '🚀 Generate Investment Plan';
      }
    }
  </script>
</body>
</html>
"""