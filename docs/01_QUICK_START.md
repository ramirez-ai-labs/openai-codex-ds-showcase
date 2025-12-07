# Quick Start Guide

A condensed version for those who want to get running fast. For detailed explanations, see [GETTING_STARTED.md](GETTING_STARTED.md).

## 🚀 Setup (One Time)

```bash
# 1. Navigate to repository
cd openai-codex-ds-showcase

# 2. Create virtual environment
python -m venv .venv

# 3. Activate virtual environment
# Windows:
.venv\Scripts\activate
# Mac/Linux:
source .venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt
```

## 📋 Step-by-Step Exploration

Run these commands **in order** to understand each component:

### Step 1: Generate Data
```bash
python app.py simulate
```
**What it does:** Creates fake developer telemetry data  
**Output:** `developer-telemetry-simulation/telemetry_events.csv`  
**Time:** ~5 seconds

---

### Step 2: Basic Analysis
```bash
python app.py analyze
```
**What it does:** Calculates acceptance rates, builds prediction model  
**Output:** Prints acceptance rates and model performance  
**Time:** ~10 seconds

---

### Step 3: SQL Analysis
```bash
python app.py sql
```
**What it does:** Runs SQL queries on the data  
**Output:** Shows query results (acceptance by segment, language, etc.)  
**Time:** ~5 seconds

---

### Step 4: A/B Testing
```bash
python app.py abtest
```
**What it does:** Statistically compares model v1 vs v2  
**Output:** P-values, effect sizes, recommendations  
**Time:** ~5 seconds

---

### Step 5: Causal Inference
```bash
python app.py causal
```
**What it does:** Estimates true causal effects (not just correlation)  
**Output:** Causal effect estimates with confidence intervals  
**Time:** ~30 seconds

---

### Step 6: Generate Code
```bash
python app.py generate
```
**What it does:** Generates code solutions for coding tasks  
**Output:** `code-evaluation-pipeline/code_solutions/*.py`  
**Time:** ~2 seconds

---

### Step 7: Evaluate Code
```bash
python app.py evaluate
```
**What it does:** Tests if generated code works correctly  
**Output:** `code-evaluation-pipeline/code_eval_results.json`  
**Time:** ~3 seconds

---

### Step 8: Launch Dashboard
```bash
python app.py dashboard
```
**What it does:** Opens interactive web dashboard  
**Output:** Browser opens at http://localhost:8501  
**Time:** Dashboard stays open until you close it

---

## 🎯 Or Run Everything at Once

```bash
python app.py all
```
**What it does:** Runs steps 1-7 automatically  
**Then:** Run `python app.py dashboard` to view results

---

## 📊 What Each Component Does

| Component | Purpose | Key Output |
|-----------|---------|------------|
| `simulate` | Create fake data | CSV with telemetry events |
| `analyze` | Basic statistics | Acceptance rates, model performance |
| `sql` | Database queries | Segmented metrics |
| `abtest` | Statistical comparison | P-values, recommendations |
| `causal` | Causal analysis | True effect estimates |
| `generate` | Create code | Python files with solutions |
| `evaluate` | Test code | Pass/fail results, edit distances |
| `dashboard` | Visualize | Interactive web interface |

---

## 🔍 Understanding the Flow

```
1. SIMULATE DATA
   ↓
2. ANALYZE (basic stats)
   ↓
3. SQL (segmentation)
   ↓
4. A/B TEST (comparison)
   ↓
5. CAUSAL (true effects)
   ↓
6. GENERATE CODE
   ↓
7. EVALUATE CODE
   ↓
8. DASHBOARD (visualize all)
```

---

## 📁 Key Files to Explore

After running the pipeline, check these files:

1. **Data:**
   - `developer-telemetry-simulation/telemetry_events.csv` - The raw data

2. **Results:**
   - `code-evaluation-pipeline/code_eval_results.json` - Code test results
   - `developer-productivity-analysis/telemetry.db` - SQL database

3. **Code:**
   - `code-evaluation-pipeline/code_solutions/*.py` - Generated code

4. **Documentation:**
   - `GETTING_STARTED.md` - Detailed beginner's guide
   - `METHODOLOGY.md` - Deep dive into methods
   - `SHOWCASE_SUMMARY.md` - Role alignment

---

## ❓ Troubleshooting

**Error: "No module named X"**
→ Run: `pip install -r requirements.txt`

**Error: "File not found: telemetry_events.csv"**
→ Run: `python app.py simulate` first

**Dashboard won't open**
→ Make sure port 8501 isn't in use, or it will use the next available port

**Import errors in code evaluation**
→ Make sure you ran `python app.py generate` before `evaluate`

---

## 🎓 Learning Path

**Day 1:** Steps 1-2 (data + basic analysis)  
**Day 2:** Steps 3-4 (SQL + A/B testing)  
**Day 3:** Steps 5-7 (causal + code evaluation)  
**Day 4:** Step 8 (dashboard exploration)  
**Day 5:** Read METHODOLOGY.md and experiment

---

## 💡 Pro Tips

1. **Run commands in order** - Each step depends on previous ones
2. **Check the output** - Each command prints what it's doing
3. **Explore the dashboard** - Use filters to see different views
4. **Read the code** - Start with simple scripts, work up to complex ones
5. **Experiment** - Change parameters and see what happens!

---

For detailed explanations of what each step does and why, see **[GETTING_STARTED.md](GETTING_STARTED.md)**.

