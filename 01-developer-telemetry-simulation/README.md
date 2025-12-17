# Developer Telemetry Simulation (Beginner Friendly)

This folder simulates **developer telemetry data** for an AI coding assistant
(similar to GitHub Copilot / Codex–style systems).

The goal is NOT to build a real IDE plugin.
The goal is to understand:

- What telemetry data AI coding tools collect
- How product + ML teams reason about acceptance, latency, and errors
- What raw data looks like *before* dashboards and metrics exist

Everything here runs locally, on CPU, and uses synthetic (fake) data.

---

## What Is Telemetry?

**Telemetry** is structured data about how users interact with a system.

For AI coding assistants, telemetry might answer questions like:
- Do developers accept suggestions?
- How long do suggestions take to appear?
- Do accepted suggestions compile and pass tests?
- Are newer models actually better?

This folder gives you a realistic dataset to explore those questions.

---

## Files in This Folder

### `simulate_telemetry.py`
**Purpose:** Generate synthetic telemetry events that *look like* real IDE data.

This script simulates:
- Coding sessions
- AI suggestions
- Developer acceptance or rejection
- Edits, compile results, test results
- Latency and hallucinations
- Model A/B testing (`model_v1` vs `model_v2`)

You run this file to **create the dataset**.

📌 This is the *source of truth* for how the data is generated :contentReference[oaicite:0]{index=0}

---

### `telemetry_events.csv`
**Purpose:** The generated dataset (rows = events, columns = signals).

Each row represents **one AI suggestion shown to a developer**.

This is what a data scientist or PM would analyze using:
- Pandas
- SQL
- Notebooks
- Dashboards

You never edit this file manually — it is produced by `simulate_telemetry.py`.

---

### `telemetry_schema.md`
**Purpose:** Documentation for every column in `telemetry_events.csv`.

This file answers:
- What does each column mean?
- What type is it?
- Why does it exist?

Think of it as the **data contract** between engineering, ML, and analytics teams :contentReference[oaicite:1]{index=1}

---

### `sample_output_head.csv`
**Purpose:** A small preview of the dataset.

This file exists so readers can:
- See real column names immediately
- Understand the shape of the data
- Avoid opening a huge CSV just to inspect it

---

## How `simulate_telemetry.py` Works (High Level)

Think of the script as a **game simulation**.

### 1. Simulate Coding Sessions
Each session represents a developer working in an IDE.

For every session, we randomly assign:
- A developer
- A task
- A programming language
- A model version (v1 or v2)
- A user experience level (beginner → expert)

---

### 2. Simulate AI Suggestions
Inside each session, the script generates multiple AI suggestions.

For every suggestion, it simulates:
- How long the model took to respond (latency)
- How long the suggested code was
- Whether the developer accepted it
- Whether they edited it
- Whether it compiled and passed tests
- Whether it hallucinated nonsense code

Each suggestion becomes **one row** in the CSV.

---

### 3. Encode Product Assumptions
The script intentionally bakes in realistic assumptions:

- `model_v2` is faster and more accurate than `model_v1`
- Beginners accept suggestions more often than experts
- Longer suggestions fail more often
- Faster models improve acceptance indirectly

This allows meaningful analysis later.

---

### 4. Write to CSV
At the end, all simulated events are written to:
```bash
telemetry_events.csv
```

This mirrors how real telemetry pipelines work:
- Raw events → stored → analyzed later

---

## How to Run the Simulation

From this folder:

```bash
python simulate_telemetry.py

Optional arguments:

python simulate_telemetry.py --sessions 1000
python simulate_telemetry.py --sessions 300 --output my_data.csv
```

After running:

- Open telemetry_events.csv
- Explore with pandas, Excel, or notebooks

---

# How to Think About This Data

This dataset lets you ask real product + ML questions, such as:

- Does lower latency improve acceptance?
- Is model_v2 actually better across user segments?
- Do hallucinations correlate with code length?
- Are experts harder to satisfy than beginners?
- What matters more: speed or correctness?

These are the same questions asked by:

- Codex Data Scientists
- Copilot PMs
- Applied ML teams

---

# Why This Folder Exists

This simulation is a bridge between:
- Raw ML models
- Real developer behavior
- Business/product decisions

Understanding telemetry is what separates:

“The model is better”
from
“The product actually improved developer productivity.”

---

# Suggested Next Steps

- Load telemetry_events.csv into a notebook
- Plot acceptance rate vs latency
- Compare model_v1 vs model_v2
- Join this data with edit-distance or RAG quality experiments

This folder is intentionally reusable across many analyses.

---

### Why this README works for your repo

- ✅ Beginner-friendly, no assumed telemetry background  
- ✅ Explains *why* each file exists, not just what it does  
- ✅ Mirrors how real AI platform teams document datasets  
- ✅ Fits perfectly into your **Codex / Developer Experience / Applied ML** narrative  

If you want, next we can:
- Add a **first analysis notebook**
- Write a **RESULTS.md** for this folder
- Or connect this telemetry to your earlier **edit-distance → acceptance** work
