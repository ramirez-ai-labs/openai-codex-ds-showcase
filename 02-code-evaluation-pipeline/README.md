# Code Evaluation Pipeline (Beginner Friendly)

This folder contains a **small, end-to-end code evaluation pipeline** that mimics how
AI-assisted coding systems are measured in practice.

The goal is **not** to build a perfect benchmark.
The goal is to understand:

- How code suggestions are generated
- How correctness is evaluated
- How *edit distance* relates to acceptance and quality
- How telemetry and evaluation data flow through a system

This mirrors real workflows used by teams building tools like Copilot, Codex, and IDE agents.

---

## High-Level Flow (Read This First)

Think of the pipeline as four stages:

1. **Generate code** for programming tasks  
2. **Run tests** to check correctness  
3. **Measure similarity** to reference solutions  
4. **Analyze results** and developer-style telemetry  

Each Python file handles *one* responsibility.

---

## Files in This Folder

### 1. `generate_code.py` — *Code Generation*

**What it does:**
- Loads a list of programming tasks
- Asks a model (or a mock fallback) to generate Python solutions
- Saves each solution to disk

**Why it exists:**
This simulates the *“AI suggestion”* step in coding assistants.

In real systems, this would call an LLM API.
Here, it’s intentionally simple so the rest of the pipeline is easy to understand.

**Output:**

```bash
code_solutions/
├── fizzbuzz.py
├── binary_search.py
├── validate_email.py
└── ...
```

---

### 2. `run_tests.py` — *Correctness Evaluation*

**What it does:**
- Dynamically imports each generated solution
- Runs task-specific unit tests
- Records whether the solution passed or failed
- Computes edit distance vs. a reference solution

**Why it exists:**
Correctness is the **minimum bar** for acceptance.
This mirrors how real systems run sandboxed tests or CI checks.

**Key concepts introduced:**
- Dynamic imports
- Programmatic testing
- Failure modes (import errors, runtime errors, failed tests)

**Output:**
- Writes a structured results file:  
  `code_eval_results.json`

---

### 3. `compute_edit_distance.py` — *Similarity Metric*

**What it does:**
- Implements classic edit distance (Levenshtein distance)
- Measures how many character edits separate two code snippets

**Why it exists:**
Research shows that **closer suggestions are accepted more often**.
Edit distance is a simple proxy for developer effort.

Lower distance → less editing → higher acceptance.

This file isolates the metric so it’s easy to swap out later.

---

### 4. `code_eval_results.json` — *Raw Evaluation Output*

**What it contains:**
A JSON list with one entry per task, including:
- Pass/fail status
- Error information (if any)
- Edit distance to reference code

**Why this matters:**
This is *machine-readable ground truth*.
Everything downstream (reports, plots, analysis) comes from this file.

---

### 5. `evaluation_report.md` — *Human-Readable Summary*

**What it does:**
- Interprets `code_eval_results.json`
- Explains what passed, what failed, and why
- Connects metrics to real developer experience

**Why this exists:**
Raw numbers don’t tell a story.
This file translates metrics into **insight**.

---

### 6. `telemetry_schema.md` — *What We Measure*

**What it defines:**
- What events are logged (accept, edit, retry, reject)
- What fields exist (latency, retries, edit distance, outcome)
- How telemetry mirrors real IDE instrumentation

**Why this matters:**
Modern AI tools are driven by telemetry.
This schema shows *what teams actually track*.

---

### 7. `simulate_telemetry.py` — *Synthetic Developer Behavior*

**What it does:**
- Generates fake developer interaction events
- Simulates acceptance vs rejection based on quality
- Writes realistic telemetry logs

**Why it exists:**
We don’t have real users here.
This lets us study **system behavior at scale** without private data.

---

### 8. `telemetry_events.csv` — *Simulated Usage Data*

**What it contains:**
- One row per simulated interaction
- Signals like acceptance, retries, latency, edits

This is what analysts and ML researchers would explore.

---

### 9. `sample_output_head.csv` — *Quick Preview*

**What it is:**
A small snapshot of telemetry output for quick inspection.

Useful for:
- Sanity checks
- Documentation
- Blog posts / explanations

---

## How to Run the Pipeline

From this folder, run in order:

```bash
python generate_code.py
python run_tests.py
```
Optional analysis / exploration:

- Open evaluation_report.md
- Inspect code_eval_results.json
- Explore telemetry with simulate_telemetry.py

---

# What You Should Learn From This

By the end of this folder, you should understand:

- Why correctness alone is not enough
- Why closeness matters as much as accuracy
- How acceptance emerges from many small signals
- How real AI coding tools evaluate themselves

This is the bridge between models and real product impact.

---

# Where This Fits in the Repo

This section connects earlier work on:

- Inference latency
- Agent retries
- Tool costs

to later work on:

- Research reproduction
- Developer productivity
- System-level evaluation

If inference is how fast a model runs,
this folder is about whether it’s actually useful.

---

# Next Logical Extension

- Replace edit distance with semantic similarity
- Add latency-weighted acceptance modeling
- Compare multiple models side-by-side
- Connect telemetry to agent retry policies

---