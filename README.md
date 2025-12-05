# OpenAI Codex Data Scientist Showcase  
*A portfolio demonstrating metrics, evaluation, and analysis for AI-assisted developer tooling*

---

## 🎯 Purpose of This Repository

This repo is designed to showcase the **skills required for the Data Scientist, Codex role at OpenAI**:

- Measuring & evaluating AI-assisted code generation  
- Simulating developer telemetry at scale  
- Understanding developer workflows and productivity metrics  
- Running automated code-evaluation pipelines  
- Designing A/B tests and analyzing model differences  
- Communicating insights clearly through notebooks and dashboards  

The goal of this project is to demonstrate **end-to-end ownership** of the DS problems Codex solves every day.

---

## 📦 Repository Structure
```bash

openai-codex-ds-showcase/
│
├── developer-telemetry-simulation/
│ ├── simulate_telemetry.py
│ ├── telemetry_schema.md
│ └── sample_output.csv
│
├── developer-productivity-analysis/
│ ├── productivity_analysis.ipynb
│ ├── acceptance_rate_model.py
│ └── charts/
│
├── code-evaluation-pipeline/
│ ├── tasks/
│ │ ├── fizzbuzz.py
│ │ ├── palindrome.py
│ │ └── …
│ ├── generate_code.py
│ ├── run_tests.py
│ ├── compute_edit_distance.py
│ └── evaluation_report.md
│
├── dashboard/
│ └── app.py
│
└── README.md <-- (this file)

```


---

## 🧠 Skills Demonstrated (Matched to Codex DS Requirements)

### ✔ 1. Understanding Developer Telemetry  
Codex DS analyzes IDE-level signals such as:

- suggestion acceptance rate  
- edit distance between suggestion and final code  
- compile/run failures  
- keystrokes saved  
- latency  
- time-to-completion  
- fallback requests  
- hallucination/failure mode categories  

The repo includes a **synthetic telemetry generator** to model thousands of “AI coding sessions” with configurable behaviors.

---

### ✔ 2. Productivity & Behavioral Analysis  
Notebook includes:

- Acceptance-rate modeling (logistic regression / XGBoost)
- Latency → satisfaction relationships
- Causal inference: *“Would this developer have been faster without AI?”*
- Developer segmentation via clustering
- Fail-case taxonomy analysis

This mirrors how the Codex DS team measures **developer experience and model improvements**.

---

### ✔ 3. Automated Code Evaluation Pipeline  
Codex is evaluated on:

- test pass rates  
- correctness  
- run-time behavior  
- static analysis results  
- refactor/edit distance  
- error types & categories  
- quality deltas between model versions  

This repository includes an automated pipeline that:

1. Sends coding tasks to the OpenAI API  
2. Executes the returned code in a safe sandbox  
3. Runs unit tests  
4. Computes edit distance & quality metrics  
5. Aggregates results into a single evaluation report  

---

### ✔ 4. Experimental Design & A/B Testing  
The analysis includes:

- model A vs. model B acceptance-rate comparisons  
- latency impact on engagement  
- code-eval pass-rate deltas  
- developer-level random effects  
- bootstrapped confidence intervals  

This demonstrates readiness to own **metric design, experiment analysis, and model evaluation** at OpenAI.

---

## 🚀 How to Run

### 1. Generate telemetry
```bash
cd developer-telemetry-simulation
python simulate_telemetry.py
```

### 2. Run the productivity analysis notebook

Open:
```bash
developer-productivity-analysis/productivity_analysis.ipynb
```

### 3. Run code evaluation pipeline
```bash
cd code-evaluation-pipeline
python generate_code.py
python run_tests.py
```

### 4. Launch the demo dashboard
```bash
cd dashboard
streamlit run app.py
```

---

### 🧪 Example Metrics Included

Suggestion Acceptance Rate

- Edit Distance From Final Code
- Test Pass Rate
- Code Quality Score
- Compilation Success Ratio
- Time-To-Completion
- Latency Buckets (P50, P90, P99)
- Failure Mode Categorization
- Overall “Developer Happiness” score

These metrics demonstrate the DS mindset required to improve model performance, developer trust, and product usability.

--- 

## 🧰 Technologies Used

- Python
- Jupyter / Colab
- Pandas, NumPy, Scikit-Learn, XGBoost
- Matplotlib / Seaborn / Plotly
- Streamlit Dashboard
- subprocess sandbox for code execution
- OpenAI API (optional)
