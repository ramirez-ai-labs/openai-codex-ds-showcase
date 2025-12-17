# Developer Productivity Analysis

Analyzes AI coding telemetry to answer: Do developers accept suggestions? How much editing effort is required? Does latency or quality matter more? These scripts mirror how AI developer experience teams turn raw telemetry into product decisions.

## Prerequisites
- Telemetry CSV from the simulation step (`../01-developer-telemetry-simulation/telemetry_events.csv`).
- Python 3 with repo dependencies installed (`pip install -r requirements.txt`).
- Note: scripts currently look for `../developer-telemetry-simulation/telemetry_events.csv`. If you’ve renamed the folder, either copy the CSV there or update the default path in each script.

## Quick Start (common runs)
From the repo root:
```bash
python 03-developer-productivity-analysis/acceptance_rate_model.py
python 03-developer-productivity-analysis/ab_testing_framework.py
python 03-developer-productivity-analysis/run_sql_analysis.py
python 03-developer-productivity-analysis/nlp_analysis.py
```
Open the notebook if you prefer an interactive walk-through:
- `03-developer-productivity-analysis/productivity_analysis_template.ipynb`

## What’s Inside
- `acceptance_rate_model.py` — trains a logistic regression to predict suggestion acceptance; prints AUC, classification report, and top features.
- `ab_testing_framework.py` — runs statistical tests (proportions, power, CIs) to compare model versions and emits rollout guidance.
- `causal_inference.py` — explores causal impacts of latency/quality on acceptance using matching and regression.
- `run_sql_analysis.py` + `sql_queries.sql` — loads telemetry into SQLite and runs segmentation/cohort SQL queries; produces `telemetry.db` if missing.
- `nlp_analysis.py` — basic text analytics on suggestion content (keywords, similarity) with results saved to `nlp_analysis_results.csv`.
- `telemetry.db` — SQLite database generated from telemetry for SQL analyses.

## Suggested Analyses
- Acceptance vs. latency and by user segment (beginner vs. expert).
- Model v1 vs. v2 uplift via A/B tests and power checks.
- Edit distance or similarity signals vs. acceptance to link quality and effort.
- Cohort trends over time using the SQL queries.
