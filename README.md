# OpenAI Codex DS Showcase

A comprehensive showcase repository demonstrating the skills required for the **Data Scientist, Codex** role at OpenAI. This repository includes:

- **Telemetry Analysis**: SQL queries, segmentation, cohort analysis
- **A/B Testing Framework**: Statistical significance testing, power analysis
- **Causal Inference**: Propensity score matching, regression adjustment
- **Code Evaluation Pipeline**: Multi-task evaluation with correctness and quality metrics
- **Interactive Dashboard**: Streamlit dashboard for self-service analytics
- **Productivity Metrics**: Acceptance rates, latency, session productivity, task completion

This demonstrates how a Data Scientist would measure and accelerate product-market fit for AI developer tools.

## Repository Structure
- `developer-telemetry-simulation/` — simulate developer telemetry streams; schema docs and sample output.
- `developer-productivity-analysis/` — notebooks and models for productivity/acceptance analysis.
- `code-evaluation-pipeline/` — tasks, reference solutions, generators, tests, and evaluation reporting.
- `dashboard/` — simple app for viewing evaluation outcomes.

## Getting Started

> **New to this repository?** 
> - **Complete beginner?** Start with **[GETTING_STARTED.md](GETTING_STARTED.md)** for a detailed step-by-step guide
> - **Want to get running fast?** Check **[QUICK_START.md](QUICK_START.md)** for a condensed version

### 1. Setup Environment
```bash
git clone <repo-url>
cd openai-codex-ds-showcase
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

### 2. Run Full Pipeline
```bash
# Run everything end-to-end
python app.py all

# Or run individual components:
python app.py simulate      # Generate synthetic telemetry
python app.py analyze        # Acceptance rate analysis
python app.py sql            # SQL queries on telemetry
python app.py abtest         # A/B testing framework
python app.py causal         # Causal inference analysis
python app.py generate       # Generate code from tasks
python app.py evaluate       # Run correctness tests
python app.py dashboard      # Launch interactive dashboard
```

### 3. Explore Results
- **Dashboard**: `python app.py dashboard` → Opens Streamlit UI
- **SQL Analysis**: `python app.py sql` → See query results
- **A/B Tests**: `python app.py abtest` → Statistical comparisons
- **Causal Analysis**: `python app.py causal` → Causal inference results

## Components
### Developer Telemetry Simulation (`developer-telemetry-simulation/`)
- `simulate_telemetry.py`: generates synthetic telemetry events (open, edit, run, test, commit).
- `telemetry_schema.md`: documents fields and event types.
- `sample_output_head.csv`: example snippet of generated telemetry.

### Developer Productivity Analysis (`developer-productivity-analysis/`)
- `productivity_analysis_template.ipynb`: Starter notebook for exploring acceptance/throughput.
- `acceptance_rate_model.py`: Logistic regression model for acceptance rates.
- `sql_queries.sql`: Comprehensive SQL queries for telemetry analysis (segmentation, cohorts, time-series).
- `run_sql_analysis.py`: Python script to run SQL queries and display results.
- `ab_testing_framework.py`: Complete A/B testing framework with statistical tests, power analysis, and effect sizes.
- `causal_inference.py`: Causal inference methods (propensity score matching, regression adjustment).

### Code Evaluation Pipeline (`code-evaluation-pipeline/`)
- `tasks/` with `tasks.json` (prompts/metadata) and `reference_solutions.py`.
- `generate_code.py`: produce candidate solutions from prompts.
- `run_tests.py`: execute task-specific tests.
- `compute_edit_distance.py`: compute Levenshtein distance to references.
- `evaluation_report.md`: README-style summary of metrics and how to interpret results.

### Dashboard (`dashboard/`)
- `app.py`: Interactive Streamlit dashboard with:
  - Code evaluation results visualization
  - Telemetry analysis (acceptance rates, latency, segmentation)
  - Model version comparison (A/B test results)
  - Failure diagnostics and error analysis
  - Interactive filters and drill-downs

## Key Features

### 📊 Metrics & Analysis
- **Acceptance Rate**: Model version, language, user segment breakdowns
- **Edit Distance**: Levenshtein distance to reference solutions
- **Latency Analysis**: Impact on acceptance, percentiles, distributions
- **Session Productivity**: Task completion rates, session-level metrics
- **Failure Diagnostics**: Error type classification, hallucination detection

### 🔬 Statistical Methods
- **A/B Testing**: Chi-square, t-tests, proportion tests with power analysis
- **Causal Inference**: Propensity score matching, regression adjustment
- **Confidence Intervals**: Bootstrap methods for uncertainty quantification
- **Effect Sizes**: Cohen's h, Cohen's d for practical significance

### 💾 SQL Analysis
- Segmentation queries (by language, user segment, model version)
- Cohort analysis (developer retention, behavior over time)
- Time-series analysis (trends, seasonality)
- Session-level aggregations
- Error and hallucination analysis

### 📈 Dashboard
- Interactive visualizations with Plotly
- Real-time filtering and segmentation
- Model comparison with statistical significance
- Failure mode diagnostics
- Self-service analytics for PM, Eng, Design teams

## Documentation

📖 **See [DOCUMENTATION.md](DOCUMENTATION.md) for a complete guide to all documentation.**

### Quick Links
- **[GETTING_STARTED.md](GETTING_STARTED.md)**: Step-by-step beginner's guide
- **[QUICK_START.md](QUICK_START.md)**: Quick reference
- **[METHODOLOGY.md](METHODOLOGY.md)**: Comprehensive methodology
- **[SHOWCASE_SUMMARY.md](SHOWCASE_SUMMARY.md)**: Role alignment
- **[docs/NLP_ANALYSIS.md](docs/NLP_ANALYSIS.md)**: NLP techniques

## Data/Outputs
- Generated solutions: `code-evaluation-pipeline/code_solutions/`
- Evaluation metrics: `code-evaluation-pipeline/code_eval_results.json`
- Telemetry data: `developer-telemetry-simulation/telemetry_events.csv`
- SQL database: `developer-productivity-analysis/telemetry.db` (created on first run)
- Reports: `code-evaluation-pipeline/evaluation_report.md`

## Skills Demonstrated

This repository showcases the following skills required for the Codex DS role:

✅ **SQL Fluency**: Complex queries for segmentation, cohorts, time-series  
✅ **Python**: Statistical analysis, modeling, data processing  
✅ **Experiment Design**: A/B testing, power analysis, statistical significance  
✅ **Causal Inference**: Propensity score matching, regression adjustment  
✅ **Product Metrics**: Acceptance rates, latency, productivity metrics  
✅ **Code Evaluation**: Correctness, quality, failure mode analysis  
✅ **Dashboard Building**: Streamlit, interactive visualizations  
✅ **Communication**: Clear documentation, actionable insights  

## Troubleshooting

- **Imports fail**: Check that generated code has no top-level side effects and meets task signatures.
- **Tests fail**: Inspect task-specific edge cases; see `reference_solutions.py` for intent.
- **Dashboard empty**: Ensure evaluation outputs exist. Run `python app.py all` first.
- **SQL errors**: Make sure telemetry simulation has run (`python app.py simulate`).
- **Missing dependencies**: Run `pip install -r requirements.txt` to install all packages.

