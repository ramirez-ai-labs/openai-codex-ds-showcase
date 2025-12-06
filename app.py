"""
Codex DS Showcase — Unified App Launcher
==========================================

This file provides a simple command-line interface to run the major
components of the OpenAI Codex Developer Tools Showcase repo.

What this app demonstrates:
---------------------------
The Data Scientist, Codex role requires:
- Understanding developer telemetry
- Running model evaluations
- Measuring code-generation quality
- Producing dashboards + analyses
- Running experiments end-to-end
- SQL fluency for data analysis
- A/B testing and causal inference

This launcher helps reviewers understand how the pieces connect.

Commands:
---------
python app.py simulate      → Generate synthetic IDE telemetry
python app.py analyze       → Run acceptance-rate + productivity analysis
python app.py sql           → Run SQL analysis queries
python app.py abtest        → Run A/B testing framework
python app.py causal        → Run causal inference analysis
python app.py generate      → Generate code from tasks using a model
python app.py evaluate      → Run correctness tests + edit distance scoring
python app.py dashboard     → Launch Streamlit dashboard
python app.py all           → Run full end-to-end pipeline

Each command corresponds to a subfolder in the repo.

Directory Structure Recap:
--------------------------
openai-codex-ds-showcase/
│
├── developer-telemetry-simulation/
├── developer-productivity-analysis/
├── code-evaluation-pipeline/
└── dashboard/

This app connects all of those parts.
"""

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).parent


def run(cmd: list, cwd: Path = None):
    """Utility: Run a child process with nice printing."""
    print(f"\n🚀 Running: {' '.join(cmd)}")
    subprocess.run(cmd, cwd=cwd or ROOT, check=True)
    print("✅ Done.\n")


def simulate():
    """Run synthetic telemetry generator."""
    sim_file = ROOT / "developer-telemetry-simulation" / "simulate_telemetry.py"
    run([sys.executable, str(sim_file)])


def analyze():
    """Run productivity + acceptance-rate analysis."""
    analysis_file = ROOT / "developer-productivity-analysis" / "acceptance_rate_model.py"
    run([sys.executable, str(analysis_file)])


def sql_analysis():
    """Run SQL analysis queries."""
    sql_file = ROOT / "developer-productivity-analysis" / "run_sql_analysis.py"
    run([sys.executable, str(sql_file)])


def ab_test():
    """Run A/B testing framework."""
    ab_file = ROOT / "developer-productivity-analysis" / "ab_testing_framework.py"
    run([sys.executable, str(ab_file)])


def causal_inference():
    """Run causal inference analysis."""
    causal_file = ROOT / "developer-productivity-analysis" / "causal_inference.py"
    run([sys.executable, str(causal_file)])


def nlp_analysis():
    """Run NLP analysis on code generation."""
    nlp_file = ROOT / "developer-productivity-analysis" / "nlp_analysis.py"
    run([sys.executable, str(nlp_file)])


def generate_code():
    """Generate model-written code for all tasks."""
    gen_file = ROOT / "code-evaluation-pipeline" / "generate_code.py"
    run([sys.executable, str(gen_file)], cwd=ROOT / "code-evaluation-pipeline")


def evaluate():
    """Run correctness tests + compute edit distance."""
    test_file = ROOT / "code-evaluation-pipeline" / "run_tests.py"
    run([sys.executable, str(test_file)], cwd=ROOT / "code-evaluation-pipeline")


def launch_dashboard():
    """Launch Streamlit dashboard."""
    dash_file = ROOT / "dashboard" / "app.py"
    run(["streamlit", "run", str(dash_file)], cwd=ROOT / "dashboard")


def run_all():
    """Full end-to-end pipeline (Codex-style)."""
    print("\n" + "="*60)
    print("Running Full Codex DS Showcase Pipeline")
    print("="*60)
    
    simulate()
    analyze()
    sql_analysis()
    ab_test()
    causal_inference()
    generate_code()
    evaluate()
    nlp_analysis()
    
    print("\n" + "="*60)
    print("🎉 All tasks completed!")
    print("="*60)
    print("\nNext steps:")
    print("   python app.py dashboard  → Launch interactive dashboard")
    print("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Codex Developer Tools Showcase App Launcher")
    parser.add_argument(
        "command",
        choices=[
            "simulate",
            "analyze",
            "sql",
            "abtest",
            "causal",
            "nlp",
            "generate",
            "evaluate",
            "dashboard",
            "all",
        ],
        help="Which component to run",
    )

    args = parser.parse_args()

    if args.command == "simulate":
        simulate()
    elif args.command == "analyze":
        analyze()
    elif args.command == "sql":
        sql_analysis()
    elif args.command == "abtest":
        ab_test()
    elif args.command == "causal":
        causal_inference()
    elif args.command == "nlp":
        nlp_analysis()
    elif args.command == "generate":
        generate_code()
    elif args.command == "evaluate":
        evaluate()
    elif args.command == "dashboard":
        launch_dashboard()
    elif args.command == "all":
        run_all()
    else:
        print("Unknown command")


if __name__ == "__main__":
    main()

