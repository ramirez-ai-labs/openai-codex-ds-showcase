"""
Codex DS Showcase — Streamlit Dashboard
========================================

Interactive dashboard for exploring:
- Code evaluation results (correctness, edit distance, failure modes)
- Developer telemetry metrics (acceptance rates, latency, productivity)
- Model version comparisons (A/B testing)
- Language and task type breakdowns
- Failure mode diagnostics

This demonstrates the kind of self-service analytics dashboards
a Codex Data Scientist would build for PM, Eng, and Design teams.
"""

import streamlit as st
import pandas as pd
import json
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from datetime import datetime

# Page config
st.set_page_config(
    page_title="Codex DS Showcase Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

ROOT = Path(__file__).parent.parent


@st.cache_data
def load_evaluation_results():
    """Load code evaluation results."""
    eval_path = ROOT / "code-evaluation-pipeline" / "code_eval_results.json"
    if not eval_path.exists():
        return None
    with open(eval_path, "r") as f:
        return json.load(f)


@st.cache_data
def load_telemetry():
    """Load telemetry events."""
    telemetry_path = ROOT / "developer-telemetry-simulation" / "telemetry_events.csv"
    if not telemetry_path.exists():
        return None
    return pd.read_csv(telemetry_path)


def main():
    st.title("📊 Codex DS Showcase Dashboard")
    st.markdown("""
    This dashboard demonstrates key metrics and analyses for AI developer tools:
    - **Code Evaluation Metrics**: Correctness, edit distance, failure modes
    - **Developer Productivity**: Acceptance rates, latency, session metrics
    - **Model Comparisons**: A/B testing results across model versions
    - **Segmentation**: Breakdowns by language, user segment, task type
    """)

    # Sidebar navigation
    page = st.sidebar.selectbox(
        "Navigation",
        ["Overview", "Code Evaluation", "Telemetry Analysis", "Model Comparison", "Failure Diagnostics"]
    )

    if page == "Overview":
        show_overview()
    elif page == "Code Evaluation":
        show_code_evaluation()
    elif page == "Telemetry Analysis":
        show_telemetry_analysis()
    elif page == "Model Comparison":
        show_model_comparison()
    elif page == "Failure Diagnostics":
        show_failure_diagnostics()


def show_overview():
    """Overview page with key metrics."""
    st.header("📈 Overview Metrics")

    eval_results = load_evaluation_results()
    telemetry_df = load_telemetry()

    col1, col2, col3, col4 = st.columns(4)

    # Code evaluation metrics
    if eval_results:
        pass_rate = sum(1 for r in eval_results if r.get("status") == "passed") / len(eval_results)
        avg_edit_dist = sum(r.get("edit_distance", 0) for r in eval_results if r.get("edit_distance")) / max(1, sum(1 for r in eval_results if r.get("edit_distance")))
        
        col1.metric("Code Pass Rate", f"{pass_rate:.1%}")
        col2.metric("Avg Edit Distance", f"{avg_edit_dist:.1f}")
    else:
        col1.info("Run code evaluation first")
        col2.info("Run code evaluation first")

    # Telemetry metrics
    if telemetry_df is not None:
        acceptance_rate = telemetry_df["accepted"].mean()
        avg_latency = telemetry_df["latency_ms"].mean()
        
        col3.metric("Acceptance Rate", f"{acceptance_rate:.1%}")
        col4.metric("Avg Latency", f"{avg_latency:.0f}ms")
    else:
        col3.info("Run telemetry simulation first")
        col4.info("Run telemetry simulation first")

    st.markdown("---")

    # Quick charts
    if telemetry_df is not None:
        st.subheader("Acceptance Rate by Model Version")
        fig = px.bar(
            telemetry_df.groupby("model_version")["accepted"].mean().reset_index(),
            x="model_version",
            y="accepted",
            labels={"accepted": "Acceptance Rate", "model_version": "Model Version"},
            color="model_version"
        )
        st.plotly_chart(fig, use_container_width=True)


def show_code_evaluation():
    """Code evaluation results page."""
    st.header("🔍 Code Evaluation Results")

    eval_results = load_evaluation_results()
    if not eval_results:
        st.warning("No evaluation results found. Run the code evaluation pipeline first.")
        st.code("python app.py generate\npython app.py evaluate", language="bash")
        return

    df = pd.DataFrame(eval_results)

    # Summary metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Tasks", len(df))
    col2.metric("Passed", sum(df["status"] == "passed"))
    col3.metric("Failed", sum(df["status"] != "passed"))

    st.markdown("---")

    # Status breakdown
    st.subheader("Status Breakdown")
    status_counts = df["status"].value_counts()
    fig = px.pie(
        values=status_counts.values,
        names=status_counts.index,
        title="Evaluation Status Distribution"
    )
    st.plotly_chart(fig, use_container_width=True)

    # Edit distance analysis
    if "edit_distance" in df.columns:
        st.subheader("Edit Distance Analysis")
        edit_dist_df = df[df["edit_distance"].notna()]
        if len(edit_dist_df) > 0:
            fig = px.histogram(
                edit_dist_df,
                x="edit_distance",
                nbins=20,
                title="Distribution of Edit Distances",
                labels={"edit_distance": "Edit Distance (Levenshtein)", "count": "Count"}
            )
            st.plotly_chart(fig, use_container_width=True)

            # Edit distance by status
            fig2 = px.box(
                edit_dist_df,
                x="status",
                y="edit_distance",
                title="Edit Distance by Status"
            )
            st.plotly_chart(fig2, use_container_width=True)

    # Detailed results table
    st.subheader("Detailed Results")
    st.dataframe(df, use_container_width=True)


def show_telemetry_analysis():
    """Telemetry analysis page."""
    st.header("📡 Developer Telemetry Analysis")

    telemetry_df = load_telemetry()
    if telemetry_df is None:
        st.warning("No telemetry data found. Run the telemetry simulation first.")
        st.code("python app.py simulate", language="bash")
        return

    # Filters
    st.sidebar.subheader("Filters")
    selected_languages = st.sidebar.multiselect(
        "Languages",
        options=telemetry_df["language"].unique(),
        default=telemetry_df["language"].unique()
    )
    selected_segments = st.sidebar.multiselect(
        "User Segments",
        options=telemetry_df["user_segment"].unique(),
        default=telemetry_df["user_segment"].unique()
    )

    # Apply filters
    filtered_df = telemetry_df[
        (telemetry_df["language"].isin(selected_languages)) &
        (telemetry_df["user_segment"].isin(selected_segments))
    ]

    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Suggestions", len(filtered_df))
    col2.metric("Acceptance Rate", f"{filtered_df['accepted'].mean():.1%}")
    col3.metric("Avg Latency", f"{filtered_df['latency_ms'].mean():.0f}ms")
    col4.metric("Compile Success", f"{filtered_df['compile_success'].mean():.1%}")

    st.markdown("---")

    # Acceptance rate by language
    st.subheader("Acceptance Rate by Language")
    lang_acceptance = filtered_df.groupby("language")["accepted"].mean().reset_index()
    fig = px.bar(
        lang_acceptance,
        x="language",
        y="accepted",
        labels={"accepted": "Acceptance Rate", "language": "Language"},
        title="Acceptance Rate by Programming Language"
    )
    st.plotly_chart(fig, use_container_width=True)

    # Latency distribution
    st.subheader("Latency Distribution")
    fig = px.histogram(
        filtered_df,
        x="latency_ms",
        nbins=50,
        title="Distribution of Suggestion Latency",
        labels={"latency_ms": "Latency (ms)", "count": "Count"}
    )
    st.plotly_chart(fig, use_container_width=True)

    # Acceptance by user segment
    st.subheader("Acceptance Rate by User Segment")
    segment_acceptance = filtered_df.groupby("user_segment")["accepted"].mean().reset_index()
    fig = px.bar(
        segment_acceptance,
        x="user_segment",
        y="accepted",
        labels={"accepted": "Acceptance Rate", "user_segment": "User Segment"},
        title="Acceptance Rate by Developer Experience Level"
    )
    st.plotly_chart(fig, use_container_width=True)

    # Latency vs Acceptance
    st.subheader("Latency vs Acceptance")
    fig = px.scatter(
        filtered_df.sample(min(1000, len(filtered_df))),  # Sample for performance
        x="latency_ms",
        y="accepted",
        color="model_version",
        trendline="ols",
        labels={"latency_ms": "Latency (ms)", "accepted": "Accepted (0/1)"},
        title="Relationship Between Latency and Acceptance"
    )
    st.plotly_chart(fig, use_container_width=True)


def show_model_comparison():
    """Model version comparison (A/B testing)."""
    st.header("🔬 Model Version Comparison (A/B Test)")

    telemetry_df = load_telemetry()
    if telemetry_df is None:
        st.warning("No telemetry data found. Run the telemetry simulation first.")
        return

    # Key comparison metrics
    comparison = telemetry_df.groupby("model_version").agg({
        "accepted": ["mean", "count"],
        "latency_ms": "mean",
        "compile_success": "mean",
        "test_pass": "mean"
    }).round(3)

    st.subheader("Key Metrics Comparison")
    st.dataframe(comparison, use_container_width=True)

    # Statistical comparison
    from scipy import stats
    import numpy as np

    v1_accepted = telemetry_df[telemetry_df["model_version"] == "model_v1"]["accepted"]
    v2_accepted = telemetry_df[telemetry_df["model_version"] == "model_v2"]["accepted"]

    if len(v1_accepted) > 0 and len(v2_accepted) > 0:
        # Chi-square test for acceptance rate
        contingency = pd.crosstab(telemetry_df["model_version"], telemetry_df["accepted"])
        chi2, p_value = stats.chi2_contingency(contingency)[:2]

        st.subheader("Statistical Significance Test")
        col1, col2 = st.columns(2)
        col1.metric("Chi-square statistic", f"{chi2:.3f}")
        col2.metric("P-value", f"{p_value:.4f}")
        
        if p_value < 0.05:
            st.success("✅ Statistically significant difference (p < 0.05)")
        else:
            st.info("ℹ️ No statistically significant difference (p >= 0.05)")

    # Visual comparison
    st.subheader("Side-by-Side Comparison")
    
    metrics_to_compare = ["accepted", "compile_success", "test_pass"]
    fig = go.Figure()
    
    for metric in metrics_to_compare:
        v1_mean = telemetry_df[telemetry_df["model_version"] == "model_v1"][metric].mean()
        v2_mean = telemetry_df[telemetry_df["model_version"] == "model_v2"][metric].mean()
        
        fig.add_trace(go.Bar(
            name=f"model_v1",
            x=[metric],
            y=[v1_mean],
            showlegend=(metric == metrics_to_compare[0])
        ))
        fig.add_trace(go.Bar(
            name=f"model_v2",
            x=[metric],
            y=[v2_mean],
            showlegend=(metric == metrics_to_compare[0])
        ))
    
    fig.update_layout(
        title="Model Comparison: Key Metrics",
        xaxis_title="Metric",
        yaxis_title="Rate",
        barmode="group"
    )
    st.plotly_chart(fig, use_container_width=True)


def show_failure_diagnostics():
    """Failure mode analysis."""
    st.header("🔧 Failure Diagnostics")

    telemetry_df = load_telemetry()
    eval_results = load_evaluation_results()

    if telemetry_df is not None:
        st.subheader("Error Type Distribution")
        error_counts = telemetry_df["error_type"].value_counts()
        fig = px.pie(
            values=error_counts.values,
            names=error_counts.index,
            title="Distribution of Error Types"
        )
        st.plotly_chart(fig, use_container_width=True)

        # Error type by model version
        st.subheader("Error Types by Model Version")
        error_by_model = pd.crosstab(telemetry_df["model_version"], telemetry_df["error_type"])
        fig = px.bar(
            error_by_model.reset_index(),
            x="model_version",
            y=error_by_model.columns.tolist(),
            title="Error Type Breakdown by Model",
            barmode="stack"
        )
        st.plotly_chart(fig, use_container_width=True)

        # Hallucination analysis
        st.subheader("Hallucination Analysis")
        hallucination_rate = telemetry_df["hallucination_flag"].mean()
        st.metric("Hallucination Rate", f"{hallucination_rate:.1%}")
        
        if hallucination_rate > 0:
            hallucination_by_model = telemetry_df.groupby("model_version")["hallucination_flag"].mean()
            fig = px.bar(
                hallucination_by_model.reset_index(),
                x="model_version",
                y="hallucination_flag",
                labels={"hallucination_flag": "Hallucination Rate"},
                title="Hallucination Rate by Model Version"
            )
            st.plotly_chart(fig, use_container_width=True)

    if eval_results:
        st.subheader("Code Evaluation Failures")
        df = pd.DataFrame(eval_results)
        failures = df[df["status"] != "passed"]
        
        if len(failures) > 0:
            st.dataframe(failures[["task_id", "status", "error", "edit_distance"]], use_container_width=True)
        else:
            st.success("✅ No failures in code evaluation!")


if __name__ == "__main__":
    main()
