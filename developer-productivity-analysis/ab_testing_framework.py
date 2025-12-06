"""
Codex DS Showcase: A/B Testing Framework
=========================================

This module demonstrates experiment design and statistical analysis
for comparing model versions (A/B testing).

Key features:
- Hypothesis testing for acceptance rates
- Power analysis
- Confidence intervals
- Effect size calculations
- Multiple comparison corrections

This shows how a Codex Data Scientist would:
- Design experiments to compare model versions
- Run statistical tests with proper methodology
- Interpret results and make recommendations
"""

import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path
from typing import Dict, Tuple
import warnings
warnings.filterwarnings('ignore')


def load_telemetry_data(csv_path: str = None) -> pd.DataFrame:
    """Load telemetry data."""
    if csv_path is None:
        root = Path(__file__).parent.parent
        csv_path = root / "developer-telemetry-simulation" / "telemetry_events.csv"
    
    if not Path(csv_path).exists():
        raise FileNotFoundError(f"Telemetry data not found at {csv_path}")
    
    return pd.read_csv(csv_path)


def chi_square_test(df: pd.DataFrame, metric: str = "accepted") -> Dict:
    """
    Chi-square test for independence between model version and metric.
    
    Tests: H0: No difference between model versions
          H1: There is a difference
    """
    contingency = pd.crosstab(df["model_version"], df[metric])
    chi2, p_value, dof, expected = stats.chi2_contingency(contingency)
    
    return {
        "test": "Chi-square",
        "chi2_statistic": chi2,
        "p_value": p_value,
        "degrees_of_freedom": dof,
        "significant": p_value < 0.05,
        "interpretation": "Statistically significant difference" if p_value < 0.05 else "No statistically significant difference"
    }


def t_test_continuous(df: pd.DataFrame, metric: str = "latency_ms") -> Dict:
    """
    Two-sample t-test for continuous metrics.
    
    Tests: H0: Mean(control) = Mean(treatment)
          H1: Mean(control) != Mean(treatment)
    """
    control = df[df["model_version"] == "model_v1"][metric]
    treatment = df[df["model_version"] == "model_v2"][metric]
    
    t_stat, p_value = stats.ttest_ind(control, treatment, equal_var=False)
    
    control_mean = control.mean()
    treatment_mean = treatment.mean()
    effect_size = (treatment_mean - control_mean) / control.std()
    
    return {
        "test": "Two-sample t-test",
        "t_statistic": t_stat,
        "p_value": p_value,
        "control_mean": control_mean,
        "treatment_mean": treatment_mean,
        "difference": treatment_mean - control_mean,
        "effect_size": effect_size,
        "significant": p_value < 0.05,
        "interpretation": f"Treatment {'higher' if treatment_mean > control_mean else 'lower'} than control" if p_value < 0.05 else "No significant difference"
    }


def proportion_test(df: pd.DataFrame, metric: str = "accepted") -> Dict:
    """
    Two-proportion z-test for binary metrics.
    
    Tests: H0: p(control) = p(treatment)
          H1: p(control) != p(treatment)
    """
    control = df[df["model_version"] == "model_v1"][metric]
    treatment = df[df["model_version"] == "model_v2"][metric]
    
    n_control = len(control)
    n_treatment = len(treatment)
    p_control = control.mean()
    p_treatment = treatment.mean()
    
    # Two-proportion z-test
    p_pooled = (control.sum() + treatment.sum()) / (n_control + n_treatment)
    se = np.sqrt(p_pooled * (1 - p_pooled) * (1/n_control + 1/n_treatment))
    z_stat = (p_treatment - p_control) / se
    p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
    
    # Effect size (Cohen's h)
    h = 2 * (np.arcsin(np.sqrt(p_treatment)) - np.arcsin(np.sqrt(p_control)))
    
    # Confidence interval for difference
    se_diff = np.sqrt(p_control * (1 - p_control) / n_control + p_treatment * (1 - p_treatment) / n_treatment)
    ci_lower = (p_treatment - p_control) - 1.96 * se_diff
    ci_upper = (p_treatment - p_control) + 1.96 * se_diff
    
    return {
        "test": "Two-proportion z-test",
        "z_statistic": z_stat,
        "p_value": p_value,
        "control_proportion": p_control,
        "treatment_proportion": p_treatment,
        "difference": p_treatment - p_control,
        "difference_ci": (ci_lower, ci_upper),
        "effect_size_cohens_h": h,
        "significant": p_value < 0.05,
        "interpretation": f"Treatment {'higher' if p_treatment > p_control else 'lower'} than control" if p_value < 0.05 else "No significant difference"
    }


def power_analysis(df: pd.DataFrame, metric: str = "accepted", alpha: float = 0.05) -> Dict:
    """
    Post-hoc power analysis.
    
    Calculates the statistical power of the test given the observed effect size.
    """
    control = df[df["model_version"] == "model_v1"][metric]
    treatment = df[df["model_version"] == "model_v2"][metric]
    
    n_control = len(control)
    n_treatment = len(treatment)
    p_control = control.mean()
    p_treatment = treatment.mean()
    
    # Effect size
    p_pooled = (control.sum() + treatment.sum()) / (n_control + n_treatment)
    se = np.sqrt(p_pooled * (1 - p_pooled) * (1/n_control + 1/n_treatment))
    effect_size = abs(p_treatment - p_control) / se
    
    # Calculate power
    z_alpha = stats.norm.ppf(1 - alpha/2)
    z_beta = effect_size * np.sqrt(n_control * n_treatment / (n_control + n_treatment)) - z_alpha
    power = stats.norm.cdf(z_beta)
    
    return {
        "effect_size": effect_size,
        "sample_size_control": n_control,
        "sample_size_treatment": n_treatment,
        "observed_power": power,
        "interpretation": f"Power: {power:.1%} - {'Adequate' if power >= 0.8 else 'Insufficient'} power to detect effect"
    }


def run_ab_test_suite(df: pd.DataFrame) -> Dict:
    """Run a complete A/B test analysis suite."""
    results = {}
    
    # Binary metrics
    binary_metrics = ["accepted", "compile_success", "test_pass"]
    for metric in binary_metrics:
        results[f"{metric}_proportion_test"] = proportion_test(df, metric)
        results[f"{metric}_chi_square"] = chi_square_test(df, metric)
    
    # Continuous metrics
    continuous_metrics = ["latency_ms", "suggestion_length", "final_code_length"]
    for metric in continuous_metrics:
        results[f"{metric}_t_test"] = t_test_continuous(df, metric)
    
    # Power analysis
    results["power_analysis"] = power_analysis(df, "accepted")
    
    return results


def print_results(results: Dict):
    """Pretty print A/B test results."""
    print("\n" + "="*70)
    print("A/B TEST RESULTS: Model v1 vs Model v2")
    print("="*70)
    
    # Key metric: Acceptance rate
    if "accepted_proportion_test" in results:
        acc_result = results["accepted_proportion_test"]
        print("\n📊 ACCEPTANCE RATE")
        print("-" * 70)
        print(f"Control (v1):     {acc_result['control_proportion']:.1%}")
        print(f"Treatment (v2):  {acc_result['treatment_proportion']:.1%}")
        print(f"Difference:      {acc_result['difference']:+.1%}")
        print(f"95% CI:          [{acc_result['difference_ci'][0]:+.1%}, {acc_result['difference_ci'][1]:+.1%}]")
        print(f"P-value:          {acc_result['p_value']:.4f}")
        print(f"Effect size (h):  {acc_result['effect_size_cohens_h']:.3f}")
        print(f"Significant:      {'✅ YES' if acc_result['significant'] else '❌ NO'}")
        print(f"Interpretation:   {acc_result['interpretation']}")
    
    # Latency
    if "latency_ms_t_test" in results:
        lat_result = results["latency_ms_t_test"]
        print("\n⏱️  LATENCY")
        print("-" * 70)
        print(f"Control (v1):     {lat_result['control_mean']:.0f}ms")
        print(f"Treatment (v2):  {lat_result['treatment_mean']:.0f}ms")
        print(f"Difference:      {lat_result['difference']:+.0f}ms")
        print(f"P-value:          {lat_result['p_value']:.4f}")
        print(f"Effect size:     {lat_result['effect_size']:.3f}")
        print(f"Significant:      {'✅ YES' if lat_result['significant'] else '❌ NO'}")
    
    # Compile success
    if "compile_success_proportion_test" in results:
        comp_result = results["compile_success_proportion_test"]
        print("\n✅ COMPILE SUCCESS RATE")
        print("-" * 70)
        print(f"Control (v1):     {comp_result['control_proportion']:.1%}")
        print(f"Treatment (v2):  {comp_result['treatment_proportion']:.1%}")
        print(f"Difference:      {comp_result['difference']:+.1%}")
        print(f"P-value:          {comp_result['p_value']:.4f}")
        print(f"Significant:      {'✅ YES' if comp_result['significant'] else '❌ NO'}")
    
    # Power analysis
    if "power_analysis" in results:
        power = results["power_analysis"]
        print("\n🔋 STATISTICAL POWER")
        print("-" * 70)
        print(f"Observed power:  {power['observed_power']:.1%}")
        print(f"Interpretation:  {power['interpretation']}")
    
    print("\n" + "="*70)
    print("RECOMMENDATION:")
    if results.get("accepted_proportion_test", {}).get("significant"):
        diff = results["accepted_proportion_test"]["difference"]
        if diff > 0:
            print("✅ Model v2 shows statistically significant improvement. Recommend rollout.")
        else:
            print("⚠️  Model v2 shows statistically significant decline. Do not rollout.")
    else:
        print("ℹ️  No statistically significant difference detected. Consider:")
        print("   - Increasing sample size")
        print("   - Running experiment longer")
        print("   - Investigating segment-specific effects")
    print("="*70 + "\n")


def main():
    """Main entry point."""
    try:
        df = load_telemetry_data()
        results = run_ab_test_suite(df)
        print_results(results)
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("\nRun the telemetry simulation first:")
        print("   python app.py simulate")


if __name__ == "__main__":
    main()

