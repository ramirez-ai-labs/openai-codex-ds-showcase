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
    Performs a Chi-square test of independence between model version and a binary metric.
    
    This test determines if there's a statistically significant association between 
    the model version (v1 vs v2) and the outcome (e.g., accepted/rejected).
    
    Hypotheses:
    - H0 (Null Hypothesis): No association between model version and the outcome
    - H1 (Alternative Hypothesis): There is an association between model version and outcome
    
    The test works by comparing observed frequencies to expected frequencies if H0 were true.
    A low p-value (<0.05) suggests we reject H0 in favor of H1.
    
    Args:
        df: DataFrame containing the data
        metric: Name of the binary metric column to test (default: "accepted")
        
    Returns:
        Dictionary containing test results and interpretation
    """
    # Create a contingency table (cross-tabulation) of model versions vs metric outcomes
    # Rows: model versions (v1, v2)
    # Columns: metric outcomes (e.g., True/False for accepted/rejected)
    contingency = pd.crosstab(df["model_version"], df[metric])
    
    # Perform the chi-square test for independence
    # chi2: The test statistic (larger values indicate greater difference from expected)
    # p_value: Probability of observing the data if H0 is true (small p-value suggests H0 is unlikely)
    # dof: Degrees of freedom (depends on table dimensions: (rows-1)*(cols-1))
    # expected: The expected frequencies under H0 (for reference)
    chi2, p_value, dof, expected = stats.chi2_contingency(contingency)
    
    return {
        "test": "Chi-square Test of Independence",
        "chi2_statistic": chi2,
        "p_value": p_value,
        "degrees_of_freedom": dof,
        "significant": p_value < 0.05,  # Using standard 5% significance level
        "interpretation": (
            "Statistically significant difference between model versions (p < 0.05)" 
            if p_value < 0.05 
            else "No statistically significant difference found between model versions"
        )
    }


def t_test_continuous(df: pd.DataFrame, metric: str = "latency_ms") -> Dict:
    """
    Performs a two-sample t-test to compare means between two independent groups.
    
    This test determines if there's a statistically significant difference between the means
    of a continuous metric (e.g., latency, code length) between two model versions.
    
    When to use:
    - Comparing means of a continuous variable between two independent groups
    - Data is approximately normally distributed
    - Variances between groups may be unequal (uses Welch's t-test)
    
    Hypotheses:
    - H0 (Null Hypothesis): The means of the two groups are equal
    - H1 (Alternative Hypothesis): The means of the two groups are different
    
    The test calculates a t-statistic and p-value. A low p-value (<0.05) suggests we reject H0.
    
    Args:
        df: DataFrame containing the data
        metric: Name of the continuous metric column to test (default: "latency_ms")
        
    Returns:
        Dictionary containing test results, effect size, and interpretation
    """
    # Split data into control (model_v1) and treatment (model_v2) groups
    control = df[df["model_version"] == "model_v1"][metric]
    treatment = df[df["model_version"] == "model_v2"][metric]
    
    # Perform Welch's t-test (does not assume equal variances between groups)
    # t_stat: The calculated t-statistic
    # p_value: Probability of observing the data if H0 is true
    t_stat, p_value = stats.ttest_ind(control, treatment, equal_var=False)
    
    # Calculate summary statistics
    control_mean = control.mean()
    treatment_mean = treatment.mean()
    
    # Calculate Cohen's d effect size: (mean1 - mean2) / pooled_std
    # This tells us how many standard deviations apart the means are
    effect_size = (treatment_mean - control_mean) / control.std()
    
    # Determine if the result is statistically significant (using 5% significance level)
    is_significant = p_value < 0.05
    
    # Create interpretation message
    if not is_significant:
        interpretation = "No statistically significant difference between groups"
    else:
        direction = "higher" if treatment_mean > control_mean else "lower"
        interpretation = f"Treatment group mean is {direction} than control group (p < 0.05)"
    
    return {
        "test": "Two-sample t-test (Welch's)",
        "t_statistic": t_stat,
        "p_value": p_value,
        "control_mean": control_mean,
        "treatment_mean": treatment_mean,
        "difference": treatment_mean - control_mean,
        "effect_size": effect_size,
        "significant": is_significant,
        "interpretation": interpretation,
        "effect_size_interpretation": get_effect_size_interpretation(abs(effect_size))
    }


def get_effect_size_interpretation(d: float) -> str:
    """Provides a qualitative interpretation of Cohen's d effect size."""
    if d >= 0.8:
        return "Large effect size"
    elif d >= 0.5:
        return "Medium effect size"
    elif d >= 0.2:
        return "Small effect size"
    return "Negligible effect size"


def proportion_test(df: pd.DataFrame, metric: str = "accepted") -> Dict:
    """
    Performs a two-proportion z-test to compare proportions between two independent groups.
    
    This test determines if there's a statistically significant difference between the 
    proportions of a binary outcome (e.g., success/failure) between two model versions.
    
    When to use:
    - Comparing proportions (percentages) between two independent groups
    - Data is binary (e.g., accepted/rejected, pass/fail, yes/no)
    - Sample sizes are large enough (np > 10 and n(1-p) > 10 for both groups)
    
    Hypotheses:
    - H0 (Null Hypothesis): The proportions are equal (p_control = p_treatment)
    - H1 (Alternative Hypothesis): The proportions are not equal (p_control ≠ p_treatment)
    
    The test calculates a z-statistic and p-value. A low p-value (<0.05) suggests we reject H0.
    
    Args:
        df: DataFrame containing the data
        metric: Name of the binary metric column to test (default: "accepted")
        
    Returns:
        Dictionary containing test results, confidence intervals, effect size, and interpretation
    """
    # Split data into control (model_v1) and treatment (model_v2) groups
    control = df[df["model_version"] == "model_v1"][metric]
    treatment = df[df["model_version"] == "model_v2"][metric]
    
    # Calculate sample sizes and proportions
    n_control = len(control)
    n_treatment = len(treatment)
    p_control = control.mean()  # Proportion in control group
    p_treatment = treatment.mean()  # Proportion in treatment group
    
    # Two-proportion z-test
    # 1. Calculate pooled proportion (assuming H0 is true)
    p_pooled = (control.sum() + treatment.sum()) / (n_control + n_treatment)
    
    # 2. Calculate standard error of the difference in proportions under H0
    se = np.sqrt(p_pooled * (1 - p_pooled) * (1/n_control + 1/n_treatment))
    
    # 3. Calculate z-statistic
    z_stat = (p_treatment - p_control) / se
    
    # 4. Calculate two-tailed p-value
    p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
    
    # Calculate Cohen's h effect size for proportions
    # This measures the difference between two proportions in a standardized way
    # h = 2 * (arcsin(√p1) - arcsin(√p2))
    # Values: 0.2=small, 0.5=medium, 0.8=large effect
    h = 2 * (np.arcsin(np.sqrt(p_treatment)) - np.arcsin(np.sqrt(p_control)))
    
    # Calculate 95% confidence interval for the difference in proportions
    # This gives us a range of plausible values for the true difference
    se_diff = np.sqrt(p_control * (1 - p_control) / n_control + 
                     p_treatment * (1 - p_treatment) / n_treatment)
    margin_of_error = 1.96 * se_diff  # 1.96 is the z-score for 95% CI
    ci_lower = (p_treatment - p_control) - margin_of_error
    ci_upper = (p_treatment - p_control) + margin_of_error
    
    # Determine if the result is statistically significant (using 5% significance level)
    is_significant = p_value < 0.05
    
    # Create interpretation message
    if not is_significant:
        interpretation = "No statistically significant difference in proportions"
    else:
        direction = "higher" if p_treatment > p_control else "lower"
        diff_pct = abs(p_treatment - p_control) * 100
        interpretation = f"Treatment proportion is {direction} by {diff_pct:.1f} percentage points (p < 0.05)"
    
    return {
        "test": "Two-proportion z-test",
        "z_statistic": z_stat,
        "p_value": p_value,
        "control_proportion": p_control,
        "treatment_proportion": p_treatment,
        "difference": p_treatment - p_control,
        "difference_ci": (ci_lower, ci_upper),
        "effect_size_cohens_h": h,
        "effect_size_interpretation": get_effect_size_interpretation(abs(h)),
        "significant": is_significant,
        "interpretation": interpretation,
        "n_control": n_control,
        "n_treatment": n_treatment
    }


def power_analysis(df: pd.DataFrame, metric: str = "accepted", alpha: float = 0.05) -> Dict:
    """
    Performs a post-hoc power analysis to evaluate the statistical power of an A/B test.
    
    Statistical power is the probability that the test will detect an effect if one exists.
    A higher power means a lower chance of a Type II error (false negative).
    
    Key Concepts:
    - Power: Probability of detecting an effect if it exists (1 - β, where β is Type II error rate)
    - Effect Size: Magnitude of the difference between groups (standardized)
    - Alpha (α): Significance level (probability of Type I error, default 0.05)
    - Sample Size: Number of observations in each group
    
    Rule of Thumb for Power:
    - 0.8 or higher: Good (80% chance to detect an effect if it exists)
    - 0.7-0.8: Acceptable but could be better
    - Below 0.7: Likely underpowered (high risk of missing real effects)
    
    Args:
        df: DataFrame containing the experimental data
        metric: Name of the metric column to analyze (default: "accepted")
        alpha: Significance level (default: 0.05)
        
    Returns:
        Dictionary containing power analysis results and interpretation
    """
    # Split data into control (model_v1) and treatment (model_v2) groups
    control = df[df["model_version"] == "model_v1"][metric]
    treatment = df[df["model_version"] == "model_v2"][metric]
    
    # Calculate sample sizes and proportions for each group
    n_control = len(control)
    n_treatment = len(treatment)
    p_control = control.mean()  # Proportion in control group
    p_treatment = treatment.mean()  # Proportion in treatment group
    
    # Calculate the pooled proportion (combined success rate)
    # This assumes the null hypothesis is true (no difference between groups)
    p_pooled = (control.sum() + treatment.sum()) / (n_control + n_treatment)
    
    # Calculate standard error of the difference in proportions
    # This measures the variability we'd expect by chance
    se = np.sqrt(p_pooled * (1 - p_pooled) * (1/n_control + 1/n_treatment))
    
    # Calculate the observed effect size (standardized difference)
    # This tells us how large the difference is in terms of standard errors
    effect_size = abs(p_treatment - p_control) / se
    
    # Calculate the critical z-value for our chosen alpha level (two-tailed test)
    # This is the threshold for statistical significance
    z_alpha = stats.norm.ppf(1 - alpha/2)  # Default: 1.96 for alpha=0.05
    
    # Calculate the z-score for power (z_beta)
    # This combines the effect size and sample sizes to estimate power
    z_beta = effect_size * np.sqrt(n_control * n_treatment / (n_control + n_treatment)) - z_alpha
    
    # Convert z_beta to a probability (power) using the standard normal CDF
    # This gives us the probability of detecting an effect if it exists
    power = stats.norm.cdf(z_beta)
    
    # Interpret the power level
    power_interpretation = (
        "Adequate (≥80% chance to detect real effects)" if power >= 0.8 else
        "Moderate (may miss some real effects)" if power >= 0.7 else
        "Low (high risk of missing real effects)"
    )
    
    return {
        "effect_size": effect_size,  # Standardized measure of the effect
        "effect_size_interpretation": get_effect_size_interpretation(effect_size),
        "sample_size_control": n_control,
        "sample_size_treatment": n_treatment,
        "total_sample_size": n_control + n_treatment,
        "observed_power": power,
        "alpha": alpha,  # Significance level (Type I error rate)
        "beta": 1 - power,  # Type II error rate
        "interpretation": (
            f"Power: {power:.1%} - {power_interpretation}. "
            f"With the current sample size and effect size, you have a {power:.1%} "
            f"chance of detecting a real effect if it exists."
        ),
        "recommendation": (
            "Consider increasing sample size for more reliable results."
            if power < 0.8 else
            "Adequate power for detecting the observed effect size."
        )
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

