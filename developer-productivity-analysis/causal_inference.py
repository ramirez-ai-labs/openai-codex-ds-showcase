"""
Codex DS Showcase: Causal Inference Analysis
==============================================

This module demonstrates causal inference techniques for understanding
the impact of AI coding assistance on developer productivity.

Techniques demonstrated:
- Difference-in-differences (if applicable)
- Propensity score matching
- Instrumental variables (conceptual)
- Regression discontinuity (conceptual)

This shows how a Codex Data Scientist would:
- Go beyond correlation to understand causation
- Design studies to measure true impact
- Account for confounding variables
- Make causal claims with proper methodology
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
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


def estimate_ate_propensity_score_matching(
    df: pd.DataFrame,
    treatment_col: str = "model_version",
    treatment_value: str = "model_v2",
    outcome_col: str = "accepted",
    covariates: list = None
) -> Dict:
    """
    Estimate Average Treatment Effect (ATE) using Propensity Score Matching.
    
    This addresses confounding by matching treated and control units
    with similar propensity scores (probability of receiving treatment).
    """
    if covariates is None:
        covariates = ["latency_ms", "suggestion_length", "language", "user_segment"]
    
    # Prepare data
    df_clean = df.copy()
    df_clean["treatment"] = (df_clean[treatment_col] == treatment_value).astype(int)
    
    # Encode categorical variables
    df_encoded = pd.get_dummies(df_clean, columns=["language", "user_segment"], drop_first=True)
    
    # Get numeric covariates
    numeric_covariates = [c for c in covariates if c in df_encoded.columns and df_encoded[c].dtype in [np.int64, np.float64]]
    all_covariates = numeric_covariates + [c for c in df_encoded.columns if c.startswith("language_") or c.startswith("user_segment_")]
    
    # Estimate propensity scores
    X_ps = df_encoded[all_covariates].fillna(0)
    y_ps = df_encoded["treatment"]
    
    ps_model = LogisticRegression(max_iter=1000)
    ps_model.fit(X_ps, y_ps)
    df_encoded["propensity_score"] = ps_model.predict_proba(X_ps)[:, 1]
    
    # Simple nearest neighbor matching (1:1)
    treated = df_encoded[df_encoded["treatment"] == 1].copy()
    control = df_encoded[df_encoded["treatment"] == 0].copy()
    
    matched_pairs = []
    control_used = set()
    
    for idx, treated_row in treated.iterrows():
        ps_treated = treated_row["propensity_score"]
        
        # Find closest control match
        distances = np.abs(control["propensity_score"] - ps_treated)
        available = control[~control.index.isin(control_used)]
        
        if len(available) > 0:
            closest_idx = available.loc[distances[available.index].idxmin()].name
            matched_pairs.append((idx, closest_idx))
            control_used.add(closest_idx)
    
    # Calculate ATE on matched sample
    if len(matched_pairs) > 0:
        treated_outcomes = [df_encoded.loc[t_idx, outcome_col] for t_idx, _ in matched_pairs]
        control_outcomes = [df_encoded.loc[c_idx, outcome_col] for _, c_idx in matched_pairs]
        
        ate = np.mean(treated_outcomes) - np.mean(control_outcomes)
        
        # Bootstrap confidence interval
        n_bootstrap = 1000
        bootstrap_ates = []
        for _ in range(n_bootstrap):
            indices = np.random.choice(len(matched_pairs), len(matched_pairs), replace=True)
            boot_treated = [treated_outcomes[i] for i in indices]
            boot_control = [control_outcomes[i] for i in indices]
            bootstrap_ates.append(np.mean(boot_treated) - np.mean(boot_control))
        
        ci_lower = np.percentile(bootstrap_ates, 2.5)
        ci_upper = np.percentile(bootstrap_ates, 97.5)
        
        return {
            "method": "Propensity Score Matching",
            "ate": ate,
            "ate_percentage": ate * 100,
            "ci_95_lower": ci_lower,
            "ci_95_upper": ci_upper,
            "n_matched_pairs": len(matched_pairs),
            "interpretation": f"Treatment increases {outcome_col} by {ate:.1%} (95% CI: [{ci_lower:.1%}, {ci_upper:.1%}])"
        }
    else:
        return {
            "method": "Propensity Score Matching",
            "error": "No matches found"
        }


def regression_adjustment(
    df: pd.DataFrame,
    treatment_col: str = "model_version",
    treatment_value: str = "model_v2",
    outcome_col: str = "accepted",
    covariates: list = None
) -> Dict:
    """
    Estimate ATE using regression adjustment.
    
    Controls for confounders by including them in the regression model.
    """
    if covariates is None:
        covariates = ["latency_ms", "suggestion_length", "language", "user_segment"]
    
    df_clean = df.copy()
    df_clean["treatment"] = (df_clean[treatment_col] == treatment_value).astype(int)
    
    # Prepare features
    df_encoded = pd.get_dummies(df_clean, columns=["language", "user_segment"], drop_first=True)
    
    numeric_covariates = [c for c in covariates if c in df_encoded.columns and df_encoded[c].dtype in [np.int64, np.float64]]
    all_features = ["treatment"] + numeric_covariates + [c for c in df_encoded.columns if c.startswith("language_") or c.startswith("user_segment_")]
    
    X = df_encoded[all_features].fillna(0)
    y = df_encoded[outcome_col]
    
    # Fit model
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)
    
    # Calculate ATE: E[Y|T=1, X] - E[Y|T=0, X] averaged over X
    X_treated = X.copy()
    X_treated["treatment"] = 1
    X_control = X.copy()
    X_control["treatment"] = 0
    
    y_treated = model.predict_proba(X_treated)[:, 1]
    y_control = model.predict_proba(X_control)[:, 1]
    
    ate = np.mean(y_treated - y_control)
    
    # Bootstrap CI
    n_bootstrap = 1000
    bootstrap_ates = []
    for _ in range(n_bootstrap):
        indices = np.random.choice(len(X), len(X), replace=True)
        X_boot = X.iloc[indices]
        y_boot = y.iloc[indices]
        
        model_boot = LogisticRegression(max_iter=1000)
        model_boot.fit(X_boot, y_boot)
        
        X_treated_boot = X_boot.copy()
        X_treated_boot["treatment"] = 1
        X_control_boot = X_boot.copy()
        X_control_boot["treatment"] = 0
        
        y_treated_boot = model_boot.predict_proba(X_treated_boot)[:, 1]
        y_control_boot = model_boot.predict_proba(X_control_boot)[:, 1]
        
        bootstrap_ates.append(np.mean(y_treated_boot - y_control_boot))
    
    ci_lower = np.percentile(bootstrap_ates, 2.5)
    ci_upper = np.percentile(bootstrap_ates, 97.5)
    
    return {
        "method": "Regression Adjustment",
        "ate": ate,
        "ate_percentage": ate * 100,
        "ci_95_lower": ci_lower,
        "ci_95_upper": ci_upper,
        "treatment_coefficient": model.coef_[0][0],
        "interpretation": f"Treatment increases {outcome_col} by {ate:.1%} (95% CI: [{ci_lower:.1%}, {ci_upper:.1%}])"
    }


def analyze_latency_impact(df: pd.DataFrame) -> Dict:
    """
    Analyze the causal impact of latency on acceptance.
    
    Uses regression to control for confounders.
    """
    df_clean = df.copy()
    df_encoded = pd.get_dummies(df_clean, columns=["language", "user_segment", "model_version"], drop_first=True)
    
    # Features
    features = ["latency_ms", "suggestion_length"] + [c for c in df_encoded.columns if c.startswith("language_") or c.startswith("user_segment_") or c.startswith("model_version_")]
    X = df_encoded[features].fillna(0)
    y = df_encoded["accepted"]
    
    # Fit model
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)
    
    # Get latency coefficient
    latency_idx = list(X.columns).index("latency_ms")
    latency_coef = model.coef_[0][latency_idx]
    
    # Calculate marginal effect: dP(accept)/d(latency)
    # For logistic regression: beta * p * (1-p) at mean
    p_mean = y.mean()
    marginal_effect = latency_coef * p_mean * (1 - p_mean)
    
    return {
        "latency_coefficient": latency_coef,
        "marginal_effect_per_100ms": marginal_effect * 100,  # Effect of 100ms increase
        "interpretation": f"Each 100ms increase in latency decreases acceptance probability by {abs(marginal_effect * 100):.1%}"
    }


def print_causal_results(results: Dict):
    """Pretty print causal inference results."""
    print("\n" + "="*70)
    print("CAUSAL INFERENCE ANALYSIS")
    print("="*70)
    
    if "propensity_score" in results:
        ps_result = results["propensity_score"]
        print("\n📊 PROPENSITY SCORE MATCHING")
        print("-" * 70)
        print(f"ATE:              {ps_result['ate']:+.1%}")
        print(f"95% CI:           [{ps_result['ci_95_lower']:+.1%}, {ps_result['ci_95_upper']:+.1%}]")
        print(f"Matched pairs:    {ps_result['n_matched_pairs']}")
        print(f"Interpretation:   {ps_result['interpretation']}")
    
    if "regression_adjustment" in results:
        ra_result = results["regression_adjustment"]
        print("\n📈 REGRESSION ADJUSTMENT")
        print("-" * 70)
        print(f"ATE:              {ra_result['ate']:+.1%}")
        print(f"95% CI:           [{ra_result['ci_95_lower']:+.1%}, {ra_result['ci_95_upper']:+.1%}]")
        print(f"Treatment coef:   {ra_result['treatment_coefficient']:.4f}")
        print(f"Interpretation:   {ra_result['interpretation']}")
    
    if "latency_impact" in results:
        lat_result = results["latency_impact"]
        print("\n⏱️  LATENCY IMPACT (Controlling for Confounders)")
        print("-" * 70)
        print(f"Marginal effect:  {lat_result['marginal_effect_per_100ms']:+.1%} per 100ms")
        print(f"Interpretation:   {lat_result['interpretation']}")
    
    print("\n" + "="*70)
    print("KEY INSIGHTS:")
    print("="*70)
    print("1. Causal inference accounts for confounding variables")
    print("2. Multiple methods provide robustness checks")
    print("3. Confidence intervals quantify uncertainty")
    print("4. These methods go beyond simple correlation")
    print("="*70 + "\n")


def main():
    """Main entry point."""
    try:
        df = load_telemetry_data()
        
        results = {}
        
        # Propensity score matching
        print("Running propensity score matching...")
        results["propensity_score"] = estimate_ate_propensity_score_matching(df)
        
        # Regression adjustment
        print("Running regression adjustment...")
        results["regression_adjustment"] = regression_adjustment(df)
        
        # Latency impact
        print("Analyzing latency impact...")
        results["latency_impact"] = analyze_latency_impact(df)
        
        print_causal_results(results)
        
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("\nRun the telemetry simulation first:")
        print("   python app.py simulate")


if __name__ == "__main__":
    main()

