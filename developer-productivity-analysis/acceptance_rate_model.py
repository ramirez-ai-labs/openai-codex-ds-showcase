# path: developer-productivity-analysis/acceptance_rate_model.py

"""
Acceptance Rate Prediction Model for AI Code Suggestions
=======================================================

Beginner-friendly machine learning pipeline that predicts whether a developer
will accept an AI code suggestion. Includes:

- Preprocessing (scaling + one-hot encoding)
- Logistic Regression classifier
- Evaluation metrics
- Feature importance analysis

This mirrors real-world analysis done by Data Scientists working on AI developer tools.
"""

from pathlib import Path
import pandas as pd
import numpy as np

# Machine learning imports
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    classification_report,
    confusion_matrix,
)

# Optional plotting
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOT_AVAILABLE = True
except ImportError:
    PLOT_AVAILABLE = False


# ===============================================================
# Load Data
# ===============================================================

def load_data(csv_path: str = None) -> pd.DataFrame:
    """Load telemetry dataset from CSV."""
    if csv_path is None:
        root = Path(__file__).parent.parent
        csv_path = root / "developer-telemetry-simulation" / "telemetry_events.csv"

    print(f"📂 Loading data from: {csv_path}")
    return pd.read_csv(csv_path)


# ===============================================================
# Preprocessing
# ===============================================================

def preprocess_data(df: pd.DataFrame):
    """
    Prepare features for machine learning.

    - Numeric values get scaled
    - Categorical values get one-hot encoded
    - Target is the `accepted` column
    """

    features = [
        "model_version",
        "latency_ms",
        "suggestion_length",
        "final_code_length",
        "user_segment",
        "time_of_day",
        "file_extension",
    ]

    features = [f for f in features if f in df.columns]

    categorical_features = [
        f for f in ["model_version", "user_segment", "time_of_day", "file_extension"]
        if f in features
    ]

    numerical_features = [f for f in features if f not in categorical_features]

    # Numerical features ➝ scale them
    numeric_transformer = Pipeline(steps=[
        ("scaler", StandardScaler())
    ])

    # Categorical features ➝ one-hot encode them
    # Compact explanation:
    # --------------------
    # OneHotEncoder converts categories like "junior", "senior", "python"
    # into separate 0/1 columns so ML models can learn from them.
    categorical_transformer = Pipeline(steps=[
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numerical_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    X = df[features]
    y = df["accepted"].astype(int)

    return X, y, preprocessor


# ===============================================================
# Model Training
# ===============================================================

def train_model(X, y, preprocessor):
    """
    Train a Logistic Regression model using a clean ML pipeline.

    Compact beginner explanation:
    -----------------------------
    - We split data so we can test the model on unseen examples.
    - Pipeline ensures preprocessing happens automatically.
    - Logistic Regression outputs probabilities → great for “likelihood of acceptance”.
    """

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42,
        )),
    ])

    print("🧠 Training Logistic Regression model...")
    model.fit(X_train, y_train)

    return model, X_test, y_test


# ===============================================================
# Model Evaluation
# ===============================================================

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the trained model.

    Compact beginner explanations of each metric:
    --------------------------------------------

    ✔ Precision, Recall, F1  
        - Precision: Of predictions marked “accepted,” how many were correct?  
        - Recall: Of actual accepted suggestions, how many did we catch?  
        - F1: Balance between precision + recall.

    ✔ Confusion Matrix  
        Shows four outcomes:
        - True Positive (TP)  → correct accept
        - False Positive (FP) → predicted accept but wrong
        - True Negative (TN)  → correct reject
        - False Negative (FN) → missed a good suggestion

    ✔ AUC Score  
        Measures how well the model *ranks* suggestions from low→high acceptance likelihood.
        0.5 = random guessing, 1.0 = perfect ranking.

    ✔ Feature Importance  
        Shows which signals matter most (e.g., latency, model version).
        Higher absolute coefficients = stronger influence.
    """

    print("\n📊 Model Evaluation\n" + "-" * 60)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # 1. Classification Report
    print("\n📘 Classification Report (Precision, Recall, F1):")
    print(classification_report(y_test, y_pred))

    # 2. AUC Score
    auc = roc_auc_score(y_test, y_prob)
    print(f"\n📈 AUC Score (ranking ability): {auc:.3f}")

    # 3. Confusion Matrix
    print("\n🧮 Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # 4. Feature Importance
    try:
        preproc = model.named_steps["preprocessor"]
        classifier = model.named_steps["classifier"]

        feature_names = []
        for name, transformer, cols in preproc.transformers_:
            if name == "num":
                feature_names.extend(cols)
            elif name == "cat":
                feature_names.extend(
                    transformer.named_steps["onehot"].get_feature_names_out(cols)
                )

        coefs = classifier.coef_[0]

        importance_df = (
            pd.DataFrame({"feature": feature_names, "importance": np.abs(coefs)})
            .sort_values("importance", ascending=False)
        )

        print("\n⭐ Top Influential Features:")
        print(importance_df.head(10).to_string(index=False))

    except Exception as e:
        print(f"\n⚠ Feature importance unavailable: {e}")


# ===============================================================
# Optional Plot
# ===============================================================

def plot_feature_importance(model, feature_names):
    """Visualize top features (optional)."""
    if not PLOT_AVAILABLE:
        print("Plotting skipped — matplotlib not installed.")
        return

    coef = model.named_steps["classifier"].coef_[0]
    df = pd.DataFrame({"Feature": feature_names, "Importance": np.abs(coef)})
    df = df.sort_values("Importance", ascending=False).head(10)

    plt.figure(figsize=(10, 5))
    sns.barplot(data=df, x="Importance", y="Feature")
    plt.title("Top 10 Important Features")
    plt.tight_layout()
    plt.show()


# ===============================================================
# Main Runner
# ===============================================================

def main():
    print("\n=== Acceptance Rate Prediction Pipeline ===")

    df = load_data()
    X, y, preprocessor = preprocess_data(df)
    model, X_test, y_test = train_model(X, y, preprocessor)
    evaluate_model(model, X_test, y_test)

    print("\n🎉 Model training and evaluation complete!\n")


if __name__ == "__main__":
    main()
