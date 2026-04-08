import sys
from pathlib import Path
import polars as pl
import xgboost as xgb
import shap
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import OrdinalEncoder
from loguru import logger

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from utils import setup_logging, init_polars


def run_explainability_suite(data_path: str = "data/raw/credit_risk.csv"):
    """Generate SHAP-based global and local explanations for the risk model."""
    setup_logging()
    init_polars()

    logger.info("Initializing RiskLens Explainability Layer.")

    # 1. Prepare Data and Model (Using Top 5 factors as validated in Phase 3)
    df = pl.read_csv(data_path)
    df = df.with_columns([((pl.col("default") == 2).cast(pl.Int8)).alias("target")])

    features = [
        "checking_balance",
        "credit_history",
        "months_loan_duration",
        "amount",
        "savings_balance",
    ]
    X = df.select(features).to_pandas()
    y = df["target"].to_numpy()

    # Encoding
    encoder = OrdinalEncoder()
    X_encoded = encoder.fit_transform(X)
    X_df = pl.DataFrame(X_encoded, schema=features).to_pandas()

    # Train validation model
    model = xgb.XGBClassifier(n_estimators=100, max_depth=4, random_state=42)
    model.fit(X_df, y)

    # 2. SHAP Analysis
    logger.info("Computing SHAP values (TreeExplainer)...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_df)

    # 3. Global Interpretation: Summary Plot
    logger.info("Generating Global Summary Plot...")
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_df, show=False)
    plt.title("RiskLens Global Factor Impact (SHAP)", fontsize=14, pad=20)

    summary_path = "reports/figures/shap_summary.png"
    plt.savefig(summary_path, bbox_inches="tight", dpi=150)
    plt.close()
    logger.info(f"Global summary saved to {summary_path}")

    # 4. Local Interpretation: Waterfall Plot for a high-risk sample
    # Find a sample with high probability of default
    probs = model.predict_proba(X_df)[:, 1]
    high_risk_idx = probs.argmax()

    logger.info(
        f"Generating Local Explanation for high-risk sample (Index {high_risk_idx})..."
    )
    plt.figure(figsize=(10, 4))

    # In newer SHAP versions, waterfall expects an Explanation object
    # Or we can use the older force_plot or bar plot logic
    # For modern, we'll use the Explanation object pattern
    exp = shap.Explanation(
        values=shap_values[high_risk_idx],
        base_values=explainer.expected_value,
        data=X_df.iloc[high_risk_idx],
        feature_names=features,
    )

    shap.plots.waterfall(exp, show=False)
    plt.title(
        f"Local Default Driver Analysis: Sample {high_risk_idx}", fontsize=12, pad=20
    )

    local_path = "reports/figures/shap_local_high_risk.png"
    plt.savefig(local_path, bbox_inches="tight", dpi=150)
    plt.close()
    logger.info(f"Local explanation saved to {local_path}")

    print("\n" + "=" * 50)
    print("XAI SUITE COMPLETE")
    print("Reports generated: reports/figures/shap_summary.png")
    print("Reports generated: reports/figures/shap_local_high_risk.png")
    print("=" * 50)


if __name__ == "__main__":
    run_explainability_suite()
